from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from langdetect import detect
from googleapiclient.discovery import build
from deep_translator import GoogleTranslator
import instaloader
import pytesseract
from PIL import Image
import sqlite3
import re
import io
from collections import Counter
from datetime import datetime

YOUTUBE_API_KEY = "AIzaSyCdKhoGPuEzuXGNxGTEu9D6i-TSIK6vrVE"

app = Flask(__name__)

print("Loading AI models... this happens once when the server starts. This will take a few minutes.")
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
credibility_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
image_detector = pipeline("image-classification", model="Organika/sdxl-detector")

print("All models loaded. Server ready.")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
insta_loader = instaloader.Instaloader()

SUPPORTED_INDIAN_LANGS = {"ta", "hi", "te", "ml", "kn"}


# ---------- Database ----------
def init_db():
    conn = sqlite3.connect("comments.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            author TEXT,
            text TEXT,
            toxicity REAL,
            sentiment TEXT,
            emotion TEXT,
            is_bot INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ---------- Translation ----------
def translate_to_english(text, lang_code):
    if lang_code not in SUPPORTED_INDIAN_LANGS:
        return None
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return None


# ---------- Bot / spam heuristics ----------
def is_bot_or_spam(text, all_texts_counter):
    reasons = []
    text_clean = text.strip().lower()

    if all_texts_counter[text_clean] >= 3:
        reasons.append("Duplicate comment posted multiple times")

    letters = sum(c.isalpha() for c in text)
    if len(text) > 5 and letters / max(len(text), 1) < 0.3:
        reasons.append("Mostly emojis / low text content")

    scam_patterns = ["dm me", "click the link", "whatsapp me", "telegram", "crypto", "investment", "giveaway", "won a prize", "claim your"]
    if any(p in text_clean for p in scam_patterns):
        reasons.append("Contains promotional/scam language")

    generic = ["first!!!", "nice video", "great content", "love this", "amazing"]
    if text_clean in generic:
        reasons.append("Generic low-effort engagement comment")

    return reasons


# ---------- Core analysis engine (shared by all sources) ----------
def analyze_comments_list(comments, source):
    text_counter = Counter(c["text"].strip().lower() for c in comments if c["text"].strip())
    conn = sqlite3.connect("comments.db")
    cur = conn.cursor()

    results = []
    for c in comments:
        text = c["text"]
        if not text.strip():
            continue

        try:
            lang = detect(text)
        except Exception:
            lang = "unknown"

        translated = translate_to_english(text, lang)
        text_for_ai = translated if translated else text

        # Limit comment length so the models don't choke on very long comments
        clean_text = text_for_ai[:250]

        toxic = toxicity_model(clean_text, truncation=True, max_length=256)[0]
        sentiment = sentiment_model(clean_text, truncation=True, max_length=256)[0]
        emotion = emotion_model(clean_text, truncation=True, max_length=256)[0]
        bot_reasons = is_bot_or_spam(text, text_counter)

        toxicity_score = round(toxic["score"] * 100, 2)

        results.append({
            "author": c["author"],
            "text": text,
            "translated_text": translated,
            "likes": c.get("likes", 0),
            "language": lang,
            "toxicity_label": toxic["label"],
            "toxicity_score": toxicity_score,
            "sentiment": sentiment["label"],
            "emotion": emotion["label"],
            "is_bot": len(bot_reasons) > 0,
            "bot_reasons": bot_reasons
        })

        cur.execute("""
            INSERT INTO comments (source, author, text, toxicity, sentiment, emotion, is_bot, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (source, c["author"], text, toxicity_score, sentiment["label"], emotion["label"], int(len(bot_reasons) > 0), datetime.now().isoformat()))

    conn.commit()
    conn.close()

    total = len(results)
    toxic_count = sum(1 for r in results if r["toxicity_label"].lower() == "toxic" and r["toxicity_score"] > 50)
    bot_count = sum(1 for r in results if r["is_bot"])
    negative_count = sum(1 for r in results if r["sentiment"].lower() == "negative")

    top_abusive = sorted(results, key=lambda r: r["toxicity_score"], reverse=True)[:10]
    top_bots = [r for r in results if r["is_bot"]][:10]

    for r in top_abusive[:3]:
        if r["toxicity_label"].lower() == "toxic" and r["toxicity_score"] > 40:
            r["rewritten"] = "Consider rephrasing this more respectfully — for example, focus on disagreeing with the idea rather than the person."
        else:
            r["rewritten"] = None

    bot_clusters = []
    seen_texts = set()
    for c in comments:
        t = c["text"].strip().lower()
        if text_counter[t] >= 3 and t not in seen_texts:
            seen_texts.add(t)
            authors = list(set(cc["author"] for cc in comments if cc["text"].strip().lower() == t))
            bot_clusters.append({"text": c["text"], "count": text_counter[t], "authors": authors[:10]})

    return {
        "total_comments": total,
        "toxic_percent": round((toxic_count / total) * 100, 2) if total else 0,
        "bot_percent": round((bot_count / total) * 100, 2) if total else 0,
        "negative_percent": round((negative_count / total) * 100, 2) if total else 0,
        "top_abusive": top_abusive,
        "top_bots": top_bots,
        "bot_clusters": bot_clusters[:5]
    }


def compute_trust_score(result):
    toxic_pct = result.get("toxic_percent", 0)
    bot_pct = result.get("bot_percent", 0)
    neg_pct = result.get("negative_percent", 0)

    penalty = (toxic_pct * 0.45) + (bot_pct * 0.30) + (neg_pct * 0.15)

    caption = result.get("caption_analysis")
    if caption and caption["credibility_label"] == "misinformation":
        penalty += caption["credibility_score"] * 0.10

    score = round(max(0, min(100, 100 - penalty)), 2)

    if score >= 70:
        badge = "Healthy"
    elif score >= 40:
        badge = "Moderate Risk"
    else:
        badge = "High Risk"

    result["trust_score"] = score
    result["trust_badge"] = badge
    return result


def analyze_caption(caption):
    if not caption or not caption.strip():
        return None

    credibility = credibility_model(caption[:512], candidate_labels=["credible content", "misinformation", "opinion"])
    ai_written = credibility_model(caption[:512], candidate_labels=["human-written text", "AI-generated text"])

    return {
        "text": caption,
        "credibility_label": credibility["labels"][0],
        "credibility_score": round(credibility["scores"][0] * 100, 2),
        "ai_written_label": ai_written["labels"][0],
        "ai_written_score": round(ai_written["scores"][0] * 100, 2)
    }


# ---------- YouTube ----------
def extract_video_id(url):
    for pattern in [r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", r"youtu\.be\/([0-9A-Za-z_-]{11})"]:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    url = request.get_json().get("url", "")
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Could not extract a valid YouTube video ID."}), 400

    try:
        video_response = youtube.videos().list(part="snippet", id=video_id).execute()
        description = video_response["items"][0]["snippet"]["description"] if video_response["items"] else ""

        comment_response = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100, textFormat="plainText", order="relevance"
        ).execute()
        comments = []
        for item in comment_response.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({"author": s["authorDisplayName"], "text": s["textDisplay"], "likes": s["likeCount"]})
    except Exception as e:
        return jsonify({"error": f"Could not fetch data: {str(e)}"}), 400

    if not comments:
        return jsonify({"error": "No comments found (or comments are disabled)."}), 400

    result = analyze_comments_list(comments, source="youtube")
    result["caption_analysis"] = analyze_caption(description)
    result = compute_trust_score(result)
    return jsonify(result)


# ---------- Instagram ----------
def extract_shortcode(url):
    match = re.search(r"instagram\.com\/(?:p|reel)\/([A-Za-z0-9_-]+)", url)
    return match.group(1) if match else None


@app.route("/analyze-instagram", methods=["POST"])
def analyze_instagram():
    url = request.get_json().get("url", "")
    shortcode = extract_shortcode(url)
    if not shortcode:
        return jsonify({"error": "Could not extract a valid Instagram post code from that link."}), 400

    try:
        post = instaloader.Post.from_shortcode(insta_loader.context, shortcode)
        comments = []
        for i, comment in enumerate(post.get_comments()):
            if i >= 100:
                break
            comments.append({"author": comment.owner.username, "text": comment.text, "likes": getattr(comment, "likes_count", 0) or 0})
        caption = post.caption or ""
    except Exception as e:
        return jsonify({"error": f"Could not fetch this post. It may be private, or Instagram is rate-limiting requests. ({str(e)})"}), 400

    if not comments:
        return jsonify({"error": "No comments found on this post."}), 400

    result = analyze_comments_list(comments, source="instagram")
    result["caption_analysis"] = analyze_caption(caption)
    result = compute_trust_score(result)
    return jsonify(result)


# ---------- Screenshot OCR ----------
@app.route("/analyze-screenshot", methods=["POST"])
def analyze_screenshot():
    if "screenshot" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["screenshot"]
    image = Image.open(io.BytesIO(file.read()))
    raw_text = pytesseract.image_to_string(image)
    lines = [line.strip() for line in raw_text.split("\n") if len(line.strip()) > 8]

    if not lines:
        return jsonify({"error": "Could not detect readable comment text. Try a clearer screenshot."}), 400

    comments = [{"author": "unknown (from screenshot)", "text": line, "likes": 0} for line in lines]
    result = analyze_comments_list(comments, source="screenshot")
    result["caption_analysis"] = None
    result = compute_trust_score(result)
    return jsonify(result)


# ---------- AI / Deepfake Image Detector ----------
@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read this image file."}), 400

    predictions = image_detector(image)
    predictions = [{"label": p["label"], "score": round(p["score"] * 100, 2)} for p in predictions]
    predictions.sort(key=lambda p: p["score"], reverse=True)

    return jsonify({
        "predictions": predictions,
        "top_label": predictions[0]["label"],
        "top_score": predictions[0]["score"],
        "note": "This is a probability estimate based on visual patterns in the image, not a certainty. AI-detection tools can be wrong, especially on heavily compressed or edited images."
    })


# ---------- Dashboard ----------
@app.route("/dashboard-data")
def dashboard_data():
    conn = sqlite3.connect("comments.db")
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM comments")
    total = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(*) FROM comments WHERE toxicity > 50")
    toxic_count = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(*) FROM comments WHERE is_bot = 1")
    bot_count = c.fetchone()[0] or 0

    c.execute("SELECT emotion, COUNT(*) FROM comments GROUP BY emotion")
    emotion_counts = c.fetchall()

    c.execute("SELECT sentiment, COUNT(*) FROM comments GROUP BY sentiment")
    sentiment_counts = c.fetchall()

    c.execute("SELECT source, COUNT(*) FROM comments GROUP BY source")
    source_counts = c.fetchall()

    c.execute("SELECT author, text, toxicity, timestamp FROM comments ORDER BY toxicity DESC LIMIT 10")
    high_risk = c.fetchall()

    c.execute("SELECT author, AVG(toxicity) as avg_tox, COUNT(*) as cnt FROM comments GROUP BY author HAVING cnt >= 2 ORDER BY avg_tox DESC LIMIT 10")
    reputation_rows = c.fetchall()

    conn.close()

    return jsonify({
        "total_comments": total,
        "toxic_percent": round((toxic_count / total) * 100, 2) if total else 0,
        "bot_percent": round((bot_count / total) * 100, 2) if total else 0,
        "emotion_distribution": {row[0]: row[1] for row in emotion_counts},
        "sentiment_distribution": {row[0].capitalize(): row[1] for row in sentiment_counts},
        "source_distribution": {row[0]: row[1] for row in source_counts},
        "high_risk_comments": [{"author": r[0], "text": r[1], "toxicity": r[2], "timestamp": r[3]} for r in high_risk],
        "reputation": [{"author": r[0], "reputation_score": round(max(0, 100 - r[1]), 2), "comment_count": r[2]} for r in reputation_rows]
    })


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(debug=True)