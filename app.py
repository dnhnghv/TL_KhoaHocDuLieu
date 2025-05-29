# app.py

import joblib
import requests
import pandas as pd
from flask import Flask, jsonify, render_template
from config import page_token, post_id, fetch_limit, model_path

app = Flask(__name__)

# --- Load mô hình và vectorizer ---
pipeline = joblib.load(model_path)
vectorizer = pipeline["vectorizer"]
model = pipeline["model"]

# --- Hàm fetch comment gốc ---
def fetch_comments_raw():
    """
    Trả về list dict chứa 'message' và 'created_time'
    """
    url = (
        f"https://graph.facebook.com/v17.0/{post_id}/comments"
        f"?access_token={page_token}&limit={fetch_limit}"
    )
    resp = requests.get(url).json().get("data", [])
    return [
        {"message": c.get("message", ""),
         "created_time": c.get("created_time", "")}
        for c in resp if c.get("message")
    ]

# --- Hàm fetch comment chỉ message ---
def fetch_comments():
    raw = fetch_comments_raw()
    return [item["message"] for item in raw]

# --- Hàm dự đoán nhãn ---
def predict_labels(comments):
    if not comments:
        return []
    X = vectorizer.transform(comments)
    y = model.predict(X)
    mapping = {2: "tích cực", 1: "trung tính", 0: "tiêu cực"}
    return [mapping[i] for i in y]

# --- Endpoint trả tỉ lệ counts ---
@app.route("/api/comments")
def api_comments():
    raw = fetch_comments_raw()
    messages = [item["message"] for item in raw]
    labels = predict_labels(messages)

    total = len(labels)
    counts = pd.Series(labels).value_counts().to_dict()
    # đảm bảo đủ 3 nhãn
    for k in ["tích cực", "trung tính", "tiêu cực"]:
        counts.setdefault(k, 0)

    return jsonify(total=total, counts=counts)

# --- Endpoint trả chi tiết comment + nhãn ---
@app.route("/api/comments/full")
def api_comments_full():
    raw = fetch_comments_raw()
    messages = [item["message"] for item in raw]
    labels = predict_labels(messages)

    result = []
    for item, label in zip(raw, labels):
        result.append({
            "message": item["message"],
            "created_time": item["created_time"],
            "label": label
        })
    return jsonify(comments=result)

# --- Trang chủ ---
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
