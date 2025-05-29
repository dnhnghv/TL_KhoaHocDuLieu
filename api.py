import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model & vectorizer
model_data = joblib.load(os.path.join("models", "sentiment_model.pkl"))
model = model_data["model"]
vectorizer = model_data["vectorizer"]
reverse_mapping = {0: "N", 1: "O", 2: "P"}

@app.route("/", methods=["GET"])
def index():
    return "API Phân tích cảm xúc đang chạy!"

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    data = request.get_json()
    if not data or "comments" not in data:
        return jsonify({"error": "Cần key 'comments'"}), 400
    comments = data["comments"]
    if not isinstance(comments, list):
        return jsonify({"error": "'comments' phải là danh sách"}), 400

    df_input = pd.DataFrame({"comment": comments})
    X_vect = vectorizer.transform(df_input["comment"])
    y_pred_numeric = model.predict(X_vect)
    y_pred_labels = [reverse_mapping[num] for num in y_pred_numeric]
    return jsonify({"predictions": y_pred_labels})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
