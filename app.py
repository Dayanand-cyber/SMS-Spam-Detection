
from flask import Flask, render_template, request, jsonify
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv
import json
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "SMSSpamCollection"
METRICS_PATH = BASE_DIR / "metrics.json"


def load_dataset():
    messages = []
    labels = []

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) >= 2:
                labels.append(row[0].strip().lower())
                messages.append("\t".join(row[1:]))

    return messages, labels


messages, labels = load_dataset()

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(messages, labels)


def load_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            pass

    return {}


@app.route("/")
def home():
    return render_template("index.html", metrics=load_metrics())


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter an SMS message."}), 400

    probabilities = model.predict_proba([message])[0]
    classes = model.named_steps["classifier"].classes_
    predicted_index = probabilities.argmax()

    prediction = classes[predicted_index].upper()
    confidence = float(probabilities[predicted_index]) * 100

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run()
