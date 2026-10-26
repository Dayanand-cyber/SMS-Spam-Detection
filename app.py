
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import json
import os

app = Flask(__name__)

MODEL_NAME = "didulantha/sms-spam-detector"

print("Loading pretrained spam detection model...")

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=-1
)

print("Model loaded successfully!")


def predict_message(message):
    result = classifier(
        message,
        truncation=True,
        max_length=128
    )[0]

    label = result["label"].strip().lower()
    confidence = float(result["score"])

    # Map the model's labels to SPAM or HAM
    if label in ["spam", "label_1", "1"]:
        prediction = "SPAM"
    elif label in ["ham", "not spam", "not_spam", "label_0", "0"]:
        prediction = "HAM"
    else:
        raise ValueError(f"Unexpected model label: {result['label']}")

    return prediction, confidence


def load_metrics():
    if os.path.exists("metrics.json"):
        try:
            with open("metrics.json", "r", encoding="utf-8") as file:
                metrics = json.load(file)

            required = ["accuracy", "precision", "recall", "f1_score"]

            if all(key in metrics for key in required):
                return metrics

        except (json.JSONDecodeError, OSError):
            pass

    return {}


@app.route("/")
def home():
    metrics = load_metrics()
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "error": "Please enter an SMS message."
        }), 400

    try:
        prediction, confidence = predict_message(message)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        })

    except Exception:
        app.logger.exception("Message prediction failed")

        return jsonify({
            "error": "Unable to classify the message. Check the model setup."
        }), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
