
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

# Load the pretrained model once when the app starts
MODEL_NAME = "didulantha/sms-spam-detector"

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=-1  # Use CPU
)


def predict_message(message):
    # Get prediction from the pretrained model
    result = classifier(message, truncation=True, max_length=128)[0]

    label = result["label"].lower()
    confidence = float(result["score"])

    # Convert model output into SPAM or HAM
    if label in ["spam", "label_1", "1"]:
        prediction = "SPAM"
    elif label in ["ham", "not spam", "not_spam", "label_0", "0"]:
        prediction = "HAM"
    else:
        # Do not silently guess if the model labels differ
        raise ValueError(f"Unknown model label: {result['label']}")

    return prediction, confidence


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter an SMS message."}), 400

    try:
        prediction, confidence = predict_message(message)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the model setup and try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
