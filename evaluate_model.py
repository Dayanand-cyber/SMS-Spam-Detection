
import json
import pandas as pd

from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load the labeled SMS dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

data = data.dropna(subset=["label", "message"])
data["label"] = data["label"].str.strip().str.lower()
data = data[data["label"].isin(["ham", "spam"])]

X = data["message"].astype(str)
y = data["label"]

# Split only for evaluation; the pretrained model is not retrained
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load the existing pretrained model
classifier = pipeline(
    "text-classification",
    model="didulantha/sms-spam-detector",
    device=-1
)

# Predict in batches
predictions = []

for start in range(0, len(X_test), 16):
    messages = X_test.iloc[start:start + 16].tolist()
    results = classifier(
        messages,
        truncation=True,
        max_length=128,
        batch_size=16
    )

    for result in results:
        label = result["label"].strip().lower()

        if label in ["spam", "label_1", "1"]:
            predictions.append("spam")
        elif label in ["ham", "not spam", "not_spam", "label_0", "0"]:
            predictions.append("ham")
        else:
            raise ValueError(
                f"Unknown model label: {result['label']}. "
                "Check the model's id2label configuration."
            )

# Calculate the required metrics
metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(
        y_test, predictions, pos_label="spam", zero_division=0
    ),
    "recall": recall_score(
        y_test, predictions, pos_label="spam", zero_division=0
    ),
    "f1_score": f1_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
}

# Save the actual results
with open("metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("Evaluation complete")
for name, value in metrics.items():
    print(f"{name}: {value * 100:.2f}%")

print("Saved results to metrics.json")
