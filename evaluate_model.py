
import csv
import json
from pathlib import Path

from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Set the dataset path
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "SMSSpamCollection"
METRICS_PATH = BASE_DIR / "metrics.json"

# 2. Load the dataset
messages = []
labels = []

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}\n"
        "Place SMSSpamCollection inside the dataset folder."
    )

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")

    for row in reader:
        if len(row) >= 2:
            label = row[0].strip().lower()
            message = "\t".join(row[1:]).strip()

            if label in ["ham", "spam"] and message:
                labels.append(label)
                messages.append(message)

print("Total messages:", len(messages))
print("Dataset loaded successfully.")

# 3. Split data for evaluation
_, X_test, _, y_test = train_test_split(
    messages,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Test messages:", len(X_test))

# 4. Load the pretrained model
print("\nLoading pretrained spam detection model...")

classifier = pipeline(
    "text-classification",
    model="didulantha/sms-spam-detector",
    device=-1
)

print("Model loaded successfully.")
print("Model labels:", classifier.model.config.id2label)

# 5. Predict test messages
predictions = []
batch_size = 16

print("\nEvaluating messages...")

for start in range(0, len(X_test), batch_size):
    batch = X_test[start:start + batch_size]

    results = classifier(
        batch,
        truncation=True,
        max_length=128,
        batch_size=batch_size
    )

    for result in results:
        label = result["label"].strip().upper()

        # Model label mapping:
        # LABEL_0 = HAM
        # LABEL_1 = SPAM
        if label in ["LABEL_1", "SPAM"]:
            predictions.append("spam")

        elif label in ["LABEL_0", "HAM", "NOT SPAM", "NOT_SPAM"]:
            predictions.append("ham")

        else:
            raise ValueError(
                f"Unknown model label: {label}. "
                "Verify the model's label configuration."
            )

    completed = min(start + batch_size, len(X_test))
    print(f"Processed {completed}/{len(X_test)} messages")

# 6. Calculate evaluation metrics
metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    ),
    "recall": recall_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    ),
    "f1_score": f1_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )
}

# 7. Display results
print("\n========== EVALUATION RESULTS ==========")
print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
print(f"Precision: {metrics['precision'] * 100:.2f}%")
print(f"Recall:    {metrics['recall'] * 100:.2f}%")
print(f"F1-score:  {metrics['f1_score'] * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        zero_division=0
    )
)

print("Confusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

# 8. Save metrics for the frontend
with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print(f"\nMetrics saved to: {METRICS_PATH}")
print("Evaluation completed successfully.")
