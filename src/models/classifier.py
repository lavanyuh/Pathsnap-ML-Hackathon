
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.features.columns import FEATURE_COLS

# Load datasets
train = pd.read_parquet("data/features/features_train.parquet")
test = pd.read_parquet("data/features/features_student_test.parquet")

# Remove unlabelled rows
train = train[train["label"] != -1].copy()
test = test[test["label"] != -1].copy()

X_train = train[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
y_train = train["label"].astype(int)

X_test = test[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
y_test = test["label"].astype(int)

# Handle class imbalance
counts = y_train.value_counts()
weights = {
    cls: len(y_train) / (2 * count)
    for cls, count in counts.items()
}
sample_weights = y_train.map(weights)

# Build classifier
model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.1,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    early_stopping=True,
    random_state=42
)

# Train
print("Training classifier...")
model.fit(X_train, y_train, sample_weight=sample_weights)

# Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n===== CLASSIFIER RESULTS =====")

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, predictions, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, probabilities))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
Path("artifacts").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_cols": FEATURE_COLS
    },
    "artifacts/classifier.joblib"
)

print("\nClassifier saved successfully!")

