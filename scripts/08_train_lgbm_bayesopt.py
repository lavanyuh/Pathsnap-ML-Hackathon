"""Train LightGBM Classifier with Bayesian Hyperparameter Optimization (Optuna).

Selected as the primary model due to achieving the highest Macro F1 score on candidate classification.
Saves model artifact to artifacts/classifier.joblib and trial logs to results/models/lgbm/.
"""
import sys
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, confusion_matrix, classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.columns import FEATURE_COLS

def main():
    out_dir = Path("results/models/lgbm")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train = pd.read_parquet("data/features/features_train.parquet")
    test = pd.read_parquet("data/features/features_student_test.parquet")

    train = train[train["label"] != -1].copy()
    test = test[test["label"] != -1].copy()

    X_train = train[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    y_train = train["label"].astype(int)

    X_test = test[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    y_test = test["label"].astype(int)

    # Class imbalance weight
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"Dataset: Train={len(train)} rows, Test={len(test)} rows, scale_pos_weight={scale_pos_weight:.2f}")

    # Validation sample for Bayesian Optimization (to keep trials fast)
    val_trips = train["trip_id"].drop_duplicates().sample(frac=0.2, random_state=42)
    val_mask = train["trip_id"].isin(val_trips)
    X_tr_sub = X_train[~val_mask]
    y_tr_sub = y_train[~val_mask]
    X_va_sub = X_train[val_mask]
    y_va_sub = y_train[val_mask]

    print("\n--- Starting Bayesian Hyperparameter Optimization (Optuna) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, scale_pos_weight * 1.2),
            "n_jobs": -1
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr_sub, y_tr_sub)
        preds = clf.predict(X_va_sub)
        macro_f1 = f1_score(y_va_sub, preds, average="macro", zero_division=0)
        return macro_f1

    study = optuna.create_study(direction="maximize")
    start_opt = time.time()
    study.optimize(objective, n_trials=15, timeout=180)
    opt_duration = time.time() - start_opt
    print(f"Optuna completed {len(study.trials)} trials in {opt_duration:.1f}s")
    print("Best Trial Macro F1:", study.best_value)
    print("Best Hyperparameters:", json.dumps(study.best_params, indent=2))

    # Save Optuna trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)

    print("\n--- Training Final LightGBM Model with Best Hyperparameters on Full Training Set ---")
    best_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
        **study.best_params
    }
    final_model = lgb.LGBMClassifier(**best_params)
    t0 = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Final model fitted in {train_time:.2f}s")

    # Evaluate on student_test
    t0 = time.time()
    test_preds = final_model.predict(X_test)
    test_probs = final_model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - t0

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    pos_f1 = f1_score(y_test, test_preds, zero_division=0)
    macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs)
    ll = log_loss(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)

    print("\n===== FINAL LIGHTGBM TEST EVALUATION =====")
    print(f"Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1 Score:   {macro_f1:.4f} (Primary Criterion)")
    print(f"Positive F1 Score:{pos_f1:.4f}")
    print(f"Precision:        {prec:.4f}")
    print(f"Recall:           {rec:.4f}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"Log Loss:         {ll:.4f}")
    print(f"Inference Latency:{inference_time*1000/len(X_test):.4f} ms/candidate")
    print("\nClassification Report:\n", classification_report(y_test, test_preds))
    print("Confusion Matrix:\n", cm)

    # Save metrics JSON
    metrics = {
        "model_name": "LightGBM",
        "best_hyperparameters": study.best_params,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "pos_f1": float(pos_f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(roc_auc),
        "log_loss": float(ll),
        "train_time_sec": float(train_time),
        "confusion_matrix": cm.tolist()
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature Importance Plot
    importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    plt.figure(figsize=(10, 7))
    importances.plot(kind="barh", color="#10b981")
    plt.title("LightGBM Feature Importance (Split Count)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=200)
    plt.close()

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("LightGBM Confusion Matrix (student_test)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Negative (0)", "Positive (1)"])
    plt.yticks(tick_marks, ["Negative (0)", "Positive (1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # Save to artifacts/classifier.joblib for production API & pipeline
    artifact_payload = {
        "model": final_model,
        "feature_cols": FEATURE_COLS,
        "model_name": "lgbm",
        "metrics": metrics
    }
    joblib.dump(artifact_payload, "artifacts/classifier.joblib")
    print(f"\nSaved production artifact to artifacts/classifier.joblib")

if __name__ == "__main__":
    main()
