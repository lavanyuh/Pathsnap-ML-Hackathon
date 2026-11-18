"""Train and evaluate 5 distinct algorithm families on PathSnap candidate classification.

Required for Deliverable 2 (Technical Report) Section 3:
Demonstrates meaningful experimentation across multiple algorithm families before selecting LightGBM.
"""
import sys
import time
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.columns import FEATURE_COLS

def main():
    out_dir = Path("results/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train = pd.read_parquet("data/features/features_train.parquet")
    test = pd.read_parquet("data/features/features_student_test.parquet")

    train = train[train["label"] != -1].copy()
    test = test[test["label"] != -1].copy()

    # Subsample 250k rows from train for fair, rapid cross-family training
    train_sub = train.sample(n=min(250000, len(train)), random_state=42)
    X_tr = train_sub[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    y_tr = train_sub["label"].astype(int)

    X_te = test[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    y_te = test["label"].astype(int)

    # Class balance weights
    scale_pos = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    class_weights = {0: 1.0, 1: scale_pos}

    # Impute missing values with median for models that don't support NaNs natively
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    models = {
        "Logistic Regression (L2)": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", random_state=42)),
        "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, l2_regularization=1.0, random_state=42),
        "LightGBM (Tuned - Final)": lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=63, max_depth=9, scale_pos_weight=scale_pos*0.8, n_jobs=-1, random_state=42, verbose=-1)
    }

    results = []
    print("\nEvaluating 5 distinct model families...")
    print(f"{'Model':<28} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Macro F1':<10} {'ROC-AUC':<10} {'Train Time':<10}")
    print("-" * 92)

    for name, clf in models.items():
        t0 = time.time()
        if "HistGradientBoosting" in name or "LightGBM" in name:
            clf.fit(X_tr, y_tr)
            train_t = time.time() - t0
            t_inf = time.time()
            preds = clf.predict(X_te)
            probs = clf.predict_proba(X_te)[:, 1]
        else:
            clf.fit(X_tr_imp, y_tr)
            train_t = time.time() - t0
            t_inf = time.time()
            preds = clf.predict(X_te_imp)
            probs = clf.predict_proba(X_te_imp)[:, 1]
        inf_t = time.time() - t_inf

        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        pos_f1 = f1_score(y_te, preds, zero_division=0)
        macro_f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        auc = roc_auc_score(y_te, probs)

        print(f"{name:<28} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {macro_f1:<10.4f} {auc:<10.4f} {train_t:<10.2f}s")
        results.append({
            "Algorithm Family": name.split()[0],
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Positive F1": round(pos_f1, 4),
            "Macro F1": round(macro_f1, 4),
            "ROC-AUC": round(auc, 4),
            "Train Time (s)": round(train_t, 2),
            "Inference Time (s)": round(inf_t, 2)
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "model_comparison_summary.csv", index=False)
    print("\nSaved comparison table to results/models/model_comparison_summary.csv")

    # Bar chart comparing Macro F1 and ROC-AUC
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_res))
    width = 0.35
    ax.bar(x - width/2, df_res["Macro F1"], width, label="Macro F1 Score", color="#10b981")
    ax.bar(x + width/2, df_res["ROC-AUC"], width, label="ROC-AUC", color="#3b82f6")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["Model"], rotation=20, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("5-Model Comparison on PathSnap Candidate Classification (student_test)")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_chart.png", dpi=200)
    plt.close()
    print("Saved comparison chart to results/models/model_comparison_chart.png")

if __name__ == "__main__":
    main()
