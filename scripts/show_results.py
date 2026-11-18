#!/usr/bin/env python3
"""Convenience script to print all ML results and Bayesian Optimization logs in the terminal."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("\n" + "=" * 80)
    print("🏆 PATHSNAP: MACHINE LEARNING MODEL SELECTION & TUNING SUMMARY")
    print("=" * 80)

    # 1. Final Tuned LightGBM (from friend's optimization)
    friend_metrics = ROOT / "results/models/lgbm_tuned/student_test_metrics.json"
    friend_params = ROOT / "results/optimization/lgbm/best_params.json"
    metrics_path = ROOT / "results/models/lgbm/metrics.json"

    if friend_metrics.exists() and friend_params.exists():
        with open(friend_metrics) as f:
            m = json.load(f)
        with open(friend_params) as f:
            p = json.load(f)
        print("\n1️⃣  TUNED LIGHTGBM TEST PERFORMANCE (student_test):")
        print(f"   • Accuracy:       {m['accuracy']*100:.2f}%")
        print(f"   • Macro F1:       {m['macro_f1']:.4f}  <-- Primary Selection Metric")
        print(f"   • Precision:      {m['precision']*100:.2f}%")
        print(f"   • Recall:         {m['recall']*100:.2f}%")
        print(f"   • Top-1 Accuracy: {m['top1']*100:.2f}%")
        print(f"   • Log Loss:       {m['log_loss']:.4f}")
        print("\n   Best Hyperparameters (Optuna Bayesian Optimization from results/optimization):")
        for k, v in p.items():
            print(f"     - {k:<18}: {v}")
    elif metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        print("\n1️⃣  TUNED LIGHTGBM TEST PERFORMANCE (student_test):")
        print(f"   • Accuracy:       {m['accuracy']*100:.2f}%")
        print(f"   • Macro F1:       {m['macro_f1']:.4f}  <-- Primary Selection Metric")
        print(f"   • Precision:      {m['precision']*100:.2f}%")
        print(f"   • Recall:         {m['recall']*100:.2f}%")
        print(f"   • ROC-AUC:        {m['roc_auc']:.4f}")
        print(f"   • Log Loss:       {m['log_loss']:.4f}")
        print("\n   Best Hyperparameters (Optuna Bayesian Optimization):")
        for k, v in m['best_hyperparameters'].items():
            print(f"     - {k:<18}: {v}")

    # 2. 5-Model Comparison Table
    comp_path = ROOT / "results/models/model_comparison_summary.csv"
    if comp_path.exists():
        df = pd.read_csv(comp_path)
        print("\n" + "-" * 80)
        print("2️⃣  5-MODEL FAMILY COMPARISON TABLE:")
        cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'Macro F1', 'ROC-AUC', 'Train Time (s)']
        print(df[cols].to_string(index=False))

    # 3. Optuna Trials Summary
    trials_path = ROOT / "results/models/lgbm/optuna_trials.csv"
    if trials_path.exists():
        df_tr = pd.read_csv(trials_path)
        print("\n" + "-" * 80)
        print("3️⃣  OPTUNA BAYESIAN OPTIMIZATION SEARCH RECORD (15 Trials):")
        cols = ['number', 'value', 'params_learning_rate', 'params_num_leaves', 'params_max_depth', 'state']
        print(df_tr[cols].rename(columns={'number': 'Trial', 'value': 'Macro F1'}).to_string(index=False))

    # 4. 10 Evaluator Trips Summary
    eval_path = ROOT / "results/models/evaluator_test_summary.csv"
    if eval_path.exists():
        df_ev = pd.read_csv(eval_path)
        print("\n" + "-" * 80)
        print("4️⃣  PHASE 2 RESERVED EVALUATION TRIPS (10 Trips with LightGBM + HMM):")
        cols = ['Trip ID', 'GPS Points', 'HMM Accuracy (%)', 'Classifier Accuracy (%)', 'HMM Gain (%)', 'Route Continuity (%)']
        print(df_ev[cols].to_string(index=False))

    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
