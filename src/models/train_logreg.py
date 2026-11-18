"""Logistic Regression (linear family) through the shared harness.

Run from the repo root:
    python -m src.models.train_logreg                          # baseline: C=1.0, no class weights  -> results/models/logreg
    python -m src.models.train_logreg --class-weight balanced --tag cw_balanced   -> results/models/logreg_cw_balanced
    python -m src.models.train_logreg --C 0.1 --tag C0.1

Every variant you run is an experiment you must document in the report (config, CV, student-test).
Choose between variants using CV results, NOT student_test.
"""
import argparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.features.columns import FEATURE_COLS
from src.models import harness

ap = argparse.ArgumentParser()
ap.add_argument("--C", type=float, default=1.0, help="inverse L2 regularisation strength")
ap.add_argument("--class-weight", choices=["none", "balanced"], default="none")
ap.add_argument("--max-iter", type=int, default=500)
ap.add_argument("--tag", default="", help="suffix for the results folder of a non-baseline variant")
a = ap.parse_args()
cw = None if a.class_weight == "none" else "balanced"


def factory():
    # imputer + scaler live INSIDE the pipeline, so the harness fits them on each fold's training part only
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=a.C, penalty="l2", solver="lbfgs", class_weight=cw,
                                            max_iter=a.max_iter, random_state=harness.SEED))


name = "logreg" + (f"_{a.tag}" if a.tag else "")
config = {"C": a.C, "penalty": "l2", "solver": "lbfgs", "class_weight": a.class_weight, "max_iter": a.max_iter,
          "preprocessing": "median imputer + StandardScaler, fitted inside each CV fold",
          "objective": "L2-regularised log loss (binary cross-entropy)", "threshold": 0.5}
res = harness.run_experiment(name, "linear", factory, config, FEATURE_COLS)
cv, te = res["cv"], res["student_test"]
print(f"\n{name}")
print(f"  CV macro F1      {cv['macro_f1']['mean']:.4f} +- {cv['macro_f1']['std']:.4f}")
print(f"  CV top-1         {cv['top1']['mean']:.4f} +- {cv['top1']['std']:.4f}")
print(f"  test macro F1    {te['macro_f1']:.4f} | top-1 {te['top1']:.4f} | log loss {te['log_loss']:.4f}")
print(f"  precision/recall {te['precision']:.4f} / {te['recall']:.4f} | miss rate {te['candidate_miss_rate']:.4f}")
print(f"  train {te['train_time_s']:.1f}s | inference {te['inference_time_s']:.2f}s")
