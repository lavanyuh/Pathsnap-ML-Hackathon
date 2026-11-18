"""Random Forest through the shared harness.  Run from the repo root:
    python -m src.models.train_rf
Runs the starting setting (saved as 'rf') and one alternative (saved as 'rf_leaf20').
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.features.columns import FEATURE_COLS
from src.models.harness import SEED, run_experiment
from src.models.reliability import reliability_plot

SETTINGS = {
    "rf":          dict(n_estimators=300, max_depth=None, min_samples_leaf=5,
                        class_weight="balanced_subsample"),
    "rf_leaf20":   dict(n_estimators=300, max_depth=None, min_samples_leaf=20,
                        class_weight="balanced_subsample"),
}

for name, params in SETTINGS.items():
    def factory(p=params):
        # per-fold median imputer (fitted on the training part of each fold only)
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_jobs=-1, random_state=SEED, **p))
    res = run_experiment(name, "bagging ensemble", factory, params, FEATURE_COLS)
    reliability_plot(name)
    st, cv = res["student_test"], res["cv"]
    print(f"{name}: CV top1 {cv['top1']['mean']:.4f} +/- {cv['top1']['std']:.4f} | "
          f"test top1 {st['top1']:.4f} | test log loss {st['log_loss']:.4f} | "
          f"CV log loss {cv['log_loss']['mean']:.4f} | train {st['train_time_s']:.1f}s | "
          f"infer/1000 cand {st['inference_time_s_per_1000_candidates']:.4f}s")
