"""LightGBM through the shared harness.  Run from the repo root:
    python -m src.models.train_lgbm
Runs the starting setting (saved as 'lgbm') and one alternative (saved as 'lgbm_small').
"""
from lightgbm import LGBMClassifier

from src.features.columns import FEATURE_COLS
from src.models.harness import SEED, run_experiment
from src.models.reliability import reliability_plot

SETTINGS = {
    "lgbm":       dict(n_estimators=500, learning_rate=0.05, num_leaves=31,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
    "lgbm_small": dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
}

for name, params in SETTINGS.items():
    def factory(p=params):
        # LightGBM handles missing values natively, so no imputer here
        return LGBMClassifier(objective="binary", n_jobs=-1, random_state=SEED, verbose=-1, **p)
    res = run_experiment(name, "boosting", factory, params, FEATURE_COLS)
    reliability_plot(name)
    st, cv = res["student_test"], res["cv"]
    print(f"{name}: CV top1 {cv['top1']['mean']:.4f} +/- {cv['top1']['std']:.4f} | "
          f"test top1 {st['top1']:.4f} | test log loss {st['log_loss']:.4f} | "
          f"CV log loss {cv['log_loss']['mean']:.4f} | train {st['train_time_s']:.1f}s | "
          f"infer/1000 cand {st['inference_time_s_per_1000_candidates']:.4f}s")
