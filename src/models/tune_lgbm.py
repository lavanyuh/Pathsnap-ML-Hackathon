
"""Bayesian hyperparameter optimisation for lgbm.

Run from the repository root:
    python -m src.models.tune_lgbm
"""

import json
import time
from pathlib import Path

import optuna
from lightgbm import LGBMClassifier

from src.features.columns import FEATURE_COLS
from src.models.harness import (
    SEED,
    cross_validate,
    run_experiment,
)

# Number of Bayesian optimisation trials
N_TRIALS = 20

# Keep optimisation outputs separate from the original model results
OUTPUT_DIR = Path("results/optimization/lgbm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def objective(trial):
    """Evaluate one parameter configuration using 5-fold GroupKFold CV."""

    max_depth = trial.suggest_int("max_depth", 3, 10)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", 7, min(40, 2 ** max_depth)
        ),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 10, 60
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }

    def factory():
        return LGBMClassifier(
            objective="binary",
            n_jobs=-1,
            random_state=SEED,
            verbose=-1,
            **params,
        )

    start = time.perf_counter()

    cv = cross_validate(
        model_factory=factory,
        feature_cols=FEATURE_COLS,
    )

    duration = time.perf_counter() - start
    score = cv["summary"]["macro_f1"]["mean"]

    trial.set_user_attr("duration_seconds", duration)

    print(
        f"Trial {trial.number + 1}/{N_TRIALS}: "
        f"Macro F1 = {score:.5f}, "
        f"time = {duration:.1f}s"
    )

    return score


def main():
    print("Starting Bayesian optimisation for lgbm...")
    print(f"Number of trials: {N_TRIALS}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="lgbm_tuning",
    )

    total_start = time.perf_counter()

    study.optimize(objective, n_trials=N_TRIALS)

    total_time = time.perf_counter() - total_start

    best_params = study.best_params

    # Restore the fixed parameter needed for LightGBM row subsampling
    best_params["subsample_freq"] = 1

    print("\nBest CV Macro F1:", study.best_value)
    print("Best parameters:", best_params)

    # Save every optimisation trial
    study.trials_dataframe().to_csv(
        OUTPUT_DIR / "trials.csv",
        index=False,
    )

    (OUTPUT_DIR / "best_params.json").write_text(
        json.dumps(best_params, indent=2)
    )

    (OUTPUT_DIR / "optimization_summary.json").write_text(
        json.dumps(
            {
                "number_of_trials": N_TRIALS,
                "best_cv_macro_f1": study.best_value,
                "total_optimization_time_seconds": total_time,
                "best_parameters": best_params,
            },
            indent=2,
        )
    )

    # Train and evaluate the final tuned model.
    # This evaluates student_test only once, after tuning.
    def best_factory():
        return LGBMClassifier(
            objective="binary",
            n_jobs=-1,
            random_state=SEED,
            verbose=-1,
            **best_params,
        )

    print("\nTraining final tuned LightGBM model...")

    result = run_experiment(
        name="lgbm_tuned",
        family="LightGBM (Bayesian Optimisation)",
        model_factory=best_factory,
        config=best_params,
        feature_cols=FEATURE_COLS,
    )

    print("\nFINAL TUNED MODEL RESULTS")
    print("CV Macro F1:", result["cv"]["macro_f1"])
    print("Student-test Top-1:", result["student_test"]["top1"])
    print("Student-test Macro F1:", result["student_test"]["macro_f1"])
    print("Student-test Log Loss:", result["student_test"]["log_loss"])
    print(
        "Inference time per 1000 candidates:",
        result["student_test"]["inference_time_s_per_1000_candidates"],
    )

    print("\nBayesian optimisation completed.")
    print("Results saved in:", OUTPUT_DIR)
    print("Final model saved in results/models/lgbm_tuned/")


if __name__ == "__main__":
    main()
