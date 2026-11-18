"""Reliability (calibration) plot from a model's saved student_test predictions."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

RESULTS_DIR = Path("results/models")


def reliability_plot(name, n_bins=10):
    d = RESULTS_DIR / name
    te = pd.read_parquet(d / "student_test_predictions.parquet")
    te = te[te["labelable"]]
    frac_pos, mean_pred = calibration_curve(te["label"], te["proba"], n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraction positive")
    ax.set_title(f"Reliability: {name} (student_test)"); ax.legend()
    fig.tight_layout(); fig.savefig(d / "reliability.png", dpi=130); plt.close(fig)
