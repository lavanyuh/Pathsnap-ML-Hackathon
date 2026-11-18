import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("results/graphs").mkdir(parents=True, exist_ok=True)

# Previously measured student_test results
models = {
    "HistGradientBoosting Classifier": {
        "Accuracy": 96.56,
        "Precision": 73.65,
        "Recall": 97.10,
        "F1 Score": 83.76
    },
    "HMM + Viterbi": {
        "Accuracy": 82.88,
        "Precision": 94.70,
        "Recall": 82.88,
        "F1 Score": 86.12
    }
}

for name, metrics in models.items():
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        metrics.keys(),
        metrics.values()
    )

    plt.title(f"{name} - Performance Metrics")
    plt.ylabel("Score (%)")
    plt.ylim(0, 110)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, metrics.values()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    filename = (
        "classifier_performance.png"
        if "Classifier" in name
        else "hmm_performance.png"
    )

    plt.savefig(f"results/graphs/{filename}", dpi=300)
    plt.close()

print("Graphs generated successfully!")
print("Saved inside results/graphs/")
