
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from src.models.harness import run_experiment, load_split

# Load the prepared training features
_, train_features = load_split("train")

# Use only numeric feature columns.
# Exclude identifiers, labels, and target information.
excluded = {
    "trip_id", "point_idx", "arc_id",
    "label", "labelable", "true_arc_id",
    "true_in_candidates"
}

feature_cols = [
    col for col in train_features.columns
    if col not in excluded
    and train_features[col].dtype.kind in "iuf"
]

print("Features being used:", feature_cols)
print("Number of features:", len(feature_cols))

if not feature_cols:
    raise ValueError("No numeric feature columns found.")

# Gaussian Naive Bayes
def make_gnb():
    return GaussianNB(var_smoothing=1e-9)

# Multi-Layer Perceptron
def make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )

print("\nTraining Gaussian Naive Bayes...")
gnb_results = run_experiment(
    name="gnb",
    family="Gaussian Naive Bayes",
    model_factory=make_gnb,
    config={"var_smoothing": 1e-9},
    feature_cols=feature_cols
)
print("GNB RESULTS:", gnb_results)

print("\nTraining MLP...")
mlp_results = run_experiment(
    name="mlp",
    family="Multi-Layer Perceptron",
    model_factory=make_mlp,
    config={
        "hidden_layer_sizes": [64, 32],
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "max_iter": 200,
        "early_stopping": True
    },
    feature_cols=feature_cols
)
print("MLP RESULTS:", mlp_results)

print("\nBOTH MODELS FINISHED.")
