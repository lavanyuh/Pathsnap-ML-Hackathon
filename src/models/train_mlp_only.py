
from sklearn.neural_network import MLPClassifier
from src.models.harness import run_experiment, load_split

# Load training features
_, train_features = load_split("train")

# Exclude metadata and target columns
excluded = {
    "trip_id", "point_idx", "arc_id",
    "label", "labelable", "true_arc_id",
    "true_in_candidates"
}

# Select numeric feature columns
feature_cols = [
    col for col in train_features.columns
    if col not in excluded
    and train_features[col].dtype.kind in "iuf"
]

print("Number of features:", len(feature_cols))

if not feature_cols:
    raise ValueError("No numeric feature columns found.")

# Smaller, faster MLP configuration
def make_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=60,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42
    )

print("\nStarting MLP training...")

mlp_results = run_experiment(
    name="mlp",
    family="Multi-Layer Perceptron",
    model_factory=make_mlp,
    config={
        "hidden_layer_sizes": [32],
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "max_iter": 60,
        "early_stopping": True,
        "n_iter_no_change": 5
    },
    feature_cols=feature_cols
)

print("\nMLP RESULTS:")
print(mlp_results)

print("\nMLP TRAINING FINISHED.")
