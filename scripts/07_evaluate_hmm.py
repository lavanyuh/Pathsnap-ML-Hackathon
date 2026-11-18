import argparse
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import list_trips, load_trip
from src.data.network import RoadNetwork, prepare_track
from src.features.columns import FEATURE_COLS
from src.models.hmm import viterbi_decode_variable


def make_transition(net, ids1, t1, ids2, t2, gps_step):
    matrix = np.full((len(ids1), len(ids2)), 1e-8)

    for i, a in enumerate(ids1):
        for j, b in enumerate(ids2):
            distance = None

            if a == b and t2[j] >= t1[i]:
                distance = (t2[j] - t1[i]) * net.length[a]

            elif net.arcs[a, 1] == net.arcs[b, 0]:
                distance = (
                    (1 - t1[i]) * net.length[a]
                    + t2[j] * net.length[b]
                )

            if distance is not None:
                scale = max(20.0, gps_step * 0.5)
                matrix[i, j] = np.exp(
                    -abs(distance - gps_step) / scale
                )

    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    train = pd.read_parquet("data/features/features_train.parquet")
    test = pd.read_parquet("data/features/features_student_test.parquet")
    points = pd.read_parquet("data/features/points_student_test.parquet")

    train = train[train.label != -1].copy()
    test = test[test.label != -1].copy()

    X_train = train[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    y_train = train.label.astype(int)
    X_test = test[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)

    counts = y_train.value_counts()
    weights = {
        c: len(y_train) / (2 * n)
        for c, n in counts.items()
    }

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=y_train.map(weights))
    test["prob"] = model.predict_proba(X_test)[:, 1]

    predictions = []
    actual = []

    for tid in list_trips(args.data_root, "student_test"):
        trip = load_trip(args.data_root, "student_test", tid, with_route=False)
        net = RoadNetwork.from_trip(trip)
        track = prepare_track(trip.track, net.to_xy)

        f = test[test.trip_id == tid]
        p = points[points.trip_id == tid].sort_values("point_idx")

        pred = np.full(len(p), -1, dtype=int)
        groups = {
            int(idx): group
            for idx, group in f.groupby("point_idx", sort=True)
        }

        sequences = []
        current = []

        for point_idx in range(len(p)):
            group = groups.get(point_idx)

            if group is None or len(group) == 0:
                if current:
                    sequences.append(current)
                    current = []
                continue

            if bool(group.gap_before.iloc[0]) and current:
                sequences.append(current)
                current = []

            current.append((point_idx, group))

        if current:
            sequences.append(current)

        for sequence in sequences:
            emissions = [
                g.prob.to_numpy()
                for _, g in sequence
            ]
            transitions = []

            for k in range(len(sequence) - 1):
                idx1, g1 = sequence[k]
                idx2, g2 = sequence[k + 1]

                step = np.linalg.norm(
                    track["xy"][idx2] - track["xy"][idx1]
                )

                transitions.append(make_transition(
                    net,
                    g1.arc_id.to_numpy(int),
                    g1.proj_t.to_numpy(float),
                    g2.arc_id.to_numpy(int),
                    g2.proj_t.to_numpy(float),
                    step
                ))

            path = viterbi_decode_variable(emissions, transitions)

            for k, (point_idx, group) in enumerate(sequence):
                pred[point_idx] = int(group.arc_id.iloc[path[k]])

        valid = (
            p.labelable.to_numpy(bool)
            & p.true_in_candidates.to_numpy(bool)
            & (pred >= 0)
        )

        predictions.extend(pred[valid])
        actual.extend(p.true_arc_id.to_numpy(int)[valid])

    predictions = np.asarray(predictions)
    actual = np.asarray(actual)

    if len(actual) == 0:
        raise RuntimeError("No evaluable test points found.")

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predictions, average="weighted", zero_division=0
    )

    print("\n===== REAL TEST HMM RESULTS =====")
    print("Evaluated points:", len(actual))
    print("Accuracy:", accuracy_score(actual, predictions))
    print("Weighted precision:", precision)
    print("Weighted recall:", recall)
    print("Weighted F1:", f1)


if __name__ == "__main__":
    main()
