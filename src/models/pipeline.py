"""Complete end-to-end map matching inference and evaluation pipeline.

Handles raw trip data (.track, .nodes, .arcs, .route), feature building,
candidate probability estimation, HMM Viterbi decoding, sequence metrics,
and geo coordinate extraction for UI / Folium rendering.
"""
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

from ..data.loader import load_trip
from ..data.network import RoadNetwork, prepare_track
from ..features.build import build_trip_features
from ..features.columns import FEATURE_COLS
from ..data.labels import derive_point_labels
from ..eval.route_metrics import sequence_metrics, route_continuity
from .hmm import viterbi_decode_variable

DEFAULT_MODEL_PATH = Path("artifacts/classifier.joblib")
_env_root = os.environ.get("DATA_ROOT", "/Users/kanishka/Downloads/PathSnap_Project/Dataset")
DEFAULT_DATA_ROOT = Path(_env_root)
if not DEFAULT_DATA_ROOT.exists():
    for _cand in [Path("../PathSnap_Project/Dataset"), Path("data"), Path("../Dataset")]:
        if _cand.exists():
            DEFAULT_DATA_ROOT = _cand.resolve()
            break


def make_transition_matrix(net: RoadNetwork, ids1, t1, ids2, t2, gps_step: float, beta: float = 12.0) -> np.ndarray:
    """Compute HMM transition probabilities between candidate sets of consecutive GPS points.

    Supports staying on same arc (handling GPS noise via absolute delta) and direct network adjacency.
    """
    matrix = np.full((len(ids1), len(ids2)), 1e-4)

    for i, a in enumerate(ids1):
        for j, b in enumerate(ids2):
            dist = None
            if a == b:
                dist = abs(t2[j] - t1[i]) * net.length[a]
            elif net.arcs[a, 1] == net.arcs[b, 0]:
                dist = (1.0 - t1[i]) * net.length[a] + t2[j] * net.length[b]

            if dist is not None:
                scale = max(beta, gps_step * 0.8)
                matrix[i, j] = np.exp(-abs(dist - gps_step) / scale)

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix /= row_sums
    return matrix


class MapMatchingPipeline:
    def __init__(self, model_path: Optional[Path] = None, data_root: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        art = joblib.load(self.model_path)
        self.model = art["model"]
        self.feature_cols = art.get("feature_cols", FEATURE_COLS)
        self.model_name = art.get("model_name", "classifier")

    def process_trip(self, trip_id: str, split: str = "evaluator_test", radius: float = 50.0) -> Dict[str, Any]:
        """Run complete map matching pipeline for a trip."""
        t_start = time.time()

        # 1. Load raw data
        t0 = time.time()
        trip = load_trip(str(self.data_root), split, trip_id, with_route=True)
        net = RoadNetwork.from_trip(trip)
        track = prepare_track(trip.track, net.to_xy)
        load_time_ms = (time.time() - t0) * 1000.0

        # 2. Server-side reference labels (if route is available)
        has_ground_truth = trip.route is not None and len(trip.route) > 0
        labels_df = None
        if has_ground_truth:
            labels_dict = derive_point_labels(net, track["xy"], trip.route)
            labels_df = pd.DataFrame({
                "labelable": labels_dict["labelable"],
                "true_arc_id": labels_dict["true_arc_id"]
            })

        # 3. Candidate & Feature Generation (labels only passed for evaluation tracking)
        t0 = time.time()
        pts, f = build_trip_features(net, track, trip.trip_id, radius=radius, labels=labels_df)
        feat_time_ms = (time.time() - t0) * 1000.0

        if len(f) == 0:
            raise RuntimeError(f"No candidate road segments found within radius {radius}m for trip {trip_id}")

        # 4. Model Emission Probability Inference
        t0 = time.time()
        X = f[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        f["prob"] = self.model.predict_proba(X)[:, 1]
        model_time_ms = (time.time() - t0) * 1000.0

        # 5. HMM Viterbi Sequential Route Decoding
        t0 = time.time()
        pred_hmm = np.full(len(pts), -1, dtype=int)
        top1_clf = np.full(len(pts), -1, dtype=int)

        groups = {int(idx): group for idx, group in f.groupby("point_idx", sort=True)}

        # Classifier-only top-1 predictions
        for idx, group in groups.items():
            top1_clf[idx] = int(group.iloc[group.prob.argmax()].arc_id)

        # Build contiguous GPS sequences (splitting on GPS gaps > 5s)
        sequences = []
        current = []
        for point_idx in range(len(pts)):
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

        # Decode each sequence with Viterbi
        for sequence in sequences:
            emissions = [g.prob.to_numpy() for _, g in sequence]
            transitions = []
            for k in range(len(sequence) - 1):
                idx1, g1 = sequence[k]
                idx2, g2 = sequence[k + 1]
                step = np.linalg.norm(track["xy"][idx2] - track["xy"][idx1])
                transitions.append(make_transition_matrix(
                    net,
                    g1.arc_id.to_numpy(int), g1.proj_t.to_numpy(float),
                    g2.arc_id.to_numpy(int), g2.proj_t.to_numpy(float),
                    step
                ))
            path = viterbi_decode_variable(emissions, transitions)
            for k, (point_idx, group) in enumerate(sequence):
                pred_hmm[point_idx] = int(group.arc_id.iloc[path[k]])

        hmm_time_ms = (time.time() - t0) * 1000.0
        total_time_ms = (time.time() - t_start) * 1000.0

        # 6. Evaluation Metrics against Server-side Reference Ground Truth
        metrics = {
            "n_gps_points": int(len(pts)),
            "n_candidate_rows": int(len(f)),
            "mean_candidates_per_point": float(pts.n_candidates.mean()),
            "total_latency_ms": round(total_time_ms, 2),
            "breakdown_ms": {
                "load": round(load_time_ms, 2),
                "features": round(feat_time_ms, 2),
                "model_inference": round(model_time_ms, 2),
                "hmm_viterbi": round(hmm_time_ms, 2)
            }
        }

        if has_ground_truth and labels_df is not None:
            valid = pts.labelable.to_numpy(bool) & pts.true_in_candidates.to_numpy(bool) & (pred_hmm >= 0)
            actual = pts.true_arc_id.to_numpy(int)[valid]

            if len(actual) > 0:
                hmm_acc = float((pred_hmm[valid] == actual).mean())
                top1_acc = float((top1_clf[valid] == actual).mean())
            else:
                hmm_acc = 0.0
                top1_acc = 0.0

            seq_prec, seq_rec, seq_f1 = sequence_metrics(net, pred_hmm, trip.route)
            cont, breaks = route_continuity(net, pred_hmm)

            metrics["accuracy"] = {
                "hmm_pointwise_accuracy": round(hmm_acc * 100.0, 2),
                "classifier_pointwise_accuracy": round(top1_acc * 100.0, 2),
                "hmm_accuracy_gain": round((hmm_acc - top1_acc) * 100.0, 2),
                "evaluated_points": int(len(actual)),
                "labelable_points": int(pts.labelable.sum()),
                "candidate_coverage": round(float(pts.true_in_candidates.mean()) * 100.0, 2)
            }
            metrics["sequence"] = {
                "precision": round(float(seq_prec) * 100.0, 2),
                "recall": round(float(seq_rec) * 100.0, 2),
                "f1_score": round(float(seq_f1) * 100.0, 2),
                "continuity": round(float(cont) * 100.0, 2),
                "breaks": int(breaks)
            }

        # 7. Extract Geographic Coordinates for UI Leaflet Map
        # GPS Observations [lat, lon]
        gps_coords = trip.track[:, [1, 0]].tolist()  # lat, lon

        # Reconstruct road segments from node lat/lon
        nodes = trip.nodes  # [lon, lat]
        arcs = trip.arcs    # [from_node, to_node]

        # Unique predicted arcs in chronological order
        collapsed_pred_arcs = []
        for a in pred_hmm:
            if a >= 0 and (not collapsed_pred_arcs or a != collapsed_pred_arcs[-1]):
                collapsed_pred_arcs.append(int(a))

        pred_route_lines = []
        for arc_id in collapsed_pred_arcs:
            if 0 <= arc_id < len(arcs):
                u, v = arcs[arc_id]
                pred_route_lines.append([
                    [float(nodes[u, 1]), float(nodes[u, 0])],  # lat, lon
                    [float(nodes[v, 1]), float(nodes[v, 0])]
                ])

        ref_route_lines = []
        if has_ground_truth:
            for arc_id in trip.route:
                if 0 <= arc_id < len(arcs):
                    u, v = arcs[arc_id]
                    ref_route_lines.append([
                        [float(nodes[u, 1]), float(nodes[u, 0])],
                        [float(nodes[v, 1]), float(nodes[v, 0])]
                    ])

        min_lat, min_lon = float(trip.track[:, 1].min()), float(trip.track[:, 0].min())
        max_lat, max_lon = float(trip.track[:, 1].max()), float(trip.track[:, 0].max())
        center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]

        geo = {
            "center": center,
            "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
            "gps_track": gps_coords,
            "predicted_route": pred_route_lines,
            "reference_route": ref_route_lines
        }

        return {
            "trip_id": trip_id,
            "split": split,
            "model_name": self.model_name,
            "metrics": metrics,
            "geo": geo,
            "predicted_arc_ids": pred_hmm.tolist()
        }
