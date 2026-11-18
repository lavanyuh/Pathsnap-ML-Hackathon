"""Shared evaluation harness (src/models/harness.py).

Every model goes through run_experiment(), so all results are produced the same way
and follow the model contract in 00_shared/INTERFACES.md.

A model must be a scikit-learn style object with fit(X, y) and predict_proba(X).
model_factory is a function returning a FRESH, untrained model (or Pipeline), so each
CV fold starts clean and scalers are fitted on the training part of the fold only.
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold

SEED = 42
DATA_DIR = Path("data/features")
RESULTS_DIR = Path("results/models")
KEYS = ["trip_id", "point_idx"]
EPS = 1e-15


def load_split(split, data_dir=None):
    """Return (points_df, features_df) for 'train' or 'student_test'."""
    d = Path(data_dir) if data_dir else DATA_DIR
    points = pd.read_parquet(d / f"points_{split}.parquet")
    feats = pd.read_parquet(d / f"features_{split}.parquet")
    return points, feats


def _proba(model, X):
    """Probability of the positive class as a 1-D array."""
    p = model.predict_proba(X)
    p = np.asarray(p)
    return p[:, 1] if p.ndim == 2 else p


def evaluate(points_df, features_df, proba, threshold=0.5, route_fn=None):
    """All metrics from INTERFACES.md section 3, on labelable rows only.

    proba: one probability per row of features_df (same order).
    route_fn: optional function(pred_df) -> dict, supplied by the decoder owner.
              pred_df has one row per labelable point: trip_id, point_idx, pred_arc_id.
    """
    df = features_df[KEYS + ["arc_id", "label", "labelable"]].copy()
    df["proba"] = np.clip(np.asarray(proba, dtype=float), EPS, 1 - EPS)
    lab = df[df["labelable"]]
    y, p = lab["label"].to_numpy(), lab["proba"].to_numpy()
    pred = (p >= threshold).astype(int)

    out = {
        "n_candidate_rows": int(len(lab)),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
        "pos_f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


    pts = points_df[points_df["labelable"]]
    top = lab.loc[lab.groupby(KEYS)["proba"].idxmax(), KEYS + ["arc_id"]]
    top = top.rename(columns={"arc_id": "pred_arc_id"})
    m = pts.merge(top, on=KEYS, how="left")
    correct = (m["pred_arc_id"] == m["true_arc_id"]).to_numpy()
    present = m["true_in_candidates"].to_numpy(dtype=bool)
    out["n_points"] = int(len(m))
    out["top1"] = float(correct.mean()) if len(m) else float("nan")
    out["top1_conditional"] = float(correct[present].mean()) if present.any() else float("nan")
    out["candidate_miss_rate"] = float(1.0 - present.mean()) if len(m) else float("nan")

    if route_fn is not None:
        out.update(route_fn(m[KEYS + ["pred_arc_id"]]))
    else:
        out["route_continuity"] = None               # filled once the decoder owner's function exists
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def cross_validate(model_factory, feature_cols, n_splits=5, route_fn=None, data_dir=None):
    """GroupKFold by trip_id on the train split. Returns per-fold metrics, mean/std, OOF predictions."""
    points, feats = load_split("train", data_dir)
    feats = feats.reset_index(drop=True)
    X, y = feats[feature_cols], feats["label"].to_numpy()
    groups = feats["trip_id"].to_numpy()
    oof = np.full(len(feats), np.nan)
    per_fold = []

    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        tr = tr[feats["labelable"].to_numpy()[tr]]
        model = model_factory()
        model.fit(X.iloc[tr], y[tr])
        oof[va] = _proba(model, X.iloc[va])
        va_trips = set(groups[va])
        m = evaluate(points[points["trip_id"].isin(va_trips)], feats.iloc[va], oof[va], route_fn=route_fn)
        m["fold"] = k
        per_fold.append(m)

    num = [c for c in per_fold[0] if isinstance(per_fold[0][c], float) and c != "fold"]
    summary = {c: {"mean": float(np.nanmean([f[c] for f in per_fold])),
                   "std": float(np.nanstd([f[c] for f in per_fold]))} for c in num}
    oof_df = feats[KEYS + ["arc_id", "label", "labelable"]].copy()
    oof_df["proba"] = oof
    return {"per_fold": per_fold, "summary": summary, "oof": oof_df}


def _confusion_png(y, pred, path, title):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_experiment(name, family, model_factory, config, feature_cols, route_fn=None, data_dir=None):
    """CV on train -> fit on all train -> evaluate on student_test -> write the model-contract files."""
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cv = cross_validate(model_factory, feature_cols, route_fn=route_fn, data_dir=data_dir)

    pts_tr, f_tr = load_split("train", data_dir)
    f_tr = f_tr[f_tr["labelable"]]
    model = model_factory()
    t0 = time.perf_counter()
    model.fit(f_tr[feature_cols], f_tr["label"])
    train_time = time.perf_counter() - t0

    # student_test (never used for tuning)
    pts_te, f_te = load_split("student_test", data_dir)
    f_te = f_te.reset_index(drop=True)
    t0 = time.perf_counter()
    proba = _proba(model, f_te[feature_cols])
    infer_time = time.perf_counter() - t0
    metrics = evaluate(pts_te, f_te, proba, route_fn=route_fn)
    metrics["train_time_s"] = train_time
    metrics["inference_time_s"] = infer_time
    metrics["inference_time_s_per_1000_candidates"] = infer_time / max(len(f_te), 1) * 1000

    cfg = {"name": name, "family": family, "seed": SEED, "feature_cols": list(feature_cols),
           "config": config}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (out_dir / "cv_metrics.json").write_text(
        json.dumps({"per_fold": cv["per_fold"], "summary": cv["summary"]}, indent=2, default=str))
    (out_dir / "student_test_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    cv["oof"].to_parquet(out_dir / "oof_predictions.parquet", index=False)
    te = f_te[KEYS + ["arc_id", "label", "labelable"]].copy()
    te["proba"] = proba
    te.to_parquet(out_dir / "student_test_predictions.parquet", index=False)
    lab = te[te["labelable"]]
    _confusion_png(lab["label"], (lab["proba"] >= 0.5).astype(int),
                   out_dir / "confusion_matrix.png", f"{name} (student_test)")
    joblib.dump(model, out_dir / "model.joblib", compress=3)
    return {"cv": cv["summary"], "student_test": metrics}

