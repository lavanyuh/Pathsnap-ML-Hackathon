"""Step 4-5: candidate generation + feature engineering for one trip.

Output rows are sorted by (point_idx, distance), so dist_rank 0 comes first for every point.
Labels are optional (the API calls this without them); they never influence any feature.
"""
import numpy as np, pandas as pd
from ..data.geo import point_segment_distance
from .columns import FEATURE_COLS

OFFSETS = [o for o in range(-8, 9) if o != 0]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def build_trip_features(net, trk, trip_id, radius=50.0, labels=None):
    """net: RoadNetwork, trk: prepare_track() dict, labels: DataFrame with true_arc_id, labelable (len n) or None.
    Returns (points_df, features_df)."""
    P, t = trk["xy"], trk["t"]
    n = len(P)
    idx = np.arange(n)
    # ---------------- point-level kinematics
    lo, hi = np.clip(idx - 2, 0, n - 1), np.clip(idx + 2, 0, n - 1)
    disp = P[hi] - P[lo]
    move = np.hypot(disp[:, 0], disp[:, 1]); tspan = t[hi] - t[lo]
    hvalid = (move >= 3.0) & (tspan > 0) & (tspan <= 15)
    heading = np.arctan2(disp[:, 1], disp[:, 0])
    speed = np.where((tspan > 0) & (tspan <= 15), move / np.maximum(tspan, 1e-9), 0.0)
    lo2, hi2 = np.clip(idx - 2, 0, n - 1), np.clip(idx + 2, 0, n - 1)
    d_in, d_out = P - P[lo2], P[hi2] - P
    ok_turn = (np.hypot(*d_in.T) >= 3) & (np.hypot(*d_out.T) >= 3) & hvalid
    turn = np.where(ok_turn, np.abs(_wrap(np.arctan2(d_out[:, 1], d_out[:, 0]) - np.arctan2(d_in[:, 1], d_in[:, 0]))), 0.0)
    dt = np.nan_to_num(trk["dt"], nan=1.0)
    # ---------------- candidates
    pi, arc, dist, proj_t = net.arcs_within(P, radius)
    order = np.lexsort((dist, pi)); pi, arc, dist, proj_t = pi[order], arc[order], dist[order], proj_t[order]
    counts = np.bincount(pi, minlength=n)
    starts = np.cumsum(counts) - counts
    rank = np.arange(len(pi)) - starts[pi]
    min_d = dist[starts[pi]] if len(pi) else dist
    deg = np.bincount(net.arcs[net.valid].ravel(), minlength=len(net.node_xy))
    # window context: distance of neighbouring points to THIS candidate arc (masked across time gaps)
    W = np.full((len(pi), len(OFFSETS)), np.nan, dtype=np.float32)
    A, B = net.A[arc], net.B[arc]
    for c, o in enumerate(OFFSETS):
        j = pi + o; inb = (j >= 0) & (j < n); jj = np.clip(j, 0, n - 1)
        okw = inb & (np.abs(t[jj] - t[pi]) <= max(5, 3 * abs(o)))
        d, _ = point_segment_distance(P[jj], A, B)
        W[:, c] = np.where(okw, d, np.nan)
    cols3 = [c for c, o in enumerate(OFFSETS) if abs(o) <= 3]
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore", RuntimeWarning)
        w3m, w3x, w8m = np.nanmean(W[:, cols3], 1), np.nanmax(W[:, cols3], 1), np.nanmean(W, 1)
    w3m, w3x, w8m = [np.where(np.isnan(x), dist, x) for x in (w3m, w3x, w8m)]   # empty window -> own distance
    hc = np.where(hvalid[pi], np.cos(heading[pi] - net.bearing[arc]), 0.0)
    f = pd.DataFrame({
        "trip_id": trip_id, "point_idx": pi.astype(np.int32), "arc_id": arc.astype(np.int64),
        "dist_m": dist, "dist_rel_min": dist - min_d, "dist_rank": rank, "proj_t": proj_t,
        "along_m": proj_t * net.length[arc], "rem_m": (1 - proj_t) * net.length[arc], "arc_len_m": net.length[arc],
        "heading_cos": hc, "heading_valid": hvalid[pi].astype(int), "is_oneway": (~net.has_reverse[arc]).astype(int),
        "turn_angle": turn[pi], "deg_from": deg[net.arcs[arc, 0]], "deg_to": deg[net.arcs[arc, 1]],
        "n_candidates": counts[pi], "speed_ms": speed[pi], "dt_s": np.minimum(dt[pi], 60.0),
        "gap_before": trk["gap_before"][pi].astype(int),
        "win3_mean_dist": w3m, "win3_max_dist": w3x, "win8_mean_dist": w8m})
    f[FEATURE_COLS] = f[FEATURE_COLS].astype(np.float32)
    # ---------------- labels / points table
    pts = pd.DataFrame({"trip_id": trip_id, "point_idx": idx, "n_candidates": counts})
    if labels is not None:
        lab = labels["labelable"].to_numpy(bool); true = labels["true_arc_id"].to_numpy()
        f["labelable"] = lab[pi]
        f["label"] = np.where(lab[pi], (arc == true[pi]).astype(int), -1).astype(np.int8)
        hit = np.bincount(pi[arc == true[pi]], minlength=n) > 0
        pts["true_arc_id"] = np.where(lab, true, -1); pts["labelable"] = lab; pts["true_in_candidates"] = hit & lab
    return pts, f
