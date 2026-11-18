"""Step 2: build the projected network + spatial index for every labelled trip, validate the
index against brute force, and summarise preprocessing + candidate-set sizes.
No labels are used here.

Usage: python scripts/03_preprocess_check.py --data-root "path/to/Student folder"
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import list_trips, load_trip
from src.data.network import RoadNetwork, prepare_track
from src.data.geo import point_segment_distance

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--splits", nargs="+", default=["train", "student_test"])
ap.add_argument("--radii", nargs="+", type=float, default=[30, 50, 100])
ap.add_argument("--out", default="results/eda")
a = ap.parse_args()
rng = np.random.default_rng(0)
rows = []
for split in a.splits:
    for tid in list_trips(a.data_root, split):
        trip = load_trip(a.data_root, split, tid, with_route=False)   # labels not needed
        t0 = time.time(); net = RoadNetwork.from_trip(trip); t_build = time.time() - t0
        trk = prepare_track(trip.track, net.to_xy)
        P = trk["xy"]
        row = dict(split=split, trip_id=tid, n_points=len(P), n_arcs=len(net.arcs),
                   invalid_arcs=int((~net.valid).sum()), oneway_arcs=int((~net.has_reverse).sum()),
                   build_s=round(t_build, 2), n_gap_points=int(trk["gap_before"].sum()),
                   n_speed_flag=int(trk["speed_flag"].sum()),
                   speed_p99_ms=float(np.nanpercentile(trk["speed"], 99)))
        for R in a.radii:
            t0 = time.time(); pi, arc, d, tt = net.arcs_within(P, R); q = time.time() - t0
            cnt = np.bincount(pi, minlength=len(P))
            row.update({f"R{int(R)}_mean_cand": cnt.mean(), f"R{int(R)}_max_cand": int(cnt.max()),
                        f"R{int(R)}_zero_cand_frac": float((cnt == 0).mean()), f"R{int(R)}_query_s": round(q, 2)})
        # brute-force validation of the index on 40 random points at the middle radius
        R = a.radii[len(a.radii) // 2]
        pi, arc, d, tt = net.arcs_within(P, R)
        ok = True
        for i in rng.choice(len(P), size=min(40, len(P)), replace=False):
            ids = np.flatnonzero(net.valid)
            dd, _ = point_segment_distance(np.tile(P[i], (len(ids), 1)), net.A[ids], net.B[ids])
            if set(ids[dd <= R]) != set(arc[pi == i]): ok = False
        row["index_matches_bruteforce"] = ok
        rows.append(row)
        print(f"{split}/{tid}: pts={len(P)} arcs={len(net.arcs)} build={t_build:.1f}s ok={ok}")
df = pd.DataFrame(rows)
Path(a.out).mkdir(parents=True, exist_ok=True)
df.to_csv(Path(a.out) / "preprocess_summary.csv", index=False)
print("\nAll index checks pass:", bool(df.index_matches_bruteforce.all()))
print(df.groupby("split")[[c for c in df.columns if "cand" in c or c in ("n_gap_points", "n_speed_flag", "invalid_arcs")]].mean().round(2).T.to_string())
print("slowest build:", df.loc[df.build_s.idxmax(), ["trip_id", "n_arcs", "build_s"]].to_dict())
