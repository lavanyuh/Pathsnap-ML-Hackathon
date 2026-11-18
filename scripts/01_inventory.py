"""Step 1a: dataset inventory + EDA plots (labelled partitions only).

Usage: python scripts/01_inventory.py --data-root "path/to/Student folder"
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import list_trips, load_trip
from src.data.geo import haversine_m

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--splits", nargs="+", default=["train", "student_test"])
ap.add_argument("--out", default="results/eda")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

rows = []
for split in a.splits:
    for tid in list_trips(a.data_root, split):
        t = load_trip(a.data_root, split, tid)
        tr, no, ar, ro = t.track, t.nodes, t.arcs, t.route
        dt = np.diff(tr[:, 2])
        step = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
        L = haversine_m(no[ar[:, 0], 0], no[ar[:, 0], 1], no[ar[:, 1], 0], no[ar[:, 1], 1])
        pairs = set(map(tuple, ar))
        deg = np.bincount(ar.ravel(), minlength=len(no))
        rows.append(dict(
            split=split, trip_id=tid, n_points=len(tr), n_nodes=len(no), n_arcs=len(ar),
            n_route_arcs=len(ro) if ro is not None else np.nan,
            n_unique_route_arcs=len(set(ro)) if ro is not None else np.nan,
            file_kb=sum(p.stat().st_size for p in (Path(a.data_root)/split/tid).iterdir())/1024,
            nan_values=int(np.isnan(tr).sum()),
            lon_ok=bool(np.all(np.abs(tr[:, 0]) <= 180)), lat_ok=bool(np.all(np.abs(tr[:, 1]) <= 90)),
            t_start=tr[0, 2], duration_s=tr[-1, 2] - tr[0, 2],
            frac_dt_eq_1=float((dt == 1).mean()), dt_max=float(dt.max()), n_dt_gt5=int((dt > 5).sum()),
            n_dt_nonpositive=int((dt <= 0).sum()),
            step_median_m=float(np.median(step)), step_max_m=float(step.max()),
            track_length_km=float(step.sum() / 1000),
            bbox_diag_km=float(haversine_m(tr[:, 0].min(), tr[:, 1].min(), tr[:, 0].max(), tr[:, 1].max()) / 1000),
            centre_lon=float(tr[:, 0].mean()), centre_lat=float(tr[:, 1].mean()),
            arc_len_median_m=float(np.median(L)), n_zero_len_arcs=int((L == 0).sum()),
            n_self_loops=int((ar[:, 0] == ar[:, 1]).sum()), n_duplicate_arcs=int(len(ar) - len(pairs)),
            frac_arcs_with_reverse=float(np.mean([(b, c) in pairs for c, b in ar])),
            n_isolated_nodes=int((deg == 0).sum()),
            route_len_km=float(L[ro].sum() / 1000) if ro is not None else np.nan,
            route_consecutive_connected=float((ar[ro[:-1], 1] == ar[ro[1:], 0]).mean())
                if ro is not None and len(ro) > 1 else np.nan,
        ))
df = pd.DataFrame(rows)
df.to_csv(out / "inventory.csv", index=False)

# --- summary tables
print(df.groupby("split")[["n_points", "n_nodes", "n_arcs", "n_route_arcs"]].describe().T.round(1).to_string())
print("\nTrips per split:\n", df.split.value_counts().to_string())
print("\nTotal missing values:", int(df.nan_values.sum()),
      "| non-positive time steps:", int(df.n_dt_nonpositive.sum()))
(df.groupby("split").agg(trips=("trip_id", "count"), points=("n_points", "sum"),
                         nodes=("n_nodes", "sum"), arcs=("n_arcs", "sum"))
   ).to_csv(out / "partition_summary.csv")

# --- plots
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for s, g in df.groupby("split"):
    ax[0, 0].hist(g.n_points, bins=25, alpha=.6, label=s)
    ax[0, 1].hist(g.frac_dt_eq_1, bins=25, alpha=.6, label=s)
    ax[0, 2].scatter(g.n_nodes, g.n_arcs, s=14, alpha=.7, label=s)
    ax[1, 0].hist(np.log10(g.dt_max), bins=25, alpha=.6, label=s)
    ax[1, 1].hist(g.bbox_diag_km, bins=25, alpha=.6, label=s)
    ax[1, 2].scatter(g.centre_lon, g.centre_lat, s=18, alpha=.8, label=s)
for axx, (t, xl) in zip(ax.ravel(), [("GPS points per trip", "points"), ("Share of steps with dt = 1 s", "fraction"),
                                     ("Road network size", "nodes"), ("Largest time gap per trip", "log10(seconds)"),
                                     ("Trip extent (bbox diagonal)", "km"), ("Trip locations (worldwide!)", "lon")]):
    axx.set_title(t); axx.set_xlabel(xl); axx.legend()
ax[0, 2].set_ylabel("arcs"); ax[1, 2].set_ylabel("lat")
plt.tight_layout(); plt.savefig(out / "eda_overview.png", dpi=130); plt.close()

print("\nData-quality flags:")
print(" trips with dt gap > 60 s:", df[df.dt_max > 60].trip_id.tolist())
print(" trips with zero-length arcs / self loops / duplicate arcs:",
      int((df.n_zero_len_arcs > 0).sum()), int((df.n_self_loops > 0).sum()), int((df.n_duplicate_arcs > 0).sum()))
print(" trips where route is not fully connected:", df[df.route_consecutive_connected < 1].trip_id.tolist())
print(f"\nWrote {out/'inventory.csv'} and {out/'eda_overview.png'}")
