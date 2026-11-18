"""Step 3: derive per-point labels for train + student_test and write a label-quality report.

Usage: python scripts/04_derive_labels.py --data-root "path/to/Student folder"
Optional: --map-trip 00000002 --map-split train   (writes a Folium map of labelable/unlabelable points)
Outputs : data/labels/labels_<split>.parquet   (git-ignored; regenerate with this script)
          results/labels/label_quality.csv, label_summary.csv, miss_vs_radius.png
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import list_trips, load_trip
from src.data.network import RoadNetwork, prepare_track
from src.data.labels import derive_point_labels

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--splits", nargs="+", default=["train", "student_test"])
ap.add_argument("--max-label-dist", type=float, default=100.0, help="metres; points farther from their assigned arc are unlabelable")
ap.add_argument("--cap", type=float, default=100.0, help="metres; distance truncation inside the assignment cost")
ap.add_argument("--unreliable-below", type=float, default=0.90, help="flag trips with a smaller labelable share")
ap.add_argument("--radii", nargs="+", type=float, default=[20, 30, 50, 75, 100])
ap.add_argument("--map-trip"); ap.add_argument("--map-split", default="train")
a = ap.parse_args()
Path("data/labels").mkdir(parents=True, exist_ok=True); Path("results/labels").mkdir(parents=True, exist_ok=True)

qrows, dists = [], {s: [] for s in a.splits}
for split in a.splits:
    frames = []
    for tid in list_trips(a.data_root, split):
        trip = load_trip(a.data_root, split, tid)
        net = RoadNetwork.from_trip(trip); P = prepare_track(trip.track, net.to_xy)["xy"]
        t0 = time.time()
        r = derive_point_labels(net, P, trip.route, a.max_label_dist, a.cap)
        frames.append(pd.DataFrame(dict(trip_id=tid, point_idx=np.arange(len(P)), route_pos=r["route_pos"],
                                        true_arc_id=r["true_arc_id"], assign_dist_m=r["assign_dist_m"], labelable=r["labelable"])))
        d, lab = r["assign_dist_m"], r["labelable"]
        row = dict(split=split, trip_id=tid, n_points=len(P), n_route_arcs=len(trip.route),
                   frac_labelable=lab.mean(), dist_median_m=np.median(d[lab]) if lab.any() else np.nan,
                   dist_p99_m=np.percentile(d[lab], 99) if lab.any() else np.nan,
                   first_pos=int(r["route_pos"][0]), last_pos=int(r["route_pos"][-1]),
                   unassigned_route_arcs=r["n_unassigned_route_arcs"], seconds=round(time.time() - t0, 1))
        for R in a.radii:   # candidate miss rate among labelable points if candidates came from radius R
            row[f"miss_R{int(R)}"] = float((d[lab] > R).mean()) if lab.any() else np.nan
        qrows.append(row); dists[split].append(d[lab])
        print(f"{split}/{tid}: pts={len(P):6d} labelable={row['frac_labelable']:.3f} median={row['dist_median_m']:.1f}m ({row['seconds']}s)")
        if a.map_trip == tid and a.map_split == split:
            import folium
            m = folium.Map(location=[trip.track[:, 1].mean(), trip.track[:, 0].mean()], zoom_start=14, tiles=None)
            folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                             attr="Tiles &copy; Esri", name="Esri").add_to(m)
            rg = folium.FeatureGroup(name="Reference route (green)")
            for k in trip.route:
                f, to = trip.arcs[k]
                folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[to, 1], trip.nodes[to, 0]]], color="#00a651", weight=4).add_to(rg)
            rg.add_to(m)
            for nm, mask, col in [("Labelable points (blue)", lab, "#1f5fff"), ("UNlabelable points (red)", ~lab, "#e00000")]:
                g = folium.FeatureGroup(name=nm); idx = np.flatnonzero(mask); step = max(1, len(idx) // 2500)
                for i in idx[::step]:
                    folium.CircleMarker([trip.track[i, 1], trip.track[i, 0]], radius=2, color=col, fill=True,
                                        tooltip=f"idx {i} | route pos {r['route_pos'][i]} | arc {r['true_arc_id'][i]} | {d[i]:.0f} m").add_to(g)
                g.add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)
            Path("maps").mkdir(exist_ok=True); fp = f"maps/labels_{tid}_{split}.html"; m.save(fp); print("  map ->", fp)
    pd.concat(frames, ignore_index=True).to_parquet(f"data/labels/labels_{split}.parquet", index=False)

q = pd.DataFrame(qrows); q["unreliable"] = q.frac_labelable < a.unreliable_below
q.to_csv("results/labels/label_quality.csv", index=False)
print("\n=== Label quality by split (per-trip means) ===")
cols = ["frac_labelable", "dist_median_m"] + [f"miss_R{int(R)}" for R in a.radii]
summ = q.groupby("split")[cols].mean().round(4); print(summ.T.to_string()); summ.to_csv("results/labels/label_summary.csv")
print("\nUnreliable trips (labelable share <", a.unreliable_below, "):")
print(q[q.unreliable][["split", "trip_id", "n_points", "frac_labelable"]].round(3).to_string(index=False) or "none")
print("\nTrips whose route had arcs never assigned to any point:", int((q.unassigned_route_arcs > 0).sum()))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s in a.splits:
    d = np.concatenate(dists[s]); xs = np.array(a.radii + [150]); 
    ax[0].plot(xs, [(d > R).mean() * 100 for R in xs], marker="o", label=s)
    ax[1].hist(np.clip(d, 0, a.max_label_dist), bins=50, alpha=.6, label=s, density=True)
ax[0].set_xlabel("candidate radius R (m)"); ax[0].set_ylabel("% labelable points whose true arc is farther than R"); ax[0].set_title("Candidate miss rate vs radius"); ax[0].legend()
ax[1].set_xlabel("distance to assigned arc (m)"); ax[1].set_title("GPS noise vs derived labels (labelable points)"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/labels/miss_vs_radius.png", dpi=130)
print("\nWrote results/labels/ and data/labels/")
