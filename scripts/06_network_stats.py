"""EDA: road-network size and connectivity per trip (required by Dataset Exploration.pdf section 4).

Weakly connected components ignore arc direction; strongly connected components respect it
(you can drive from any node to any other). The HMM can only link arcs that are mutually reachable,
so this matters for step 8.
Usage: python scripts/06_network_stats.py --data-root "path/to/Student folder"
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import list_trips, load_trip
from src.data.network import RoadNetwork

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--splits", nargs="+", default=["train", "student_test"])
a = ap.parse_args()
Path("results/eda").mkdir(parents=True, exist_ok=True)
rows = []
for split in a.splits:
    for tid in list_trips(a.data_root, split):
        trip = load_trip(a.data_root, split, tid)
        net = RoadNetwork.from_trip(trip)
        m = len(net.node_xy); v = net.valid
        g = coo_matrix((np.ones(v.sum()), (net.arcs[v, 0], net.arcs[v, 1])), shape=(m, m)).tocsr()
        nw, lw = connected_components(g, directed=True, connection="weak")
        ns, ls = connected_components(g, directed=True, connection="strong")
        big_w, big_s = np.bincount(lw).argmax(), np.bincount(ls).argmax()
        deg = np.bincount(net.arcs[v].ravel(), minlength=m)
        route_nodes = np.unique(trip.arcs[trip.route].ravel())
        rows.append(dict(split=split, trip_id=tid, n_nodes=m, n_arcs=len(net.arcs), n_valid_arcs=int(v.sum()),
                         weak_components=nw, largest_weak_share=float((lw == big_w).mean()),
                         strong_components=ns, largest_strong_share=float((ls == big_s).mean()),
                         dead_end_nodes=int((deg == 1).sum()), isolated_nodes=int((deg == 0).sum()),
                         mean_degree=float(deg.mean()), max_degree=int(deg.max()),
                         route_nodes_in_largest_strong=float((ls[route_nodes] == big_s).mean()),
                         route_nodes_in_largest_weak=float((lw[route_nodes] == big_w).mean())))
df = pd.DataFrame(rows); df.to_csv("results/eda/network_stats.csv", index=False)
print(df.groupby("split")[["weak_components", "largest_weak_share", "strong_components", "largest_strong_share",
                           "dead_end_nodes", "route_nodes_in_largest_strong", "route_nodes_in_largest_weak"]].describe().T.round(3).to_string())
print("\nTrips whose route leaves the largest strongly connected component:",
      df[df.route_nodes_in_largest_strong < 1][["trip_id", "route_nodes_in_largest_strong"]].round(3).values.tolist())
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(df.largest_weak_share, bins=30); ax[0].set_title("Share of nodes in largest weak component")
ax[1].hist(df.largest_strong_share, bins=30); ax[1].set_title("Share of nodes in largest strong component")
ax[2].scatter(df.n_nodes, df.strong_components, s=14); ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_title("Strong components vs network size"); ax[2].set_xlabel("nodes"); ax[2].set_ylabel("components")
plt.tight_layout(); plt.savefig("results/eda/network_connectivity.png", dpi=130)
