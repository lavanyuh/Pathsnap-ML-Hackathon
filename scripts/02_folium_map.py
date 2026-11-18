"""Step 1b: interactive Folium map of one trip: GPS points, road network, reference route.

Usage: python scripts/02_folium_map.py --data-root "path/to/Student folder" --split train --trip 00000005
Later (step 8) you will add the predicted route as another layer.
"""
import argparse, sys
from pathlib import Path
import numpy as np, folium
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import load_trip

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--trip", required=True)
ap.add_argument("--max-points", type=int, default=3000, help="subsample GPS dots for responsiveness")
ap.add_argument("--buffer-deg", type=float, default=0.003, help="network drawn within bbox of track + buffer")
ap.add_argument("--out", default="maps")
a = ap.parse_args()

t = load_trip(a.data_root, a.split, a.trip)
tr, no, ar, ro = t.track, t.nodes, t.arcs, t.route
m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])

# Road network (only near the track, otherwise huge networks make the HTML unusable)
b = a.buffer_deg
inbox = ((no[:, 0] >= tr[:, 0].min() - b) & (no[:, 0] <= tr[:, 0].max() + b) &
         (no[:, 1] >= tr[:, 1].min() - b) & (no[:, 1] <= tr[:, 1].max() + b))
net = folium.FeatureGroup(name="Road network (arcs)", show=True)
for f, to in ar[inbox[ar[:, 0]] & inbox[ar[:, 1]]]:
    folium.PolyLine([[no[f, 1], no[f, 0]], [no[to, 1], no[to, 0]]], color="#999999", weight=1.5, opacity=.7).add_to(net)
net.add_to(m)

if ro is not None:
    rt = folium.FeatureGroup(name="Reference route", show=True)
    for k in ro:
        f, to = ar[k]
        folium.PolyLine([[no[f, 1], no[f, 0]], [no[to, 1], no[to, 0]]], color="#00a651", weight=4, opacity=.9,
                        tooltip=f"arc {k}").add_to(rt)
    rt.add_to(m)

gp = folium.FeatureGroup(name="Raw GPS observations", show=True)
step = max(1, len(tr) // a.max_points)
for i in range(0, len(tr), step):
    folium.CircleMarker([tr[i, 1], tr[i, 0]], radius=2, color="#1f5fff", fill=True, fill_opacity=.8,
                        tooltip=f"idx {i}, t={tr[i,2]:.0f}s").add_to(gp)
gp.add_to(m)
folium.Marker([tr[0, 1], tr[0, 0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)

legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;
border:1px solid #888;font-size:13px"><b>Trip %s (%s)</b><br>
<span style="color:#1f5fff">&#9679;</span> raw GPS<br><span style="color:#00a651">&#9632;</span> reference route<br>
<span style="color:#999">&#9632;</span> road network</div>""" % (a.trip, a.split)
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl(collapsed=False).add_to(m)
Path(a.out).mkdir(exist_ok=True)
fp = Path(a.out) / f"trip_{a.trip}_{a.split}.html"
m.save(fp); print("saved", fp)
