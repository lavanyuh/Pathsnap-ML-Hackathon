"""Steps 4-5: candidates + features for train and student_test; fit the train-only preprocessor.

Run AFTER 04_derive_labels.py:
  python scripts/05_build_features.py --data-root "path/to/Student folder" --radius 50
Outputs: data/features/points_<split>.parquet, features_<split>.parquet   (git-ignored; share via Drive)
         artifacts/preprocess.joblib, results/features/*.csv|png
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import list_trips, load_trip
from src.data.network import RoadNetwork, prepare_track
from src.features.build import build_trip_features
from src.features.columns import FEATURE_COLS
from src.features.preprocess import fit_preprocessor

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--radius", type=float, default=50.0)
ap.add_argument("--exclude-train", nargs="*", default=["00000033", "00000054"],
                help="train trips whose reference route is mostly unusable (see label_quality.csv). Student_test is never filtered.")
a = ap.parse_args()
Path("data/features").mkdir(parents=True, exist_ok=True); Path("results/features").mkdir(parents=True, exist_ok=True)

stats, feats = [], {}
for split in ["train", "student_test"]:
    labels_all = pd.read_parquet(f"data/labels/labels_{split}.parquet")
    P_list, F_list = [], []
    for tid in list_trips(a.data_root, split):
        if split == "train" and tid in a.exclude_train:
            print(f"skip {split}/{tid} (excluded)"); continue
        trip = load_trip(a.data_root, split, tid, with_route=False)      # route not needed: labels come from step 3
        net = RoadNetwork.from_trip(trip); trk = prepare_track(trip.track, net.to_xy)
        lab = labels_all[labels_all.trip_id == tid].sort_values("point_idx")
        t0 = time.time(); pts, f = build_trip_features(net, trk, tid, a.radius, lab)
        P_list.append(pts); F_list.append(f)
        lp = pts[pts.labelable]
        stats.append(dict(split=split, trip_id=tid, n_points=len(pts), labelable_points=int(pts.labelable.sum()), rows=len(f),
                          mean_candidates=pts.n_candidates.mean(), zero_cand_frac=(pts.n_candidates == 0).mean(),
                          miss_rate=1 - lp.true_in_candidates.mean() if len(lp) else np.nan, seconds=round(time.time() - t0, 1)))
        print(f"{split}/{tid}: rows={len(f):7d} miss={stats[-1]['miss_rate']:.4f} ({stats[-1]['seconds']}s)")
    pts_df, f_df = pd.concat(P_list, ignore_index=True), pd.concat(F_list, ignore_index=True)
    pts_df.to_parquet(f"data/features/points_{split}.parquet", index=False); f_df.to_parquet(f"data/features/features_{split}.parquet", index=False)
    feats[split] = (pts_df, f_df)

pipe = fit_preprocessor(feats["train"][1])          # TRAIN ONLY
st = pd.DataFrame(stats); st.to_csv("results/features/candidate_stats_per_trip.csv", index=False)
rows = []
for s, (p, f) in feats.items():
    lp = p[p.labelable]
    rows.append(dict(split=s, radius_m=a.radius, trips=p.trip_id.nunique(), points=len(p), labelable_points=len(lp), feature_rows=len(f),
                     mean_candidates=p.n_candidates.mean(), points_with_zero_candidates=int((p.n_candidates == 0).sum()),
                     candidate_miss_rate_labelable=1 - lp.true_in_candidates.mean(),
                     positive_rate_labelable_rows=(f[f.labelable].label == 1).mean()))
summ = pd.DataFrame(rows); summ.to_csv("results/features/candidate_stats.csv", index=False); print("\n", summ.T.to_string())
tr = feats["train"][1]; tr = tr[tr.labelable]
tr[FEATURE_COLS].groupby(tr.label).mean().T.rename(columns={0: "mean_neg", 1: "mean_pos"}).to_csv("results/features/feature_means_by_label.csv")
fig, ax = plt.subplots(2, 4, figsize=(16, 7)); smp = tr.sample(min(len(tr), 200000), random_state=42)
for axx, c in zip(ax.ravel(), ["dist_m", "dist_rel_min", "heading_cos", "win3_mean_dist", "win8_mean_dist", "proj_t", "dist_rank", "arc_len_m"]):
    hi = smp[c].quantile(.99)
    for lv, nm in [(0, "not true arc"), (1, "true arc")]:
        axx.hist(smp[smp.label == lv][c].clip(upper=hi), bins=40, alpha=.55, density=True, label=nm)
    axx.set_title(c); axx.legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/features/feature_distributions.png", dpi=120)
print("\nPreprocessor saved to artifacts/preprocess.joblib (fit on train only). Done.")
