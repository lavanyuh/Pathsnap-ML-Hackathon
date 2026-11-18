"""Batch evaluate all 10 reserved evaluator trips.

Generates evaluation results summary CSV and Markdown table for the final report.
"""
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.pipeline import MapMatchingPipeline
from src.data.loader import list_trips

def main():
    pipe = MapMatchingPipeline()
    trips = list_trips(str(pipe.data_root), "evaluator_test")
    print(f"Evaluating {len(trips)} reserved evaluator trips with {pipe.model_name.upper()} + HMM Viterbi...")

    records = []
    for tid in trips:
        res = pipe.process_trip(tid, split="evaluator_test")
        m = res["metrics"]
        acc = m.get("accuracy", {})
        seq = m.get("sequence", {})

        records.append({
            "Trip ID": tid,
            "GPS Points": m["n_gps_points"],
            "Candidates": m["n_candidate_rows"],
            "Mean Cands/Pt": round(m["mean_candidates_per_point"], 2),
            "HMM Accuracy (%)": acc.get("hmm_pointwise_accuracy", 0.0),
            "Classifier Accuracy (%)": acc.get("classifier_pointwise_accuracy", 0.0),
            "HMM Gain (%)": acc.get("hmm_accuracy_gain", 0.0),
            "Route Continuity (%)": seq.get("continuity", 0.0),
            "Breaks": seq.get("breaks", 0),
            "Sequence Precision (%)": seq.get("precision", 0.0),
            "Sequence Recall (%)": seq.get("recall", 0.0),
            "Sequence F1 (%)": seq.get("f1_score", 0.0),
            "Latency (ms)": m["total_latency_ms"]
        })

    df = pd.DataFrame(records)
    out_path = Path("results/models/evaluator_test_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("PHASE 2 RESERVED EVALUATION TRIPS SUMMARY (Model: LightGBM + HMM)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Average metrics
    avg_hmm = df["HMM Accuracy (%)"].mean()
    avg_clf = df["Classifier Accuracy (%)"].mean()
    avg_cont = df["Route Continuity (%)"].mean()
    avg_lat = df["Latency (ms)"].mean()
    print(f"\nOverall Summary Across All 10 Evaluation Trips:")
    print(f"  • Average HMM Pointwise Accuracy: {avg_hmm:.2f}%")
    print(f"  • Average Classifier Accuracy:    {avg_clf:.2f}%")
    print(f"  • Average Route Continuity:       {avg_cont:.2f}%")
    print(f"  • Average Processing Latency:     {avg_lat:.1f} ms/trip")
    print(f"\nSaved summary to {out_path}")

if __name__ == "__main__":
    main()
