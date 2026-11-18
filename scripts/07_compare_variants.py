"""Print CV and student-test metrics side by side for every results/models/<prefix>* variant.

Usage (repo root):  python scripts/07_compare_variants.py logreg
Select variants using the CV columns, NOT the student-test columns.
"""
import json, sys
from pathlib import Path
import pandas as pd

prefix = sys.argv[1] if len(sys.argv) > 1 else "logreg"
rows = []
for d in sorted(Path("results/models").glob(prefix + "*")):
    cv = json.loads((d / "cv_metrics.json").read_text())["summary"]
    te = json.loads((d / "student_test_metrics.json").read_text())
    rows.append({"variant": d.name,
                 "CV macroF1": f"{cv['macro_f1']['mean']:.4f} +- {cv['macro_f1']['std']:.4f}",
                 "CV top1": f"{cv['top1']['mean']:.4f} +- {cv['top1']['std']:.4f}",
                 "CV logloss": f"{cv['log_loss']['mean']:.4f} +- {cv['log_loss']['std']:.4f}",
                 "test macroF1": round(te["macro_f1"], 4), "test top1": round(te["top1"], 4),
                 "test logloss": round(te["log_loss"], 4), "train_s": round(te["train_time_s"], 1)})
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).to_string(index=False))
