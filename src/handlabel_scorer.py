"""
HAND-LABEL SCORER  -  NLP Classification Evaluation (R2-02)
============================================================
After you finish labelling Paths.HANDLABEL_TASK, this script joins your
labels to the model's predictions (Paths.HANDLABEL_KEY) and computes:
  - overall macro-F1  (the number for the manuscript paragraph)
  - overall accuracy
  - per-category precision / recall / F1 / support
  - a confusion table of the most common disagreements

Results are saved to Paths.HANDLABEL_RESULT.

All file paths come from config.py — no hardcoded paths in this script.

Usage:
  python handlabel_scorer.py

Requires: pandas, scikit-learn
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# Import config — required (script is part of the pipeline)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Paths


def main():
    # ── Load files ────────────────────────────────────────────────────────────
    if not Paths.HANDLABEL_TASK.exists():
        raise SystemExit(
            f"Task file not found: {Paths.HANDLABEL_TASK}\n"
            "Run handlabel_sampler.py first, then complete the labelling."
        )
    if not Paths.HANDLABEL_KEY.exists():
        raise SystemExit(
            f"Key file not found: {Paths.HANDLABEL_KEY}\n"
            "Run handlabel_sampler.py first."
        )

    task = pd.read_csv(Paths.HANDLABEL_TASK)
    key  = pd.read_csv(Paths.HANDLABEL_KEY)

    # ── Validate labels ───────────────────────────────────────────────────────
    task["true_category"] = task["true_category"].astype(str).str.strip()
    blank = task["true_category"].eq("") | task["true_category"].str.lower().eq("nan")
    if blank.any():
        print(f"[warn] {blank.sum()} rows have no label yet — they are excluded.")
        task = task[~blank]

    # ── Merge on activity_id ──────────────────────────────────────────────────
    merged = task.merge(key, on="activity_id", how="inner")
    y_true = merged["true_category"]
    y_pred = merged["primary_category"]
    n      = len(merged)

    if n == 0:
        raise SystemExit("No labelled rows to score. Fill in true_category in the task file.")

    # ── Overall metrics ───────────────────────────────────────────────────────
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc         = accuracy_score(y_true, y_pred)

    print("=" * 64)
    print(f"  Records scored : {n}")
    print(f"  Macro-F1       : {macro_f1:.3f}   <-- report this in the manuscript")
    print(f"  Weighted-F1    : {weighted_f1:.3f}")
    print(f"  Accuracy       : {acc:.3f}")
    print("=" * 64)

    # ── Per-category metrics ──────────────────────────────────────────────────
    print("\nPer-category metrics:")
    print(classification_report(y_true, y_pred, zero_division=0, digits=3))

    # ── Top disagreements ─────────────────────────────────────────────────────
    wrong = merged[y_true.values != y_pred.values]
    if len(wrong):
        print(f"Top disagreements (human label → model prediction), {len(wrong)} total:")
        combo = wrong["true_category"] + "  →  " + wrong["primary_category"]
        for label, cnt in combo.value_counts().head(12).items():
            print(f"  {cnt:>4}  {label}")

    # ── Save summary ──────────────────────────────────────────────────────────
    Paths.ensure_output_dirs()
    pd.DataFrame([{
        "n_records":    n,
        "macro_f1":     round(macro_f1,    4),
        "weighted_f1":  round(weighted_f1, 4),
        "accuracy":     round(acc,         4),
    }]).to_csv(Paths.HANDLABEL_RESULT, index=False)
    print(f"\nSaved summary → {Paths.HANDLABEL_RESULT}")


if __name__ == "__main__":
    main()
