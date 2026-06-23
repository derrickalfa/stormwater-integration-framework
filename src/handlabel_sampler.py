"""
HAND-LABEL SAMPLER  -  NLP Classification Evaluation (R2-02)
=============================================================
Draws a stratified blind sample of activity strings for manual labelling,
so the paper can report a real macro-F1 for the 14-category NLP classifier.

This script is the companion to handlabel_scorer.py. Run this first, then
label handlabel_TASK.csv, then run the scorer to get per-category metrics.

Outputs TWO files (paths from config.py):
  1) Paths.HANDLABEL_TASK   -- what YOU fill in.
                               Columns: activity_id, activity_text, true_category (BLANK)
                               The model's prediction is NOT shown so you label blind.
  2) Paths.HANDLABEL_KEY    -- the model's primary_category for each row.
                               Do NOT open until after labelling is complete.
                               The scorer joins on activity_id.

Stratified sampling:
  Allocation is proportional to each category's share of the corpus,
  with a floor of Params.HANDLABEL_MIN_PER_CAT so rare categories
  (e.g. Defective connection, Corrosion and abrasion) have enough records
  for reliable per-category F1.
  Only stormwater pipe features (feature_prefix == 'S') are included
  when Params.HANDLABEL_STORMWATER_ONLY is True, matching the paper scope.

All parameters (sample size, floor, seed, scope) are controlled via config.py
under the Params class — no hardcoded values in this script.

Usage:
  python handlabel_sampler.py             # uses config.py paths and params
  python handlabel_sampler.py --dir PATH  # override by_sheet directory only

Requires: pandas
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Import config — required (script is part of the pipeline)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Paths, Params

# ---------------------------------------------------------------------------
# All parameters come from config.py — do not edit here
# ---------------------------------------------------------------------------
SAMPLE_SIZE      = Params.HANDLABEL_SAMPLE_SIZE
MIN_PER_CATEGORY = Params.HANDLABEL_MIN_PER_CAT
RANDOM_STATE     = Params.HANDLABEL_RANDOM_STATE
STORMWATER_ONLY  = Params.HANDLABEL_STORMWATER_ONLY

SHEET_FILES = [
    "Export Worksheet_activities.csv",
    "Sheet1_activities.csv",
    "Sheet2_activities.csv",
    "Sheet3_activities.csv",
    "Sheet4_activities.csv",
    "Sheet5_activities.csv",
]

CATEGORY_ORDER = [
    "Blockages",
    "Blockage and Inspection",
    "General Cleaning",
    "Infiltration and inflow",
    "Flow velocity/ hydraulic conditions",
    "Cracks and fractures",
    "Defective lining",
    "Deformation",
    "Corrosion and abrasion of pipes",
    "Defective connection",
    "Broken manhole covers/ damaged manhole walls",
    "Environmental",
    "Inspection",
    "General Maintenance",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw stratified NLP evaluation sample for handlabelling"
    )
    parser.add_argument(
        "--dir",
        default=str(Paths.MAINTENANCE_BY_SHEET),
        help="Path to by_sheet directory (default: Paths.MAINTENANCE_BY_SHEET from config.py)",
    )
    return parser.parse_args()


def load_all(by_sheet_dir: str) -> pd.DataFrame:
    frames = []
    for fn in SHEET_FILES:
        path = os.path.join(by_sheet_dir, fn)
        if not os.path.exists(path):
            print(f"  [skip] not found: {path}")
            continue
        df = pd.read_csv(
            path,
            usecols=lambda c: c in (
                "activity_id", "activity_text",
                "feature_number", "feature_prefix",
                "primary_category",
            ),
            low_memory=False,
        )
        frames.append(df)
        print(f"  [load] {fn}: {len(df):,} rows")
    if not frames:
        raise SystemExit("No activity CSV files loaded. Check Paths.MAINTENANCE_BY_SHEET in config.py")
    return pd.concat(frames, ignore_index=True)


def build_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Infer feature_prefix from feature_number if not already present."""
    if "feature_prefix" not in df.columns or df["feature_prefix"].isna().all():
        df["feature_prefix"] = (
            df["feature_number"]
            .astype(str)
            .str.strip()
            .str[0]
            .str.upper()
        )
    return df


def allocate(counts: pd.Series, cats: list, total: int, floor: int) -> dict:
    """
    Proportional allocation with a per-category floor.
    Any category where the corpus has fewer records than floor
    is fully included (no oversampling).
    """
    alloc = {c: min(floor, int(counts[c])) for c in cats}
    used  = sum(alloc.values())

    remaining  = max(0, total - used)
    total_pool = sum(counts[c] for c in cats)

    if remaining > 0 and total_pool > 0:
        for c in cats:
            extra = int(round(remaining * counts[c] / total_pool))
            cap   = int(counts[c]) - alloc[c]
            alloc[c] += max(0, min(extra, cap))

    return alloc


def main():
    args = parse_args()

    print(f"\nHAND-LABEL SAMPLER")
    print(f"  Source dir   : {args.dir}")
    print(f"  Sample size  : {SAMPLE_SIZE}  (Params.HANDLABEL_SAMPLE_SIZE)")
    print(f"  Floor        : {MIN_PER_CATEGORY} per category  (Params.HANDLABEL_MIN_PER_CAT)")
    print(f"  Scope        : {'Stormwater (S-prefix) only' if STORMWATER_ONLY else 'All prefixes'}"
          f"  (Params.HANDLABEL_STORMWATER_ONLY)")
    print(f"  Random seed  : {RANDOM_STATE}  (Params.HANDLABEL_RANDOM_STATE)")
    print(f"  Task output  : {Paths.HANDLABEL_TASK}")
    print(f"  Key output   : {Paths.HANDLABEL_KEY}\n")

    # Load
    df = load_all(args.dir)
    df = build_prefix(df)

    # Filter to stormwater scope
    if STORMWATER_ONLY:
        before = len(df)
        df = df[df["feature_prefix"].astype(str).str.upper().str.startswith("S")]
        print(f"Stormwater filter: {before:,} → {len(df):,} rows")

    # Drop rows without usable text or category
    df = df.dropna(subset=["activity_text", "primary_category"])
    df = df[df["activity_text"].astype(str).str.strip().ne("")]
    print(f"After cleaning : {len(df):,} activity strings across "
          f"{df['primary_category'].nunique()} categories")

    # Category counts
    counts    = df["primary_category"].value_counts()
    cats      = [c for c in CATEGORY_ORDER if counts.get(c, 0) > 0]
    missing_cats = [c for c in CATEGORY_ORDER if counts.get(c, 0) == 0]
    if missing_cats:
        print(f"\n[warn] These categories have no records and will be skipped:")
        for c in missing_cats:
            print(f"  - {c}")

    # Allocation
    alloc = allocate(counts, cats, SAMPLE_SIZE, MIN_PER_CATEGORY)

    # Draw stratified sample
    parts = []
    for c in cats:
        sub = df[df["primary_category"] == c]
        n   = min(alloc[c], len(sub))
        parts.append(sub.sample(n=n, random_state=RANDOM_STATE))
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Output files — paths from config.py
    Paths.ensure_output_dirs()

    task = sample[["activity_id", "activity_text"]].copy()
    task["true_category"] = ""   # labeller fills this in
    task.to_csv(Paths.HANDLABEL_TASK, index=False)

    key = sample[["activity_id", "primary_category"]].copy()
    key.to_csv(Paths.HANDLABEL_KEY, index=False)

    # Summary
    print(f"\nStratified allocation ({len(sample)} total):")
    print(f"  {'Category':<48} {'Pool':>8} {'Sampled':>8}")
    print(f"  {'-'*48} {'-'*8} {'-'*8}")
    for c in cats:
        print(f"  {c:<48} {int(counts[c]):>8,} {alloc[c]:>8}")

    print(f"\nOutput files:")
    print(f"  {Paths.HANDLABEL_TASK}  ← fill the true_category column (label blind)")
    print(f"  {Paths.HANDLABEL_KEY}   ← do not open until labelling is complete")

    print(f"\nValid category labels (copy exactly):")
    for c in CATEGORY_ORDER:
        print(f"  {c}")

    print(f"\nNext step: label {Paths.HANDLABEL_TASK.name}, then run:")
    print(f"  python handlabel_scorer.py")


if __name__ == "__main__":
    main()
