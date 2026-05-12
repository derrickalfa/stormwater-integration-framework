"""
07_data_quality_validation.py
==============================
Removes physically impossible records; retains records with missing data.

Removal criteria (physically impossible values only):
  - Water_Depth or Silt_Depth exceeds Prop_Manhole_Depth
  - Silt_Depth > Water_Depth when water is present (invalid measurement)
  - Negative Water_Depth, Silt_Depth, or Prop_Manhole_Depth
  - Prop_Manhole_Depth < MIN_MANHOLE_DEPTH_MM
  - Avg_Invert_Level outside plausible range for Hong Kong [-50, 200 m]

Records with missing data are KEPT — ML algorithms handle imputation.

Author: Research Team
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, warnings
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

warnings.filterwarnings("ignore")


def _print(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def check_physical(df: pd.DataFrame) -> tuple[pd.Series, list]:
    """Return (removal_mask, summary_list)."""
    masks, summary = {}, []

    def _add(key, mask, label):
        masks[key] = mask
        summary.append({"Issue": label, "Count": mask.sum(),
                         "Percentage": mask.sum() / len(df) * 100, "Action": "REMOVE"})
        _print(f"  {label}: {mask.sum():,} ({mask.sum()/len(df)*100:.2f}%)")

    if {"Water_Depth", "Prop_Manhole_Depth"}.issubset(df.columns):
        _add("water_gt_depth", df["Water_Depth"] > df["Prop_Manhole_Depth"],
             "Water_Depth > Manhole_Depth")

    if {"Silt_Depth", "Prop_Manhole_Depth"}.issubset(df.columns):
        _add("silt_gt_depth", df["Silt_Depth"] > df["Prop_Manhole_Depth"],
             "Silt_Depth > Manhole_Depth")

    if {"Silt_Depth", "Water_Depth"}.issubset(df.columns):
        w = df["Water_Depth"].fillna(0)
        s = df["Silt_Depth"].fillna(0)
        _add("silt_gt_water", (s > w) & (w > 0), "Silt_Depth > Water_Depth (with flowing water)")
        dry = (w == 0) & (s > 0)
        _print(f"  Dry conditions (Water=0, Silt>0): {dry.sum():,} — KEPT")

    for col in ["Water_Depth", "Silt_Depth", "Prop_Manhole_Depth"]:
        if col in df.columns:
            mask = df[col] < 0
            if mask.sum() > 0:
                _add(f"{col}_negative", mask, f"Negative {col}")

    if "Prop_Manhole_Depth" in df.columns:
        mask = (df["Prop_Manhole_Depth"] > 0) & (df["Prop_Manhole_Depth"] < Params.MIN_MANHOLE_DEPTH_MM)
        if mask.sum() > 0:
            _add("depth_too_small", mask, f"Prop_Manhole_Depth < {Params.MIN_MANHOLE_DEPTH_MM} mm")

    if "Avg_Invert_Level" in df.columns:
        mask = (df["Avg_Invert_Level"] < -50) | (df["Avg_Invert_Level"] > 200)
        if mask.sum() > 0:
            _add("elevation_extreme", mask, "Avg_Invert_Level outside [-50, 200] m")

    combined = pd.Series(False, index=df.index)
    for m in masks.values():
        combined = combined | m

    return combined, summary


def main():
    Paths.ensure_output_dirs()

    _print("Loading data")
    df = pd.read_csv(Paths.INTEGRATED_V2, low_memory=False)
    if Params.SYSTEM != "All" and "System" in df.columns:
        df = df[df["System"] == Params.SYSTEM].copy()
    _print(f"  {len(df):,} records")

    # Missing data report (informational only)
    critical = ["Water_Depth", "Silt_Depth", "Prop_Manhole_Depth",
                "Surcharge_Ratio", "Avg_Pipe_Diameter"]
    _print("\nMissing data (records will be KEPT):")
    for col in [c for c in critical if c in df.columns]:
        n = df[col].isna().sum()
        _print(f"  {col}: {n:,} missing ({n/len(df)*100:.2f}%) — KEPT")

    # Physical impossibility removal
    _print("\nPhysical impossibility checks:")
    remove_mask, summary = check_physical(df)
    _print(f"\nTotal to remove: {remove_mask.sum():,} ({remove_mask.sum()/len(df)*100:.2f}%)")

    clean_df   = df[~remove_mask].copy()
    removed_df = df[remove_mask].copy()
    _print(f"Retained: {len(clean_df):,}")

    # Save
    clean_df.to_csv(Paths.DATA_CLEANED, index=False)
    _print(f"Saved → {Paths.DATA_CLEANED}")

    removed_path = Paths.OUTPUT_ROOT / "removed_records.csv"
    if len(removed_df) > 0:
        removed_df.to_csv(removed_path, index=False)
        _print(f"Removed records → {removed_path}")

    # Quality report
    with pd.ExcelWriter(str(Paths.QUALITY_REPORT), engine="openpyxl") as w:
        pd.DataFrame([{
            "Original_Records": len(df),
            "Removed_Impossible": remove_mask.sum(),
            "Retained": len(clean_df),
            "Retention_Rate_%": len(clean_df) / len(df) * 100,
        }]).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame(summary).to_excel(w, sheet_name="Removal_Detail", index=False)
        if "System" in df.columns:
            sys_df = df.groupby("System").agg(
                Total=("Manhole_ID","count"),
                Removed=(remove_mask, "sum"),
            ).reset_index()
            sys_df["Retained"] = sys_df["Total"] - sys_df["Removed"]
            sys_df.to_excel(w, sheet_name="By_System", index=False)

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Retained\n(complete)", "Retained\n(missing data)", "Removed\n(impossible)"]
    miss = df[~remove_mask][critical].isna().any(axis=1)
    sizes = [(~remove_mask & ~miss).sum(), (~remove_mask & miss).sum(), remove_mask.sum()]
    axes[0].pie(sizes, labels=labels, autopct="%1.1f%%",
                colors=["#2ecc71", "#3498db", "#e74c3c"], explode=(0, 0, 0.1))
    axes[0].set_title("Data Retention Strategy")

    missing_counts = [df[c].isna().sum() for c in critical if c in df.columns]
    avail_critical = [c for c in critical if c in df.columns]
    axes[1].barh(avail_critical, missing_counts, color="#3498db")
    axes[1].set_xlabel("Missing Records (KEPT)")
    axes[1].set_title("Missing Data by Column")
    plt.tight_layout()
    fig_path = Paths.OUTPUT_ROOT / "data_quality_chart.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    _print(f"Chart → {fig_path}")

    _print(f"\nQuality report → {Paths.QUALITY_REPORT}")


if __name__ == "__main__":
    main()
