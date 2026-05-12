"""
09_feature_validation_and_split.py
=====================================
Final feature validation and dataset splitting.

Operations:
  1. Apply physical plausibility caps to pipe slope and length
  2. Produce the final CLASSIFICATION dataset (all surcharged + unsurcharged records)
  3. Produce the final REGRESSION dataset (surcharged records only, for severity modelling)
  4. Save a correlation matrix for multicollinearity assessment

Physical plausibility caps applied:
  Pipe slope : clipped to [MIN_PIPE_SLOPE, MAX_PIPE_SLOPE]
  Pipe length: values below MIN_PIPE_LENGTH_M set to NaN

Author: Research Team
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder
import os, sys, warnings
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

warnings.filterwarnings("ignore")


def _print(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


SLOPE_COLS  = ["Upstream_Pipe_Slope_Avg",  "Downstream_Pipe_Slope_Avg"]
LENGTH_COLS = ["Upstream_Pipe_Length_Avg", "Downstream_Pipe_Length_Avg"]


def apply_physical_caps(df: pd.DataFrame) -> pd.DataFrame:
    """Cap pipe slope and floor pipe length in-place."""
    df = df.copy()
    for col in SLOPE_COLS:
        if col not in df.columns: continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n = ((df[col] > Params.MAX_PIPE_SLOPE) | (df[col] < Params.MIN_PIPE_SLOPE)).sum()
        df[col] = df[col].clip(Params.MIN_PIPE_SLOPE, Params.MAX_PIPE_SLOPE)
        _print(f"  {col}: {n:,} values outside [{Params.MIN_PIPE_SLOPE}, {Params.MAX_PIPE_SLOPE}] → NaN")

    for col in LENGTH_COLS:
        if col not in df.columns: continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n = (df[col] < Params.MIN_PIPE_LENGTH_M).sum()
        df.loc[df[col] < Params.MIN_PIPE_LENGTH_M, col] = np.nan
        _print(f"  {col}: {n:,} values < {Params.MIN_PIPE_LENGTH_M} m → NaN")

    return df


def compute_correlation(df: pd.DataFrame, out_dir) -> pd.DataFrame:
    """Compute and save full correlation matrix."""
    exclude = ["Manhole_ID", "Inspection_Date", "System",
               "Surcharge_Ratio", "First_Inspection_Date", "Latest_Inspection_Date"]

    numeric_cols = [c for c in df.columns if c not in exclude
                    and pd.api.types.is_numeric_dtype(df[c])]

    cat_encode = ["District_Insp", "Road_Type", "Manhole_Type", "Manhole_Shape",
                  "Material", "Type_of_Cover"]
    df_c = df[numeric_cols].copy()
    le = LabelEncoder()
    for col in cat_encode:
        if col in df.columns and col not in df_c.columns:
            df_c[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))

    miss_thresh = Params.MISSING_COL_THRESHOLD
    df_c = df_c[[c for c in df_c.columns if df_c[c].isna().mean() < miss_thresh]]
    df_c = df_c.dropna()
    if len(df_c) > Params.SAMPLE_SIZE_CORR:
        df_c = df_c.sample(Params.SAMPLE_SIZE_CORR, random_state=42)
    df_c = df_c[[c for c in df_c.columns if df_c[c].nunique() > 1]]

    corr = df_c.corr()
    corr.to_csv(Paths.CORRELATION_MATRIX)
    _print(f"  Correlation matrix: {corr.shape[0]}×{corr.shape[0]} → {Paths.CORRELATION_MATRIX}")

    # Heatmap (high correlations only)
    high_feats = set()
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if abs(corr.iloc[i, j]) > Params.HIGH_CORR_THRESHOLD:
                high_feats.add(corr.index[i])
                high_feats.add(corr.index[j])

    if high_feats:
        sub = corr.loc[sorted(high_feats), sorted(high_feats)]
        fig, ax = plt.subplots(figsize=(max(10, len(high_feats)*0.4),
                                        max(10, len(high_feats)*0.4)))
        sns.heatmap(sub, annot=len(high_feats) <= 40, fmt=".2f", cmap="RdYlBu_r",
                    center=0, square=True, vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"High Correlation Features (|r| > {Params.HIGH_CORR_THRESHOLD})")
        plt.tight_layout()
        fig_path = str(out_dir) + "/correlation_heatmap_high.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        _print(f"  Heatmap → {fig_path}")

    # High-pair CSV
    pairs = []
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            r = corr.iloc[i, j]
            if abs(r) > Params.VERY_HIGH_CORR:
                pairs.append({"Feature_1": corr.index[i], "Feature_2": corr.columns[j],
                               "Correlation": round(r, 4)})
    if pairs:
        pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
        pairs_path = str(out_dir) + "/high_correlation_pairs.csv"
        pairs_df.to_csv(pairs_path, index=False)
        _print(f"  {len(pairs_df)} pairs with |r|>{Params.VERY_HIGH_CORR} → {pairs_path}")

    return corr


def main():
    Paths.ensure_output_dirs()

    _print("Loading maintenance-integrated dataset")
    df = pd.read_csv(Paths.MAINTENANCE_FIXED, low_memory=False)
    if Params.SYSTEM != "All" and "System" in df.columns:
        df = df[df["System"] == Params.SYSTEM].copy()
    _print(f"  {len(df):,} records × {df.shape[1]} cols")

    # Physical caps
    _print("\nApplying physical plausibility caps")
    df = apply_physical_caps(df)

    # Classification dataset — all records
    _print("\nSaving classification dataset")
    df.to_csv(Paths.FINAL_CLASSIFICATION, index=False)
    _print(f"  {len(df):,} records → {Paths.FINAL_CLASSIFICATION}")
    pos = df["Is_Surcharged"].sum() if "Is_Surcharged" in df.columns else "N/A"
    _print(f"  Positive class (Is_Surcharged=1): {pos:,}")

    # Regression dataset — surcharged records only
    if "Is_Surcharged" in df.columns:
        df_reg = df[df["Is_Surcharged"] == 1].copy()
        df_reg.to_csv(Paths.FINAL_REGRESSION, index=False)
        _print(f"\n  Regression subset: {len(df_reg):,} records → {Paths.FINAL_REGRESSION}")

    # Correlation analysis
    _print("\nComputing correlation matrix")
    compute_correlation(df, Paths.OUTPUT_ROOT)

    # Final summary
    _print("\n" + "="*60)
    _print("FINAL DATASET SUMMARY")
    _print("="*60)
    _print(f"  Classification dataset: {len(df):,} records × {df.shape[1]} features")
    if "Is_Surcharged" in df.columns:
        pos_pct = df["Is_Surcharged"].mean() * 100
        _print(f"  Positive class (surcharged): {pos_pct:.2f}%")
    _print(f"  Correlation matrix: {Paths.CORRELATION_MATRIX}")


if __name__ == "__main__":
    main()
