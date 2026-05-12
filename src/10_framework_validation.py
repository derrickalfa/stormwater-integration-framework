"""
10_framework_validation.py
===========================
Generates all validation statistics, figures, and tables for
Section 6 of the paper: Framework Validation.

Sections covered
  6.1  Internal consistency
       6.1.1  Surcharge label agreement (computed vs field-recorded)
       6.1.2  Maintenance aggregation completeness
  6.2  Physical plausibility
       6.2.1  Manhole depth distribution
       6.2.2  Surcharge ratio distribution
       6.2.3  Record retention after physical filtering
  6.3  Spatial consistency
       6.3.1  Climate station coverage after propagation
       6.3.2  Coordinate coverage after ID harmonisation
       6.3.3  Spatial distribution of surcharge events by district
  6.4  Temporal consistency
       6.4.1  Inspection record density by year
       6.4.2  Climate data coverage by year

Outputs (all written to OUTPUT_ROOT/validation/)
  validation_statistics.xlsx    — master table (one sheet per section)
  fig_6_1_label_agreement.png
  fig_6_2_surcharge_ratio_dist.png
  fig_6_3_depth_distribution.png
  fig_6_4_spatial_surcharge.png
  fig_6_5_temporal_density.png
  fig_6_6_climate_coverage.png

Author: Research Team
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────

BLUE   = "#2471A3"
ORANGE = "#E67E22"
GREEN  = "#1E8449"
RED    = "#C0392B"
GREY   = "#707B7C"
LIGHT  = "#D6EAF8"

plt.rcParams.update({
    "font.family":      "Arial",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    _print(f"  Saved → {path.name}")


def _pct(n, total):
    return f"{n:,} ({n/total*100:.1f}%)" if total > 0 else "N/A"


def _load_final():
    _print("Loading final classification dataset")
    df = pd.read_csv(Paths.FINAL_CLASSIFICATION, low_memory=False)
    if Params.SYSTEM != "All" and "System" in df.columns:
        df = df[df["System"] == Params.SYSTEM].copy()
    df["Inspection_Date"] = pd.to_datetime(df["Inspection_Date"], errors="coerce")
    df["Inspection_Year"] = df["Inspection_Date"].dt.year
    _print(f"  {len(df):,} records | {df['Manhole_ID'].nunique():,} manholes")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.1  INTERNAL CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

def validate_internal_consistency(df, val_dir):
    _print("6.1  Internal consistency")
    results = {}

    # ── 6.1.1  Surcharge label agreement ────────────────────────────────────
    _print("  6.1.1  Surcharge label agreement")

    has_both = (df["Surcharged_Binary_Original"].notna() &
                df["Is_Surcharged"].notna())
    sub = df[has_both].copy()
    n   = len(sub)

    tp = ((sub["Surcharged_Binary_Original"] == 1) & (sub["Is_Surcharged"] == 1)).sum()
    tn = ((sub["Surcharged_Binary_Original"] == 0) & (sub["Is_Surcharged"] == 0)).sum()
    fp = ((sub["Surcharged_Binary_Original"] == 0) & (sub["Is_Surcharged"] == 1)).sum()
    fn = ((sub["Surcharged_Binary_Original"] == 1) & (sub["Is_Surcharged"] == 0)).sum()

    agree_pct    = (tp + tn) / n * 100
    precision    = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall       = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    label_stats = {
        "Total records with both labels": n,
        "Exact agreement":   f"{tp+tn:,} ({agree_pct:.1f}%)",
        "True Positive (both=1)":  f"{tp:,} ({tp/n*100:.1f}%)",
        "True Negative (both=0)":  f"{tn:,} ({tn/n*100:.1f}%)",
        "False Positive (computed=1, field=0)": f"{fp:,} ({fp/n*100:.1f}%)",
        "False Negative (computed=0, field=1)": f"{fn:,} ({fn/n*100:.1f}%)",
        "Precision (computed vs field)": f"{precision:.1f}%",
        "Recall (computed vs field)":    f"{recall:.1f}%",
        "Interpretation": (
            "False positives are cases where computed ratio>1 but inspector "
            "did not flag surcharge — physically plausible if water receded before inspection. "
            "False negatives are cases where inspector flagged surcharge but ratio≤1 — "
            "may indicate pipe diameter data gap."
        )
    }
    results["6.1.1_Label_Agreement"] = pd.DataFrame(
        label_stats.items(), columns=["Metric", "Value"])

    # Confusion matrix figure
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n({cm[i,j]/n*100:.1f}%)",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()*0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Computed = 0\n(Not Surcharged)",
                         "Computed = 1\n(Surcharged)"], fontsize=10)
    ax.set_yticklabels(["Field = 0\n(Not Surcharged)",
                         "Field = 1\n(Surcharged)"], fontsize=10)
    ax.set_xlabel("Computed Label (Is_Surcharged)", labelpad=10)
    ax.set_ylabel("Field-Recorded Label", labelpad=10)
    ax.set_title(f"Surcharge Label Agreement Matrix\n"
                 f"Overall agreement: {agree_pct:.1f}%  |  n = {n:,}")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Record count")
    plt.tight_layout()
    _save(fig, val_dir / "fig_6_1_label_agreement.png")

    # ── 6.1.2  Maintenance aggregation completeness ──────────────────────────
    _print("  6.1.2  Maintenance aggregation completeness")

    act_cols = [c for c in df.columns
                if c.endswith(("_count", "Blockages", "General_Cleaning",
                               "Infiltration_and_inflow", "Cracks_and_fractures",
                               "Defective_lining", "Deformation",
                               "General_Maintenance", "Inspection",
                               "Environmental"))
                and c != "Total_Maintenance_Events"
                and pd.api.types.is_numeric_dtype(df[c])]

    # Use maintenance category columns that exist
    maint_cats = [c for c in [
        "Blockages", "General_Cleaning", "Infiltration_and_inflow",
        "Flow_velocity__hydraulic_conditions", "Cracks_and_fractures",
        "Defective_lining", "Deformation", "Corrosion_and_abrasion_of_pipes",
        "Defective_connection", "Broken_manhole_covers__damaged_manhole_walls",
        "Environmental", "Inspection", "General_Maintenance",
    ] if c in df.columns]

    if "Total_Maintenance_Events" in df.columns and maint_cats:
        cat_sum  = df[maint_cats].sum(axis=1)
        total    = df["Total_Maintenance_Events"]
        exact    = (abs(cat_sum - total) < 0.01).sum()
        near     = (abs(cat_sum - total) < 1.0).sum()

        maint_stats = {
            "Total records":                   f"{len(df):,}",
            "Records with maintenance events": _pct((total > 0).sum(), len(df)),
            "Category columns summed":         len(maint_cats),
            "Exact sum match (diff < 0.01)":   _pct(exact, len(df)),
            "Near match (diff < 1.0)":         _pct(near,  len(df)),
            "Total maintenance events":        f"{total.sum():,}",
            "Mean events per manhole":         f"{total.mean():.2f}",
            "Max events at one manhole":       f"{total.max():.0f}",
        }
    else:
        maint_stats = {"Note": "Total_Maintenance_Events column not found"}

    results["6.1.2_Maintenance_Completeness"] = pd.DataFrame(
        maint_stats.items(), columns=["Metric", "Value"])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.2  PHYSICAL PLAUSIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def validate_physical_plausibility(df, val_dir):
    _print("6.2  Physical plausibility")
    results = {}

    # ── 6.2.1  Manhole depth distribution ────────────────────────────────────
    _print("  6.2.1  Manhole depth distribution")

    if "Prop_Manhole_Depth" in df.columns:
        depth = df["Prop_Manhole_Depth"].dropna()
        # Convert to metres for display if stored in mm
        if depth.median() > 10:
            depth_m = depth / 1000
            unit    = "m (converted from mm)"
        else:
            depth_m = depth
            unit    = "m"

        depth_stats = {
            "Total records":          f"{len(df):,}",
            "Records with depth":     _pct(depth.notna().sum(), len(df)),
            "Unit":                   unit,
            "Min depth":              f"{depth_m.min():.2f} m",
            "5th percentile":         f"{depth_m.quantile(0.05):.2f} m",
            "Median depth":           f"{depth_m.median():.2f} m",
            "Mean depth":             f"{depth_m.mean():.2f} m",
            "95th percentile":        f"{depth_m.quantile(0.95):.2f} m",
            "Max depth":              f"{depth_m.max():.2f} m",
            "Plausible range (HK)":   "0.8 m – 4.0 m",
            "Within plausible range": _pct(
                ((depth_m >= 0.8) & (depth_m <= 4.0)).sum(), len(depth_m)),
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Histogram
        axes[0].hist(depth_m.clip(0, 6), bins=60, color=BLUE,
                     edgecolor="white", linewidth=0.3)
        axes[0].axvline(depth_m.median(), color=RED, lw=2,
                        linestyle="--", label=f"Median {depth_m.median():.2f} m")
        axes[0].axvspan(0.8, 4.0, alpha=0.12, color=GREEN,
                        label="Plausible range (0.8–4.0 m)")
        axes[0].set_xlabel("Manhole Depth (m)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Manhole Depth Distribution")
        axes[0].legend(fontsize=9)

        # Box plot by quality flag
        if "Depth_Data_Quality" in df.columns:
            order = ["Complete", "Missing_Cover", "Missing_Invert",
                     "Invalid_Negative", "Invalid_Zero"]
            cats  = [c for c in order if c in df["Depth_Data_Quality"].unique()]
            counts = df["Depth_Data_Quality"].value_counts()
            labels = [f"{c}\n(n={counts.get(c,0):,})" for c in cats]
            axes[1].bar(labels,
                        [counts.get(c, 0) for c in cats],
                        color=[GREEN, ORANGE, ORANGE, RED, RED][:len(cats)])
            axes[1].set_title("Depth Data Quality Classification")
            axes[1].set_ylabel("Record Count")
            axes[1].tick_params(axis="x", labelsize=8)
        else:
            axes[1].text(0.5, 0.5, "Depth_Data_Quality\nnot in dataset",
                         ha="center", va="center", transform=axes[1].transAxes)

        plt.suptitle("Section 6.2.1 — Manhole Depth Validation",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        _save(fig, val_dir / "fig_6_3_depth_distribution.png")

    else:
        depth_stats = {"Note": "Prop_Manhole_Depth not in dataset"}

    results["6.2.1_Depth_Distribution"] = pd.DataFrame(
        depth_stats.items(), columns=["Metric", "Value"])

    # ── 6.2.2  Surcharge ratio distribution ──────────────────────────────────
    _print("  6.2.2  Surcharge ratio distribution")

    if "Surcharge_Ratio" in df.columns:
        ratio = df["Surcharge_Ratio"].dropna()
        binary_pct = ((ratio == 0) | (ratio == 1.0)).mean() * 100

        ratio_stats = {
            "Total records":            f"{len(df):,}",
            "Records with valid ratio": _pct(len(ratio), len(df)),
            "Min":                      f"{ratio.min():.4f}",
            "5th percentile":           f"{ratio.quantile(0.05):.4f}",
            "Median":                   f"{ratio.median():.4f}",
            "Mean":                     f"{ratio.mean():.4f}",
            "95th percentile":          f"{ratio.quantile(0.95):.4f}",
            "Max":                      f"{ratio.max():.4f}",
            "% exactly 0 or 1 (binary check)": f"{binary_pct:.1f}%",
            "Unit conversion valid?":    "YES" if binary_pct < 70 else "WARNING — check units",
            "Normal flow (ratio < 0.75)":     _pct((ratio < 0.75).sum(),   len(ratio)),
            "Near surcharge (0.75–1.0)":      _pct(((ratio >= 0.75) & (ratio <= 1.0)).sum(), len(ratio)),
            "Moderate surcharge (1.0–1.5)":   _pct(((ratio > 1.0)  & (ratio <= 1.5)).sum(), len(ratio)),
            "Severe surcharge (>1.5)":        _pct((ratio > 1.5).sum(),    len(ratio)),
        }

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Full distribution histogram
        axes[0].hist(ratio.clip(0, 3), bins=80, color=BLUE,
                     edgecolor="white", linewidth=0.3, density=True)
        for val, label, col in [
            (0.75, "Near surcharge\n(0.75)", ORANGE),
            (1.0,  "Surcharge\nthreshold (1.0)", RED),
            (1.5,  "Severe\n(1.5)", "#8E44AD"),
        ]:
            axes[0].axvline(val, color=col, lw=1.8, linestyle="--", label=label)
        axes[0].set_xlabel("Surcharge Ratio (Water Depth / Pipe Diameter)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Surcharge Ratio Distribution")
        axes[0].legend(fontsize=8)
        axes[0].set_xlim(0, 3)

        # Severity breakdown bar chart
        sev_labels = ["Normal Flow\n(<0.75)", "Near Surcharge\n(0.75–1.0)",
                      "Moderate\n(1.0–1.5)", "Severe\n(>1.5)"]
        sev_counts = [
            (ratio < 0.75).sum(),
            ((ratio >= 0.75) & (ratio <= 1.0)).sum(),
            ((ratio > 1.0) & (ratio <= 1.5)).sum(),
            (ratio > 1.5).sum(),
        ]
        bars = axes[1].bar(sev_labels, sev_counts,
                           color=[GREEN, ORANGE, "#E74C3C", "#8E44AD"])
        for bar, count in zip(bars, sev_counts):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + max(sev_counts)*0.01,
                         f"{count:,}\n({count/len(ratio)*100:.1f}%)",
                         ha="center", va="bottom", fontsize=9)
        axes[1].set_title("Surcharge Severity Classification")
        axes[1].set_ylabel("Number of Inspection Records")

        plt.suptitle("Section 6.2.2 — Surcharge Ratio Validation",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        _save(fig, val_dir / "fig_6_2_surcharge_ratio_dist.png")

    else:
        ratio_stats = {"Note": "Surcharge_Ratio not in dataset"}

    results["6.2.2_Surcharge_Ratio"] = pd.DataFrame(
        ratio_stats.items(), columns=["Metric", "Value"])

    # ── 6.2.3  Record retention ───────────────────────────────────────────────
    _print("  6.2.3  Record retention")

    # Try to load the quality report
    ret_stats = {}
    try:
        qr = pd.read_excel(Paths.QUALITY_REPORT, sheet_name="Summary", engine="openpyxl")
        for _, row in qr.iterrows():
            ret_stats[str(row["Metric"])] = str(row["Value"])
    except Exception:
        # Approximate from the final dataset vs cleaned dataset
        try:
            n_cleaned = len(pd.read_csv(Paths.DATA_CLEANED, usecols=["Manhole_ID"]))
            n_final   = len(df)
            ret_stats = {
                "Records after physical validation": f"{n_cleaned:,}",
                "Records in final dataset":          f"{n_final:,}",
                "Note": "See data_quality_report.xlsx for full breakdown"
            }
        except Exception:
            ret_stats = {"Note": "Quality report not found — run script 07 first"}

    results["6.2.3_Record_Retention"] = pd.DataFrame(
        ret_stats.items(), columns=["Metric", "Value"])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.3  SPATIAL CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

def validate_spatial_consistency(df, val_dir):
    _print("6.3  Spatial consistency")
    results = {}

    # ── 6.3.1  Climate station coverage ──────────────────────────────────────
    _print("  6.3.1  Climate station coverage")

    climate_vars = {
        "Temperature (daily)":    "Temp_Mean_Day",
        "Rainfall (daily)":       "Rain_Day",
        "Humidity (daily)":       "Humidity_Day",
        "Wind speed (daily)":     "Windspeed_Day",
        "Dewpoint (daily)":       "Dewpoint_Day",
        "Pressure (monthly)":     "Mean_Pressure_Monthly",
        "Wind direction (monthly)":"Wind_Direction_Monthly",
    }

    cov_rows = []
    for label, col in climate_vars.items():
        if col in df.columns:
            rec_n    = df[col].notna().sum()
            rec_pct  = rec_n / len(df) * 100
            mh_n     = df[df[col].notna()]["Manhole_ID"].nunique()
            mh_total = df["Manhole_ID"].nunique()
            mh_pct   = mh_n / mh_total * 100
            cov_rows.append({
                "Climate Variable":        label,
                "Records with data":       rec_n,
                "Record coverage (%)":     round(rec_pct, 1),
                "Manholes with data":      mh_n,
                "Manhole coverage (%)":    round(mh_pct, 1),
            })

    cov_df = pd.DataFrame(cov_rows)
    results["6.3.1_Climate_Coverage"] = cov_df

    if not cov_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(cov_df))
        w = 0.38
        b1 = ax.bar(x - w/2, cov_df["Record coverage (%)"],  w,
                    color=BLUE,   label="Record coverage (%)")
        b2 = ax.bar(x + w/2, cov_df["Manhole coverage (%)"], w,
                    color=ORANGE, label="Manhole coverage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(cov_df["Climate Variable"], rotation=25,
                           ha="right", fontsize=9)
        ax.set_ylabel("Coverage (%)")
        ax.set_ylim(0, 115)
        ax.axhline(100, color=GREY, lw=0.8, linestyle=":")
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title("Section 6.3.1 — Climate Station Coverage After "
                     "Network Propagation & Distance Assignment")
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, val_dir / "fig_6_6_climate_coverage.png")

    # ── 6.3.2  Coordinate coverage ────────────────────────────────────────────
    _print("  6.3.2  Coordinate coverage")

    coord_rows = []
    for col, label in [("X_Coord", "Easting"), ("Y_Coord", "Northing")]:
        if col in df.columns:
            n     = df[col].notna().sum()
            mh_n  = df[df[col].notna()]["Manhole_ID"].nunique()
            coord_rows.append({
                "Coordinate":          label,
                "Records with data":   n,
                "Record coverage (%)": round(n / len(df) * 100, 1),
                "Manholes with data":  mh_n,
                "Manhole coverage (%)":round(mh_n / df["Manhole_ID"].nunique() * 100, 1),
            })

    results["6.3.2_Coordinate_Coverage"] = pd.DataFrame(coord_rows) if coord_rows else \
        pd.DataFrame([{"Note": "X_Coord / Y_Coord not in dataset"}])

    # ── 6.3.3  Spatial surcharge distribution by district ────────────────────
    _print("  6.3.3  Spatial distribution by district")

    if "District_Insp" in df.columns and "Is_Surcharged" in df.columns:
        dist = (df.groupby("District_Insp")
                  .agg(Total_Inspections=("Is_Surcharged", "count"),
                       Surcharged_Events=("Is_Surcharged", "sum"))
                  .reset_index())
        dist["Surcharge_Rate_%"] = (dist["Surcharged_Events"] /
                                     dist["Total_Inspections"] * 100).round(1)
        dist = dist[dist["Total_Inspections"] >= 50]  # exclude tiny samples
        dist = dist.sort_values("Surcharge_Rate_%", ascending=False)

        results["6.3.3_District_Surcharge"] = dist

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Surcharge rate by district
        top_n    = min(20, len(dist))
        top_dist = dist.head(top_n)
        colors   = [RED if r > dist["Surcharge_Rate_%"].quantile(0.75) else
                    ORANGE if r > dist["Surcharge_Rate_%"].median() else
                    BLUE for r in top_dist["Surcharge_Rate_%"]]
        bars = axes[0].barh(top_dist["District_Insp"].astype(str),
                             top_dist["Surcharge_Rate_%"], color=colors)
        med  = dist["Surcharge_Rate_%"].median()
        axes[0].axvline(med, color=GREY, lw=1.5, linestyle="--",
                        label=f"Median {med:.1f}%")
        axes[0].set_xlabel("Surcharge Rate (%)")
        axes[0].set_title(f"Surcharge Rate by District (top {top_n})")
        axes[0].legend(fontsize=9)
        axes[0].invert_yaxis()

        # Total inspections vs surcharged
        axes[1].scatter(dist["Total_Inspections"],
                        dist["Surcharge_Rate_%"],
                        c=dist["Surcharge_Rate_%"], cmap="RdYlGn_r",
                        s=70, alpha=0.8, edgecolors="white", linewidth=0.5)
        axes[1].set_xlabel("Total Inspection Records")
        axes[1].set_ylabel("Surcharge Rate (%)")
        axes[1].set_title("Surcharge Rate vs Inspection Volume by District\n"
                           "(validates spatial non-randomness)")
        # Label top 5 districts
        for _, row in dist.head(5).iterrows():
            axes[1].annotate(str(row["District_Insp"]),
                             (row["Total_Inspections"], row["Surcharge_Rate_%"]),
                             textcoords="offset points", xytext=(5, 3),
                             fontsize=8, color=RED)

        plt.suptitle("Section 6.3.3 — Spatial Distribution of Surcharge Events",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        _save(fig, val_dir / "fig_6_4_spatial_surcharge.png")

    else:
        results["6.3.3_District_Surcharge"] = pd.DataFrame(
            [{"Note": "District_Insp or Is_Surcharged not in dataset"}])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.4  TEMPORAL CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

def validate_temporal_consistency(df, val_dir):
    _print("6.4  Temporal consistency")
    results = {}

    # ── 6.4.1  Inspection record density by year ─────────────────────────────
    _print("  6.4.1  Inspection density by year")

    if "Inspection_Year" in df.columns:
        yr = (df.groupby("Inspection_Year")
                .agg(Records=("Manhole_ID", "count"),
                     Unique_Manholes=("Manhole_ID", "nunique"),
                     Surcharged=("Is_Surcharged", "sum")
                     if "Is_Surcharged" in df.columns
                     else ("Manhole_ID", lambda x: 0))
                .reset_index()
                .sort_values("Inspection_Year"))
        yr = yr[yr["Inspection_Year"].between(2000, 2030)]
        if "Is_Surcharged" in df.columns:
            yr["Surcharge_Rate_%"] = (yr["Surcharged"] / yr["Records"] * 100).round(1)

        results["6.4.1_Yearly_Density"] = yr

        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

        axes[0].bar(yr["Inspection_Year"], yr["Records"],
                    color=BLUE, edgecolor="white", linewidth=0.3)
        axes[0].set_ylabel("Inspection Records")
        axes[0].set_title("Section 6.4.1 — Inspection Record Density by Year")
        mean_r = yr["Records"].mean()
        axes[0].axhline(mean_r, color=RED, lw=1.5, linestyle="--",
                        label=f"Mean {mean_r:,.0f}")
        axes[0].legend(fontsize=9)

        axes[1].bar(yr["Inspection_Year"], yr["Unique_Manholes"],
                    color=ORANGE, edgecolor="white", linewidth=0.3)
        axes[1].set_ylabel("Unique Manholes Inspected")
        axes[1].set_xlabel("Year")

        plt.tight_layout()
        _save(fig, val_dir / "fig_6_5_temporal_density.png")

    else:
        results["6.4.1_Yearly_Density"] = pd.DataFrame(
            [{"Note": "Inspection_Year not in dataset"}])

    # ── 6.4.2  Climate coverage by year ──────────────────────────────────────
    _print("  6.4.2  Climate coverage by year")

    climate_cols = {
        "Temperature": "Temp_Mean_Day",
        "Rainfall":    "Rain_Day",
        "Humidity":    "Humidity_Day",
    }
    avail = {k: v for k, v in climate_cols.items() if v in df.columns}

    if avail and "Inspection_Year" in df.columns:
        yr_groups = df.groupby("Inspection_Year")
        rows = []
        for year, grp in yr_groups:
            if not (2000 <= year <= 2030): continue
            row = {"Year": year, "Total_Records": len(grp)}
            for label, col in avail.items():
                row[f"{label}_coverage_%"] = round(
                    grp[col].notna().mean() * 100, 1)
            rows.append(row)

        cov_yr = pd.DataFrame(rows).sort_values("Year")
        results["6.4.2_Climate_Coverage_By_Year"] = cov_yr

        fig, ax = plt.subplots(figsize=(13, 5))
        line_styles = ["-o", "-s", "-^"]
        for (label, col), ls in zip(avail.items(), line_styles):
            col_name = f"{label}_coverage_%"
            if col_name in cov_yr.columns:
                ax.plot(cov_yr["Year"], cov_yr[col_name],
                        ls, label=label, markersize=5, linewidth=1.8)

        ax.axhline(95, color=GREY, lw=1, linestyle=":",
                   label="95% target coverage")
        ax.set_xlabel("Inspection Year")
        ax.set_ylabel("Climate Data Coverage (%)")
        ax.set_title("Section 6.4.2 — Climate Data Coverage by Year\n"
                     "(validates temporal completeness of station assignment)")
        ax.set_ylim(0, 115)
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, val_dir / "fig_6_6b_climate_coverage_year.png")

    else:
        results["6.4.2_Climate_Coverage_By_Year"] = pd.DataFrame(
            [{"Note": "Climate columns or Inspection_Year not in dataset"}])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_table(df):
    """One-page summary of all validation metrics for the paper."""
    _print("Building master summary table")

    n   = len(df)
    mhs = df["Manhole_ID"].nunique()

    rows = [
        # Section, Metric, Value, Interpretation
        ("6.1.1", "Total records with both surcharge labels",
         f"{df[['Surcharged_Binary_Original','Is_Surcharged']].dropna().shape[0]:,}", ""),

        ("6.1.1", "Overall label agreement (computed vs field)",
         f"{((df['Surcharged_Binary_Original']==df['Is_Surcharged']).mean()*100):.1f}%"
         if {"Surcharged_Binary_Original","Is_Surcharged"}.issubset(df.columns) else "N/A",
         "Validates surcharge computation methodology"),

        ("6.1.2", "Maintenance category-to-total exact match",
         "See 6.1.2 sheet", "Validates reaggregation losslessness"),

        ("6.2.1", "Median manhole depth",
         f"{df['Prop_Manhole_Depth'].median()/1000:.2f} m"
         if "Prop_Manhole_Depth" in df.columns and df['Prop_Manhole_Depth'].median() > 10
         else f"{df['Prop_Manhole_Depth'].median():.2f} m"
         if "Prop_Manhole_Depth" in df.columns else "N/A",
         "Should be 0.8–4.0 m for HK urban manholes"),

        ("6.2.2", "Records with valid surcharge ratio",
         _pct(df["Surcharge_Ratio"].notna().sum(), n)
         if "Surcharge_Ratio" in df.columns else "N/A",
         "Validates diameter hierarchy logic"),

        ("6.2.2", "% binary ratios (unit conversion check)",
         f"{((df['Surcharge_Ratio']==0)|(df['Surcharge_Ratio']==1)).mean()*100:.1f}%"
         if "Surcharge_Ratio" in df.columns else "N/A",
         "< 70% confirms correct unit conversion"),

        ("6.2.3", "Record retention after physical filtering",
         "See quality_report.xlsx", "Validates conservative removal criteria"),

        ("6.3.1", "Temperature data coverage (records)",
         f"{df['Temp_Mean_Day'].notna().mean()*100:.1f}%"
         if "Temp_Mean_Day" in df.columns else "N/A",
         "Validates station propagation strategy"),

        ("6.3.1", "Rainfall data coverage (records)",
         f"{df['Rain_Day'].notna().mean()*100:.1f}%"
         if "Rain_Day" in df.columns else "N/A", ""),

        ("6.3.2", "Coordinate coverage (manholes)",
         f"{df[df['X_Coord'].notna()]['Manhole_ID'].nunique()/mhs*100:.1f}%"
         if "X_Coord" in df.columns else "N/A",
         "Validates ID harmonisation across sources"),

        ("6.3.3", "Districts with surcharge data",
         str(df["District_Insp"].nunique())
         if "District_Insp" in df.columns else "N/A",
         "Validates spatial non-randomness of surcharge events"),

        ("6.4.1", "Year range of inspection records",
         f"{int(df['Inspection_Year'].min())}–{int(df['Inspection_Year'].max())}"
         if "Inspection_Year" in df.columns else "N/A",
         "Validates temporal completeness of raw data"),

        ("6.4.2", "Mean annual climate coverage",
         "See 6.4.2 sheet", "Validates temporal stability of station assignment"),
    ]

    return pd.DataFrame(rows, columns=["Section", "Metric", "Value", "Interpretation"])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Paths.ensure_output_dirs()

    val_dir = Paths.OUTPUT_ROOT / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    _print(f"Output directory: {val_dir}")

    # Load data
    df = _load_final()

    # Run all validation sections
    r1 = validate_internal_consistency(df,    val_dir)
    r2 = validate_physical_plausibility(df,   val_dir)
    r3 = validate_spatial_consistency(df,     val_dir)
    r4 = validate_temporal_consistency(df,    val_dir)

    # Master summary
    summary = build_summary_table(df)

    # Write master Excel workbook
    out_xlsx = val_dir / "validation_statistics.xlsx"
    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="0_Master_Summary", index=False)
        for section_results in [r1, r2, r3, r4]:
            for sheet_name, table in section_results.items():
                safe = sheet_name[:31]
                table.to_excel(writer, sheet_name=safe, index=False)

    _print(f"\nMaster Excel → {out_xlsx}")

    # Print master summary to console
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))

    print("\n" + "="*80)
    print("ALL OUTPUTS")
    print("="*80)
    for f in sorted(val_dir.iterdir()):
        size = f.stat().st_size / 1024
        print(f"  {f.name:<50}  {size:6.1f} KB")


if __name__ == "__main__":
    main()
