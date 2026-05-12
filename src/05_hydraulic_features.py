"""
05_hydraulic_features.py
=========================
Computes hydraulic features and the surcharge target variables.

Corrections embedded (no separate fix scripts needed):
  1. Pipe diameter priority: inspection-level inlet/outlet diameters first,
     network-derived average as fallback.
  2. Effective water depth uses Water_Depth only (not max of water + silt).
  3. Invert_Level and Cover_Level from the manhole inspection record,
     not averaged from connected pipes.
  4. Pipe slope capped to [-0.5, 1.0]; pipe length floor at 0.5 m.

Outputs:
  manhole_hydraulic_features_complete.csv

Target variables produced:
  Is_Surcharged            — binary  (Surcharge_Ratio > 1.0)
  Surcharge_Ratio          — continuous severity target
  Surcharge_Severity       — ordinal label (Normal / Near / Moderate / Severe)

Author: Research Team
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MAINTENANCE_CATEGORIES = [
    "Blockages", "General_Cleaning", "Infiltration_and_inflow",
    "Flow_velocity__hydraulic_conditions", "Cracks_and_fractures",
    "Defective_lining", "Deformation", "Corrosion_and_abrasion_of_pipes",
    "Defective_connection", "Broken_manhole_covers__damaged_manhole_walls",
    "Environmental", "Inspection", "General_Maintenance",
]
SEVERITY_CATEGORIES = ["EMERGENCY", "HIGH", "MEDIUM", "LOW", "ROUTINE"]

PHYSICAL_FEATURES = [
    "Water_Depth", "Silt_Depth", "Prop_Manhole_Depth", "Size_of_Cover",
    "Inlet_Pipe_1_Dia", "Inlet_Pipe_2_Dia", "Inlet_Pipe_3_Dia", "Inlet_Pipe_4_Dia",
    "Outlet_Pipe_1_Dia", "Outlet_Pipe_2_Dia",
]
CATEGORICAL_FEATURES = [
    "District_Insp", "Manhole_Shape", "Manhole_Type",
    "Type_of_Cover", "Material", "Road_Type",
]
INLET_COLS  = [f"Inlet_Pipe_{i}_Dia"  for i in range(1, 5)]
OUTLET_COLS = [f"Outlet_Pipe_{i}_Dia" for i in range(1, 3)]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _clean(v):
    if pd.isna(v) or v in ("-", "", "NA"): return np.nan
    try:    return float(v)
    except: return np.nan


def _infer_system(identifier):
    s = str(identifier).upper()
    if s.startswith("F") or "FMH" in s: return "Sewer"
    if s.startswith("S") or "SMH" in s: return "Stormwater"
    if s.startswith("C") or "CMH" in s: return "Combined"
    return None


def _parse_dates(series): return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def load_network(network_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pipe_manhole_map, pipe_geometry)."""
    _print("STEP 1: Loading network data")
    sheets = {"Combined": "Combined_System_Links",
               "Sewer":    "Sewer_System_Links",
               "Stormwater": "Stormwater_System_Links"}
    frames = []
    for system, sheet in sheets.items():
        df = pd.read_excel(network_file, sheet_name=sheet, engine="openpyxl")
        df["System"] = system
        frames.append(df)
        _print(f"  {system}: {len(df):,} pipes")

    net = pd.concat(frames, ignore_index=True)

    # pipe → manhole map
    up   = net[["Link_ID", "Upstream_Node",   "System"]].rename(columns={"Link_ID": "Pipe_ID", "Upstream_Node":   "Manhole_ID"})
    down = net[["Link_ID", "Downstream_Node", "System"]].rename(columns={"Link_ID": "Pipe_ID", "Downstream_Node": "Manhole_ID"})
    pm_map = pd.concat([up, down], ignore_index=True).dropna(subset=["Pipe_ID", "Manhole_ID"])
    pm_map["Pipe_ID"]    = pm_map["Pipe_ID"].astype(str).str.strip().str.upper()
    pm_map["Manhole_ID"] = pm_map["Manhole_ID"].astype(str).str.strip().str.upper()

    # pipe geometry
    for col in ["Upstream_Invert_Level", "Downstream_Invert_Level", "Nominal_Width", "Nominal_Height"]:
        if col in net.columns:
            net[col] = net[col].apply(_clean)

    def _equiv_diam(row):
        shape  = str(row.get("Shape", "")).upper()
        width  = _clean(row.get("Nominal_Width"))
        height = _clean(row.get("Nominal_Height"))
        if "CIRC" in shape or "ROUND" in shape or pd.isna(height): return width
        return height if pd.notna(height) else width

    net["Pipe_Diameter"] = net.apply(_equiv_diam, axis=1)
    net["Pipe_ID"]           = net["Link_ID"].astype(str).str.strip().str.upper()
    net["Upstream_Node"]     = net["Upstream_Node"].astype(str).str.strip().str.upper()
    net["Downstream_Node"]   = net["Downstream_Node"].astype(str).str.strip().str.upper()

    # slope (capped per Params)
    if {"Upstream_Invert_Level", "Downstream_Invert_Level", "Computed_Length"}.issubset(net.columns):
        net["Pipe_Slope"] = ((net["Upstream_Invert_Level"] - net["Downstream_Invert_Level"])
                             / net["Computed_Length"].replace(0, np.nan))
        net["Pipe_Slope"] = net["Pipe_Slope"].clip(Params.MIN_PIPE_SLOPE, Params.MAX_PIPE_SLOPE)

    # length floor
    if "Computed_Length" in net.columns:
        net.loc[net["Computed_Length"] < Params.MIN_PIPE_LENGTH_M, "Computed_Length"] = np.nan

    _print(f"  Pipe-manhole map: {len(pm_map):,} rows")
    return pm_map, net


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MAINTENANCE
# ─────────────────────────────────────────────────────────────────────────────

def load_maintenance(sheet_dir: Path) -> pd.DataFrame:
    """Concatenate per-sheet maintenance CSVs."""
    _print("STEP 2: Loading maintenance events")
    frames = []
    for f in sorted(sheet_dir.glob("*_activities.csv")):
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
        _print(f"  {f.name}: {len(df):,} events")

    maint = pd.concat(frames, ignore_index=True)

    date_col = next((c for c in ["date_of_completion", "date_of_commencement"] if c in maint.columns), None)
    if date_col is None:
        raise ValueError("No date column found in maintenance files")

    maint["Maintenance_Date"] = _parse_dates(maint[date_col])
    maint = maint[maint["Maintenance_Date"].notna()].copy()

    pipe_col = next((c for c in ["feature_number", "Feature_Number"] if c in maint.columns), None)
    if pipe_col is None:
        raise ValueError("No feature_number column found in maintenance files")

    maint["Pipe_ID"]  = maint[pipe_col].astype(str).str.strip().str.upper()
    maint["Category"] = maint.get("primary_category", pd.Series("Unknown", index=maint.index)).fillna("Unknown")
    maint["Severity"] = (maint.get("severity_keyword_based") or maint.get("severity_response_based",
                         pd.Series("Unknown", index=maint.index))).fillna("Unknown")

    _print(f"  Total maintenance events: {len(maint):,}")
    return maint


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — INSPECTIONS + FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def load_inspections(inspection_file) -> pd.DataFrame:
    """Load merged inspection/properties file and extract all features."""
    _print("STEP 3: Loading inspections and extracting features")
    df = pd.read_csv(inspection_file, low_memory=False)
    _print(f"  {len(df):,} records")

    manhole_col = "Manhole" if "Manhole" in df.columns else "Manhole_ID"
    df["Manhole_ID"] = df[manhole_col].astype(str).str.strip().str.upper()
    df["Inspection_Date"] = _parse_dates(df["Inspection_Date"])
    df = df[df["Inspection_Date"].notna()].copy()

    surcharge_col = "Surcharged_Binary" if "Surcharged_Binary" in df.columns else "Surcharged"
    df["Surcharged_Binary_Original"] = (
        pd.to_numeric(df.get(surcharge_col, 0), errors="coerce").fillna(0).astype(int)
    )

    if "System" not in df.columns:
        df["System"] = df["Manhole_ID"].apply(_infer_system)

    # Physical columns
    for col in [c for c in PHYSICAL_FEATURES if c in df.columns]:
        df[col] = df[col].apply(_clean)

    # Categorical columns
    for col in [c for c in CATEGORICAL_FEATURES if c in df.columns]:
        df[col] = df[col].astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown", "NA": "Unknown"})

    # Temporal
    df["Inspection_Year"]   = df["Inspection_Date"].dt.year
    df["Inspection_Month"]  = df["Inspection_Date"].dt.month
    df["Inspection_Season"] = df["Inspection_Month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Autumn",  10: "Autumn", 11: "Autumn",
    })

    # Derived pipe geometry from inspection
    avail_inlet  = [c for c in INLET_COLS  if c in df.columns]
    avail_outlet = [c for c in OUTLET_COLS if c in df.columns]
    if avail_inlet:
        df["Total_Inlet_Diameter"] = df[avail_inlet].sum(axis=1)
        df["Num_Inlet_Pipes"]      = (df[avail_inlet] > 0).sum(axis=1)
    if avail_outlet:
        df["Total_Outlet_Diameter"] = df[avail_outlet].sum(axis=1)
    if "Total_Inlet_Diameter" in df and "Total_Outlet_Diameter" in df:
        df["Capacity_Ratio"] = df["Total_Outlet_Diameter"] / df["Total_Inlet_Diameter"].replace(0, np.nan)
    if {"Silt_Depth", "Water_Depth"}.issubset(df.columns):
        df["Silt_Water_Ratio"] = df["Silt_Depth"] / df["Water_Depth"].replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PIPE DIAMETER (priority: inspection > network)
# ─────────────────────────────────────────────────────────────────────────────

def attach_pipe_diameter(df: pd.DataFrame, pipe_geometry: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Avg_Pipe_Diameter using a two-tier hierarchy:
      Tier 1 — mean of non-zero inlet + outlet diameters from inspection record
               (capped at MAX_PIPE_DIAMETER_MM to remove implausible values)
      Tier 2 — mean diameter from connected pipes in network file (fallback)
    """
    _print("STEP 4: Attaching pipe diameter (inspection-first hierarchy)")

    # Tier 1: inspection-level diameters
    avail = [c for c in INLET_COLS + OUTLET_COLS if c in df.columns]
    if avail:
        dia_mat = df[avail].copy()
        dia_mat[dia_mat > Params.MAX_PIPE_DIAMETER_MM] = np.nan  # cap extreme values
        dia_mat[dia_mat <= 0] = np.nan
        df["Avg_Pipe_Diameter_Inspection"] = dia_mat.mean(axis=1)

    # Tier 2: network-derived average
    melt = pipe_geometry.melt(
        id_vars=["Pipe_ID", "Pipe_Diameter"],
        value_vars=["Upstream_Node", "Downstream_Node"],
        value_name="Manhole_ID",
    ).drop(columns=["variable"])
    net_avg = melt.groupby("Manhole_ID")["Pipe_Diameter"].mean().reset_index(name="Avg_Pipe_Diameter_Network")
    df = df.merge(net_avg, on="Manhole_ID", how="left")

    # Combine: use inspection tier if available, else network
    if "Avg_Pipe_Diameter_Inspection" in df.columns:
        df["Avg_Pipe_Diameter"] = np.where(
            df["Avg_Pipe_Diameter_Inspection"].notna(),
            df["Avg_Pipe_Diameter_Inspection"],
            df["Avg_Pipe_Diameter_Network"],
        )
        df["Diameter_Source"] = np.where(
            df["Avg_Pipe_Diameter_Inspection"].notna(),
            "Inspection_Data",
            "Network_Data_Fallback",
        )
        tier1_pct = (df["Diameter_Source"] == "Inspection_Data").mean() * 100
        _print(f"  Inspection-level diameter used for {tier1_pct:.1f}% of records")
    else:
        df["Avg_Pipe_Diameter"] = df["Avg_Pipe_Diameter_Network"]
        df["Diameter_Source"]   = "Network_Data_Fallback"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — HYDRAULIC CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_hydraulics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all hydraulic metrics from depth measurements and pipe geometry."""
    _print("STEP 5: Computing hydraulic features")
    df = df.copy()

    water = df["Water_Depth"].fillna(0) if "Water_Depth" in df.columns else pd.Series(0, index=df.index)
    silt  = df["Silt_Depth"].fillna(0)  if "Silt_Depth"  in df.columns else pd.Series(0, index=df.index)

    # Effective water depth = Water_Depth only (silt is sediment, not flowing water)
    df["Effective_Water_Depth"] = np.where(water > 0, water, np.nan)
    df["Clear_Water_Thickness"] = np.maximum(water - silt, 0)
    df["Exposed_Silt_Depth"]    = np.maximum(silt - water, 0)

    # Flow condition
    def _flow_cond(row):
        w = row.get("Water_Depth") or 0
        s = row.get("Silt_Depth")  or 0
        if w == 0 and s == 0: return "No_Data"
        if w > s:             return "Normal_Flow"
        if s > w:             return "Low_Flow_High_Silt"
        return "Fully_Saturated_Silt"
    df["Flow_Condition"] = df.apply(_flow_cond, axis=1)

    # Elevations in mm
    if "Invert_Level" in df.columns:
        df["Invert_Level"] = pd.to_numeric(df["Invert_Level"], errors="coerce")
        df["Invert_Level_mm"] = df["Invert_Level"] * 1000
    else:
        df["Invert_Level_mm"] = np.nan

    has_cover = "Cover_Level" in df.columns and df["Cover_Level"].notna().any()
    if has_cover:
        df["Cover_Level"] = pd.to_numeric(df["Cover_Level"], errors="coerce")
        df["Cover_Level_mm"] = df["Cover_Level"] * 1000

    # Water surface and pipe crown
    df["Water_Surface_Elevation"] = df["Invert_Level_mm"] + df["Effective_Water_Depth"]
    df["Pipe_Crown_Elevation"]    = df["Invert_Level_mm"] + df["Avg_Pipe_Diameter"]
    df["Surcharge_Depth"]         = df["Water_Surface_Elevation"] - df["Pipe_Crown_Elevation"]

    # Surcharge ratio (primary target variable)
    df["Surcharge_Ratio"] = (df["Effective_Water_Depth"] / df["Avg_Pipe_Diameter"].replace(0, np.nan))

    # Freeboard
    if has_cover:
        df["Freeboard"] = df["Cover_Level_mm"] - df["Water_Surface_Elevation"]

    # Silt ratio
    df["Silt_Coverage_Ratio"] = silt / df["Avg_Pipe_Diameter"].replace(0, np.nan)

    # ── Classifications ────────────────────────────────────────────────────
    def _surcharge_sev(r):
        if pd.isna(r):           return "Unknown"
        if r < Params.SURCHARGE_NEAR:  return "Normal_Flow"
        if r <= Params.SURCHARGE_THRESHOLD: return "Near_Surcharge"
        if r <= Params.SURCHARGE_MODERATE:  return "Moderate_Surcharge"
        return "Severe_Surcharge"

    df["Surcharge_Severity"] = df["Surcharge_Ratio"].apply(_surcharge_sev)

    def _silt_risk(r):
        if pd.isna(r):                         return "Unknown"
        if r < 0.3:                            return "Low_Silt"
        if r < Params.SILT_BLOCKAGE_HIGH:      return "Moderate_Silt"
        if r < Params.SILT_BLOCKAGE_CRITICAL:  return "High_Silt"
        return "CRITICAL_Silt_Blockage"

    df["Silt_Blockage_Risk"] = df["Silt_Coverage_Ratio"].apply(_silt_risk)

    if has_cover:
        def _flood_risk(fb):
            if pd.isna(fb):  return "Unknown"
            if fb < 0:       return "CRITICAL_Surface_Flooding"
            if fb < 200:     return "HIGH_Near_Surface"
            if fb < 500:     return "MODERATE_Limited_Freeboard"
            return "LOW_Adequate_Freeboard"
        df["Flooding_Risk"] = df["Freeboard"].apply(_flood_risk)

    # ── Binary flags ────────────────────────────────────────────────────────
    df["Is_Surcharged"] = (df["Surcharge_Ratio"] > Params.SURCHARGE_THRESHOLD).fillna(False).astype(int)
    df["Needs_Cleaning"] = (df["Silt_Coverage_Ratio"] > Params.SILT_BLOCKAGE_HIGH).fillna(False).astype(int)
    if has_cover:
        df["Has_Surface_Flooding"] = (df["Freeboard"] < 0).fillna(False).astype(int)

    # ── Hydraulic risk score (0–10) ──────────────────────────────────────────
    def _risk_score(row):
        score = 0
        r = row.get("Surcharge_Ratio")
        if pd.notna(r):
            if r > 1.5: score += 4
            elif r > 1.0: score += 3
            elif r > 0.75: score += 2
            elif r > 0.5: score += 1
        if has_cover:
            fb = row.get("Freeboard")
            if pd.notna(fb):
                if fb < 0:    score += 4
                elif fb < 200: score += 3
                elif fb < 500: score += 2
                elif fb < 1000: score += 1
        sr = row.get("Silt_Coverage_Ratio")
        if pd.notna(sr):
            if sr > 0.75: score += 2
            elif sr > 0.5: score += 1
        return score

    df["Hydraulic_Risk_Score"] = df.apply(_risk_score, axis=1)

    def _priority(s):
        if s >= 8: return "CRITICAL_Immediate"
        if s >= 6: return "HIGH_Schedule_Soon"
        if s >= 4: return "MODERATE_Plan_Maintenance"
        return "LOW_Routine_Monitoring"

    df["Maintenance_Priority"] = df["Hydraulic_Risk_Score"].apply(_priority)

    # Summary
    valid_sr = df["Surcharge_Ratio"].notna().sum()
    _print(f"  Surcharge ratio valid for {valid_sr:,} / {len(df):,} records")
    for sev in ["Normal_Flow", "Near_Surcharge", "Moderate_Surcharge", "Severe_Surcharge"]:
        n = (df["Surcharge_Severity"] == sev).sum()
        _print(f"    {sev}: {n:,} ({n/len(df)*100:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — INTERVAL-BASED MAINTENANCE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_maintenance_features(df: pd.DataFrame, maintenance: pd.DataFrame,
                                  pm_map: pd.DataFrame) -> pd.DataFrame:
    """Count maintenance events per category/severity in the interval before each inspection."""
    _print("STEP 6: Computing interval-based maintenance features")

    df = df.sort_values(["Manhole_ID", "Inspection_Date"]).reset_index(drop=True)
    df["Previous_Inspection_Date"] = df.groupby("Manhole_ID")["Inspection_Date"].shift(1)
    system_start = maintenance["Maintenance_Date"].min()
    df["Previous_Inspection_Date"] = df["Previous_Inspection_Date"].fillna(system_start)
    df["Inspection_ID"] = range(len(df))

    # Link maintenance to manholes via pipe network
    maint_mh = maintenance.merge(pm_map[["Pipe_ID", "Manhole_ID"]], on="Pipe_ID", how="inner")

    # Join to inspection intervals
    merged = maint_mh.merge(
        df[["Inspection_ID", "Manhole_ID", "Inspection_Date", "Previous_Inspection_Date"]],
        on="Manhole_ID", how="inner",
    )
    merged = merged[
        (merged["Maintenance_Date"] > merged["Previous_Inspection_Date"]) &
        (merged["Maintenance_Date"] <= merged["Inspection_Date"])
    ]

    # Pivot category counts
    cat_pivot = (merged.groupby(["Inspection_ID", "Category"]).size()
                 .reset_index(name="Count")
                 .pivot(index="Inspection_ID", columns="Category", values="Count")
                 .fillna(0))

    sev_pivot = (merged.groupby(["Inspection_ID", "Severity"]).size()
                 .reset_index(name="Count")
                 .pivot(index="Inspection_ID", columns="Severity", values="Count")
                 .fillna(0))

    total = merged.groupby("Inspection_ID").size().reset_index(name="Total_Maintenance_Events")

    df = df.set_index("Inspection_ID")
    for cat in MAINTENANCE_CATEGORIES:
        df[cat] = cat_pivot[cat].reindex(df.index, fill_value=0) if cat in cat_pivot.columns else 0
    for sev in SEVERITY_CATEGORIES:
        df[sev] = sev_pivot[sev].reindex(df.index, fill_value=0) if sev in sev_pivot.columns else 0

    df = df.merge(total, left_index=True, right_on="Inspection_ID", how="left")
    df["Total_Maintenance_Events"] = df["Total_Maintenance_Events"].fillna(0).astype(int)
    df = df.reset_index()

    _print(f"  Records with maintenance: {(df['Total_Maintenance_Events'] > 0).sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Paths.ensure_output_dirs()

    pm_map, pipe_geometry = load_network(Paths.NETWORK_ANALYSIS)
    maintenance = load_maintenance(Paths.MAINTENANCE_BY_SHEET)
    df = load_inspections(Paths.MERGED_PROPERTIES)

    # Filter to selected system
    if Params.SYSTEM != "All" and "System" in df.columns:
        df = df[df["System"] == Params.SYSTEM].copy()
        _print(f"Filtered to {Params.SYSTEM}: {len(df):,} records")

    df = attach_pipe_diameter(df, pipe_geometry)
    df = compute_hydraulics(df)
    df = compute_maintenance_features(df, maintenance, pm_map)

    # Column ordering
    base_cols       = ["Manhole_ID", "Inspection_Date", "System", "Surcharged_Binary_Original"]
    maint_cols      = ["Total_Maintenance_Events"] + MAINTENANCE_CATEGORIES + SEVERITY_CATEGORIES
    temporal_cols   = ["Inspection_Year", "Inspection_Month", "Inspection_Season"]
    hydraulic_cols  = [
        "Effective_Water_Depth", "Clear_Water_Thickness", "Exposed_Silt_Depth",
        "Flow_Condition", "Water_Surface_Elevation", "Pipe_Crown_Elevation",
        "Surcharge_Depth", "Surcharge_Ratio", "Surcharge_Severity",
        "Silt_Coverage_Ratio", "Silt_Blockage_Risk",
        "Is_Surcharged", "Needs_Cleaning", "Hydraulic_Risk_Score", "Maintenance_Priority",
        "Avg_Pipe_Diameter", "Diameter_Source", "Invert_Level", "Cover_Level",
    ]
    optional_cols = ["Freeboard", "Flooding_Risk", "Has_Surface_Flooding"]
    derived_cols    = ["Total_Inlet_Diameter", "Total_Outlet_Diameter", "Num_Inlet_Pipes",
                       "Capacity_Ratio", "Silt_Water_Ratio"]
    phys_avail = [c for c in PHYSICAL_FEATURES if c in df.columns]
    cat_avail  = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    all_cols = (base_cols + maint_cols + temporal_cols + hydraulic_cols +
                optional_cols + derived_cols + phys_avail + cat_avail)
    out_cols = [c for c in all_cols if c in df.columns]
    df = df[out_cols].sort_values(["System", "Manhole_ID", "Inspection_Date"])

    df.to_csv(Paths.HYDRAULIC_FEATURES, index=False)
    _print(f"\nSaved → {Paths.HYDRAULIC_FEATURES}  ({len(df):,} rows × {df.shape[1]} cols)")
    _print(f"Target: Is_Surcharged positive cases = {df['Is_Surcharged'].sum():,}")
    _print(f"Target: Surcharge_Ratio median = {df['Surcharge_Ratio'].median():.3f}")


if __name__ == "__main__":
    main()
