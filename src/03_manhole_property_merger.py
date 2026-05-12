"""
03_manhole_property_merger.py
==============================
Merges multiple manhole properties files with the inspection dataset.

Key operations:
  1. Load and deduplicate properties from Combined / Sewer / Stormwater files
  2. Parse coordinates, dates, and numeric fields
  3. Compute Prop_Manhole_Depth (Cover_Level − Invert_Level)
  4. Detect and convert depth units (meters → mm) before ratio calculation
  5. Merge with inspection records
  6. Compute Water_Depth_Ratio and Total_Depth_Ratio (unit-safe)
  7. Create binary condition scores

Unit convention enforced throughout:
  Water_Depth, Silt_Depth          → millimetres (mm)
  Cover_Level, Invert_Level        → metres (m) in raw data
  Prop_Manhole_Depth               → converted to mm before any ratio
  Water_Depth_Ratio, Total_Depth_Ratio → dimensionless [0, 1]

Author: Research Team
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# COLUMNS TO DROP (mostly empty in source data)
# ─────────────────────────────────────────────────────────────────────────────

EMPTY_COLUMNS = [
    "Bottom_Level", "Disconnecting_Trap_IL", "Number_of_Cover",
    "Cover_1_-_Type", "Cover_1_-_Opening_Width", "Cover_1_-_Opening_Length",
    "Cover_2_-_Type", "Cover_2_-_Opening_Width", "Cover_2_-_Opening_Length",
    "Cover_3_-_Type", "Cover_3_-_Opening_Width", "Cover_3_-_Opening_Length",
    "Internal_Size_-_Width", "Internal_Size_-_Length", "Internal_Headroom",
    "Presence_of_Ladder", "Presence_of_Step_Iron", "Presence_of_Cover_Restraint",
    "ID_Tag_Type", "ID_Tag_Installation_Date", "Parent_Feature_Number", "Missing",
    "Description_1", "Description_2", "Description_3", "Description_4",
    "Section_ID", "Design_Capacity", "Maximum_Water_Level", "Design_Flow_Rate",
    "Flood_Protection_Standard", "Remarks", "Inspection_Frequency",
    "Data_Source", "Data_Source_Reference", "Data_Source_Remarks",
    "Abandoned_Date", "Year_In_Service", "Commission_Date", "Stop_Log",
    "Installation_Date", "Maintenance_Agent",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _print(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _load_file(path) -> pd.DataFrame:
    """Load xlsx / xlsb / csv, auto-detecting engine."""
    path = str(path)
    if path.endswith(".xlsb"):
        return pd.read_excel(path, engine="pyxlsb")
    elif path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, engine="openpyxl")
    else:
        return pd.read_csv(path)


def detect_and_convert_depth(series: pd.Series) -> pd.Series:
    """
    Return depth values guaranteed to be in MILLIMETRES.

    Detection rule:
      median < 10  → assume metres → multiply by 1000
      median ≥ 10  → already millimetres
    """
    valid = series.dropna()
    if valid.empty:
        return series
    if valid.median() < 10:
        _print(f"   Unit detection: depth in METRES (median {valid.median():.2f} m) → converting to mm")
        return series * 1000
    _print(f"   Unit detection: depth already in mm (median {valid.median():.0f} mm)")
    return series


def validate_ratios(series: pd.Series, name: str):
    """Print distribution diagnostics for a ratio column."""
    valid = series.dropna()
    if valid.empty:
        return
    binary_pct = ((valid == 0) | (valid == 1.0)).sum() / len(valid) * 100
    status = "✅" if binary_pct < 70 else "⚠️  HIGH binary pct — check units"
    _print(f"   {name}: min={valid.min():.3f} median={valid.median():.3f} "
           f"max={valid.max():.3f}  binary={binary_pct:.1f}%  {status}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — COMBINE PROPERTIES FILES
# ─────────────────────────────────────────────────────────────────────────────

def combine_properties(properties_files: list, strategy: str = "keep_latest") -> pd.DataFrame:
    """Load and deduplicate records from multiple properties files."""
    _print("STEP 1: Combining properties files")
    frames = []
    for path in properties_files:
        _print(f"  Loading {os.path.basename(str(path))}")
        df = _load_file(path)
        df["Source_File"] = os.path.basename(str(path))
        frames.append(df)
        _print(f"    {len(df):,} records")

    combined = pd.concat(frames, ignore_index=True)
    _print(f"  Combined: {len(combined):,} records")

    if "Feature_Number" not in combined.columns:
        raise ValueError("'Feature_Number' column not found in properties files.")

    dupes = len(combined) - combined["Feature_Number"].nunique()
    if dupes > 0:
        _print(f"  {dupes:,} duplicate manholes — applying strategy: {strategy}")
        if strategy == "keep_latest" and "Last_Modified_Date" in combined.columns:
            combined["Last_Modified_Date"] = pd.to_datetime(
                combined["Last_Modified_Date"], format="%d/%m/%Y", errors="coerce"
            )
            combined = (combined
                        .sort_values("Last_Modified_Date")
                        .drop_duplicates("Feature_Number", keep="last"))
        else:
            keep = "last" if strategy in ("keep_latest", "keep_last") else "first"
            combined = combined.drop_duplicates("Feature_Number", keep=keep)
        _print(f"  After deduplication: {len(combined):,} unique manholes")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PREPARE PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────

def prepare_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, parse, and engineer features from the combined properties table."""
    _print("STEP 2: Preparing properties data")
    df = df.copy()

    # Drop empty columns
    to_drop = [c for c in EMPTY_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    _print(f"  Dropped {len(to_drop)} empty columns; {df.shape[1]} remaining")

    # Replace missing-value markers
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].replace(["-", " ", "", "  "], np.nan)

    # Parse dates
    for col in ["Completion_Date", "Placement_Date", "Handover_Date", "Last_Modified_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")

    # Parse coordinates
    if "Coordinate" in df.columns:
        split = df["Coordinate"].str.split(",", expand=True)
        if split.shape[1] >= 2:
            df["Easting"]  = pd.to_numeric(split[0], errors="coerce")
            df["Northing"] = pd.to_numeric(split[1], errors="coerce")

    # Numeric elevations
    for col in ["Cover_Level", "Invert_Level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Manhole depth (metres at this point — converted to mm later, after merge)
    if {"Cover_Level", "Invert_Level"}.issubset(df.columns):
        df["Prop_Manhole_Depth_m"] = df["Cover_Level"] - df["Invert_Level"]

    # Age
    ref = pd.Timestamp("2026-01-13")
    for date_col in ["Completion_Date", "Placement_Date"]:
        if date_col in df.columns and df[date_col].notna().any():
            df["Prop_Age_Years"] = (ref - df[date_col]).dt.days / 365.25
            break

    # Binary flags
    for col in ["Presence_of_Platform", "Presence_of_ID_Tag", "Pile_Foundation"]:
        if col in df.columns:
            df[col + "_Binary"] = df[col].map({"Yes": 1, "No": 0})

    if "Road_Type"     in df.columns: df["Is_Carriageway"] = (df["Road_Type"] == "Carriageway").astype(int)
    if "Material"      in df.columns: df["Is_Concrete"]    = (df["Material"]  == "Concrete").astype(int)
    if "Present_State" in df.columns: df["Is_Active"]      = (df["Present_State"] == "Existing").astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PREPARE INSPECTION DATA
# ─────────────────────────────────────────────────────────────────────────────

def prepare_inspection(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and type-convert the inspection dataset."""
    _print("STEP 3: Preparing inspection data")
    df = df.copy()

    if "Inspection_Date" in df.columns:
        df["Inspection_Date"] = pd.to_datetime(
            df["Inspection_Date"], format="%d/%m/%Y", errors="coerce"
        )

    numeric_cols = [
        "Inlet_Pipe_1_Dia", "Inlet_Pipe_2_Dia", "Inlet_Pipe_3_Dia", "Inlet_Pipe_4_Dia",
        "Outlet_Pipe_1_Dia", "Outlet_Pipe_2_Dia",
        "Water_Depth", "Silt_Depth",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Y/N → binary
    yn_cols = [
        "Surcharged", "Inconsistent_Manhole_Cover", "Manhole_ID_Tag_Damaged",
        "Manhole_ID_Tag_Missing", "Wrong_Manhole_ID_Tag", "Subsidence_of_Invert",
        "Cracking,_Spalling", "Grease,_Slime", "Sign_of_Sewage_in_SWD",
        "Stinky_Smell", "Flammable_Fluid", "Flammable_Fumes",
    ]
    for col in yn_cols:
        if col in df.columns:
            df[col + "_Binary"] = df[col].map({"Y": 1, "N": 0})

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MERGE
# ─────────────────────────────────────────────────────────────────────────────

def merge(df_insp: pd.DataFrame, df_prop: pd.DataFrame) -> pd.DataFrame:
    """Left-join inspection records onto properties on manhole ID."""
    _print("STEP 4: Merging inspection and properties")
    merged = df_insp.merge(
        df_prop,
        left_on="Manhole",
        right_on="Feature_Number",
        how="left",
        suffixes=("_Insp", "_Prop"),
    )
    matched = merged["Feature_Number"].notna().sum()
    _print(f"  {len(merged):,} records  |  "
           f"{matched:,} matched ({matched/len(merged)*100:.1f}%)  |  "
           f"{len(merged)-matched:,} unmatched")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — COMPUTE RATIOS (unit-safe)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute depth ratios after guaranteeing unit consistency.

    Prop_Manhole_Depth_m is stored in metres from the properties prep step.
    Water_Depth and Silt_Depth are in millimetres from the inspection file.
    This function converts Prop_Manhole_Depth to mm, then divides.
    """
    _print("STEP 5: Computing depth ratios (unit-safe)")
    df = df.copy()

    if "Prop_Manhole_Depth_m" not in df.columns:
        _print("  ⚠  Prop_Manhole_Depth_m not found — skipping ratios")
        return df

    # Convert to mm — detect_and_convert_depth handles both m and mm input
    df["Prop_Manhole_Depth"] = detect_and_convert_depth(df["Prop_Manhole_Depth_m"])
    df = df.drop(columns=["Prop_Manhole_Depth_m"])

    safe_depth = df["Prop_Manhole_Depth"].abs() + 0.01   # avoid divide-by-zero

    if "Water_Depth" in df.columns:
        df["Water_Depth_Ratio"] = (df["Water_Depth"] / safe_depth).clip(0, 1)
        validate_ratios(df["Water_Depth_Ratio"], "Water_Depth_Ratio")

    if {"Water_Depth", "Silt_Depth"}.issubset(df.columns):
        df["Total_Depth"] = df["Water_Depth"].fillna(0) + df["Silt_Depth"].fillna(0)
        df["Total_Depth_Ratio"] = (df["Total_Depth"] / safe_depth).clip(0, 1)
        validate_ratios(df["Total_Depth_Ratio"], "Total_Depth_Ratio")

    # Condition scores
    structural_cols = [c for c in [
        "Inconsistent_Manhole_Cover_Binary", "Manhole_ID_Tag_Damaged_Binary",
        "Manhole_ID_Tag_Missing_Binary", "Subsidence_of_Invert_Binary",
        "Cracking,_Spalling_Binary",
    ] if c in df.columns]
    if structural_cols:
        df["Structural_Issue_Count"] = df[structural_cols].sum(axis=1)

    hazard_cols = [c for c in [
        "Grease,_Slime_Binary", "Sign_of_Sewage_in_SWD_Binary",
        "Stinky_Smell_Binary", "Flammable_Fluid_Binary", "Flammable_Fumes_Binary",
    ] if c in df.columns]
    if hazard_cols:
        df["Hazard_Score"] = df[hazard_cols].sum(axis=1)

    _print(f"  Ratios complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Paths.ensure_output_dirs()

    properties_files = [
        Paths.COMBINED_MANHOLE,
        Paths.SEWER_MANHOLE,
        Paths.STORMWATER_MANHOLE,
    ]

    # --- Properties ---
    df_prop_raw  = combine_properties(properties_files, Params.DUPLICATE_STRATEGY)
    df_prop      = prepare_properties(df_prop_raw)

    # --- Inspection ---
    _print("Loading inspection data")
    df_insp_raw = _load_file(Paths.INSPECTION_XLSB)
    _print(f"  {len(df_insp_raw):,} inspection records")
    df_insp = prepare_inspection(df_insp_raw)

    # --- Merge ---
    df_merged = merge(df_insp, df_prop)

    # --- Ratios (unit-safe) ---
    df_final = compute_ratios(df_merged)

    # --- Save ---
    out = Paths.MERGED_PROPERTIES
    df_final.to_csv(out, index=False)
    _print(f"Saved → {out}  ({len(df_final):,} rows × {df_final.shape[1]} cols)")


if __name__ == "__main__":
    main()
