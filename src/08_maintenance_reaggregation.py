"""
08_maintenance_reaggregation.py
================================
Re-aggregates pipe-level maintenance history to manhole level and
attaches it to the cleaned ML dataset.

This is the final maintenance integration step. The script:
  1. Loads the cleaned dataset and the pipe shapefile (for pipe-manhole mapping)
  2. Loads the aggregated maintenance statistics from Script 01
  3. Maps pipe maintenance counts to their connected manholes (upstream + downstream)
  4. Sums category and severity counts across all connected pipes
  5. Recomputes Total_Maintenance_Events as the sum of activity categories

Author: Research Team
"""

import pandas as pd
import numpy as np
import os, sys, warnings
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

warnings.filterwarnings("ignore")


def _print(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


ACTIVITY_CATEGORIES = [
    "Blockages", "Blockage_and_Inspection", "General_Cleaning",
    "Infiltration_and_inflow", "Flow_velocity__hydraulic_conditions",
    "Cracks_and_fractures", "Defective_lining", "Deformation",
    "Corrosion_and_abrasion_of_pipes", "Defective_connection",
    "Broken_manhole_covers__damaged_manhole_walls",
    "Environmental", "Inspection", "General_Maintenance",
]
SEVERITY_LEVELS = ["EMERGENCY", "HIGH", "MEDIUM", "LOW", "ROUTINE"]


def main():
    Paths.ensure_output_dirs()

    # Load main dataset
    _print("Loading cleaned dataset")
    df = pd.read_csv(Paths.DATA_CLEANED, low_memory=False)
    if Params.SYSTEM != "All" and "System" in df.columns:
        df = df[df["System"] == Params.SYSTEM].copy()
    _print(f"  {len(df):,} records | {df['Manhole_ID'].nunique():,} unique manholes")

    # Load pipe network (shapefile CSV) for pipe-manhole mapping
    _print("Building pipe-manhole map")
    pipes = pd.read_csv(str(Paths.PIPE_SHAPEFILE_CSV), low_memory=False)
    col_map = {}
    for col in pipes.columns:
        u = col.upper()
        if "FEAT" in u and "NUM" in u:  col_map[col] = "FEAT_NUM"
        elif "FR"  in u and "PNT" in u: col_map[col] = "FR_PNT"
        elif "TO"  in u and "PNT" in u: col_map[col] = "TO_PNT"
    pipes.rename(columns=col_map, inplace=True)

    for req in ["FEAT_NUM", "FR_PNT", "TO_PNT"]:
        if req not in pipes.columns:
            raise ValueError(f"Required column '{req}' not found in pipe shapefile")

    manhole_pipes: dict[str, list[str]] = {}
    for _, row in pipes.iterrows():
        feat = str(row["FEAT_NUM"]).strip()
        fr   = str(row["FR_PNT"]).strip()
        to_  = str(row["TO_PNT"]).strip()
        manhole_pipes.setdefault(fr, []).append(feat)
        manhole_pipes.setdefault(to_, []).append(feat)
    _print(f"  {len(manhole_pipes):,} manholes with connected pipes")

    # Load maintenance statistics
    _print("Loading maintenance statistics")
    maint = pd.read_excel(str(Paths.MAINTENANCE_COMBINED), sheet_name="Feature_History_S",
                          engine="openpyxl")
    maint["FEAT_NUM"] = maint["feature_number"].astype(str).str.strip()

    # Identify available activity and severity count columns
    act_count_cols = [c for c in maint.columns
                      if c.endswith("_count") and "severity" not in c.lower()
                      and "response" not in c.lower()]
    sev_count_cols = [c for c in maint.columns
                      if "severity_" in c.lower() and c.endswith("_count")]

    act_name_map = {c: c.replace("_count", "") for c in act_count_cols}
    sev_name_map = {c: c.replace("severity_", "").replace("_count", "") for c in sev_count_cols}
    all_new_cols  = list(act_name_map.values()) + list(sev_name_map.values()) + \
                    ["connected_pipes", "Total_Maintenance_Events"]

    pipe_maint = {str(r["FEAT_NUM"]): r.to_dict() for _, r in maint.iterrows()}
    _print(f"  Maintenance data for {len(pipe_maint):,} pipes")

    # Remove old maintenance columns if present
    old_cols = [c for c in df.columns if c.replace("_x","").replace("_y","") in set(all_new_cols)]
    df.drop(columns=old_cols, inplace=True, errors="ignore")

    # Aggregate to manholes
    _print("Aggregating maintenance to manholes")
    records = []
    for manhole_id in df["Manhole_ID"].unique():
        rec = {"Manhole_ID": manhole_id, "connected_pipes": 0}
        for c in list(act_name_map.values()) + list(sev_name_map.values()):
            rec[c] = 0

        mh_str = str(manhole_id).strip()
        if mh_str in manhole_pipes:
            pipe_ids = manhole_pipes[mh_str]
            rec["connected_pipes"] = len(pipe_ids)
            for pid in pipe_ids:
                if pid not in pipe_maint: continue
                data = pipe_maint[pid]
                for old, new in act_name_map.items():
                    v = data.get(old)
                    if pd.notna(v): rec[new] += float(v)
                for old, new in sev_name_map.items():
                    v = data.get(old)
                    if pd.notna(v): rec[new] += float(v)

        records.append(rec)

    maint_agg = pd.DataFrame(records)

    # Total maintenance events = sum of activity categories
    act_cols_present = [c for c in act_name_map.values() if c in maint_agg.columns]
    maint_agg["Total_Maintenance_Events"] = maint_agg[act_cols_present].sum(axis=1)

    df = df.merge(maint_agg, on="Manhole_ID", how="left")

    # Fill and type
    for col in act_cols_present + list(sev_name_map.values()) + ["connected_pipes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["Total_Maintenance_Events"] = df["Total_Maintenance_Events"].fillna(0).astype(int)

    # Validation
    n_with_maint = (df["Total_Maintenance_Events"] > 0).sum()
    _print(f"  Manholes with maintenance events: {n_with_maint:,} ({n_with_maint/len(df)*100:.1f}%)")
    _print(f"  Total events aggregated: {df['Total_Maintenance_Events'].sum():,}")

    # Verify activity sum equals total
    act_sum = df[act_cols_present].sum(axis=1)
    exact = (abs(act_sum - df["Total_Maintenance_Events"]) < 0.01).mean() * 100
    _print(f"  Category-total consistency: {exact:.1f}% rows exact match")

    df.to_csv(Paths.MAINTENANCE_FIXED, index=False)
    _print(f"\nSaved → {Paths.MAINTENANCE_FIXED}  ({len(df):,} rows × {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
