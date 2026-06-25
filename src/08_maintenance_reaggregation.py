#!/usr/bin/env python3
"""
08_maintenance_reaggregation.py
================================
Temporal interval-based maintenance reaggregation.

Replaces the earlier lifetime-total approach that caused data leakage.

RULE:
  Each maintenance event is assigned to exactly ONE inspection: the FIRST
  inspection of that manhole whose Inspection_Date is on or after the event's
  date_of_completion. Events are bucketed into the inter-inspection interval
  in which they were completed. The first inspection of each manhole absorbs
  ALL prior events (full history back to the start of the maintenance records).
  Features are between-inspection counts only — no cumulative totals.

WHAT CHANGED vs the original:
  Original: pipe events → single per-pipe lifetime total (dates dropped) →
            merged onto every inspection of a manhole by Manhole_ID.
            Result: lifetime totals identical across all inspections → DATA LEAKAGE.
  This:     dated activity-level events → mapped to manholes via connected
            pipes → each event assigned to one (Manhole_ID, Inspection_Date)
            bucket by completion date → between-inspection counts per record.

INPUTS (configure via config.py):
  1. Paths.MAINTENANCE_BY_SHEET : folder of activity-level CSV files from
                                   Script 01 (one per source sheet).
                                   Each must contain: feature_number,
                                   feature_prefix, primary_category,
                                   severity_keyword_based, date_of_completion.
                                   Only stormwater rows (feature_prefix == 'S')
                                   are used when Params.SYSTEM == 'Stormwater'.
  2. Paths.PIPE_SHAPEFILE_CSV   : pipe shapefile CSV with FEAT_NUM, FR_PNT,
                                   TO_PNT (same file as the original Script 08).
  3. Paths.DATA_CLEANED         : cleaned ML inspection dataset (output of
                                   Script 07) with at least Manhole_ID and
                                   Inspection_Date.

OUTPUT:
  Paths.MAINTENANCE_FIXED : DATA_CLEANED with the 14 category columns, 6
                             severity columns, connected_pipes, and
                             Total_Maintenance_Events replaced by temporally
                             correct between-inspection counts.

Author: Research Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Attachment mode: how maintenance feature_numbers map to manholes.
#   "hybrid" — route each event by its own ID type: IDs matching an inspected
#              manhole attach directly; IDs matching a pipe attach via connected
#              manholes. Correct for mixed SWD-pipe + SMH-manhole maintenance.
ATTACH_MODE    = "hybrid"

DATE_FORMAT    = "%d/%m/%Y"   # DSD date convention: day-first dd/mm/YYYY
MANHOLE_COL    = "Manhole_ID"
INSPDATE_COL   = "Inspection_Date"

# The 14 maintenance category columns.
# Left = primary_category label from Script 01; Right = column name in dataset.
CATEGORY_MAP = {
    "Blockages":                                        "Blockages",
    "Blockage and Inspection":                          "Blockage_and_Inspection",
    "General Cleaning":                                 "General_Cleaning",
    "Infiltration and inflow":                          "Infiltration_and_inflow",
    "Flow velocity/ hydraulic conditions":              "Flow_velocity__hydraulic_conditions",
    "Cracks and fractures":                             "Cracks_and_fractures",
    "Defective lining":                                 "Defective_lining",
    "Deformation":                                      "Deformation",
    "Corrosion and abrasion of pipes":                  "Corrosion_and_abrasion_of_pipes",
    "Defective connection":                             "Defective_connection",
    "Broken manhole covers/ damaged manhole walls":     "Broken_manhole_covers__damaged_manhole_walls",
    "Environmental":                                    "Environmental",
    "Inspection":                                       "Inspection",
    "General Maintenance":                              "General_Maintenance",
}
CATEGORY_COLUMNS = list(CATEGORY_MAP.values())

# Severity: Script 01 writes severity_keyword_based as "EMERGENCY - Keyword" etc.
# Strip the " - Keyword" suffix to recover the 6 ML severity columns.
SEVERITY_SOURCE_COL = "severity_keyword_based"
SEVERITY_COLUMNS    = ["EMERGENCY", "HIGH", "MEDIUM", "LOW", "ROUTINE", "DITTO"]

ACTIVITY_GLOB = "*_activities*.csv"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def log(msg): print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")


def read_robust(path, **kwargs):
    """Read CSV with fallback encoding and delimiter detection."""
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, **kwargs)
    for enc in ("utf-8", "cp1252", "latin-1"):
        for sep in (",", "\t", ";"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep,
                                 low_memory=False, **kwargs)
                if df.shape[1] > 1:
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    return pd.read_csv(path, encoding="latin-1", sep=None,
                       engine="python", encoding_errors="replace", **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PIPE-MANHOLE MAP
# ─────────────────────────────────────────────────────────────────────────────

def build_pipe_manhole_maps(pipe_csv):
    """Return (pipe_to_manholes, manhole_to_pipes) dicts."""
    log("Building pipe-manhole map")
    pipes = read_robust(pipe_csv)

    # Normalise column names
    col_map = {}
    for col in pipes.columns:
        u = col.upper()
        if "FEAT" in u and "NUM" in u:  col_map[col] = "FEAT_NUM"
        elif "FR"  in u and "PNT" in u: col_map[col] = "FR_PNT"
        elif "TO"  in u and "PNT" in u: col_map[col] = "TO_PNT"
        elif "UPS" in u and "NOD" in u: col_map[col] = "FR_PNT"
        elif "DOW" in u and "NOD" in u: col_map[col] = "TO_PNT"
    pipes = pipes.rename(columns=col_map)

    for req in ["FEAT_NUM", "FR_PNT", "TO_PNT"]:
        if req not in pipes.columns:
            raise ValueError(f"Required column '{req}' not found in pipe file. "
                             f"Available: {list(pipes.columns)}")

    pipe_to_manholes = {}   # pipe_id -> [manhole_id, ...]
    manhole_to_pipes = defaultdict(list)

    for _, row in pipes.iterrows():
        feat = str(row["FEAT_NUM"]).strip().upper()
        fr   = str(row["FR_PNT"]).strip().upper()
        to_  = str(row["TO_PNT"]).strip().upper()
        if feat and fr and fr != "NAN":
            pipe_to_manholes.setdefault(feat, [])
            if fr not in pipe_to_manholes[feat]:
                pipe_to_manholes[feat].append(fr)
            manhole_to_pipes[fr].append(feat)
        if feat and to_ and to_ != "NAN":
            pipe_to_manholes.setdefault(feat, [])
            if to_ not in pipe_to_manholes[feat]:
                pipe_to_manholes[feat].append(to_)
            manhole_to_pipes[to_].append(feat)

    log(f"  {len(pipe_to_manholes):,} pipes | {len(manhole_to_pipes):,} manholes")
    return pipe_to_manholes, dict(manhole_to_pipes)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD AND FILTER ACTIVITY EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def load_activities(by_sheet_dir):
    """Load all activity CSV files, filter to stormwater, parse dates."""
    log("Loading activity files")
    frames = []
    for f in sorted(Path(by_sheet_dir).glob(ACTIVITY_GLOB)):
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
        log(f"  {f.name}: {len(df):,} rows")

    if not frames:
        raise FileNotFoundError(
            f"No activity files found in {by_sheet_dir}. "
            "Run Script 01 first.")

    acts = pd.concat(frames, ignore_index=True)
    log(f"  Combined: {len(acts):,} activity strings")

    # Stormwater filter
    if Params.SYSTEM == "Stormwater":
        if "feature_prefix" not in acts.columns:
            acts["feature_prefix"] = (acts["feature_number"]
                                      .astype(str).str.strip().str[0].str.upper())
        before = len(acts)
        acts = acts[acts["feature_prefix"].astype(str)
                    .str.strip().str.upper().str.startswith("S")].copy()
        log(f"  Stormwater filter: {before:,} → {len(acts):,}")

    # Normalise feature_number
    acts["feature_number"] = (acts["feature_number"]
                              .astype(str).str.strip().str.upper())

    # Parse completion date
    date_col = next((c for c in ["date_of_completion", "date_of_commencement"]
                     if c in acts.columns), None)
    if date_col is None:
        raise ValueError("No date column found in activity files.")
    acts["date_of_completion"] = pd.to_datetime(
        acts[date_col].astype(str).str.strip(),
        format=DATE_FORMAT, errors="coerce", dayfirst=True)
    acts = acts[acts["date_of_completion"].notna()].copy()
    log(f"  After date filter: {len(acts):,} events")

    # Severity: strip " - Keyword" suffix
    if SEVERITY_SOURCE_COL in acts.columns:
        acts["__sev"] = (acts[SEVERITY_SOURCE_COL]
                         .astype(str)
                         .str.replace(r"\s*-\s*Keyword$", "", regex=True)
                         .str.strip())
    else:
        acts["__sev"] = "ROUTINE"

    return acts


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Paths.ensure_output_dirs()

    # ── Load pipe network ─────────────────────────────────────────────────────
    pipe_to_manholes, manhole_to_pipes = build_pipe_manhole_maps(
        Paths.PIPE_SHAPEFILE_CSV)

    # ── Load activity events ──────────────────────────────────────────────────
    acts = load_activities(Paths.MAINTENANCE_BY_SHEET)

    # ── Load inspection dataset ───────────────────────────────────────────────
    log("Loading inspection dataset")
    insp = pd.read_csv(Paths.DATA_CLEANED, low_memory=False)
    if Params.SYSTEM != "All" and "System" in insp.columns:
        insp = insp[insp["System"] == Params.SYSTEM].copy()
    log(f"  {len(insp):,} records | {insp[MANHOLE_COL].nunique():,} manholes")

    # Normalise IDs and dates
    insp[MANHOLE_COL] = insp[MANHOLE_COL].astype(str).str.strip().str.upper()
    insp[INSPDATE_COL] = pd.to_datetime(
        insp[INSPDATE_COL], errors="coerce", dayfirst=True)
    insp = insp[insp[INSPDATE_COL].notna()].copy()

    # Drop old maintenance columns before rebuild
    old_cols = [c for c in (CATEGORY_COLUMNS + SEVERITY_COLUMNS +
                             ["connected_pipes", "Total_Maintenance_Events"])
                if c in insp.columns]
    insp = insp.drop(columns=old_cols, errors="ignore")
    log(f"  Dropped {len(old_cols)} old maintenance columns for rebuild")

    # ── Build per-manhole inspection date index ───────────────────────────────
    insp = insp.sort_values([MANHOLE_COL, INSPDATE_COL]).reset_index(drop=True)
    mh_dates = (insp.dropna(subset=[INSPDATE_COL])
                    .groupby(MANHOLE_COL)[INSPDATE_COL]
                    .apply(lambda s: np.sort(s.unique()))
                    .to_dict())

    # ── ID-space check ────────────────────────────────────────────────────────
    maint_ids      = set(acts["feature_number"].unique())
    pipe_ids_set   = set(pipe_to_manholes.keys())
    manhole_ids_set = set(mh_dates.keys())
    pct_pipe    = len(maint_ids & pipe_ids_set)    / max(len(maint_ids), 1) * 100
    pct_manhole = len(maint_ids & manhole_ids_set) / max(len(maint_ids), 1) * 100
    log(f"ID-space: {pct_pipe:.1f}% match PIPE | {pct_manhole:.1f}% match MANHOLE")
    log(f"  Attachment mode: {ATTACH_MODE}")

    # ── Assign each event to its inter-inspection bucket ──────────────────────
    log("Assigning events to inter-inspection buckets")
    bucket = defaultdict(lambda: defaultdict(float))
    n_assigned = n_no_insp = n_no_manhole = 0
    n_via_pipe = n_via_manhole = 0
    insp_mh_set = set(mh_dates.keys())

    for feat, grp in acts.groupby("feature_number"):
        if ATTACH_MODE == "pipe":
            manholes = pipe_to_manholes.get(feat)
        elif ATTACH_MODE == "manhole":
            manholes = [feat] if feat in mh_dates else None
        else:  # hybrid
            if feat in insp_mh_set:
                manholes = [feat]; n_via_manhole += len(grp)
            elif feat in pipe_to_manholes:
                manholes = pipe_to_manholes.get(feat); n_via_pipe += len(grp)
            else:
                manholes = None

        if not manholes:
            n_no_manhole += len(grp)
            continue

        comp_all = grp["date_of_completion"].values.astype("datetime64[ns]")
        cats_all = (grp["primary_category"].astype(str).str.strip()
                    .map(CATEGORY_MAP).values)
        sevs_all = grp["__sev"].astype(str).values

        for mh in manholes:
            dates = mh_dates.get(mh)
            if dates is None or len(dates) == 0:
                n_no_insp += len(grp)
                continue
            idx = np.searchsorted(dates, comp_all, side="left")
            for k in range(len(comp_all)):
                if idx[k] >= len(dates):   # event after last inspection → drop
                    continue
                key = (mh, pd.Timestamp(dates[idx[k]]))
                col = cats_all[k]
                if col is not None and not (isinstance(col, float)
                                             and np.isnan(col)):
                    bucket[key][col] += 1.0
                sev = sevs_all[k]
                if sev in SEVERITY_COLUMNS:
                    bucket[key][sev] += 1.0
                n_assigned += 1

    log(f"  Assigned {n_assigned:,} event×manhole attributions")
    if ATTACH_MODE == "hybrid":
        log(f"  Routing: {n_via_manhole:,} via direct manhole | "
            f"{n_via_pipe:,} via pipe network")
    log(f"  Skipped: {n_no_manhole:,} unmatched IDs | "
        f"{n_no_insp:,} on manholes not in inspection set")

    # ── Build per-inspection maintenance frame ────────────────────────────────
    log("Building per-inspection maintenance frame")
    rows = []
    for (mh, d), counts in bucket.items():
        rec = {MANHOLE_COL: mh, INSPDATE_COL: d}
        for c in CATEGORY_COLUMNS + SEVERITY_COLUMNS:
            rec[c] = counts.get(c, 0.0)
        rows.append(rec)
    maint_insp = pd.DataFrame(rows)
    if maint_insp.empty:
        raise RuntimeError("No maintenance assigned — check date formats and "
                           "ID matching. Verify PIPE_SHAPEFILE_CSV path.")

    # Connected pipes: static manhole property
    conn = {mh: len(p) for mh, p in manhole_to_pipes.items()}
    maint_insp["connected_pipes"] = maint_insp[MANHOLE_COL].map(conn).fillna(0)

    # ── Merge back onto inspections ───────────────────────────────────────────
    insp = insp.merge(maint_insp, on=[MANHOLE_COL, INSPDATE_COL], how="left")

    fill_cols = CATEGORY_COLUMNS + SEVERITY_COLUMNS + ["connected_pipes"]
    for c in fill_cols:
        if c in insp.columns:
            insp[c] = pd.to_numeric(insp[c], errors="coerce").fillna(0).astype(int)
        else:
            insp[c] = 0
    # Ensure connected_pipes reflects network even where no events in interval
    insp["connected_pipes"] = (insp[MANHOLE_COL].map(conn)
                               .fillna(0).astype(int))

    # Total = sum of 14 mutually exclusive primary categories
    insp["Total_Maintenance_Events"] = (insp[CATEGORY_COLUMNS]
                                        .sum(axis=1).astype(int))

    # ── Validation checks ─────────────────────────────────────────────────────
    n_with = (insp["Total_Maintenance_Events"] > 0).sum()
    log(f"  Inspections with ≥1 prior-interval event: {n_with:,} "
        f"({n_with/len(insp)*100:.1f}%)")
    log(f"  Total between-inspection events: "
        f"{insp['Total_Maintenance_Events'].sum():,}")
    log(f"  Mean per record : {insp['Total_Maintenance_Events'].mean():.2f}")
    log(f"  Mean per manhole: "
        f"{insp.groupby(MANHOLE_COL)['Total_Maintenance_Events'].sum().mean():.2f}")

    # Category sum consistency check (should be 100%)
    cat_sum = insp[CATEGORY_COLUMNS].sum(axis=1)
    exact   = (abs(cat_sum - insp["Total_Maintenance_Events"]) < 0.01).mean() * 100
    log(f"  Category sum consistency: {exact:.1f}%")

    # Leakage check: counts must now vary across inspections of the same manhole
    var = (insp.groupby(MANHOLE_COL)["Total_Maintenance_Events"]
               .nunique().gt(1).mean() * 100)
    log(f"  Manholes with varying event counts across inspections: {var:.1f}% "
        f"(was ~0% under the leaky lifetime-total scheme)")

    # ── Save ──────────────────────────────────────────────────────────────────
    insp.to_csv(Paths.MAINTENANCE_FIXED, index=False)
    log(f"Saved → {Paths.MAINTENANCE_FIXED}  "
        f"({len(insp):,} rows × {insp.shape[1]} cols)")


if __name__ == "__main__":
    main()
