"""
02_network_topology.py
=======================
Analyses pipe network topology for Combined, Sewer, and Stormwater systems.

For each system computes:
  - Connectivity metrics (in-degree, out-degree, isolation flags)
  - Manhole depth and data quality classification
  - Pipe slope (capped to physical plausibility range)
  - Connected pipe diameter statistics
  - Pipe material inventory per manhole

Outputs are used as pipe geometry lookups in script 05.

Input:  Combined/Sewer/Stormwater Manhole.xlsx and Pipe.xlsx
Output: All_Systems_Network_Analysis.xlsx + Comparative_QC_Report.txt

Author: Research Team
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SYSTEMS = {
    "Combined_System": {
        "manhole": Paths.COMBINED_MANHOLE,
        "pipe":    Paths.MANHOLE_DIR / "Combined_Pipe.xlsx"
    },
    "Sewer_System": {
        "manhole": Paths.SEWER_MANHOLE,
        "pipe":    Paths.MANHOLE_DIR / "Sewer_Pipe.xlsx"
    },
    "Stormwater_System": {
        "manhole": Paths.STORMWATER_MANHOLE,
        "pipe":    Paths.STORMWATER_PIPE
    }
}

output_dir    = Paths.NETWORK_ANALYSIS.parent
output_excel  = Paths.NETWORK_ANALYSIS
output_report = output_dir / "Comparative_QC_Report.txt"


# ─────────────────────────────────────────────────────────────────────────────
# CORE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_system(system_name, manhole_file, pipe_file):
    """Process one drainage system; return node and link tables."""
    print(f"\n{'='*70}\nProcessing: {system_name}\n{'='*70}")

    try:
        manholes = pd.read_excel(manhole_file, engine='openpyxl')
        pipes    = pd.read_excel(pipe_file,    engine='openpyxl')
        print(f"✓ Loaded {len(manholes):,} manholes and {len(pipes):,} pipes")

        # Standardise IDs
        manholes["Node_ID"]       = manholes["Feature_Number"].astype(str).str.strip()
        pipes["Link_ID"]          = pipes["Feature_Number"].astype(str).str.strip()
        pipes["Upstream_Node"]    = pipes["Upstream_Node"].astype(str).str.strip()
        pipes["Downstream_Node"]  = pipes["Downstream_Node"].astype(str).str.strip()

        # ── Node table ────────────────────────────────────────────────────────
        node_columns = [
            "Node_ID", "Feature_Type", "Manhole_Type", "Manhole_Standard",
            "Material", "Cover_Level", "Invert_Level",
            "Road_Type", "District", "Sub-district", "Coordinate",
            "Present_State", "Year_In_Service", "Installation_Date"
        ]
        node_table = manholes[[c for c in node_columns if c in manholes.columns]].copy()

        # Manhole depth
        if {"Cover_Level", "Invert_Level"}.issubset(node_table.columns):
            node_table["Manhole_Depth"] = (node_table["Cover_Level"]
                                           - node_table["Invert_Level"])
            cov_miss  = node_table["Cover_Level"].isna()
            inv_miss  = node_table["Invert_Level"].isna()
            node_table["Depth_Data_Quality"] = "Complete"
            node_table.loc[cov_miss & inv_miss,  "Depth_Data_Quality"] = "Missing_Both"
            node_table.loc[cov_miss & ~inv_miss, "Depth_Data_Quality"] = "Missing_Cover"
            node_table.loc[~cov_miss & inv_miss, "Depth_Data_Quality"] = "Missing_Invert"
            node_table.loc[node_table["Manhole_Depth"] < 0,
                           "Depth_Data_Quality"] = "Invalid_Negative"
            node_table.loc[(node_table["Manhole_Depth"] >= 0)
                           & (node_table["Manhole_Depth"] < 0.1),
                           "Depth_Data_Quality"] = "Invalid_Zero"
        else:
            node_table["Manhole_Depth"]      = np.nan
            node_table["Depth_Data_Quality"] = "No_Data_Available"

        # ── Link table ────────────────────────────────────────────────────────
        link_columns = [
            "Link_ID", "Upstream_Node", "Downstream_Node", "Material", "Shape",
            "Nominal_Width", "Nominal_Height", "Computed_Length", "Measured_Length",
            "Upstream_Invert_Level", "Downstream_Invert_Level",
            "Present_State", "Year_In_Service", "Installation_Date"
        ]
        link_table = pipes[[c for c in link_columns if c in pipes.columns]].copy()

        # Topology validation
        node_set = set(node_table["Node_ID"])
        link_table["Missing_Upstream_Node"]   = ~link_table["Upstream_Node"].isin(node_set)
        link_table["Missing_Downstream_Node"] = ~link_table["Downstream_Node"].isin(node_set)
        link_table["Topology_Status"] = np.where(
            link_table["Missing_Upstream_Node"] | link_table["Missing_Downstream_Node"],
            "Orphan_Pipe", "Connected"
        )

        # Connected pipe inventory per node
        up_pipes   = link_table.groupby("Upstream_Node")["Link_ID"].apply(
            lambda x: "; ".join(sorted(x)))
        down_pipes = link_table.groupby("Downstream_Node")["Link_ID"].apply(
            lambda x: "; ".join(sorted(x)))
        node_table["Upstream_Pipe_IDs"]   = node_table["Node_ID"].map(up_pipes).fillna("")
        node_table["Downstream_Pipe_IDs"] = node_table["Node_ID"].map(down_pipes).fillna("")

        # Connectivity
        up_cnt   = link_table.groupby("Upstream_Node").size()
        down_cnt = link_table.groupby("Downstream_Node").size()
        node_table["Out_Degree"]   = node_table["Node_ID"].map(up_cnt).fillna(0).astype(int)
        node_table["In_Degree"]    = node_table["Node_ID"].map(down_cnt).fillna(0).astype(int)
        node_table["Total_Degree"] = node_table["Out_Degree"] + node_table["In_Degree"]
        node_table["Is_Isolated"]  = node_table["Total_Degree"] == 0

        # Pipe slope (capped to physical plausibility)
        req = {"Upstream_Invert_Level", "Downstream_Invert_Level", "Computed_Length"}
        if req.issubset(link_table.columns):
            link_table["Pipe_Slope"] = (
                (link_table["Upstream_Invert_Level"] - link_table["Downstream_Invert_Level"])
                / link_table["Computed_Length"].replace(0, np.nan)
            )
            link_table["Pipe_Slope"] = link_table["Pipe_Slope"].clip(
                Params.MIN_PIPE_SLOPE, Params.MAX_PIPE_SLOPE
            )
        else:
            link_table["Pipe_Slope"] = np.nan

        # Length floor
        if "Computed_Length" in link_table.columns:
            link_table.loc[
                link_table["Computed_Length"] < Params.MIN_PIPE_LENGTH_M,
                "Computed_Length"
            ] = np.nan

        # Advanced node metrics (connected pipes only)
        la = link_table[link_table["Topology_Status"] == "Connected"].copy()

        if "Pipe_Slope" in la.columns:
            up_sl   = la.groupby("Upstream_Node")["Pipe_Slope"].mean()
            dn_sl   = la.groupby("Downstream_Node")["Pipe_Slope"].mean()
            all_sl  = pd.concat([up_sl, dn_sl]).groupby(level=0).mean()
            node_table["Avg_Connected_Pipe_Slope"] = node_table["Node_ID"].map(all_sl)

        if "Nominal_Width" in la.columns:
            up_max  = la.groupby("Upstream_Node")["Nominal_Width"].max()
            dn_max  = la.groupby("Downstream_Node")["Nominal_Width"].max()
            up_min  = la.groupby("Upstream_Node")["Nominal_Width"].min()
            dn_min  = la.groupby("Downstream_Node")["Nominal_Width"].min()
            all_max = pd.concat([up_max, dn_max]).groupby(level=0).max()
            all_min = pd.concat([up_min, dn_min]).groupby(level=0).min()
            node_table["Max_Connected_Pipe_Diameter"] = node_table["Node_ID"].map(all_max)
            node_table["Min_Connected_Pipe_Diameter"] = node_table["Node_ID"].map(all_min)
            mx = node_table["Max_Connected_Pipe_Diameter"]
            mn = node_table["Min_Connected_Pipe_Diameter"]
            node_table["Has_Size_Change"] = (
                mx.notna() & mn.notna() & (mn > 0) & ((mx - mn) / mn > 0.2)
            )

        if "Material" in la.columns:
            def _mats(node_id):
                up  = la[la["Upstream_Node"]   == node_id]["Material"].dropna().astype(str).unique()
                dn  = la[la["Downstream_Node"]  == node_id]["Material"].dropna().astype(str).unique()
                return "; ".join(sorted(set(list(up) + list(dn))))
            node_table["Connected_Pipe_Materials"] = node_table["Node_ID"].apply(_mats)

        print(f"✅ {system_name} complete")
        return {"success": True, "node_table": node_table,
                "link_table": link_table, "system_name": system_name}

    except Exception as e:
        print(f"❌ Error processing {system_name}: {e}")
        return {"success": False, "error": str(e), "system_name": system_name}


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def generate_system_stats(result):
    if not result["success"]:
        return None
    nt = result["node_table"]
    lt = result["link_table"]
    stats = {
        "system_name":       result["system_name"],
        "total_nodes":       len(nt),
        "total_links":       len(lt),
        "orphan_pipes":      (lt["Topology_Status"] == "Orphan_Pipe").sum(),
        "isolated_nodes":    nt["Is_Isolated"].sum(),
        "nodes_size_change": nt.get("Has_Size_Change", pd.Series(False)).sum(),
    }
    vd = nt[nt["Depth_Data_Quality"] == "Complete"]["Manhole_Depth"]
    for k, fn in [("mean_depth", "mean"), ("median_depth", "median"),
                  ("max_depth", "max"), ("min_depth", "min")]:
        stats[k] = getattr(vd, fn)() if not vd.empty else np.nan
    stats["max_connections"] = nt["Total_Degree"].max()
    stats["avg_connections"] = nt["Total_Degree"].mean()
    if "Pipe_Slope" in lt.columns:
        stats["adverse_slopes"] = (lt["Pipe_Slope"] < 0).sum()
        stats["avg_slope"]      = lt["Pipe_Slope"].mean()
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_report(results):
    lines = ["=" * 80,
             "COMPARATIVE DRAINAGE NETWORK QUALITY CONTROL REPORT",
             "=" * 80, ""]

    for system_name, result in results.items():
        lines += [f"\n{'='*80}", system_name.upper(), f"{'='*80}"]
        if not result["success"]:
            lines += [f"❌ FAILED: {result['error']}", ""]
            continue

        nt = result["node_table"]
        lt = result["link_table"]
        lines += [
            f"\n📊 NETWORK SIZE",
            f"   Nodes: {len(nt):,}  |  Links: {len(lt):,}",
            f"\n🔗 TOPOLOGY",
            f"   Orphan pipes:    {(lt['Topology_Status'] == 'Orphan_Pipe').sum():,}",
            f"   Isolated nodes:  {nt['Is_Isolated'].sum():,}",
            f"\n📏 DEPTH QUALITY",
        ]
        for q, n in nt["Depth_Data_Quality"].value_counts().items():
            lines.append(f"   {q}: {n:,}")
        vd = nt[nt["Depth_Data_Quality"] == "Complete"]["Manhole_Depth"]
        if not vd.empty:
            lines += [
                f"\n📐 DEPTH STATISTICS",
                f"   Mean: {vd.mean():.2f} m  |  Max: {vd.max():.2f} m  |  Min: {vd.min():.2f} m",
            ]
        lines += [
            f"\n🔀 CONNECTIVITY",
            f"   Max connections: {nt['Total_Degree'].max()}",
            f"   Avg connections: {nt['Total_Degree'].mean():.2f}",
        ]
        if "Pipe_Slope" in lt.columns:
            lines.append(f"\n⚠️  Adverse slopes: {(lt['Pipe_Slope'] < 0).sum():,}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Paths.ensure_output_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting network topology analysis for all drainage systems...")
    results = {}
    for system_name, files in SYSTEMS.items():
        results[system_name] = process_system(
            system_name, files["manhole"], files["pipe"])

    # Export to Excel
    successful = {k: v for k, v in results.items() if v["success"]}
    if successful:
        print(f"\nExporting to {output_excel}")
        with pd.ExcelWriter(str(output_excel), engine='openpyxl') as writer:
            for system_name, result in successful.items():
                result["node_table"].to_excel(
                    writer, sheet_name=f"{system_name}_Nodes", index=False)
                result["link_table"].to_excel(
                    writer, sheet_name=f"{system_name}_Links", index=False)
        print(f"✅ Excel saved: {output_excel}")
    else:
        print("❌ No systems processed successfully")

    # Generate and save report
    report = build_report(results)
    print("\n" + report)
    output_report.write_text(report, encoding='utf-8')
    print(f"\n✅ Report saved: {output_report}")


if __name__ == "__main__":
    main()
