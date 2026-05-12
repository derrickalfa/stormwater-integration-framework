"""
06_comprehensive_integration.py
=================================
Merges all data sources into one ML-ready dataset.

Both integration passes are combined in this single script:
  Pass 1 — Base integration (climate, environmental, network, pipe properties, CCTV)
  Pass 2 — Coverage enhancement for manholes that had missing station assignments
            (network propagation + distance-based fallback)

Data sources integrated:
  1.  Base hydraulic features + maintenance (from script 05)
  2.  Manhole coordinates (from Stormwater_Manhole.xlsx)
  3.  Daily climate (temperature, rainfall, humidity, windspeed, dewpoint)
  4.  Monthly climate (pressure, wind direction, wind speed)
  5.  Environmental covariates (elevation, slope, population, buildings)
  6.  Main water network (corrosion class, land use, pipe type)
  7.  Traffic (AADT by year)
  8.  Geology (soil / rock type)
  9.  Reservoir distance
  10. Pipe properties (upstream / downstream separately)
  11. CCTV inspection grades

Author: Research Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

warnings.filterwarnings("ignore")

try:
    import xlrd
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xlrd"])
    import xlrd


# ─────────────────────────────────────────────────────────────────────────────
# STATION NAME MAPPING
# ─────────────────────────────────────────────────────────────────────────────

STATION_NAMES = {
    "CCH": "Cheung Chau", "CWB": "Clear Water Bay", "HKA": "Hong Kong International Airport",
    "HKO": "Hong Kong Observatory", "HKP": "Hong Kong Park", "HKS": "Wong Chuk Hang",
    "HPV": "Happy Valley", "JKB": "Tseung Kwan O", "KLT": "Kowloon City",
    "KP":  "King's Park", "KTG": "Kwun Tong", "LFS": "Lau Fau Shan",
    "NGP": "Ngong Ping", "PEN": "Peng Chau", "PLC": "Tai Mei Tuk",
    "SE1": "Kai Tak Runway Park", "SEK": "Shek Kong", "SHA": "Sha Tin",
    "SKG": "Sai Kung", "SKW": "Shau Kei Wan", "SSH": "Sheung Shui",
    "SSP": "Sham Shui Po", "STY": "Stanley", "TC":  "Tate's Cairn",
    "TKL": "Ta Kwu Ling", "TMS": "Tai Mo Shan", "TPO": "Tai Po (Conservation Studies Centre)",
    "TU1": "Tuen Mun Children and Juvenile Home", "TW": "Tsuen Wan Shing Mun Valley",
    "TWN": "Tsuen Wan", "TY1": "New Tsing Yi Station", "TYW": "Pak Tam Chung (Tsak Yue Wu)",
    "VP1": "The Peak", "WLP": "Wetland Park", "WTS": "Wong Tai Sin",
    "YCT": "Tai Po (Yuan Chau Tsai Park)", "YLP": "Yuen Long Park",
    "WGL": "Waglan Island", "BHD": "Bluff Head", "CP1": "Central Pier",
    "SE":  "Kai Tak", "GI":  "Green Island", "LAM": "Lamma Island",
    "NP":  "North Point", "SC":  "Sha Chau",
}
STATION_ABBREV = {v: k for k, v in STATION_NAMES.items()}

CHINESE_TO_ENGLISH = {
    "沙洲": "Sha Chau", "中环码头": "Central Pier",
    "香港国际机场": "Hong Kong International Airport", "长洲": "Cheung Chau",
    "青洲": "Green Island", "启德": "Kai Tak", "京士柏": "King's Park",
    "南丫岛": "Lamma Island", "流浮山": "Lau Fau Shan", "昂坪": "Ngong Ping",
    "北角": "North Point", "坪洲": "Peng Chau", "西贡": "Sai Kung",
    "沙田": "Sha Tin", "石岗": "Shek Kong", "将军澳": "Tseung Kwan O",
    "屯门政府合署": "Tuen Mun Government Offices", "湿地公园": "Wetland Park",
    "黄竹坑": "Wong Chuk Hang",
}


def _map_station(name):
    if pd.isna(name): return None
    name = str(name).strip()
    if name in STATION_ABBREV: return STATION_ABBREV[name]
    eng = CHINESE_TO_ENGLISH.get(name)
    if eng: return STATION_ABBREV.get(eng)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _read_xls(path, sheet=0):
    try:    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    except: return pd.read_excel(path, sheet_name=sheet, engine="xlrd")


def _load_conn(path, station_col=None, sep=None):
    """Load a station-connection text file, auto-detecting delimiter."""
    for s in (sep, ",", "\t"):
        if s is None: continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", sep=s, on_bad_lines="skip")
            if len(df.columns) > 1: break
        except Exception:
            continue
    # standardise manhole ID
    for c in ["FEAT_NUM", "Manhole_ID", "FID_1"]:
        if c in df.columns:
            df["Manhole_ID"] = df[c].astype(str).str.strip().str.upper()
            break
    # standardise station
    if station_col and station_col in df.columns:
        df["Station_Abbr"] = df[station_col].astype(str).str.strip()
    elif "WeatherSta" in df.columns:
        df["Station_Abbr"] = df["WeatherSta"].apply(_map_station)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLIMATE LOOKUP BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_temp_lookup(csv_path) -> dict:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for c in ("Year", "Month", "Day"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Year", "Month", "Day"])
    df["Date"] = pd.to_datetime(
        df["Year"].astype(int).astype(str) + "-" +
        df["Month"].astype(int).astype(str).str.zfill(2) + "-" +
        df["Day"].astype(int).astype(str).str.zfill(2), format="%Y-%m-%d", errors="coerce",
    )
    df = df.dropna(subset=["Date"])
    lookup = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("Station")):
            lookup[(row["Station"], row["Date"])] = {
                "max":  pd.to_numeric(row.get("Max_Temp"),  errors="coerce"),
                "mean": pd.to_numeric(row.get("Mean_Temp"), errors="coerce"),
                "min":  pd.to_numeric(row.get("Min_Temp"),  errors="coerce"),
            }
    return lookup


def _build_scalar_lookup(csv_path, value_col) -> dict:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for c in ("Year", "Month", "Day"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Year", "Month", "Day"])
    df["Date"] = pd.to_datetime(
        df["Year"].astype(int).astype(str) + "-" +
        df["Month"].astype(int).astype(str).str.zfill(2) + "-" +
        df["Day"].astype(int).astype(str).str.zfill(2), format="%Y-%m-%d", errors="coerce",
    )
    df = df.dropna(subset=["Date"])
    return {(row["Station"], row["Date"]): pd.to_numeric(row.get(value_col), errors="coerce")
            for _, row in df.iterrows() if pd.notna(row.get("Station"))}


def _build_rain_lookup(csv_path) -> dict:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date_"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"])
    return {(row["Raingauge_No"], row["Date"]): pd.to_numeric(row["Rainfall"], errors="coerce")
            for _, row in df.iterrows() if pd.notna(row.get("Raingauge_No"))}


def _rolling(lookup, station, end_date, n_days, key="mean"):
    vals = []
    for i in range(n_days):
        d = end_date - timedelta(days=i)
        v = lookup.get((station, d))
        if v is None: continue
        val = v[key] if isinstance(v, dict) else v
        if pd.notna(val): vals.append(val)
    if not vals: return None
    return np.mean(vals) if key == "mean" else np.sum(vals)


# ─────────────────────────────────────────────────────────────────────────────
# STATION PROPAGATION THROUGH PIPE NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def _propagate_stations(seed_dict: dict, network: dict) -> dict:
    """BFS propagation of station assignments through connected manholes."""
    result = dict(seed_dict)
    visited = set()
    queue   = list(result.keys())
    while queue:
        cur = queue.pop(0)
        if cur in visited: continue
        visited.add(cur)
        station = result[cur]
        for nbr in network.get(cur, []):
            if nbr not in result:
                result[nbr] = station
                queue.append(nbr)
    return result


def _distance_assignment(df: pd.DataFrame, seed_dict: dict,
                           x_col="X_Coord", y_col="Y_Coord") -> dict:
    """Nearest-neighbour fallback for manholes not reached by propagation."""
    result = dict(seed_dict)
    if x_col not in df.columns or y_col not in df.columns:
        return result

    mh_unique = df[["Manhole_ID", x_col, y_col]].drop_duplicates("Manhole_ID").dropna()
    with_station    = mh_unique[mh_unique["Manhole_ID"].isin(seed_dict)].copy()
    without_station = mh_unique[~mh_unique["Manhole_ID"].isin(seed_dict)].copy()

    if with_station.empty or without_station.empty:
        return result

    ws_arr  = with_station[[x_col, y_col]].values
    wo_iter = without_station[["Manhole_ID", x_col, y_col]].values

    for mh_id, x, y in wo_iter:
        dists = np.sqrt((ws_arr[:, 0] - x) ** 2 + (ws_arr[:, 1] - y) ** 2)
        nearest_mh = with_station.iloc[np.argmin(dists)]["Manhole_ID"]
        result[mh_id] = seed_dict[nearest_mh]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Paths.ensure_output_dirs()

    # ── Load base data ────────────────────────────────────────────────────────
    _print("STEP 1: Loading base hydraulic features")
    df = pd.read_csv(Paths.HYDRAULIC_FEATURES, low_memory=False)
    _print(f"  {len(df):,} records")

    if Params.SYSTEM != "All" and "System" in df.columns:
        df = df[df["System"] == Params.SYSTEM].copy()
        _print(f"  Filtered to {Params.SYSTEM}: {len(df):,} records")

    df["Manhole_ID"]       = df["Manhole_ID"].astype(str).str.strip().str.upper()
    df["Inspection_Date"]  = pd.to_datetime(df["Inspection_Date"], errors="coerce")
    df = df[df["Inspection_Date"].notna()].copy()
    df["Inspection_Year"]  = df["Inspection_Date"].dt.year
    df["Inspection_Month"] = df["Inspection_Date"].dt.month
    start_n = len(df)

    # ── Attach coordinates ────────────────────────────────────────────────────
    _print("STEP 2: Attaching manhole coordinates")
    try:
        coord_df = pd.read_excel(Paths.STORMWATER_MANHOLE, engine="openpyxl")
        if "Coordinate" in coord_df.columns:
            spl = coord_df["Coordinate"].astype(str).str.split(",", expand=True)
            coord_df["X_Coord"] = pd.to_numeric(spl[0], errors="coerce")
            coord_df["Y_Coord"] = pd.to_numeric(spl[1], errors="coerce")
        coord_df["Feature_Number"] = coord_df["Feature_Number"].astype(str).str.strip().str.upper()
        lookup = coord_df[["Feature_Number", "X_Coord", "Y_Coord"]].dropna().drop_duplicates("Feature_Number")
        df = df.merge(lookup.rename(columns={"Feature_Number": "Manhole_ID"}),
                      on="Manhole_ID", how="left")
        covered = df[["X_Coord", "Y_Coord"]].notna().all(axis=1).sum()
        _print(f"  Coordinates: {covered:,} / {len(df):,} records ({covered/len(df)*100:.1f}%)")
    except Exception as e:
        _print(f"  ⚠  Could not load coordinates: {e}")

    # ── Build pipe network adjacency (for station propagation) ─────────────
    _print("STEP 3: Building pipe network adjacency")
    try:
        pipes = pd.read_excel(Paths.STORMWATER_PIPE, engine="openpyxl")
        network_adj = {}
        for _, row in pipes.iterrows():
            u = str(row.get("Upstream_Node",   "")).strip().upper()
            d = str(row.get("Downstream_Node", "")).strip().upper()
            if u and d and u != "NAN" and d != "NAN":
                network_adj.setdefault(u, set()).add(d)
                network_adj.setdefault(d, set()).add(u)
        _print(f"  {len(network_adj):,} manholes in adjacency graph")
    except Exception as e:
        _print(f"  ⚠  Could not build adjacency: {e}")
        network_adj = {}

    # ── Daily climate — station assignments ──────────────────────────────────
    _print("STEP 4: Building station assignments (Pass 1 + Pass 2 combined)")

    def _load_seeds(conn_file, station_col=None):
        try:
            df_c = _load_conn(conn_file, station_col)
            return {row["Manhole_ID"]: row["Station_Abbr"]
                    for _, row in df_c.iterrows()
                    if pd.notna(row.get("Station_Abbr")) and pd.notna(row.get("Manhole_ID"))}
        except Exception as e:
            _print(f"  ⚠  {os.path.basename(str(conn_file))}: {e}")
            return {}

    def _combined_seeds(f1, f2, station_col=None):
        s1 = _load_seeds(f1, station_col)
        s2 = _load_seeds(f2, station_col)
        s1.update(s2)   # pass-2 additions
        return s1

    seeds = {
        "temperature": _combined_seeds(Paths.CONN_TEMPERATURE, Paths.CONN2_TEMPERATURE),
        "humidity":    _combined_seeds(Paths.CONN_HUMIDITY,    Paths.CONN2_HUMIDITY),
        "windspeed":   _combined_seeds(Paths.CONN_WINDSPEED,   Paths.CONN2_WINDSPEED),
        "dewpoint":    _combined_seeds(Paths.CONN_DEWPOINT,    Paths.CONN2_DEWPOINT),
        "rainfall":    _combined_seeds(Paths.CONN_RAINFALL,    Paths.CONN2_RAINFALL,
                                       station_col="Raingauge_"),
    }

    # Propagate through pipe network, then distance fallback
    station_maps = {}
    for var, seed_dict in seeds.items():
        propagated = _propagate_stations(seed_dict, network_adj)
        final      = _distance_assignment(df, propagated)
        station_maps[var] = final
        improvement = len(final) - len(seed_dict)
        _print(f"  {var}: {len(seed_dict):,} seeds → {len(final):,} after propagation (+{improvement:,})")

    # Attach station columns to df
    col_map = {
        "temperature": "Temp_Station",
        "humidity":    "Humid_Station",
        "windspeed":   "Wind_Station",
        "dewpoint":    "Dew_Station",
        "rainfall":    "Rain_Station",
    }
    for var, col in col_map.items():
        df[col] = df["Manhole_ID"].map(station_maps.get(var, {}))

    # ── Load climate lookups ──────────────────────────────────────────────────
    _print("STEP 5: Building climate data lookups")
    t_lookup = r_lookup = h_lookup = w_lookup = d_lookup = {}
    try:
        _print("  Temperature..."); t_lookup = _build_temp_lookup(Paths.CLIMATE_TEMP_COMBINED)
        _print(f"    {len(t_lookup):,} entries")
    except Exception as e: _print(f"  ⚠  Temperature: {e}")
    try:
        _print("  Rainfall...");    r_lookup = _build_rain_lookup(Paths.RAIN_DATA)
        _print(f"    {len(r_lookup):,} entries")
    except Exception as e: _print(f"  ⚠  Rainfall: {e}")
    try:
        _print("  Humidity...");    h_lookup = _build_scalar_lookup(Paths.CLIMATE_HUMIDITY_COMBINED, "Value")
        _print(f"    {len(h_lookup):,} entries")
    except Exception as e: _print(f"  ⚠  Humidity: {e}")
    try:
        _print("  Windspeed...");   w_lookup = _build_scalar_lookup(Paths.CLIMATE_WIND_COMBINED, "Value")
        _print(f"    {len(w_lookup):,} entries")
    except Exception as e: _print(f"  ⚠  Windspeed: {e}")
    try:
        _print("  Dewpoint...");    d_lookup = _build_scalar_lookup(Paths.CLIMATE_DEW_COMBINED, "Value")
        _print(f"    {len(d_lookup):,} entries")
    except Exception as e: _print(f"  ⚠  Dewpoint: {e}")

    # ── Link daily climate ────────────────────────────────────────────────────
    _print("STEP 6: Linking daily climate data")
    for col in ["Temp_Max_Day", "Temp_Mean_Day", "Temp_Min_Day", "Temp_Mean_7d_Avg",
                "Rain_Day", "Rain_7d_Sum", "Humidity_Day", "Windspeed_Day", "Dewpoint_Day"]:
        df[col] = np.nan

    n = len(df)
    N = Params.ANTECEDENT_DAYS
    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 50_000 == 0: _print(f"  {i:,}/{n:,}")
        dt = row["Inspection_Date"]
        if pd.isna(dt): continue

        ts = row.get("Temp_Station")
        if pd.notna(ts):
            k = (ts, dt)
            if k in t_lookup:
                df.at[idx, "Temp_Max_Day"]  = t_lookup[k]["max"]
                df.at[idx, "Temp_Mean_Day"] = t_lookup[k]["mean"]
                df.at[idx, "Temp_Min_Day"]  = t_lookup[k]["min"]
            df.at[idx, "Temp_Mean_7d_Avg"] = _rolling(t_lookup, ts, dt, N, "mean")

        rs = row.get("Rain_Station")
        if pd.notna(rs):
            df.at[idx, "Rain_Day"]     = r_lookup.get((rs, dt))
            df.at[idx, "Rain_7d_Sum"]  = _rolling(r_lookup, rs, dt, N, "sum")

        hs = row.get("Humid_Station")
        if pd.notna(hs): df.at[idx, "Humidity_Day"]  = h_lookup.get((hs, dt))

        ws = row.get("Wind_Station")
        if pd.notna(ws): df.at[idx, "Windspeed_Day"] = w_lookup.get((ws, dt))

        ds = row.get("Dew_Station")
        if pd.notna(ds): df.at[idx, "Dewpoint_Day"]  = d_lookup.get((ds, dt))

    # Drop temporary station columns
    df.drop(columns=[c for c in df.columns if c.endswith("_Station")], inplace=True, errors="ignore")

    # ── Monthly climate ───────────────────────────────────────────────────────
    _print("STEP 7: Merging monthly climate data")
    try:
        mc = _read_xls(Paths.MONTHLY_CLIMATE)
        if "station" in mc.columns: mc["Station_Abbr"] = mc["station"].apply(_map_station)
        mc["Year"]  = pd.to_numeric(mc["Year"],  errors="coerce").astype("Int64")
        mc["Month"] = pd.to_numeric(mc["Month"], errors="coerce").astype("Int64")
        for col in ["Mean Pressure (hPa)", "Prevailing Wind Direction (degrees)", "Mean Wind Speed (km/h)"]:
            if col in mc.columns:
                mc[col] = mc[col].replace(["***", "---", "NA", "N/A", ""], np.nan)
                mc[col] = pd.to_numeric(mc[col], errors="coerce")
        mc = mc.drop_duplicates(subset=["Station_Abbr", "Year", "Month"], keep="first")

        # get station for each manhole (reuse temperature connection)
        t_conn = pd.read_csv(Paths.CONN_TEMPERATURE, encoding="utf-8-sig")
        t_conn["FEAT_NUM"]     = t_conn["FEAT_NUM"].astype(str).str.strip().str.upper()
        t_conn["Station_Abbr"] = t_conn["WeatherSta"].apply(_map_station)
        df = df.merge(t_conn[["FEAT_NUM", "Station_Abbr"]].rename(columns={"FEAT_NUM": "Manhole_ID"}),
                      on="Manhole_ID", how="left")
        df = df.merge(
            mc[["Station_Abbr", "Year", "Month",
                "Mean Pressure (hPa)", "Prevailing Wind Direction (degrees)", "Mean Wind Speed (km/h)"]],
            left_on=["Station_Abbr", "Inspection_Year", "Inspection_Month"],
            right_on=["Station_Abbr", "Year", "Month"],
            how="left",
        ).rename(columns={
            "Mean Pressure (hPa)":                 "Mean_Pressure_Monthly",
            "Prevailing Wind Direction (degrees)":  "Wind_Direction_Monthly",
            "Mean Wind Speed (km/h)":               "Wind_Speed_Monthly",
        }).drop(columns=["Year", "Month", "Station_Abbr"], errors="ignore")
        _print(f"  Monthly climate records: {len(df):,}")
    except Exception as e:
        _print(f"  ⚠  Monthly climate: {e}")

    # ── Environmental covariates ──────────────────────────────────────────────
    _print("STEP 8: Merging environmental covariates")
    ENV_COLS = ["FEAT_NUM", "Elev", "SlopeDeg", "AspectDeg", "Curvature", "FlowAcc",
                "CatchArea_m2", "TWI", "SPI", "STI", "TPI", "Imperv_Point",
                "PopTotal", "PopDensity_m2", "PopDensity_ha",
                "BldgCnt_100m", "BldgArea_100m2", "BldgCntDen_100m", "BldgAreaFrac_100m",
                "X_Coord", "Y_Coord"]

    def _merge_env(df, env_data, id_col="FEAT_NUM"):
        env_data[id_col] = env_data[id_col].astype(str).str.strip().str.upper()
        env_data = env_data.drop_duplicates(subset=[id_col], keep="first")
        available = [c for c in ENV_COLS if c in env_data.columns]
        merged = df.merge(env_data[available].rename(columns={id_col: "Manhole_ID"}),
                          on="Manhole_ID", how="left", suffixes=("", "_new"))
        # fill missing columns from new data
        for col in [c for c in available if c != id_col]:
            new_col = col + "_new"
            if new_col in merged.columns:
                if col not in merged.columns:
                    merged[col] = merged[new_col]
                else:
                    merged[col] = merged[col].fillna(merged[new_col])
                merged.drop(columns=[new_col], inplace=True)
        return merged

    for env_path in [Paths.ENVIRONMENTAL_FILE, Paths.ENV_REMAINING_1, Paths.ENV_REMAINING_2]:
        try:
            if str(env_path).endswith(".xls"):
                env = pd.read_excel(env_path, sheet_name="manholes_with_covariates_simple",
                                    engine="xlrd")
            else:
                env = pd.read_csv(env_path)
            id_col = "FEAT_NUM" if "FEAT_NUM" in env.columns else "Manhole_ID"
            df = _merge_env(df, env, id_col)
            _print(f"  Merged {os.path.basename(str(env_path))}")
        except Exception as e:
            _print(f"  ⚠  {os.path.basename(str(env_path))}: {e}")

    # ── Water network, Traffic, Geology, Reservoir ────────────────────────────
    _print("STEP 9: Merging network / traffic / geology / reservoir")

    def _simple_merge(df, path, sheet, cols, id_col="FEAT_NUM"):
        try:
            d = _read_xls(path, sheet)
            d[id_col] = d[id_col].astype(str).str.strip().str.upper()
            avail = [c for c in cols if c in d.columns]
            return df.merge(d[[id_col] + avail].rename(columns={id_col: "Manhole_ID"}),
                            on="Manhole_ID", how="left")
        except Exception as e:
            _print(f"  ⚠  {sheet}: {e}")
            return df

    df = _simple_merge(df, Paths.ENVIRONMENTAL_FILE, "connect_to_mainwaternetwrok",
                       ["LPR_Corros", "TYPE", "LANDUSE"])
    df = _simple_merge(df, Paths.ENVIRONMENTAL_FILE, "manholes_with_geology", ["Geol_Type"])
    df = _simple_merge(df, Paths.ENVIRONMENTAL_FILE, "distance_to_reseoviour",
                       ["Distance"])
    if "Distance" in df.columns:
        df.rename(columns={"Distance": "Reservoir_Distance"}, inplace=True)

    # Traffic (time-varying by year)
    try:
        tlink = _read_xls(Paths.ENVIRONMENTAL_FILE, "manholes_with_traffic")
        tdata = _read_xls(Paths.ENVIRONMENTAL_FILE, "traffic_daat")
        tlink["FEAT_NUM"] = tlink["FEAT_NUM"].astype(str).str.strip().str.upper()
        df = df.merge(tlink[["FEAT_NUM", "ATC_STATIO", "Distance"]].rename(
                          columns={"FEAT_NUM": "Manhole_ID", "ATC_STATIO": "Traffic_Station",
                                   "Distance": "Traffic_Station_Distance"}),
                      on="Manhole_ID", how="left")
        df = df.merge(tdata[["STATION", "YEAR", "AADT"]].rename(
                          columns={"STATION": "Traffic_Station", "YEAR": "Traffic_Year"}),
                      left_on=["Traffic_Station", "Inspection_Year"],
                      right_on=["Traffic_Station", "Traffic_Year"],
                      how="left").drop(columns=["Traffic_Year"], errors="ignore")
    except Exception as e:
        _print(f"  ⚠  Traffic: {e}")

    # ── Pipe properties ───────────────────────────────────────────────────────
    _print("STEP 10: Computing pipe properties")
    try:
        net = pd.read_excel(Paths.NETWORK_ANALYSIS, sheet_name="Stormwater_System_Links", engine="openpyxl")
        for col in ["Nominal_Width", "Computed_Length", "Pipe_Slope"]:
            if col in net.columns:
                net[col] = pd.to_numeric(net[col], errors="coerce")
        # cap slope and length
        if "Pipe_Slope" in net.columns:
            net["Pipe_Slope"] = net["Pipe_Slope"].clip(Params.MIN_PIPE_SLOPE, Params.MAX_PIPE_SLOPE)
        if "Computed_Length" in net.columns:
            net.loc[net["Computed_Length"] < Params.MIN_PIPE_LENGTH_M, "Computed_Length"] = np.nan

        for direction in ("Upstream", "Downstream"):
            node_col = f"{direction}_Node"
            agg = (net.groupby(node_col).agg(
                       **{f"{direction}_Pipe_Diameter_Avg": ("Nominal_Width",  "mean"),
                          f"{direction}_Pipe_Length_Avg":   ("Computed_Length","mean"),
                          f"{direction}_Pipe_Slope_Avg":    ("Pipe_Slope",     "mean"),
                          f"Num_{direction}_Pipes":         ("Link_ID",        "count")})
                   .reset_index().rename(columns={node_col: "Manhole_ID"}))
            agg["Manhole_ID"] = agg["Manhole_ID"].astype(str).str.strip().str.upper()
            df = df.merge(agg, on="Manhole_ID", how="left")
    except Exception as e:
        _print(f"  ⚠  Pipe properties: {e}")

    # ── CCTV ─────────────────────────────────────────────────────────────────
    _print("STEP 11: Merging CCTV grades")
    cctv_frames = []
    CCTV_SOURCES = [
        (Paths.CCTV_FILE_1, "THEDATA_ExcelToTable", "pipe", "I C G", "S C G", "S P G", "from", "to"),
        (Paths.CCTV_FILE_2, None, "manhole", "ICG", "SCG", "SPG", "Start_Node", None),
        (Paths.CCTV_FILE_3, None, "pipe", "ICG_Grade", "SCG_Peak", "SPG", "Start_Node", "Finish_Node"),
    ]
    for (fpath, sheet, ftype, icg, scg, spg, from_c, to_c) in CCTV_SOURCES:
        try:
            d = _read_xls(fpath, sheet) if sheet else _read_xls(fpath, 0)
            for col in [icg, scg, spg]:
                if col in d.columns: d[col] = pd.to_numeric(d[col], errors="coerce")
            if ftype == "pipe" and to_c:
                fr = d[[from_c, icg, scg, spg]].rename(columns={from_c:"Manhole_ID",icg:"ICG",scg:"SCG",spg:"SPG"})
                to = d[[to_c,  icg, scg, spg]].rename(columns={to_c:  "Manhole_ID",icg:"ICG",scg:"SCG",spg:"SPG"})
                chunk = pd.concat([fr, to])
            else:
                chunk = d[[from_c, icg, scg, spg]].rename(columns={from_c:"Manhole_ID",icg:"ICG",scg:"SCG",spg:"SPG"})
            chunk["Manhole_ID"] = chunk["Manhole_ID"].astype(str).str.strip().str.upper()
            chunk = chunk[chunk["Manhole_ID"].str.startswith("S", na=False)]
            cctv_frames.append(chunk.drop_duplicates())
        except Exception as e:
            _print(f"  ⚠  CCTV {os.path.basename(str(fpath))}: {e}")

    if cctv_frames:
        cctv_all = pd.concat(cctv_frames, ignore_index=True).drop_duplicates()
        cctv_agg = cctv_all.groupby("Manhole_ID").agg(
            I_C_G_Avg=("ICG","mean"), S_C_G_Avg=("SCG","mean"),
            S_P_G_Avg=("SPG","mean"), CCTV_Survey_Count=("ICG","count"),
        ).reset_index()
        df = df.merge(cctv_agg, on="Manhole_ID", how="left")
        pct = df["I_C_G_Avg"].notna().mean() * 100
        _print(f"  CCTV coverage: {pct:.1f}%")

    # ── Validate record count ─────────────────────────────────────────────────
    assert len(df) == start_n, f"Record count changed: {start_n} → {len(df)}"
    _print(f"✅ Record count preserved: {len(df):,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(Paths.INTEGRATED_V2, index=False)
    _print(f"\nSaved → {Paths.INTEGRATED_V2}  ({len(df):,} rows × {df.shape[1]} cols)")

    # Coverage summary
    key_cols = ["Temp_Mean_Day", "Rain_Day", "Humidity_Day", "Elev", "AADT",
                "Geol_Type", "Upstream_Pipe_Diameter_Avg"]
    _print("\nCoverage summary:")
    for col in key_cols:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            _print(f"  {col:<35}: {pct:5.1f}%")


if __name__ == "__main__":
    main()
