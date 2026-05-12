"""
04_climate_standardisation.py
===============================
Combines per-station daily climate CSV files into single wide-format tables,
one table per climate variable.

Run this script once. It handles temperature (Max / Mean / Min) in wide format.
The same `combine_climate_files_wide()` function is used internally for
humidity, wind speed, and dewpoint — the `main()` function calls it for
each variable using the correct directory paths from config.py.

Input:  Per-station CSV files in climate subdirectories (one file per station)
        File naming convention:  daily_<STATION>_<VARIABLE>_ALL.csv

Output: combined_temperature_wide_format.csv
        combined_relative_humidity.csv
        combined_wind_speed.csv
        combined_dew_data.csv

Author: Research Team
"""

import pandas as pd
from pathlib import Path
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params


# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURE — WIDE FORMAT (Max / Mean / Min as separate columns)
# ─────────────────────────────────────────────────────────────────────────────

def combine_climate_files_wide(input_folders: dict, output_file: str,
                                file_suffix: str = "ALL"):
    """
    Combine temperature data from Max / Mean / Min folders into one wide CSV.

    Parameters
    ----------
    input_folders : dict
        Keys are temperature type labels ('Max', 'Mean', 'Min');
        values are folder paths containing the station CSV files.
    output_file : str
        Full path for the output CSV.
    file_suffix : str
        Only files whose stem ends with this suffix are processed.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dataframes = {}

    print("=" * 70)
    print("CLIMATE STANDARDISATION — TEMPERATURE (WIDE FORMAT)")
    print("=" * 70)

    COLUMN_MAPPING = {
        'Year/年/年':   'Year',
        'Month/月/月':  'Month',
        'Day/日/日':    'Day',
        'Value/數值/数值': 'Value',
        'data Completeness/數據完整性/数据完整性': 'Data_Completeness'
    }

    for temp_type, input_folder in input_folders.items():
        folder_path = Path(input_folder)
        if not folder_path.exists():
            print(f"\n⚠  Folder not found, skipping {temp_type}: {input_folder}")
            continue

        print(f"\n{'─'*70}\nProcessing {temp_type} temperature\n{'─'*70}")
        type_data, n_files = [], 0

        for file_path in sorted(folder_path.iterdir()):
            if not (file_path.is_file() and file_path.stem.endswith(file_suffix)):
                continue
            try:
                parts        = file_path.stem.split('_')
                station_code = parts[1] if len(parts) > 1 else "UNKNOWN"
                print(f"  ✓ {file_path.name}  (station: {station_code})")

                df = pd.read_csv(file_path, encoding='utf-8-sig', skiprows=3)
                df.columns = df.columns.str.strip()
                df = df.rename(columns=COLUMN_MAPPING)
                df.insert(0, 'Station', station_code)
                df = df[['Station', 'Year', 'Month', 'Day', 'Value', 'Data_Completeness']]
                df = df.dropna(how='all')

                # Remove footer rows (non-numeric Year values)
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df = df[df['Year'].notna()]
                df = df[df['Value'].astype(str).str.len() <= 10]

                df['Year']  = df['Year'].astype(int)
                df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
                df['Day']   = pd.to_numeric(df['Day'],   errors='coerce')
                df = df[df['Month'].notna() & df['Day'].notna()]
                df['Month'] = df['Month'].astype(int)
                df['Day']   = df['Day'].astype(int)

                df = df.rename(columns={
                    'Value':            f'{temp_type}_Temp',
                    'Data_Completeness': f'{temp_type}_Data_Completeness'
                })
                type_data.append(df)
                n_files += 1

            except Exception as e:
                print(f"  ✗ Error: {e}")

        print(f"  Files processed: {n_files}")
        if type_data:
            temp_dataframes[temp_type] = pd.concat(type_data, ignore_index=True)

    if not temp_dataframes:
        print("\n✗ No temperature data loaded")
        return

    # Merge all temperature types
    print(f"\n{'='*70}\nMerging temperature types...\n{'='*70}")
    merged = None
    for temp_type, df in temp_dataframes.items():
        merged = df if merged is None else pd.merge(
            merged, df, on=['Station', 'Year', 'Month', 'Day'], how='outer')

    merged = merged.sort_values(['Station', 'Year', 'Month', 'Day']).reset_index(drop=True)

    # Reorder columns
    base = ['Station', 'Year', 'Month', 'Day']
    temp_cols  = sorted([c for c in merged.columns if '_Temp' in c])
    comp_cols  = sorted([c for c in merged.columns if '_Data_Completeness' in c])
    merged = merged[base + temp_cols + comp_cols]

    merged.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved: {output_file}")
    print(f"  Rows: {len(merged):,}  |  Stations: {merged['Station'].nunique()}")
    print(f"  Year range: {merged['Year'].min()}–{merged['Year'].max()}")
    for col in temp_cols:
        n = merged[col].notna().sum()
        print(f"  {col}: {n:,} values ({n/len(merged)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# SCALAR CLIMATE VARIABLES (humidity, wind speed, dewpoint)
# ─────────────────────────────────────────────────────────────────────────────

def combine_climate_files_scalar(input_folder: str, output_file: str,
                                  variable_name: str, file_suffix: str = "ALL"):
    """
    Combine per-station CSV files for a single scalar climate variable.

    Parameters
    ----------
    input_folder  : str  — folder containing per-station CSV files
    output_file   : str  — full output CSV path
    variable_name : str  — label for the Value column (e.g. 'Humidity')
    file_suffix   : str  — only files ending with this suffix are processed
    """
    folder_path = Path(input_folder)
    if not folder_path.exists():
        print(f"⚠  Folder not found: {input_folder}")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CLIMATE STANDARDISATION — {variable_name.upper()}")
    print(f"{'='*70}")

    COLUMN_MAPPING = {
        'Year/年/年':   'Year',
        'Month/月/月':  'Month',
        'Day/日/日':    'Day',
        'Value/數值/数值': 'Value',
        'data Completeness/數據完整性/数据完整性': 'Data_Completeness'
    }

    all_data, n_files = [], 0

    for file_path in sorted(folder_path.iterdir()):
        if not (file_path.is_file() and file_path.stem.endswith(file_suffix)):
            continue
        try:
            parts        = file_path.stem.split('_')
            station_code = parts[1] if len(parts) > 1 else "UNKNOWN"
            print(f"  ✓ {file_path.name}  (station: {station_code})")

            df = pd.read_csv(file_path, encoding='utf-8-sig', skiprows=3)
            df.columns = df.columns.str.strip()
            df = df.rename(columns=COLUMN_MAPPING)
            df.insert(0, 'Station', station_code)
            df = df[['Station', 'Year', 'Month', 'Day', 'Value', 'Data_Completeness']]
            df = df.dropna(how='all')

            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df = df[df['Year'].notna()]
            df['Year']  = df['Year'].astype(int)
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
            df['Day']   = pd.to_numeric(df['Day'],   errors='coerce')
            df = df[df['Month'].notna() & df['Day'].notna()]
            df['Month'] = df['Month'].astype(int)
            df['Day']   = df['Day'].astype(int)
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

            all_data.append(df)
            n_files += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")

    if not all_data:
        print(f"  ✗ No files processed for {variable_name}")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['Station', 'Year', 'Month', 'Day']).reset_index(drop=True)
    combined.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✓ Saved: {output_file}")
    print(f"  Rows: {len(combined):,}  |  Stations: {combined['Station'].nunique()}")
    n_valid = combined['Value'].notna().sum()
    print(f"  Valid values: {n_valid:,} ({n_valid/len(combined)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    Paths.ensure_output_dirs()
    sfx = Params.CLIMATE_FILE_SUFFIX

    # ── Temperature (wide format: Max / Mean / Min) ───────────────────────────
    combine_climate_files_wide(
        input_folders={
            'Max':  str(Paths.CLIMATE_TEMP_DIR),
            'Mean': str(Paths.CLIMATE_TEMP_MEAN_DIR),
            'Min':  str(Paths.CLIMATE_TEMP_MIN_DIR),
        },
        output_file=str(Paths.CLIMATE_TEMP_COMBINED),
        file_suffix=sfx,
    )

    # ── Humidity (scalar) ────────────────────────────────────────────────────
    combine_climate_files_scalar(
        input_folder  = str(Paths.CLIMATE_HUMIDITY_DIR),
        output_file   = str(Paths.CLIMATE_HUMIDITY_COMBINED),
        variable_name = "Relative Humidity",
        file_suffix   = sfx,
    )

    # ── Wind speed (scalar) ──────────────────────────────────────────────────
    combine_climate_files_scalar(
        input_folder  = str(Paths.CLIMATE_WIND_DIR),
        output_file   = str(Paths.CLIMATE_WIND_COMBINED),
        variable_name = "Wind Speed",
        file_suffix   = sfx,
    )

    # ── Dewpoint (scalar) ────────────────────────────────────────────────────
    combine_climate_files_scalar(
        input_folder  = str(Paths.CLIMATE_DEW_DIR),
        output_file   = str(Paths.CLIMATE_DEW_COMBINED),
        variable_name = "Dew Point",
        file_suffix   = sfx,
    )

    print("\n" + "=" * 70)
    print("CLIMATE STANDARDISATION COMPLETE")
    print("=" * 70)
    print(f"  Temperature : {Paths.CLIMATE_TEMP_COMBINED}")
    print(f"  Humidity    : {Paths.CLIMATE_HUMIDITY_COMBINED}")
    print(f"  Wind Speed  : {Paths.CLIMATE_WIND_COMBINED}")
    print(f"  Dew Point   : {Paths.CLIMATE_DEW_COMBINED}")


if __name__ == "__main__":
    main()
