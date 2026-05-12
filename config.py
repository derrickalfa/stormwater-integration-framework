"""
config.py — Central configuration for the Stormwater Integration Framework
===========================================================================
All file paths and global parameters are defined here.
Update INPUT_ROOT and OUTPUT_ROOT to match your environment before running.

Usage:
    from config import Paths, Params
"""

from pathlib import Path

# =============================================================================
# ROOT DIRECTORIES — update these two lines for your machine
# =============================================================================
INPUT_ROOT  = Path(r"C:\Users\Acer\Desktop\PolyU\Research\Data")
OUTPUT_ROOT = Path(r"C:\Users\Acer\Desktop\PolyU\Research\Data\ML_Aggregated_Analysis")


class Paths:
    """All input and output file paths, derived from INPUT_ROOT / OUTPUT_ROOT."""

    # ── Raw inputs ────────────────────────────────────────────────────────────
    RAW_MAINTENANCE        = INPUT_ROOT / "Asset maintenance record.xlsb"
    CLEANED_MAINTENANCE    = INPUT_ROOT / "CLeanedAllSheets.xlsx"
    INSPECTION_XLSB        = INPUT_ROOT / "3) M&M Manhole inspection record.xlsb"
    NETWORK_ANALYSIS       = INPUT_ROOT / "Network_Analysis_All_Systems" / "All_Systems_Network_Analysis Alfa 050125.xlsx"

    # ── Manhole property files ─────────────────────────────────────────────────
    MANHOLE_DIR            = INPUT_ROOT / "Manholes"
    COMBINED_MANHOLE       = MANHOLE_DIR / "Combined_Manhole.xlsx"
    SEWER_MANHOLE          = MANHOLE_DIR / "Sewer_Manhole.xlsx"
    STORMWATER_MANHOLE     = MANHOLE_DIR / "Stormwater_Manhole.xlsx"
    STORMWATER_PIPE        = MANHOLE_DIR / "Stormwater_Pipe.xlsx"
    PIPE_SHAPEFILE_CSV     = MANHOLE_DIR / "stromwaterpipesinfrofromshapefile.csv"
    PIPE_AGE               = MANHOLE_DIR / "storm_age_12022026.xlsx"

    # ── Climate data ──────────────────────────────────────────────────────────
    CLIMATE_DIR            = INPUT_ROOT / "Climate data to strom manholes _01022026_EA_to ALFAA" / "Climate data to strom manholes _01022026_EA_to ALFAA"
    CLIMATE_TEMP_DIR       = CLIMATE_DIR / "Temperature"
    CLIMATE_TEMP_MEAN_DIR  = CLIMATE_DIR / "Temperature Daily Mean"
    CLIMATE_TEMP_MIN_DIR   = CLIMATE_DIR / "Temperature Daily Minimum"
    CLIMATE_HUMIDITY_DIR   = CLIMATE_DIR / "Relative Humidity"
    CLIMATE_WIND_DIR       = CLIMATE_DIR / "Wind Speed"
    CLIMATE_DEW_DIR        = CLIMATE_DIR / "Dew Point"
    RAIN_DATA              = CLIMATE_DIR / "rainguage data.csv"
    MONTHLY_CLIMATE        = INPUT_ROOT / "New Data 50226" / "climate full record.xlsx"

    # Station-to-manhole connection files (first-pass)
    CONN_TEMPERATURE       = CLIMATE_DIR / "connection_to_tepreature_.txt"
    CONN_RAINFALL          = CLIMATE_DIR / "connection_to_rainguages.txt"
    CONN_HUMIDITY          = CLIMATE_DIR / "connection_to_humididty_.txt"
    CONN_WINDSPEED         = CLIMATE_DIR / "connection_to_wiindspeed_.txt"
    CONN_DEWPOINT          = CLIMATE_DIR / "connection_to_dewpoints_.txt"

    # Station-to-manhole connection files (second-pass / remaining manholes)
    CONN2_DIR              = INPUT_ROOT / "Missing manhole data" / "reminaing manholes" / "reminaing manholes"
    CONN2_TEMPERATURE      = CONN2_DIR / "connection_to_temprature.txt"
    CONN2_HUMIDITY         = CONN2_DIR / "connection_to_humidty.txt"
    CONN2_WINDSPEED        = CONN2_DIR / "connection_to_windspeed.txt"
    CONN2_DEWPOINT         = CONN2_DIR / "connection_to_dew.txt"
    CONN2_RAINFALL         = CONN2_DIR / "connection_to_rainguages.txt"
    CONN2_PRESSURE         = CONN2_DIR / "connection_to_pressure.txt"
    CONN2_TRAFFIC          = CONN2_DIR / "connection_to_traffic_stations.txt"
    CONN2_WATER_NETWORK    = CONN2_DIR / "connection_to_water_netwrok.txt"
    CONN2_GEOLOGY          = CONN2_DIR / "connection_to_geolgy.txt"

    # ── Environmental & spatial covariates (GIS outputs) ──────────────────────
    ENVIRONMENTAL_FILE     = INPUT_ROOT / "New Data 50226" / "stromwatermnaholes conncetions with data.xls"
    ENV_REMAINING_1        = CONN2_DIR / "remainaingstromwatermanholes connecetion to paramters.csv"
    ENV_REMAINING_2        = CONN2_DIR / "manholes_with_covariates_simple2.csv"

    # ── Maintenance analysis results ──────────────────────────────────────────
    MAINTENANCE_RESULTS    = INPUT_ROOT.parent / "IdeaProjects" / "Sewer" / "sewer_analysis_results"
    MAINTENANCE_COMBINED   = MAINTENANCE_RESULTS / "combined_statistics.xlsx"
    MAINTENANCE_BY_SHEET   = MAINTENANCE_RESULTS / "by_sheet"

    # ── CCTV inspection files ─────────────────────────────────────────────────
    CCTV_FILE_1            = INPUT_ROOT / "THEDATA_ExcelToTable_Original_OPERATIONAL_EA.xlsx"
    CCTV_FILE_2            = INPUT_ROOT / "HKI soultions" / "Batch" / "Form_A_Batch_files_FINAL.xlsx"
    CCTV_FILE_3            = INPUT_ROOT / "HKI soultions" / "Batch" / "Inspection_sheets_Batch_files_FINAL.xlsx"

    # ── Intermediate outputs ───────────────────────────────────────────────────
    MERGED_PROPERTIES      = OUTPUT_ROOT / "merged_manhole_properties.csv"
    CLIMATE_TEMP_COMBINED  = CLIMATE_DIR / "combined_temperature_wide_format.csv"
    CLIMATE_HUMIDITY_COMBINED = CLIMATE_DIR / "combined_relative_humidity.csv"
    CLIMATE_WIND_COMBINED  = CLIMATE_DIR / "combined_wind_speed.csv"
    CLIMATE_DEW_COMBINED   = CLIMATE_DIR / "combined_dew_data.csv"
    HYDRAULIC_FEATURES     = OUTPUT_ROOT / "manhole_hydraulic_features_complete.csv"
    INTEGRATED_V1          = OUTPUT_ROOT / "complete_integrated_dataset_stormwater.csv"
    INTEGRATED_V2          = OUTPUT_ROOT / "complete_integrated_dataset_stormwater_v2.csv"
    DATA_CLEANED           = OUTPUT_ROOT / "manhole_data_cleaned_for_ml.csv"
    MAINTENANCE_FIXED      = OUTPUT_ROOT / "manhole_data_with_maintenance.csv"

    # ── Final outputs ──────────────────────────────────────────────────────────
    FINAL_CLASSIFICATION   = OUTPUT_ROOT / "final_dataset_classification.csv"
    FINAL_REGRESSION       = OUTPUT_ROOT / "final_dataset_regression.csv"
    CORRELATION_MATRIX     = OUTPUT_ROOT / "correlation_matrix.csv"
    QUALITY_REPORT         = OUTPUT_ROOT / "data_quality_report.xlsx"

    @classmethod
    def ensure_output_dirs(cls):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class Params:
    """Global algorithm parameters."""

    # System selection
    SYSTEM                 = "Stormwater"   # "Stormwater" | "Sewer" | "Combined" | "All"

    # Manhole property merging
    DUPLICATE_STRATEGY     = "keep_latest"  # "keep_latest" | "keep_first" | "keep_last"

    # Physical validation thresholds
    MIN_MANHOLE_DEPTH_MM   = 100            # mm — below this is physically implausible
    MAX_PIPE_DIAMETER_MM   = 5000           # mm — cap for outlier detection
    MIN_PIPE_LENGTH_M      = 0.5            # m  — below this is likely a data error
    MAX_PIPE_SLOPE         = 1.0            # dimensionless — 45 degrees
    MIN_PIPE_SLOPE         = -0.5           # mild adverse gradient

    # Surcharge thresholds
    SURCHARGE_THRESHOLD    = 1.0            # ratio > 1.0 → surcharged
    SURCHARGE_NEAR         = 0.75           # 0.75–1.0 → near surcharge
    SURCHARGE_MODERATE     = 1.5            # 1.0–1.5 → moderate surcharge
    SILT_BLOCKAGE_HIGH     = 0.5            # ratio > 0.5 → needs cleaning
    SILT_BLOCKAGE_CRITICAL = 0.75           # ratio > 0.75 → critical

    # Climate data
    CLIMATE_FILE_SUFFIX    = "ALL"
    ANTECEDENT_DAYS        = 7              # rolling window for climate aggregations

    # Maintenance classification
    MIN_ACTIVITY_LENGTH    = 3              # minimum characters for an activity to be processed
    CHUNK_SIZE             = 300_000
    EXCEL_ROW_LIMIT        = 1_048_576      # actual Excel limit

    # Correlation analysis
    HIGH_CORR_THRESHOLD    = 0.7
    VERY_HIGH_CORR         = 0.9
    MISSING_COL_THRESHOLD  = 0.5           # drop columns with > 50% missing
    SAMPLE_SIZE_CORR       = 50_000

    # Response time categories (days)
    RESPONSE_TIME_CATEGORIES = {
        "Emergency - Same Day": 0,
        "Urgent - 1 Day":       1,
        "High - 3 Days":        3,
        "Medium - 1 Week":      7,
        "Low - 1 Month":        30,
        "Backlog - Over 1 Month": float("inf"),
    }
