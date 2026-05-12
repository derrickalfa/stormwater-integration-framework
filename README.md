# Stormwater Infrastructure Analytics — Multi-Source Data Integration Framework

A reproducible Python pipeline that integrates ten heterogeneous data sources into a single ML-ready dataset for stormwater manhole surcharge prediction.

This repository is the companion to the paper:
> **A Multi-Source Data Integration Framework for Stormwater Infrastructure Analytics**

The integrated dataset produced here feeds directly into the XAI paper:
> **Explainable AI for Identifying Key Drivers of Stormwater Manhole Surcharge Occurrence and Severity**

---

## Quick Start

```bash
git clone https://github.com/<your-username>/stormwater-framework.git
cd stormwater-framework
pip install -r requirements.txt

# 1. Edit INPUT_ROOT and OUTPUT_ROOT in config.py
# 2. Run stages in order:
python src/00_raw_maintenance_preprocessing.py
python src/01_maintenance_classification.py
python src/02_network_topology.py
python src/03_manhole_property_merger.py
python src/04_climate_standardisation.py
# [GIS step — see docs/gis_spatial_attribution.md]
python src/05_hydraulic_features.py
python src/06_comprehensive_integration.py
python src/07_data_quality_validation.py
python src/08_maintenance_reaggregation.py
python src/09_feature_validation_and_split.py
```

---

## Pipeline Overview

```
Stage 0  00_raw_maintenance_preprocessing.py
         Raw work order xlsb → keyword-filtered CLeanedAllSheets.xlsx

Stage 1  01_maintenance_classification.py
         Activity text → 14-category classification + severity + temporal aggregation

Stage 2  02_network_topology.py
         Pipe network GIS files → topology analysis for all 3 drainage systems

Stage 3  03_manhole_property_merger.py
         Manhole properties (3 files) + inspection records → merged dataset
         [Unit conversion embedded: Cover/Invert levels m → mm before ratios]

Stage 4  04_climate_standardisation.py
         Per-station raw CSV files → combined wide-format climate tables

Stage 4b [GIS — ArcMap]
         LULC raster reclassification → imperviousness surface
         Spatial joins for population, geology, traffic, climate stations
         See: docs/gis_spatial_attribution.md

Stage 5  05_hydraulic_features.py
         Hydraulic metrics + target variable computation
         [Diameter priority embedded: inspection > network fallback]
         [Slope/length caps embedded: no separate fix scripts]

Stage 6  06_comprehensive_integration.py
         All 10 sources merged into one dataset
         [Both integration passes combined: base + coverage enhancement]
         [Station propagation through pipe network embedded]

Stage 7  07_data_quality_validation.py
         Physical impossibility removal; missing data retained

Stage 8  08_maintenance_reaggregation.py
         Pipe-level maintenance → manhole-level aggregation

Stage 9  09_feature_validation_and_split.py
         Final physical caps + classification/regression split + correlation matrix
```

---

## Data Sources

| Source | Format | Attached via |
|---|---|---|
| Manhole inspection records | .xlsb | Direct merge |
| Manhole property registry | .xlsx (×3) | Feature_Number join |
| Pipe network (GIS) | .xlsx | Link_ID → Node_ID |
| Maintenance work orders | .xlsb | Pipe_ID → manhole via network |
| Climate stations (daily) | CSV per station/variable | Nearest-station assignment |
| Climate stations (monthly) | .xlsx | Station + Year + Month |
| Environmental covariates | .xls (GIS output) | Point-in-polygon / raster sampling |
| Traffic AADT | .xls sheet | Traffic station + Year |
| Geology | .xls sheet | Point-in-polygon |
| CCTV inspection grades | .xlsx (×3) | Manhole_ID |

---

## Target Variables

| Variable | Type | Definition |
|---|---|---|
| `Is_Surcharged` | Binary | Surcharge_Ratio > 1.0 |
| `Surcharge_Ratio` | Continuous | Water_Depth / Avg_Pipe_Diameter |
| `Surcharge_Severity` | Ordinal | Normal (<0.75) / Near (0.75–1.0) / Moderate (1.0–1.5) / Severe (>1.5) |

---

## Configuration

All paths and parameters are in **`config.py`**. Update two lines:

```python
INPUT_ROOT  = Path(r"path/to/your/raw/data")
OUTPUT_ROOT = Path(r"path/to/your/output/folder")
```

---

## Key Design Decisions

**Unit safety** — Cover_Level and Invert_Level are in metres in the raw data. Water_Depth and Silt_Depth are in millimetres. Script 03 detects and converts depths to mm before computing ratios using a statistical detection rule (median < 10 → metres).

**Pipe diameter hierarchy** — Script 05 uses inspection-level inlet/outlet diameters as the primary source (mean of non-zero values, capped at 5000 mm). The network-derived average diameter is the fallback.

**Station coverage** — Script 06 first loads direct station assignments (Pass 1), then propagates through the pipe network adjacency graph (BFS), then applies nearest-neighbour distance assignment for remaining isolated manholes.

**Missing data** — Records with missing predictor values are retained throughout. Only physically impossible measurements are removed (Script 07).

**Physical plausibility** — Pipe slopes outside [-0.5, 1.0] and lengths below 0.5 m are set to NaN in Script 09. These caps are applied once, cleanly, not as separate patch scripts.

---

## Repository Structure

```
stormwater-framework/
├── config.py                        ← All paths and parameters
├── requirements.txt
├── README.md
├── src/
│   ├── 00_raw_maintenance_preprocessing.py
│   ├── 01_maintenance_classification.py
│   ├── 02_network_topology.py
│   ├── 03_manhole_property_merger.py
│   ├── 04_climate_standardisation.py
│   ├── 05_hydraulic_features.py
│   ├── 06_comprehensive_integration.py
│   ├── 07_data_quality_validation.py
│   ├── 08_maintenance_reaggregation.py
│   ├── 09_feature_validation_and_split.py
│   └── utils/
├── data/sample/                     ← Synthetic example data (no real records)
│   └── schemas/                     ← Field definitions for each source
├── docs/
│   └── gis_spatial_attribution.md   ← ArcMap spatial join documentation
└── tests/
    └── test_surcharge_computation.py
```

---

## Requirements

```
pandas>=1.5
numpy>=1.23
openpyxl>=3.0
pyxlsb>=1.0
xlrd>=2.0
matplotlib>=3.5
seaborn>=0.12
scipy>=1.9
scikit-learn>=1.1
pathlib
psutil
```

---

## Citation

If you use this framework, please cite:

> [Author et al.] (under review). A Multi-Source Data Integration Framework for Stormwater Infrastructure Analytics.

---

## License

MIT License — code is freely reusable. Data files are not included and remain proprietary to the Drainage Services Department, Hong Kong.
