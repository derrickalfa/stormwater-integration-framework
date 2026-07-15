# Stormwater Infrastructure Analytics — Multi-Source Data Integration Framework

A reproducible Python pipeline that integrates ten heterogeneous data sources into a single ML-ready dataset for stormwater manhole surcharge prediction.

This repository is the companion to the paper:
> **A Multi-Source Data Integration Framework for Stormwater Infrastructure Analytics**
> Alfa et al. (under review), *Environmental Modelling & Software*

The integrated dataset produced here feeds directly into the XAI paper:
> **Explainable AI for Identifying Key Drivers of Stormwater Manhole Surcharge Occurrence and Severity**

---

## Quick Start

```bash
git clone https://github.com/derrickalfa/stormwater-integration-framework.git
cd stormwater-integration-framework
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
python src/10_framework_validation.py

# Optional: NLP classifier evaluation
python src/handlabel_sampler.py   # draw stratified blind sample
# [label handlabel_TASK.csv — fill true_category column]
python src/handlabel_scorer.py    # compute macro-F1 and per-category metrics
```

---

## Pipeline Overview

```
Stage 0  00_raw_maintenance_preprocessing.py
         Raw work order xlsb → keyword-filtered CLeanedAllSheets.xlsx
         2,344,147 stormwater records → 1,100,634 after keyword filtering

Stage 1  01_maintenance_classification.py
         Activity text → 14-category classification + severity + temporal aggregation
         1,100,634 filtered records → 2,635,508 activity strings (2.4× expansion)

Stage 2  02_network_topology.py
         Pipe network GIS files → topology analysis for all 3 drainage systems

Stage 3  03_manhole_property_merger.py
         Manhole properties (3 files) + inspection records → merged dataset
         [Unit conversion embedded: Cover/Invert levels m → mm before ratios]

Stage 4  04_climate_standardisation.py
         Per-station raw CSV files → combined wide-format climate tables
         Temperature: wide format (Max/Mean/Min); all other variables: long format

Stage 4b [GIS — ArcMap]
         LULC raster reclassification → imperviousness surface
         Spatial joins for population, geology, traffic, climate stations
         See: docs/gis_spatial_attribution.md

Stage 5  05_hydraulic_features.py
         Hydraulic metrics + target variable computation
         [Diameter priority: inspection-level > network fallback]
         [Slope/length caps embedded: no separate fix scripts]

Stage 6  06_comprehensive_integration.py
         All 10 sources merged into one dataset (291,116 records)
         [Three-pass climate station assignment: ArcMap seeds → BFS propagation
          → Euclidean distance fallback]

Stage 7  07_data_quality_validation.py
         Physical impossibility removal; missing data retained

Stage 8  08_maintenance_reaggregation.py  ← TEMPORAL INTERVAL-BASED (corrected)
         Pipe-level maintenance → manhole-level BETWEEN-INSPECTION counts
         Fixes temporal leakage in earlier lifetime-total approach
         Output: 2,429,271 total events; mean 8.34 events per inspection record

Stage 9  09_feature_validation_and_split.py
         Final physical caps + classification/regression split
         Classification: 291,116 records (all inspections)
         Regression (V3): 152,949 records with Water_Depth > 0 and valid
                          Surcharge_Ratio (58,477 manholes)

Stage 10 10_framework_validation.py
         Four-dimension validation: internal consistency, physical plausibility,
         spatial consistency, temporal consistency
         Key results: 98.4% surcharge label agreement; 100% maintenance
         aggregation consistency; 97.9% rainfall coverage; 98.7% coordinate coverage

NLP evaluation (optional, for reproducing Section 4.2.3 of the paper):
  handlabel_sampler.py  — stratified blind sample (300 records, 14 categories)
  handlabel_scorer.py   — macro-F1, per-category precision/recall/F1
  Achieved: macro-F1 = 0.805, accuracy = 0.888 (n = 286)
```

---

## Data Sources

| Source | Format | Attached via |
|---|---|---|
| Manhole inspection records | .xlsb | Direct merge |
| Manhole property registry | .xlsx (×3) | Feature_Number join |
| Pipe network (GIS) | .xlsx | Link_ID → Node_ID |
| Maintenance work orders | .xlsb | Pipe_ID → manhole via network |
| Climate stations (daily) | CSV per station/variable | Three-pass station assignment |
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

All paths and parameters are in **`config.py`**. Update two lines before running:

```python
INPUT_ROOT  = Path(r"path/to/your/raw/data")
OUTPUT_ROOT = Path(r"path/to/your/output/folder")
```

All other paths are derived automatically from these two roots.

---

## Key Design Decisions

**Temporal maintenance features** — Script 08 assigns each maintenance event to
the first inspection of that manhole on or after the event's completion date,
producing between-inspection counts rather than lifetime totals. The first
inspection absorbs all prior history. This eliminates temporal leakage that
was present in earlier implementations.

**Unit safety** — Cover_Level and Invert_Level are in metres in the raw data.
Water_Depth and Silt_Depth are in millimetres. Script 03 detects and converts
depths to mm before computing ratios using a statistical detection rule
(median < 10 → metres assumed).

**Pipe diameter hierarchy** — Script 05 uses inspection-level inlet/outlet
diameters as the primary source (mean of non-zero values, capped at 5000 mm).
The network-derived average diameter is the fallback. Coverage: 99.2% of records.

**Three-pass climate assignment** — Script 06 assigns climate stations via:
Pass 1: ArcMap nearest-station spatial join (seed assignments);
Pass 2: BFS propagation through the pipe network adjacency graph (corrects
Euclidean distance errors in complex terrain);
Pass 3: Euclidean distance fallback for remaining isolated manholes.
Result: 97.9% rainfall coverage, 95.7% temperature coverage.

**Missing data** — Records with missing predictor values are retained
throughout. Only physically impossible measurements are removed (Script 07).
This preserves the full inspection record while allowing downstream ML
algorithms to handle imputation.

**Regression dataset (V3)** — The regression target dataset includes all
records with Water_Depth > 0 and a valid Surcharge_Ratio (152,949 records),
not surcharged records only. This spans the full hydraulic loading spectrum
and supports continuous severity modelling.

**NLP classifier** — The 14-category rule-based classifier uses a two-tier
keyword matching strategy (primary: confidence 1.00; secondary: confidence
0.60) with morphological stemming. Three explicit design rules: manhole-first
override, ditto inheritance, and Blockages/General Cleaning deconfliction.
Evaluated at macro-F1 = 0.805 on a 300-record stratified blind sample.

---

## Repository Structure

```
stormwater-framework/
├── config.py                          ← All paths and parameters
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
│   ├── 08_maintenance_reaggregation.py  ← temporal interval-based (corrected)
│   ├── 09_feature_validation_and_split.py
│   ├── 10_framework_validation.py
│   ├── handlabel_sampler.py             ← NLP evaluation: stratified sampling
│   ├── handlabel_scorer.py              ← NLP evaluation: macro-F1 scoring
│   └── utils/
├── data/sample/                         ← Synthetic example data (no real records)
│   └── schemas/                         ← Field definitions for each source
├── docs/
│   └── gis_spatial_attribution.md       ← ArcMap spatial join documentation
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

> Alfa, D., Ali, E., & Zayed, T. (under review). A Multi-Source Data Integration
> Framework for Stormwater Infrastructure Analytics. *Environmental Modelling & Software*.

---

---

## Paper B — Surcharge Risk Assessment and Asset Prioritisation (ML Scripts)

The scripts in `ml/` implement the two-tier explainable machine-learning framework
reported in:

> **Maintenance-Informed Explainable Machine Learning for Surcharge Risk Assessment
> and Prioritisation of Assets in Underground Stormwater Drainage Networks**
> Alfa, D., Ali, E., & Zayed, T. (under review), *Automation in Construction*

These scripts consume the datasets produced by the integration pipeline above
(`Paths.FINAL_CLASSIFICATION` and `Paths.FINAL_REGRESSION`).

### ML Pipeline

```
Stage 11  ml/11_classification_primary.py
          Manhole-grouped 8-model classification benchmark
          Target: Is_Surcharged (binary)
          Key result: XGBoost F1=0.702, LightGBM F1=0.701 (ROC-AUC 0.969/0.968)

Stage 12  ml/12_severity_primary.py
          Manhole-grouped 8-model regression benchmark
          Target: Surcharge_Ratio (continuous)
          Key result: LightGBM R2=0.715, MAE=0.065

Holdout R2-13a  ml/holdout_spatial_classification.py
          Spatial hold-out: train 80% of districts, test 20% (5 seeds)
          Key result: LightGBM F1=0.574 +/- 0.145; Random Forest best at 0.579

Holdout R2-13b  ml/holdout_spatial_regression.py
          Spatial hold-out: regression tier, same 80/20 district split (5 seeds)
          Key result: LightGBM R2=0.502 +/- 0.137; Gradient Boosting best at 0.543

Holdout R2-13c  ml/holdout_spatial_regression_B.py
          Spatial hold-out Design B: re-tuned 3-way 60/20/20 split, top-2 only
          Supplementary robustness check — not reported in primary Table 8

Holdout R2-04a  ml/holdout_temporal_classification.py
          Temporal hold-out: train <=2018, test 2019-2021
          Key result: Random Forest F1=0.725 (best); LightGBM drops to 0.525

Holdout R2-04b  ml/holdout_temporal_regression.py
          Temporal hold-out: regression tier, same <=2018 / 2019-2021 split
          Key result: XGBoost R2=0.573 (best); no ranking reversal in severity tier
```

### Running the ML scripts

```bash
# Primary benchmarks (run first)
python ml/11_classification_primary.py
python ml/12_severity_primary.py

# Holdout robustness (requires primary benchmark _params.json outputs)
python ml/holdout_spatial_classification.py
python ml/holdout_spatial_regression.py
python ml/holdout_temporal_classification.py
python ml/holdout_temporal_regression.py

# Optional supplementary
python ml/holdout_spatial_regression_B.py
```

Runtime: approximately 1-2 hours per script on a mid-range GPU workstation.
Holdout scripts load hyperparameters from primary benchmark `{Model}_params.json`
files without re-tuning (param-reuse mode).

### Updated repository structure

```
stormwater-framework/
├── config.py
├── requirements.txt
├── README.md
├── src/                                  <- Integration pipeline (Stages 00-10)
│   ├── 00_raw_maintenance_preprocessing.py
│   ├── ...
│   └── 10_framework_validation.py
├── ml/                                   <- Paper B ML scripts
│   ├── 11_classification_primary.py
│   ├── 12_severity_primary.py
│   ├── holdout_spatial_classification.py
│   ├── holdout_spatial_regression.py
│   ├── holdout_spatial_regression_B.py   <- supplementary only
│   ├── holdout_temporal_classification.py
│   └── holdout_temporal_regression.py
├── data/sample/
├── docs/
└── tests/
```

## License

MIT License — code is freely reusable. Data files are not included and remain
proprietary to the Drainage Services Department, Hong Kong.
