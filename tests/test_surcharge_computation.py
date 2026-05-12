"""
tests/test_surcharge_computation.py
Unit tests for the surcharge ratio and target variable logic.
Run with: pytest tests/
"""
import sys, os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.os_hydraulic_features import compute_hydraulics, attach_pipe_diameter


def _make_df(**kwargs):
    base = {"Manhole_ID": ["S001"], "Inspection_Date": ["2020-01-01"],
            "Water_Depth": [500.0], "Silt_Depth": [100.0],
            "Invert_Level": [5.0], "Cover_Level": [8.5],
            "Avg_Pipe_Diameter": [600.0]}
    base.update(kwargs)
    df = pd.DataFrame(base)
    df["Inspection_Date"] = pd.to_datetime(df["Inspection_Date"])
    return df


class TestSurchargeRatio:
    def test_basic_ratio(self):
        df = _make_df(Water_Depth=[600.0], Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert abs(result["Surcharge_Ratio"].iloc[0] - 1.0) < 0.01

    def test_not_surcharged(self):
        df = _make_df(Water_Depth=[300.0], Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert result["Is_Surcharged"].iloc[0] == 0

    def test_surcharged(self):
        df = _make_df(Water_Depth=[700.0], Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert result["Is_Surcharged"].iloc[0] == 1

    def test_severity_normal(self):
        df = _make_df(Water_Depth=[400.0], Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert result["Surcharge_Severity"].iloc[0] == "Normal_Flow"

    def test_severity_near(self):
        df = _make_df(Water_Depth=[550.0], Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert result["Surcharge_Severity"].iloc[0] == "Near_Surcharge"

    def test_severity_moderate(self):
        df = _make_df(Water_Depth=[750.0], Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert result["Surcharge_Severity"].iloc[0] == "Moderate_Surcharge"

    def test_severity_severe(self):
        df = _make_df(Water_Depth=[1000.0], Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert result["Surcharge_Severity"].iloc[0] == "Severe_Surcharge"

    def test_effective_depth_uses_water_only(self):
        """Silt depth must not inflate the effective water depth."""
        df = _make_df(Water_Depth=[200.0], Silt_Depth=[400.0], Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert result["Effective_Water_Depth"].iloc[0] == pytest.approx(200.0, abs=1)

    def test_zero_water_gives_nan_ratio(self):
        df = _make_df(Water_Depth=[0.0], Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert pd.isna(result["Surcharge_Ratio"].iloc[0])

    def test_freeboard_positive_when_not_flooded(self):
        df = _make_df(Water_Depth=[200.0], Invert_Level=[5.0], Cover_Level=[8.5],
                      Avg_Pipe_Diameter=[600.0])
        result = compute_hydraulics(df)
        assert result["Freeboard"].iloc[0] > 0


class TestUnitConversion:
    """Ensure depth ratios are plausible — not mostly 0 or 1."""

    def test_ratio_distributed(self):
        """With varied water depths, ratios should span (0, 1)."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "Manhole_ID":      [f"S{i:04d}" for i in range(n)],
            "Inspection_Date": pd.date_range("2015-01-01", periods=n),
            "Water_Depth":     np.random.uniform(50, 900, n),
            "Silt_Depth":      np.random.uniform(0, 200, n),
            "Invert_Level":    np.random.uniform(1, 10, n),
            "Cover_Level":     np.random.uniform(11, 15, n),
            "Avg_Pipe_Diameter": np.full(n, 600.0),
        })
        result = compute_hydraulics(df)
        valid = result["Surcharge_Ratio"].dropna()
        binary_pct = ((valid == 0) | (valid == 1.0)).mean() * 100
        assert binary_pct < 70, f"Too many binary ratios: {binary_pct:.1f}%"
