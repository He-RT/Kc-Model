"""Unit tests for the Kcact training table builder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import pytest

from kcact.data.kcact_builder import (
    _normalize_patch_id,
    prepare_era5_daily,
    prepare_mod16_etc,
    prepare_s2_features,
    aggregate_daily_weather_to_mod16_windows,
    add_temporal_features,
    quality_control,
    build_training_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_mod16_df():
    """Minimal MOD16-like table matching GEE point-based export."""
    pts = [f"pt_{i}" for i in range(3)]
    rows = []
    for pt in pts:
        for w in range(4):
            d = pd.Timestamp("2025-03-01") + pd.Timedelta(days=8 * w)
            rows.append({
                "point_id": pt,
                "centroid_lat": 37.5,
                "centroid_lon": 115.0,
                "date_start": d.strftime("%Y-%m-%d"),
                "date_end": (d + pd.Timedelta(days=8)).strftime("%Y-%m-%d"),
                "date": (d + pd.Timedelta(days=8)).strftime("%Y-%m-%d"),
                "etc_8d_mm": 15.0 + w * 2,
                "qc_mod16": 1,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_era5_df():
    """Minimal ERA5-like daily table."""
    pts = [f"pt_{i}" for i in range(3)]
    rows = []
    for pt in pts:
        for d in pd.date_range("2025-03-01", "2025-04-01", freq="D"):
            rows.append({
                "point_id": pt,
                "centroid_lat": 37.5,
                "centroid_lon": 115.0,
                "date": d.strftime("%Y-%m-%d"),
                "tmean_c": 12.0,
                "tmin_c": 7.0,
                "tmax_c": 18.0,
                "solar_rad_mj_m2_d": 15.0,
                "wind_10m_m_s": 3.0,
                "dewpoint_c": 4.0,
                "precip_mm": 0.5,
                "pressure_kpa": 101.3,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_s2_df():
    """Minimal S2-like table."""
    pts = [f"pt_{i}" for i in range(3)]
    rows = []
    for pt in pts:
        for w in range(4):
            d = pd.Timestamp("2025-03-01") + pd.Timedelta(days=8 * w)
            rows.append({
                "point_id": pt,
                "centroid_lat": 37.5,
                "centroid_lon": 115.0,
                "date_start": d.strftime("%Y-%m-%d"),
                "date_end": (d + pd.Timedelta(days=8)).strftime("%Y-%m-%d"),
                "date": (d + pd.Timedelta(days=8)).strftime("%Y-%m-%d"),
                "ndvi": 0.3 + 0.05 * w,
                "evi": 0.25 + 0.04 * w,
                "savi": 0.2 + 0.03 * w,
                "gndvi": 0.2 + 0.03 * w,
                "lswi": 0.15 + 0.02 * w,
                "nirv": 0.1 + 0.02 * w,
                "re_ndvi": 0.2 + 0.03 * w,
                "valid_obs": 3,
                "obs_count_s2": 3,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _normalize_patch_id
# ---------------------------------------------------------------------------

class TestNormalizePatchId:
    def test_patch_id_passthrough(self):
        df = pd.DataFrame({"patch_id": ["a", "b"], "val": [1, 2]})
        result = _normalize_patch_id(df)
        assert result["patch_id"].tolist() == ["a", "b"]

    def test_point_id_mapped(self):
        df = pd.DataFrame({"point_id": ["pt_0", "pt_1"], "val": [1, 2]})
        result = _normalize_patch_id(df)
        assert result["patch_id"].tolist() == ["pt_0", "pt_1"]

    def test_both_columns_prefer_patch_id(self):
        df = pd.DataFrame({"patch_id": ["a"], "point_id": ["b"], "val": [1]})
        result = _normalize_patch_id(df)
        assert result["patch_id"].iloc[0] == "a"

    def test_missing_raises(self):
        df = pd.DataFrame({"val": [1, 2]})
        with pytest.raises(ValueError, match="patch_id.*point_id"):
            _normalize_patch_id(df)


# ---------------------------------------------------------------------------
# prepare_mod16_etc
# ---------------------------------------------------------------------------

class TestPrepareMod16Etc:
    def test_keeps_expected_columns(self, sample_mod16_df):
        result = prepare_mod16_etc(sample_mod16_df)
        assert "patch_id" in result.columns
        assert "etc_8d_mm" in result.columns
        assert "date_start" in result.columns

    def test_datetime_conversion(self, sample_mod16_df):
        result = prepare_mod16_etc(sample_mod16_df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_converts_ET_column(self):
        df = pd.DataFrame({
            "patch_id": ["a"], "date": ["2025-03-08"],
            "date_start": ["2025-03-01"], "date_end": ["2025-03-08"],
            "ET": 150,  # kg/m2/8day * 0.1 = 15 mm
        })
        result = prepare_mod16_etc(df)
        assert abs(result["etc_8d_mm"].iloc[0] - 15.0) < 0.01


# ---------------------------------------------------------------------------
# prepare_s2_features
# ---------------------------------------------------------------------------

class TestPrepareS2Features:
    def test_renames_obs_count(self):
        df = pd.DataFrame({
            "point_id": ["pt_0"], "date_start": ["2025-03-01"],
            "date_end": ["2025-03-08"], "date": ["2025-03-08"],
            "ndvi": [0.5], "obs_count": [3],
        })
        result = prepare_s2_features(df)
        assert "obs_count_s2" in result.columns


# ---------------------------------------------------------------------------
# prepare_era5_daily
# ---------------------------------------------------------------------------

class TestPrepareEra5Daily:
    def test_computes_et0(self, sample_era5_df):
        result = prepare_era5_daily(sample_era5_df)
        assert "et0_pm_mm" in result.columns
        assert all(result["et0_pm_mm"] >= 0)


# ---------------------------------------------------------------------------
# aggregate_daily_weather_to_mod16_windows
# ---------------------------------------------------------------------------

class TestAggregateWeather:
    @pytest.fixture
    def daily_with_et0(self, sample_era5_df):
        """ERA5 data after prepare_era5_daily, which adds et0_pm_mm, vpd_kpa, wind_2m_m_s."""
        return prepare_era5_daily(sample_era5_df)

    def test_returns_8d_windows(self, daily_with_et0, sample_mod16_df):
        result = aggregate_daily_weather_to_mod16_windows(daily_with_et0, sample_mod16_df)
        assert len(result) > 0
        assert "et0_pm_8d_mm" in result.columns
        assert "tmean_c" in result.columns
        assert "gdd_8d" in result.columns

    def test_merged_on_patch_id(self, daily_with_et0, sample_mod16_df):
        result = aggregate_daily_weather_to_mod16_windows(daily_with_et0, sample_mod16_df)
        assert set(result["patch_id"].unique()) == {"pt_0", "pt_1", "pt_2"}

    def test_handles_missing_optional_columns(self, daily_with_et0, sample_mod16_df):
        """area_ha and elevation_m are optional; aggregation must not KeyError."""
        result = aggregate_daily_weather_to_mod16_windows(daily_with_et0, sample_mod16_df)
        assert "patch_id" in result.columns


# ---------------------------------------------------------------------------
# add_temporal_features
# ---------------------------------------------------------------------------

class TestAddTemporalFeatures:
    def test_adds_lag_and_cumulative(self, sample_mod16_df, sample_era5_df, sample_s2_df):
        _, valid = build_training_table(sample_s2_df, sample_era5_df, sample_mod16_df)
        result = add_temporal_features(valid)
        assert "gdd_cum" in result.columns
        assert "ndvi_lag1" in result.columns
        assert "precip_7d" in result.columns

    def test_gdd_cum_monotonic(self, sample_mod16_df, sample_era5_df, sample_s2_df):
        _, valid = build_training_table(sample_s2_df, sample_era5_df, sample_mod16_df)
        result = add_temporal_features(valid)
        for pt, grp in result.groupby("patch_id"):
            gdd = grp.sort_values("date")["gdd_cum"].values
            assert all(np.diff(gdd) >= -1e-9), f"gdd_cum not monotonic for {pt}"


# ---------------------------------------------------------------------------
# quality_control
# ---------------------------------------------------------------------------

class TestQualityControl:
    def test_valid_range_passes(self, sample_mod16_df, sample_era5_df, sample_s2_df):
        _, valid = build_training_table(sample_s2_df, sample_era5_df, sample_mod16_df)
        result = quality_control(valid)
        assert "kcact" in result.columns
        assert "qc_valid" in result.columns

    def test_negative_etc_fails(self):
        df = pd.DataFrame({
            "patch_id": ["a"], "date": [pd.Timestamp("2025-03-08")],
            "etc_8d_mm": [-5.0], "et0_pm_8d_mm": [20.0],
        })
        result = quality_control(df)
        assert not result["qc_valid"].iloc[0]

    def test_zero_et0_fails(self):
        df = pd.DataFrame({
            "patch_id": ["a"], "date": [pd.Timestamp("2025-03-08")],
            "etc_8d_mm": [15.0], "et0_pm_8d_mm": [0.0],
        })
        result = quality_control(df)
        assert not result["qc_valid"].iloc[0]

    def test_kcact_above_2_rejected(self):
        df = pd.DataFrame({
            "patch_id": ["a"], "date": [pd.Timestamp("2025-03-08")],
            "etc_8d_mm": [60.0], "et0_pm_8d_mm": [20.0],  # kcact=3.0
        })
        result = quality_control(df)
        assert not result["qc_valid"].iloc[0]

    def test_kcact_near_zero_rejected(self):
        df = pd.DataFrame({
            "patch_id": ["a"], "date": [pd.Timestamp("2025-03-08")],
            "etc_8d_mm": [0.001], "et0_pm_8d_mm": [20.0],  # kcact ~ 0.00005
        })
        result = quality_control(df)
        assert not result["qc_valid"].iloc[0]


# ---------------------------------------------------------------------------
# build_training_table (integration)
# ---------------------------------------------------------------------------

class TestBuildTrainingTable:
    def test_returns_two_dataframes(self, sample_s2_df, sample_era5_df, sample_mod16_df):
        all_rows, valid_rows = build_training_table(sample_s2_df, sample_era5_df, sample_mod16_df)
        assert len(all_rows) > 0
        assert len(valid_rows) >= 0
        assert len(valid_rows) <= len(all_rows)

    def test_valid_rows_have_kcact(self, sample_s2_df, sample_era5_df, sample_mod16_df):
        _, valid_rows = build_training_table(sample_s2_df, sample_era5_df, sample_mod16_df)
        if len(valid_rows) > 0:
            assert all(valid_rows["kcact"] > 0)
            assert all(valid_rows["kcact"] <= 2.0)

    def test_output_columns(self, sample_s2_df, sample_era5_df, sample_mod16_df):
        all_rows, _ = build_training_table(sample_s2_df, sample_era5_df, sample_mod16_df)
        expected = ["patch_id", "date", "kcact", "qc_valid", "ndvi",
                    "et0_pm_8d_mm", "etc_8d_mm", "province", "crop_type"]
        for col in expected:
            assert col in all_rows.columns, f"Missing column: {col}"

    def test_reproducible(self, sample_s2_df, sample_era5_df, sample_mod16_df):
        a1, _ = build_training_table(sample_s2_df, sample_era5_df, sample_mod16_df)
        a2, _ = build_training_table(sample_s2_df, sample_era5_df, sample_mod16_df)
        assert len(a1) == len(a2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
