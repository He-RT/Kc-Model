"""Unit tests for FAO-56 Penman-Monteith ET0 computation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import pytest

from kcact.features.et0 import (
    saturation_vapor_pressure,
    actual_vapor_pressure_from_dewpoint,
    slope_svp_curve,
    atmospheric_pressure_kpa,
    psychrometric_constant_kpa_c,
    extraterrestrial_radiation_mj_m2_d,
    clear_sky_radiation_mj_m2_d,
    convert_wind_10m_to_2m,
    compute_et0_fao56,
)


class TestSaturationVaporPressure:
    def test_standard_temp(self):
        """At 20°C, es ≈ 2.34 kPa (FAO-56 reference)."""
        es = saturation_vapor_pressure(pd.Series([20.0]))
        assert 2.30 < es.iloc[0] < 2.40, f"Expected ~2.34 kPa, got {es.iloc[0]:.3f}"

    def test_freezing(self):
        """At 0°C, es ≈ 0.61 kPa."""
        es = saturation_vapor_pressure(pd.Series([0.0]))
        assert 0.60 < es.iloc[0] < 0.62

    def test_numpy_array(self):
        es = saturation_vapor_pressure(np.array([10.0, 30.0]))
        assert es[0] < es[1]  # warmer = higher vapor pressure


class TestActualVaporPressure:
    def test_from_dewpoint(self):
        """At dewpoint 10°C, ea ≈ 1.23 kPa."""
        ea = actual_vapor_pressure_from_dewpoint(pd.Series([10.0]))
        assert 1.20 < ea.iloc[0] < 1.26


class TestSlopeSVP:
    def test_at_20c(self):
        """Slope at 20°C ≈ 0.145 kPa/°C."""
        delta = slope_svp_curve(pd.Series([20.0]))
        assert 0.14 < delta.iloc[0] < 0.15


class TestAtmosphericPressure:
    def test_sea_level(self):
        p = atmospheric_pressure_kpa(pd.Series([0.0]))
        assert 100 < p.iloc[0] < 102

    def test_high_elevation(self):
        """At 1000m, pressure < sea level."""
        p_sea = atmospheric_pressure_kpa(pd.Series([0.0]))
        p_high = atmospheric_pressure_kpa(pd.Series([1000.0]))
        assert p_high.iloc[0] < p_sea.iloc[0]


class TestPsychrometricConstant:
    def test_sea_level(self):
        gamma = psychrometric_constant_kpa_c(pd.Series([101.3]))
        assert 0.066 < gamma.iloc[0] < 0.068


class TestExtraterrestrialRadiation:
    def test_beijing_summer(self):
        """Beijing ~40°N, DOY 180 (summer solstice) → high Ra."""
        ra = extraterrestrial_radiation_mj_m2_d(pd.Series([40.0]), pd.Series([180]))
        assert ra.iloc[0] > 30, f"Expected high summer Ra, got {ra.iloc[0]:.1f}"

    def test_beijing_winter(self):
        """Beijing ~40°N, DOY 1 (winter) → low Ra."""
        ra = extraterrestrial_radiation_mj_m2_d(pd.Series([40.0]), pd.Series([1]))
        assert ra.iloc[0] < 20, f"Expected low winter Ra, got {ra.iloc[0]:.1f}"


class TestClearSkyRadiation:
    def test_positive(self):
        rso = clear_sky_radiation_mj_m2_d(pd.Series([20.0]), pd.Series([100.0]))
        assert rso.iloc[0] > 0


class TestWindConversion:
    def test_10m_to_2m(self):
        """Wind at 2m is lower than at 10m."""
        u2 = convert_wind_10m_to_2m(pd.Series([5.0]))
        assert u2.iloc[0] < 5.0
        assert 3.0 < u2.iloc[0] < 4.5


class TestComputeET0FAO56:
    def test_basic_run(self):
        """Smoke test: compute_et0_fao56 runs without error on valid input."""
        df = pd.DataFrame({
            "date": pd.date_range("2025-04-01", periods=10, freq="D"),
            "point_id": ["pt_1"] * 10,
            "tmin_c": [8.0, 9.0, 10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0],
            "tmax_c": [20.0, 21.0, 22.0, 23.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0],
            "tmean_c": [14.0, 15.0, 16.0, 17.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0],
            "solar_rad_mj_m2_d": [15.0] * 10,
            "wind_10m_m_s": [3.0] * 10,
            "centroid_lat": [37.5] * 10,
            "centroid_lon": [115.0] * 10,
            "dewpoint_c": [5.0] * 10,
            "precip_mm": [0.0] * 10,
            "pressure_kpa": [101.3] * 10,
        })
        result = compute_et0_fao56(df)
        assert "et0_pm_mm" in result.columns
        assert all(result["et0_pm_mm"] >= 0)
        # Spring conditions in Hebei should give ET0 ~2-6 mm/day
        assert 1.0 < result["et0_pm_mm"].mean() < 8.0

    def test_produces_required_columns(self):
        df = pd.DataFrame({
            "date": pd.date_range("2025-06-01", periods=5, freq="D"),
            "tmin_c": [15.0] * 5,
            "tmax_c": [30.0] * 5,
            "tmean_c": [22.5] * 5,
            "solar_rad_mj_m2_d": [20.0] * 5,
            "wind_10m_m_s": [2.0] * 5,
            "centroid_lat": [37.5] * 5,
            "centroid_lon": [115.0] * 5,
            "dewpoint_c": [12.0] * 5,
            "precip_mm": [0.0] * 5,
            "pressure_kpa": [101.3] * 5,
        })
        result = compute_et0_fao56(df)
        for col in ["et0_pm_mm", "es_kpa", "ea_kpa", "vpd_kpa", "rn_mj_m2_d"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_et0_increases_with_temp(self):
        """All else equal, higher temp → higher ET0."""
        base = {
            "date": pd.date_range("2025-05-01", periods=3, freq="D"),
            "solar_rad_mj_m2_d": [18.0] * 3,
            "wind_10m_m_s": [2.5] * 3,
            "centroid_lat": [37.5] * 3,
            "centroid_lon": [115.0] * 3,
            "dewpoint_c": [8.0] * 3,
            "precip_mm": [0.0] * 3,
            "pressure_kpa": [101.3] * 3,
        }
        cool = pd.DataFrame({**base, "tmin_c": [10]*3, "tmax_c": [20]*3, "tmean_c": [15]*3})
        warm = pd.DataFrame({**base, "tmin_c": [18]*3, "tmax_c": [30]*3, "tmean_c": [24]*3})
        et0_cool = compute_et0_fao56(cool)["et0_pm_mm"].mean()
        et0_warm = compute_et0_fao56(warm)["et0_pm_mm"].mean()
        assert et0_warm > et0_cool, f"{et0_warm:.2f} <= {et0_cool:.2f}"

    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3, freq="D")})
        with pytest.raises(ValueError):
            compute_et0_fao56(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
