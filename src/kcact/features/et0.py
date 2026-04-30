"""FAO Penman-Monteith ET0 utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


STEFAN_BOLTZMANN = 4.903e-9


def saturation_vapor_pressure(temp_c: pd.Series | np.ndarray) -> pd.Series:
    values = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    return pd.Series(values, index=getattr(temp_c, "index", None))


def actual_vapor_pressure_from_dewpoint(
    dewpoint_c: pd.Series | np.ndarray,
) -> pd.Series:
    values = 0.6108 * np.exp((17.27 * dewpoint_c) / (dewpoint_c + 237.3))
    return pd.Series(values, index=getattr(dewpoint_c, "index", None))


def slope_svp_curve(temp_c: pd.Series | np.ndarray) -> pd.Series:
    es = saturation_vapor_pressure(temp_c)
    values = 4098 * es / np.power(temp_c + 237.3, 2)
    return pd.Series(values, index=getattr(temp_c, "index", None))


def atmospheric_pressure_kpa(
    elevation_m: pd.Series | np.ndarray,
) -> pd.Series:
    values = 101.3 * np.power((293.0 - 0.0065 * elevation_m) / 293.0, 5.26)
    return pd.Series(values, index=getattr(elevation_m, "index", None))


def psychrometric_constant_kpa_c(
    pressure_kpa: pd.Series | np.ndarray,
) -> pd.Series:
    values = 0.000665 * pressure_kpa
    return pd.Series(values, index=getattr(pressure_kpa, "index", None))


def extraterrestrial_radiation_mj_m2_d(
    latitude_deg: pd.Series | np.ndarray,
    doy: pd.Series | np.ndarray,
) -> pd.Series:
    lat_rad = np.deg2rad(latitude_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    solar_declination = 0.409 * np.sin((2 * np.pi * doy / 365.0) - 1.39)
    sunset_hour_angle = np.arccos(-np.tan(lat_rad) * np.tan(solar_declination))
    values = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (
            sunset_hour_angle * np.sin(lat_rad) * np.sin(solar_declination)
            + np.cos(lat_rad)
            * np.cos(solar_declination)
            * np.sin(sunset_hour_angle)
        )
    )
    return pd.Series(values, index=getattr(latitude_deg, "index", None))


def clear_sky_radiation_mj_m2_d(
    extraterrestrial_radiation: pd.Series | np.ndarray,
    elevation_m: pd.Series | np.ndarray,
) -> pd.Series:
    values = (0.75 + 2e-5 * elevation_m) * extraterrestrial_radiation
    return pd.Series(values, index=getattr(extraterrestrial_radiation, "index", None))


def convert_wind_10m_to_2m(
    wind_10m_m_s: pd.Series | np.ndarray,
) -> pd.Series:
    factor = 4.87 / np.log(67.8 * 10 - 5.42)
    values = wind_10m_m_s * factor
    return pd.Series(values, index=getattr(wind_10m_m_s, "index", None))


def compute_et0_fao56(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily ET0 using FAO-56 Penman-Monteith.

    Required columns:
    - date
    - tmin_c
    - tmax_c
    - tmean_c
    - solar_rad_mj_m2_d
    - wind_10m_m_s or wind_2m_m_s
    - centroid_lat
    - elevation_m or pressure_kpa
    - dewpoint_c or rh_mean
    """
    df = weather_df.copy()
    if "date" not in df.columns:
        raise ValueError("weather_df must include a 'date' column")

    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear

    if "wind_2m_m_s" not in df.columns:
        if "wind_10m_m_s" not in df.columns:
            raise ValueError("Need wind_2m_m_s or wind_10m_m_s to compute ET0")
        df["wind_2m_m_s"] = convert_wind_10m_to_2m(df["wind_10m_m_s"])

    if "pressure_kpa" not in df.columns:
        if "elevation_m" not in df.columns:
            raise ValueError("Need pressure_kpa or elevation_m to compute ET0")
        df["pressure_kpa"] = atmospheric_pressure_kpa(df["elevation_m"])

    es_tmax = saturation_vapor_pressure(df["tmax_c"])
    es_tmin = saturation_vapor_pressure(df["tmin_c"])
    df["es_kpa"] = (es_tmax + es_tmin) / 2.0

    if "dewpoint_c" in df.columns:
        df["ea_kpa"] = actual_vapor_pressure_from_dewpoint(df["dewpoint_c"])
    elif "rh_mean" in df.columns:
        df["ea_kpa"] = df["es_kpa"] * df["rh_mean"] / 100.0
    else:
        raise ValueError("Need dewpoint_c or rh_mean to compute actual vapor pressure")

    df["delta_kpa_c"] = slope_svp_curve(df["tmean_c"])
    df["gamma_kpa_c"] = psychrometric_constant_kpa_c(df["pressure_kpa"])
    df["ra_mj_m2_d"] = extraterrestrial_radiation_mj_m2_d(
        df["centroid_lat"], df["doy"]
    )
    df["rso_mj_m2_d"] = clear_sky_radiation_mj_m2_d(
        df["ra_mj_m2_d"], df.get("elevation_m", 0.0)
    )
    df["rns_mj_m2_d"] = (1.0 - 0.23) * df["solar_rad_mj_m2_d"]

    safe_rso = df["rso_mj_m2_d"].replace(0, np.nan)
    rs_rso = (df["solar_rad_mj_m2_d"] / safe_rso).clip(lower=0.0, upper=1.0).fillna(0.0)

    tmax_k = df["tmax_c"] + 273.16
    tmin_k = df["tmin_c"] + 273.16
    df["rnl_mj_m2_d"] = (
        STEFAN_BOLTZMANN
        * ((np.power(tmax_k, 4) + np.power(tmin_k, 4)) / 2.0)
        * (0.34 - 0.14 * np.sqrt(df["ea_kpa"].clip(lower=0.0)))
        * (1.35 * rs_rso - 0.35)
    )
    df["rn_mj_m2_d"] = df["rns_mj_m2_d"] - df["rnl_mj_m2_d"]
    df["vpd_kpa"] = (df["es_kpa"] - df["ea_kpa"]).clip(lower=0.0)

    numerator = (
        0.408 * df["delta_kpa_c"] * df["rn_mj_m2_d"]
        + df["gamma_kpa_c"]
        * (900.0 / (df["tmean_c"] + 273.0))
        * df["wind_2m_m_s"]
        * df["vpd_kpa"]
    )
    denominator = df["delta_kpa_c"] + df["gamma_kpa_c"] * (1.0 + 0.34 * df["wind_2m_m_s"])
    df["et0_pm_mm"] = (numerator / denominator).clip(lower=0.0)
    return df
