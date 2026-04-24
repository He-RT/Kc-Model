"""Build patch-date Kcact training tables from exported GEE tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kcact.features.et0 import compute_et0_fao56


def _normalize_patch_id(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "patch_id" not in result.columns:
        raise ValueError("Expected column 'patch_id' in input table")
    result["patch_id"] = result["patch_id"].astype(str)
    return result


def prepare_era5_daily(era5_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_patch_id(era5_df)
    df["date"] = pd.to_datetime(df["date"])
    return compute_et0_fao56(df)


def aggregate_daily_weather_to_mod16_windows(
    era5_daily_df: pd.DataFrame,
    mod16_df: pd.DataFrame,
    gdd_base_c: float = 0.0,
) -> pd.DataFrame:
    weather = _normalize_patch_id(era5_daily_df)
    weather["date"] = pd.to_datetime(weather["date"])
    weather["gdd_daily"] = np.maximum(weather["tmean_c"] - gdd_base_c, 0.0)

    windows = _normalize_patch_id(mod16_df)[["patch_id", "date_start", "date_end", "date"]].copy()
    windows["date_start"] = pd.to_datetime(windows["date_start"])
    windows["date_end"] = pd.to_datetime(windows["date_end"])
    windows["date"] = pd.to_datetime(windows["date"])
    windows = windows.drop_duplicates()

    joined = weather.merge(windows, on="patch_id", how="inner")
    joined = joined[(joined["date_x"] >= joined["date_start"]) & (joined["date_x"] < joined["date_end"])].copy()
    joined = joined.rename(columns={"date_x": "weather_date", "date_y": "window_date"})

    aggregated = (
        joined.groupby(["patch_id", "date_start", "date_end", "window_date"], as_index=False)
        .agg(
            et0_pm_8d_mm=("et0_pm_mm", "sum"),
            precip_mm_8d=("precip_mm", "sum"),
            tmean_c=("tmean_c", "mean"),
            tmax_c=("tmax_c", "mean"),
            tmin_c=("tmin_c", "mean"),
            wind_2m_m_s_mean_8d=("wind_2m_m_s", "mean"),
            solar_rad_mj_m2_d_sum_8d=("solar_rad_mj_m2_d", "sum"),
            dewpoint_c_mean_8d=("dewpoint_c", "mean"),
            pressure_kpa_mean_8d=("pressure_kpa", "mean"),
            vpd_kpa_mean_8d=("vpd_kpa", "mean"),
            gdd_8d=("gdd_daily", "sum"),
            centroid_lat=("centroid_lat", "first"),
            centroid_lon=("centroid_lon", "first"),
            area_ha=("area_ha", "first"),
            elevation_m=("elevation_m", "first"),
        )
        .rename(columns={"window_date": "date"})
    )
    return aggregated


def prepare_mod16_etc(mod16_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_patch_id(mod16_df)
    for column in ["date_start", "date_end", "date"]:
        df[column] = pd.to_datetime(df[column])
    if "etc_8d_mm" not in df.columns and "ET" in df.columns:
        df["etc_8d_mm"] = df["ET"] * 0.1
    keep_cols = [
        "patch_id",
        "date_start",
        "date_end",
        "date",
        "etc_8d_mm",
        "qc_mod16",
        "area_ha",
        "centroid_lat",
        "centroid_lon",
        "elevation_m",
    ]
    existing = [col for col in keep_cols if col in df.columns]
    return df[existing].copy()


def prepare_s2_features(s2_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_patch_id(s2_df)
    for column in ["date_start", "date_end", "date"]:
        df[column] = pd.to_datetime(df[column])
    if "obs_count_s2" not in df.columns and "obs_count" in df.columns:
        df = df.rename(columns={"obs_count": "obs_count_s2"})
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["patch_id", "date"]).copy()
    result["year"] = result["date"].dt.year
    result["doy"] = result["date"].dt.dayofyear
    result["gdd_cum"] = result.groupby("patch_id")["gdd_8d"].cumsum()
    result["precip_7d"] = result.groupby("patch_id")["precip_mm_8d"].transform(lambda s: s.rolling(1, min_periods=1).sum())
    result["precip_15d"] = result.groupby("patch_id")["precip_mm_8d"].transform(lambda s: s.rolling(2, min_periods=1).sum())
    result["precip_30d"] = result.groupby("patch_id")["precip_mm_8d"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    result["ndvi_lag1"] = result.groupby("patch_id")["ndvi"].shift(1)
    result["ndvi_mean_prev_3win"] = (
        result.groupby("patch_id")["ndvi"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    return result


def quality_control(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["kcact"] = result["etc_8d_mm"] / result["et0_pm_8d_mm"]
    result["qc_valid"] = (
        (result["et0_pm_8d_mm"] > 0.01)
        & (result["etc_8d_mm"] >= 0.0)
        & (result["kcact"] >= 0.01)
        & (result["kcact"] <= 2.0)
    )
    if "obs_count_s2" in result.columns:
        result["qc_valid"] &= result["obs_count_s2"].fillna(0) >= 1
    return result


def build_training_table(
    s2_df: pd.DataFrame,
    era5_daily_df: pd.DataFrame,
    mod16_df: pd.DataFrame,
    crop_type: str = "winter_wheat_candidate",
    province: str = "Hebei",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    s2 = prepare_s2_features(s2_df)
    era5_daily = prepare_era5_daily(era5_daily_df)
    mod16 = prepare_mod16_etc(mod16_df)
    weather_8d = aggregate_daily_weather_to_mod16_windows(era5_daily, mod16)

    merged = mod16.merge(
        weather_8d,
        on=["patch_id", "date_start", "date_end", "date"],
        how="inner",
        suffixes=("", "_weather"),
    )
    merged = merged.merge(
        s2,
        on=["patch_id", "date_start", "date_end", "date"],
        how="left",
        suffixes=("", "_s2"),
    )

    for static_col in ["area_ha", "centroid_lat", "centroid_lon", "elevation_m"]:
        if f"{static_col}_weather" in merged.columns:
            merged[static_col] = merged[static_col].fillna(merged[f"{static_col}_weather"])
    merged["province"] = province
    merged["crop_type"] = crop_type
    merged = add_temporal_features(merged)
    merged = quality_control(merged)

    valid = merged[merged["qc_valid"]].copy()
    return merged, valid
