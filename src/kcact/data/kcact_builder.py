"""Build patch-date Kcact training tables from exported GEE tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kcact.features.et0 import compute_et0_fao56


def _normalize_patch_id(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "patch_id" not in result.columns:
        if "point_id" in result.columns:
            result["patch_id"] = result["point_id"].astype(str)
        else:
            raise ValueError("Expected column 'patch_id' or 'point_id' in input table")
    else:
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

    agg_spec = {
        "et0_pm_8d_mm": ("et0_pm_mm", "sum"),
        "precip_mm_8d": ("precip_mm", "sum"),
        "tmean_c": ("tmean_c", "mean"),
        "tmax_c": ("tmax_c", "mean"),
        "tmin_c": ("tmin_c", "mean"),
        "wind_2m_m_s_mean_8d": ("wind_2m_m_s", "mean"),
        "solar_rad_mj_m2_d_sum_8d": ("solar_rad_mj_m2_d", "sum"),
        "dewpoint_c_mean_8d": ("dewpoint_c", "mean"),
        "pressure_kpa_mean_8d": ("pressure_kpa", "mean"),
        "vpd_kpa_mean_8d": ("vpd_kpa", "mean"),
        "gdd_8d": ("gdd_daily", "sum"),
        "centroid_lat": ("centroid_lat", "first"),
        "centroid_lon": ("centroid_lon", "first"),
    }
    for opt_col in ["area_ha", "elevation_m"]:
        if opt_col in joined.columns:
            agg_spec[opt_col] = (opt_col, "first")

    aggregated = (
        joined.groupby(["patch_id", "date_start", "date_end", "window_date"], as_index=False)
        .agg(**agg_spec)
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

    # GDD accumulation from start of each patch's record
    result["gdd_cum"] = result.groupby("patch_id")["gdd_8d"].cumsum()

    # Rolling precipitation sums (denominated in 8-day window counts)
    result["precip_7d"] = result.groupby("patch_id")["precip_mm_8d"].transform(
        lambda s: s.rolling(1, min_periods=1).sum())
    result["precip_15d"] = result.groupby("patch_id")["precip_mm_8d"].transform(
        lambda s: s.rolling(2, min_periods=1).sum())
    result["precip_30d"] = result.groupby("patch_id")["precip_mm_8d"].transform(
        lambda s: s.rolling(4, min_periods=1).sum())

    # NDVI temporal derivatives — capture growth trajectory shape
    result["ndvi_lag1"] = result.groupby("patch_id")["ndvi"].shift(1)
    result["ndvi_lag1"] = result["ndvi_lag1"].fillna(result["ndvi"])
    result["ndvi_diff"] = result["ndvi"] - result["ndvi_lag1"]          # 1st derivative
    result["ndvi_accel"] = result.groupby("patch_id")["ndvi_diff"].diff()  # 2nd derivative
    result["ndvi_accel"] = result["ndvi_accel"].fillna(0.0)

    result["ndvi_mean_prev_3win"] = (
        result.groupby("patch_id")["ndvi"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    result["ndvi_mean_prev_3win"] = result["ndvi_mean_prev_3win"].fillna(result["ndvi"])

    # Feature interactions — capture joint signals
    if "vpd_kpa_mean_8d" in result.columns:
        result["ndvi_vpd"] = result["ndvi"] * result["vpd_kpa_mean_8d"]
        if "lswi" in result.columns:
            result["lswi_vpd"] = result["lswi"] * result["vpd_kpa_mean_8d"]

    # Phenology features — tell the model where each window sits in the season
    # Winter wheat: planting ~Oct 1 (doy 274), harvest ~end of Jul (doy ~210)
    result["doy_season"] = np.where(
        result["doy"] >= 274, result["doy"] - 274, result["doy"] + 91
    )  # days since Oct 1, continuous 0→302

    # GDD fraction: how much of this patch's total heat accumulation so far
    patch_gdd_total = result.groupby("patch_id")["gdd_cum"].transform("max")
    result["gdd_frac"] = np.where(
        patch_gdd_total > 0, result["gdd_cum"] / patch_gdd_total, 0.0
    )

    # NDVI relative position within each patch's observed range
    patch_ndvi_min = result.groupby("patch_id")["ndvi"].transform("min")
    patch_ndvi_max = result.groupby("patch_id")["ndvi"].transform("max")
    ndvi_range = patch_ndvi_max - patch_ndvi_min
    result["ndvi_rel"] = np.where(
        ndvi_range > 0.01,
        (result["ndvi"] - patch_ndvi_min) / ndvi_range,
        0.5,
    )

    # ---- Advanced interactions & senescence indicators ----

    # Nonlinear VPD response — atmospheric demand has diminishing returns
    if "vpd_kpa_mean_8d" in result.columns:
        vpd = result["vpd_kpa_mean_8d"]
        result["vpd_sq"] = vpd ** 2
        result["vpd_doy_season"] = vpd * result["doy_season"]
        result["vpd_gdd_frac"] = vpd * result["gdd_frac"]

    # Days since peak NDVI — signals senescence depth
    patch_ndvi_cummax_idx = result.groupby("patch_id")["ndvi"].transform(
        lambda s: s.expanding().apply(lambda x: x.argmax(), raw=False)
    )
    # For each patch, how many windows since the highest NDVI so far
    result["ndvi_peak_dist"] = result.groupby("patch_id").cumcount() - result.groupby("patch_id")["ndvi"].transform(
        lambda s: s.expanding().apply(lambda x: x.argmax(), raw=False)
    )
    result["ndvi_peak_dist"] = result["ndvi_peak_dist"].clip(lower=0)

    # NDVI decline rate — how fast vegetation is browning (8-day window diff)
    result["ndvi_decline"] = -result["ndvi_diff"]  # positive = declining
    result["ndvi_decline"] = result["ndvi_decline"].clip(lower=0)

    # LSWI change — water status trajectory (declining LSWI = water stress)
    if "lswi" in result.columns:
        lswi_lag1 = result.groupby("patch_id")["lswi"].shift(1).fillna(result["lswi"])
        result["lswi_diff"] = result["lswi"] - lswi_lag1

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
