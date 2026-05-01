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
    """Aggregate daily ERA5 weather to MOD16 8-day windows.

    Processes one unique 8-day window at a time to keep memory low.
    MOD16 windows are identical across patches, so we iterate over ~161
    unique windows rather than doing a cross-join per patch_id.
    """
    weather = _normalize_patch_id(era5_daily_df)
    weather["date"] = pd.to_datetime(weather["date"])
    weather["gdd_daily"] = np.maximum(weather["tmean_c"] - gdd_base_c, 0.0)

    windows = _normalize_patch_id(mod16_df)[
        ["patch_id", "date_start", "date_end", "date"]
    ].copy()
    windows["date_start"] = pd.to_datetime(windows["date_start"])
    windows["date_end"] = pd.to_datetime(windows["date_end"])
    windows["date"] = pd.to_datetime(windows["date"])
    windows = windows.drop_duplicates()

    # Build patch_id → set of valid dates for fast lookup
    patch_window_dates = windows.groupby("patch_id")["date"].apply(set).to_dict()

    # Unique (date_start, date_end, date) across all patches (~161 windows)
    unique_win = windows[["date_start", "date_end", "date"]].drop_duplicates()
    unique_win = unique_win.sort_values("date_start").reset_index(drop=True)

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

    aggregated_frames = []
    n_win = len(unique_win)
    for i, (_, win_row) in enumerate(unique_win.iterrows()):
        w_start = win_row["date_start"]
        w_end = win_row["date_end"]
        w_date = win_row["date"]

        # Filter weather rows within this 8-day window
        in_window = weather[
            (weather["date"] >= w_start) & (weather["date"] < w_end)
        ].copy()
        if len(in_window) == 0:
            continue

        # Only keep patches that have this MOD16 window
        valid_patches = {pid for pid, dates in patch_window_dates.items() if w_date in dates}
        in_window = in_window[in_window["patch_id"].isin(valid_patches)]

        if len(in_window) == 0:
            continue

        # Add window metadata
        in_window["date_start"] = w_start
        in_window["date_end"] = w_end
        in_window["window_date"] = w_date

        # Aggregate per patch
        agged = (
            in_window.groupby(["patch_id", "date_start", "date_end", "window_date"], as_index=False)
            .agg(**agg_spec)
        )
        aggregated_frames.append(agged)
        del in_window, agged

    if not aggregated_frames:
        return pd.DataFrame()

    result = pd.concat(aggregated_frames, ignore_index=True)
    result = result.rename(columns={"window_date": "date"})
    return result


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


def add_temporal_features(df: pd.DataFrame, crop_type: str = "winter_wheat") -> pd.DataFrame:
    result = df.sort_values(["patch_id", "date"]).copy()

    # Idempotent: skip if already computed
    if "greenup_doy" in result.columns:
        return result

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

    # ---- Per-patch greenup onset detection ----
    # Winter wheat: find the NDVI minimum in the dormancy period (doy 0-90,
    # ~Jan–Mar), then the first window after the minimum where NDVI exceeds
    # the dormancy floor by a clear margin and is still rising.
    # This replaces the fixed Oct-1 season reference with a data-driven one.

    is_wheat = crop_type == "winter_wheat"

    def _detect_greenup(grp: pd.DataFrame) -> pd.Series:
        grp = grp.sort_values("doy")
        ndvi = grp["ndvi"].values
        doy = grp["doy"].values
        n = len(ndvi)

        if is_wheat:
            # Wheat: find winter dormancy minimum (doy 1-90), then greenup
            dorm_mask = doy <= 90
            dorm_ndvi = ndvi[dorm_mask] if dorm_mask.any() else ndvi[:max(1, n // 4)]
            baseline = float(dorm_ndvi.min())
            min_idx = int(np.argmin(ndvi[dorm_mask]) if dorm_mask.any() else 0)
            threshold = baseline + 0.10
        else:
            # Summer maize: no dormancy. Baseline is early-season minimum.
            # Find first NDVI > 0.30 that keeps rising (germination->vegetative)
            baseline = float(ndvi[:max(1, n // 3)].min())
            min_idx = 0
            threshold = 0.30

        greenup_idx = n
        for i in range(min_idx, n - 1):
            if ndvi[i] > threshold and ndvi[i + 1] > ndvi[i]:
                greenup_idx = i
                break

        greenup_doy = float(doy[greenup_idx]) if greenup_idx < n else (91.0 if is_wheat else 170.0)
        greenup_ndvi = float(ndvi[greenup_idx]) if greenup_idx < n else float(baseline)
        greenup_gdd = float(grp["gdd_8d"].iloc[:greenup_idx+1].sum()) if greenup_idx < n else 0.0

        return pd.Series({
            "greenup_doy": greenup_doy,
            "greenup_ndvi": greenup_ndvi,
            "greenup_gdd_cum": greenup_gdd,
        })

    greenup_lookup = result.groupby("patch_id").apply(_detect_greenup).reset_index()
    result = result.merge(greenup_lookup, on="patch_id", how="left")
    result["days_since_greenup"] = result["doy"] - result["greenup_doy"]
    result["gdd_since_greenup"] = result.groupby("patch_id")["gdd_8d"].cumsum() - result["greenup_gdd_cum"]
    del result["greenup_gdd_cum"]

    # ---- Phenology features (greenup-referenced) ----
    # days_since_greenup: 0 at greenup, negative before (dormancy),
    # positive during growth/senescence. Clipped to avoid extreme values.
    result["days_since_greenup"] = result["days_since_greenup"].clip(-90, 250)

    # Keep old doy_season for backward compatibility (still useful as calendar ref)
    result["doy_season"] = np.where(
        result["doy"] >= 274, result["doy"] - 274, result["doy"] + 91
    )

    result["gdd_since_greenup"] = result["gdd_since_greenup"].clip(lower=0.0)

    # GDD fraction: heat accumulation relative to patch total (kept)
    patch_gdd_total = result.groupby("patch_id")["gdd_cum"].transform("max")
    result["gdd_frac"] = np.where(
        patch_gdd_total > 0, result["gdd_cum"] / patch_gdd_total, 0.0
    )

    # GDD fraction since greenup
    patch_gdd_greenup_total = result.groupby("patch_id")["gdd_since_greenup"].transform("max")
    result["gdd_frac_greenup"] = np.where(
        patch_gdd_greenup_total > 0,
        result["gdd_since_greenup"] / patch_gdd_greenup_total,
        0.0,
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
        # Greenup-referenced interactions
        result["vpd_days_greenup"] = vpd * result["days_since_greenup"]
        result["vpd_gdd_frac_greenup"] = vpd * result["gdd_frac_greenup"]

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
    merged = add_temporal_features(merged, crop_type=crop_type)
    merged = quality_control(merged)

    valid = merged[merged["qc_valid"]].copy()
    return merged, valid
