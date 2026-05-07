#!/usr/bin/env python3
"""
Compute ET0 from ERA5-Land for 4 flux station sites (Yucheng, Weishan, Guantao, Luancheng),
then merge with observed ETc to compute Kcact.

ERA5-Land daily data fetched via GEE Python API, then ET0 computed locally
using the project's existing FAO-56 Penman-Monteith implementation.

Output: data/processed/station_etc_with_et0.csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kcact.features.et0 import compute_et0_fao56

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBS_PATH = PROJECT_ROOT / "data" / "processed" / "station_etc_observations.csv"
STN_PATH = PROJECT_ROOT / "data" / "processed" / "station_coordinates.csv"
ERA5_RAW_PATH = PROJECT_ROOT / "data" / "processed" / "station_era5_daily_raw.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "station_etc_with_et0.csv"

ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
SRTM = "USGS/SRTMGL1_003"

# ---------------------------------------------------------------------------
# 1. GEE: fetch ERA5-Land daily data for station points
# ---------------------------------------------------------------------------


def init_gee(project_id: str = "chuang-yaogan") -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def get_station_points(stn_df: pd.DataFrame) -> ee.FeatureCollection:
    features = []
    for _, row in stn_df.iterrows():
        pt = ee.Geometry.Point([row["lon"], row["lat"]])
        features.append(
            ee.Feature(pt, {"station": row["station"], "lat": row["lat"], "lon": row["lon"]})
        )
    return ee.FeatureCollection(features)


def get_elevation_for_stations(stn_df: pd.DataFrame) -> dict[str, float]:
    points = get_station_points(stn_df)
    elev_img = ee.Image(SRTM)
    reduced = elev_img.reduceRegions(
        collection=points, reducer=ee.Reducer.mean(), scale=90
    )
    feats = reduced.getInfo()["features"]
    return {
        f["properties"]["station"]: float(f["properties"].get("elevation", 0))
        for f in feats
    }


def _make_daily_image(img: ee.Image) -> ee.Image:
    """Extract and rename bands from one ERA5-Land daily image."""
    tmean = img.select("temperature_2m").subtract(273.15)
    wind_10m = img.expression(
        "sqrt(u*u + v*v)",
        {
            "u": img.select("u_component_of_wind_10m"),
            "v": img.select("v_component_of_wind_10m"),
        },
    )
    return ee.Image.cat([
        tmean.rename("tmean_c"),
        img.select("temperature_2m_min").subtract(273.15).rename("tmin_c"),
        img.select("temperature_2m_max").subtract(273.15).rename("tmax_c"),
        img.select("dewpoint_temperature_2m").subtract(273.15).rename("dewpoint_c"),
        img.select("surface_solar_radiation_downwards_sum")
        .divide(1e6)
        .rename("solar_rad_mj_m2_d"),
        img.select("total_precipitation_sum").multiply(1000).rename("precip_mm"),
        img.select("surface_pressure").divide(1000).rename("pressure_kpa"),
        wind_10m.rename("wind_10m_m_s"),
    ]).set("system:time_start", img.get("system:time_start"))


def _reduce_and_annotate(img: ee.Image, points: ee.FeatureCollection) -> ee.FeatureCollection:
    date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
    reduced = img.reduceRegions(
        collection=points, reducer=ee.Reducer.mean(), scale=11132, tileScale=4
    )
    return reduced.map(lambda f: f.set({"date": date_str}))


def fetch_era5_daily_batch(
    points: ee.FeatureCollection, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch one batch (e.g. one year) of ERA5-Land daily data."""
    col = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(start_date, end_date)
        .select([
            "temperature_2m", "temperature_2m_min", "temperature_2m_max",
            "dewpoint_temperature_2m", "surface_solar_radiation_downwards_sum",
            "total_precipitation_sum", "surface_pressure",
            "u_component_of_wind_10m", "v_component_of_wind_10m",
        ])
    )

    n_imgs = col.size().getInfo()
    if n_imgs == 0:
        return pd.DataFrame()
    print(f"  {start_date} → {end_date}: {n_imgs} images")

    def map_fn(img):
        return _reduce_and_annotate(_make_daily_image(img), points)

    fc = ee.FeatureCollection(col.map(map_fn).flatten())
    geojson = fc.getInfo()

    rows = []
    for feat in geojson.get("features", []):
        props = feat["properties"]
        rows.append({k: props.get(k) for k in props})

    return pd.DataFrame(rows)


def fetch_all_era5_data(
    stn_df: pd.DataFrame, years: range
) -> pd.DataFrame:
    """Fetch ERA5-Land daily data for all stations, year by year."""
    points = get_station_points(stn_df)
    all_frames = []
    for yr in years:
        df = fetch_era5_daily_batch(points, f"{yr}-01-01", f"{yr+1}-01-01")
        if len(df) > 0:
            all_frames.append(df)
    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Compute ET0 and merge with observations
# ---------------------------------------------------------------------------


# =========================================================
# 步骤2: 计算逐日ET0并匹配到观测窗口
# =========================================================
def compute_et0_and_merge(
    era5_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    stn_df: pd.DataFrame,
    elevations: dict[str, float],
) -> pd.DataFrame:
    """Compute daily ET0, then match to observation periods."""

    # ---- 2a. Prepare daily weather DataFrame ----
    weather = era5_df.copy()
    # Merge station lat/lon/elev
    stn_lookup = stn_df.set_index("station")
    weather["centroid_lat"] = weather["station"].map(stn_lookup["lat"].to_dict())
    weather["centroid_lon"] = weather["station"].map(stn_lookup["lon"].to_dict())
    weather["elevation_m"] = weather["station"].map(elevations)
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.dropna(subset=["centroid_lat"])

    # ---- 2b. Compute daily ET0 per station ----
    et0_frames = []
    for stn_name, grp in weather.groupby("station"):
        grp = grp.sort_values("date")
        try:
            with_et0 = compute_et0_fao56(grp)
            et0_frames.append(with_et0[["station", "date", "et0_pm_mm"]])
        except Exception as exc:
            print(f"  WARNING: ET0 computation failed for {stn_name}: {exc}")
    et0_daily = pd.concat(et0_frames, ignore_index=True).sort_values(["station", "date"])

    # ---- 2c. Match ET0 to observation windows ----
    # Each observation at date t represents mean ET over (t_prev, t].
    # We average daily ET0 over the same window, then Kcact = ETc_obs / mean(ET0).
    obs = obs_df.copy()
    obs["date"] = pd.to_datetime(obs["date"])

    matched = []
    for stn_name in obs["station"].unique():
        stn_obs = obs[obs["station"] == stn_name].sort_values("date")
        stn_et0 = et0_daily[et0_daily["station"] == stn_name].sort_values("date").set_index("date")

        for i, (_, row) in enumerate(stn_obs.iterrows()):
            t_curr = row["date"]
            etc_obs = row["etc_8d_mm_d"]

            # ---- 确定观测窗口 ----
            # 窗口: (上一个观测日, 当前观测日]
            # 第一个观测回退8天作为名义窗口
            if i == 0:
                t_prev = t_curr - pd.Timedelta(days=8)
            else:
                t_prev = stn_obs.iloc[i - 1]["date"]

            # ---- 时区修正 ----
            # 站点观测时间是北京时间 (UTC+8)
            # ERA5-Land数据日期是UTC时间
            # 例如: 观测"2003-01-01 00:00"北京时间 = "2002-12-31 16:00" UTC
            # 所以匹配时减8小时: 本地时间 → UTC
            TZ_OFFSET = pd.Timedelta(hours=8)
            t_prev_utc = t_prev - TZ_OFFSET
            t_curr_utc = t_curr - TZ_OFFSET

            # Mean daily ET0 over the observation window in UTC
            window_et0 = stn_et0.loc[(stn_et0.index > t_prev_utc) & (stn_et0.index <= t_curr_utc)]
            if len(window_et0) > 0 and etc_obs > 0:
                et0_mean = window_et0["et0_pm_mm"].mean()
                et0_sum = window_et0["et0_pm_mm"].sum()
                n_days = len(window_et0)
            else:
                # Fall back to daily ET0 on exact date
                if t_curr in stn_et0.index:
                    et0_mean = stn_et0.loc[t_curr, "et0_pm_mm"]
                    et0_sum = et0_mean * 1
                    n_days = 1
                else:
                    et0_mean = np.nan
                    et0_sum = np.nan
                    n_days = 0

            matched.append({
                "station": stn_name,
                "date": t_curr.strftime("%Y-%m-%d"),
                "date_prev": t_prev.strftime("%Y-%m-%d"),
                "etc_obs_mm_d": etc_obs,
                "et0_pm_mean_mm_d": round(et0_mean, 6) if not np.isnan(et0_mean) else np.nan,
                "et0_pm_sum_mm": round(et0_sum, 6) if not np.isnan(et0_sum) else np.nan,
                "n_days_window": n_days,
            })

    result = pd.DataFrame(matched)
    result["kcact"] = np.where(
        result["et0_pm_mean_mm_d"].notna() & (result["et0_pm_mean_mm_d"] > 0.01),
        result["etc_obs_mm_d"] / result["et0_pm_mean_mm_d"],
        np.nan,
    )
    result["kcact"] = result["kcact"].round(6)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """站点ET0计算主流程

    数据流:
      站点观测Excel → 整洁CSV
      GEE提取ERA5-Land逐日气象 → 本地CSV(已缓存)
      ERA5逐日数据 → FAO-56 PM公式 → 逐日ET0
      逐日ET0 → 按观测窗口聚合 → 与ETc匹配 → Kcact = ETc/ET0
    """
    print("=" * 60)
    print("Station ET0 Computation Pipeline")
    print("=" * 60)

    # Load station metadata and observations
    stn_df = pd.read_csv(STN_PATH)
    obs_df = pd.read_csv(OBS_PATH)

    print(f"\nStations: {stn_df['station'].tolist()}")
    print(f"Observations: {len(obs_df)} records")
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    obs_yr_min, obs_yr_max = obs_df["date"].dt.year.min(), obs_df["date"].dt.year.max()
    print(f"Date range: {obs_yr_min} – {obs_yr_max}")

    # ---- Fetch ERA5 data ----
    print("\n--- Fetching ERA5-Land daily data ---")
    init_gee()

    # Get station elevations from SRTM
    print("Getting elevations from SRTM...")
    elevations = get_elevation_for_stations(stn_df)
    for stn, elev in elevations.items():
        print(f"  {stn}: {elev:.1f} m")

    # Fetch ERA5 data year by year
    era5_path = ERA5_RAW_PATH
    if era5_path.exists():
        print(f"\nLoading cached ERA5 data from {era5_path}")
        era5_daily = pd.read_csv(era5_path)
    else:
        era5_daily = fetch_all_era5_data(stn_df, range(obs_yr_min, obs_yr_max + 1))
        era5_daily.to_csv(era5_path, index=False)
        print(f"Saved raw ERA5 data to {era5_path} ({len(era5_daily)} rows)")

    print(f"ERA5 daily rows: {len(era5_daily)}")

    # ---- Compute ET0 and merge ----
    print("\n--- Computing FAO-56 ET0 and matching to observations ---")
    result = compute_et0_and_merge(era5_daily, obs_df, stn_df, elevations)

    # ---- 保存结果 ----
    # Kcact: 作物实际系数 = ETc(实测) / ET0(计算)
    # Kc > 1: 作物蒸腾超过草地参照 → 作物旺盛生长期
    # Kc < 0.5: 裸土/休眠期/收割后
    # 理论范围: 0–1.2 (实际可能略超)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH} ({len(result)} rows)")

    # ---- 输出概要 ----
    valid = result[result["kcact"].notna()]
    print(f"\nValid Kcact records: {len(valid)} / {len(result)}")
    print(f"Kcact range: {valid['kcact'].min():.4f} – {valid['kcact'].max():.4f}")
    print(f"Kcact mean: {valid['kcact'].mean():.4f}, median: {valid['kcact'].median():.4f}")

    # Per-station summary
    print("\n--- Per-station Kcact summary ---")
    for stn in sorted(result["station"].unique()):
        s = result[result["station"] == stn]
        sv = s[s["kcact"].notna()]
        print(f"  {stn}: {len(sv)}/{len(s)} valid, "
              f"Kcact mean={sv['kcact'].mean():.4f}, "
              f"ET0 mean={sv['et0_pm_mean_mm_d'].mean():.2f} mm/d")

    return result


if __name__ == "__main__":
    main()
