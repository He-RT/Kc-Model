"""Estimate current Kcact using the trained XGBoost model and near-real-time GEE data.

Usage:
  python scripts/python/estimate_current_kc.py --project-id chuang-yaogan
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import ee
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys
sys.path.insert(0, str(SRC))

from kcact.features.et0 import compute_et0_fao56


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--model-path",
                        default=str(ROOT / "outputs/models/kcact_xgb_20260424.pkl"))
    parser.add_argument("--n-points", type=int, default=5,
                        help="Number of sample points in Hebei winter wheat area")
    parser.add_argument("--gdd-base-c", type=float, default=0.0)
    return parser.parse_args()


def get_hebei_geometry():
    return (ee.FeatureCollection("FAO/GAUL/2015/level1")
            .filter(ee.Filter.eq("ADM0_NAME", "China"))
            .filter(ee.Filter.eq("ADM1_NAME", "Hebei Sheng"))
            .geometry())


def sample_winter_wheat_points(year: int, hebei: ee.Geometry, n: int) -> ee.FeatureCollection:
    """Get a handful of points using the same candidate mask logic (simplified)."""
    cropland = (ee.ImageCollection("ESA/WorldCover/v200").first()
                .select("Map").eq(40))
    pts = cropland.selfMask().sample(
        region=hebei, scale=5000, numPixels=max(n * 10, 100),
        seed=42, geometries=True, tileScale=4,
    )
    def annotate(f):
        coords = f.geometry().coordinates()
        return f.set({
            "point_id": ee.String("pt_").cat(f.id()),
            "centroid_lon": coords.get(0),
            "centroid_lat": coords.get(1),
        })
    return pts.map(annotate)


def extract_era5_daily(points: ee.FeatureCollection, start: str, end: str) -> pd.DataFrame:
    """Extract ERA5 daily aggregates for a set of points."""
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start, end)
            .sort("system:time_start"))
    image_list = era5.toList(era5.size())
    n = image_list.size().getInfo()
    rows = []
    for i in range(n):
        img = ee.Image(image_list.get(i))
        date_str = img.date().format("YYYY-MM-dd").getInfo()
        wind_10m = img.expression(
            "sqrt(u*u + v*v)",
            {"u": img.select("u_component_of_wind_10m"),
             "v": img.select("v_component_of_wind_10m")},
        ).rename("wind_10m_m_s")
        derived = ee.Image.cat([
            img.select("temperature_2m").subtract(273.15).rename("tmean_c"),
            img.select("temperature_2m_min").subtract(273.15).rename("tmin_c"),
            img.select("temperature_2m_max").subtract(273.15).rename("tmax_c"),
            img.select("dewpoint_temperature_2m").subtract(273.15).rename("dewpoint_c"),
            img.select("surface_solar_radiation_downwards_sum").divide(1e6).rename("solar_rad_mj_m2_d"),
            img.select("total_precipitation_sum").multiply(1000).rename("precip_mm"),
            img.select("surface_pressure").divide(1000).rename("pressure_kpa"),
            wind_10m,
        ])
        samples = derived.reduceRegions(
            collection=points, reducer=ee.Reducer.mean(), scale=11132, tileScale=4,
        ).getInfo()
        for f in samples["features"]:
            row = {"point_id": f["properties"]["point_id"], "date": date_str}
            row.update({k: v for k, v in f["properties"].items()
                       if k not in ("point_id", "system:index")})
            rows.append(row)
    return pd.DataFrame(rows)


def extract_s2_composite(points: ee.FeatureCollection, date_start: str, date_end: str,
                         hebei: ee.Geometry) -> pd.DataFrame:
    """Extract S2 composite for a single 8-day window."""
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(hebei)
          .filterDate(date_start, date_end)
          .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80)))

    def add_vis(img):
        scaled = img.select(["B2","B3","B4","B5","B8","B11"]).divide(10000)
        ndvi = scaled.normalizedDifference(["B8","B4"]).rename("ndvi")
        evi = scaled.expression(
            "2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1.0))",
            {"nir": scaled.select("B8"), "red": scaled.select("B4"),
             "blue": scaled.select("B2")},
        ).rename("evi")
        savi = scaled.expression(
            "1.5 * ((nir - red) / (nir + red + 0.5))",
            {"nir": scaled.select("B8"), "red": scaled.select("B4")},
        ).rename("savi")
        gndvi = scaled.normalizedDifference(["B8","B3"]).rename("gndvi")
        lswi = scaled.normalizedDifference(["B8","B11"]).rename("lswi")
        nirv = ndvi.multiply(scaled.select("B8")).rename("nirv")
        re_ndvi = scaled.normalizedDifference(["B8","B5"]).rename("re_ndvi")
        valid = img.select("B8").mask().rename("valid_obs")
        return img.addBands([ndvi,evi,savi,gndvi,lswi,nirv,re_ndvi,valid])

    bands = ["ndvi","evi","savi","gndvi","lswi","nirv","re_ndvi","valid_obs"]
    n = s2.size().getInfo()
    if n == 0:
        # fallback
        results = []
        pts_info = points.getInfo()
        for f in pts_info["features"]:
            row = {"point_id": f["properties"]["point_id"],
                   "date_start": date_start, "date_end": date_end, "date": date_end}
            for b in bands:
                row[b] = 0.0
            row["obs_count_s2"] = 0
            row["valid_obs"] = 0
            results.append(row)
        return pd.DataFrame(results)

    composite = s2.map(add_vis).select(bands).mean()
    obs_count = s2.map(add_vis).select("valid_obs").sum().rename("obs_count_s2")
    stack = composite.addBands(obs_count)

    samples = stack.reduceRegions(
        collection=points, reducer=ee.Reducer.mean(),
        scale=10, tileScale=4,
    ).getInfo()

    results = []
    for f in samples["features"]:
        row = {"point_id": f["properties"]["point_id"],
               "date_start": date_start, "date_end": date_end, "date": date_end}
        row.update({k: v for k, v in f["properties"].items()
                   if k not in ("point_id", "system:index")})
        results.append(row)
    return pd.DataFrame(results)


def compute_8day_features(era5_df: pd.DataFrame, s2_df: pd.DataFrame,
                          gdd_base_c: float = 0.0) -> pd.DataFrame:
    """Compute all features needed for the model from raw ERA5+S2 data."""
    # Compute ET0 from daily ERA5
    era5_daily = pd.DataFrame(era5_df)
    era5_daily["date"] = pd.to_datetime(era5_daily["date"])
    # Ensure required columns for compute_et0_fao56
    for col in ["centroid_lat", "centroid_lon"]:
        if col not in era5_daily.columns:
            era5_daily[col] = 37.5
    era5_with_et0 = compute_et0_fao56(era5_daily)

    # Merge S2地望着
    s2 = pd.DataFrame(s2_df)
    for c in ["date_start", "date_end", "date"]:
        s2[c] = pd.to_datetime(s2[c])

    # Create 8-day windows
    era5_with_et0["gdd_daily"] = np.maximum(era5_with_et0["tmean_c"] - gdd_base_c, 0.0)
    era5_with_et0["date_start"] = pd.to_datetime(s2["date_start"].iloc[0])
    era5_with_et0["date_end"] = pd.to_datetime(s2["date_end"].iloc[0])

    # Filter to window
    window_dates = (era5_with_et0["date"] >= era5_with_et0["date_start"]) & \
                   (era5_with_et0["date"] < era5_with_et0["date_end"])
    window_data = era5_with_et0[window_dates]

    # Aggregate
    agg = window_data.groupby("point_id").agg(
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
    ).reset_index()

    # Merge S2
    s2_meta = s2.drop(columns=["centroid_lat", "centroid_lon"], errors="ignore")
    merged = agg.merge(s2_meta, on="point_id", how="left")

    # Add temporal features
    merged["date"] = pd.to_datetime(s2["date"].iloc[0])
    merged["year"] = merged["date"].dt.year
    merged["doy"] = merged["date"].dt.dayofyear

    # For real-time: these would need historical data. We compute rolling features
    # from all previous windows for each point. Since we only have one window here,
    # we use placeholder values.
    merged["gdd_cum"] = merged["gdd_8d"]  # Simplified for single-window
    merged["precip_7d"] = merged["precip_mm_8d"]
    merged["precip_15d"] = merged["precip_mm_8d"]
    merged["precip_30d"] = merged["precip_mm_8d"]
    merged["ndvi_lag1"] = merged["ndvi"]  # No prior window available
    merged["ndvi_mean_prev_3win"] = merged["ndvi"]

    # Add missing_columns that the model expects
    for col in ["season_year", "winter_wheat_candidate", "obs_count_s2",
                ".geo", "system:index", "valid_obs_s2", "area_ha", "elevation_m"]:
        if col not in merged.columns:
            merged[col] = 0.0

    return merged


def predict_kc(model_path: str, features_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Load model and predict Kcact."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    expected = bundle["feature_cols"]

    # Align columns
    X = features_df.copy()
    for col in expected:
        if col not in X.columns:
            X[col] = 0.0
    X = X[expected].fillna(0.0)

    y_pred = model.predict(X)
    result = features_df[["point_id", "date", "centroid_lat", "centroid_lon"]].copy()
    result["kcact_predicted"] = y_pred
    result["kcact_predicted"] = result["kcact_predicted"].clip(0.01, 2.0)
    return result


def main():
    args = parse_args()
    ee.Initialize(project=args.project_id)

    hebei = get_hebei_geometry()
    print("Sampling points in Hebei cropland...")
    points = sample_winter_wheat_points(2026, hebei, args.n_points)

    # Current window: April 16-23, 2026 (most recent complete 8-day window with ERA5)
    date_start = "2026-04-16"
    date_end = "2026-04-24"

    print(f"Extracting ERA5 daily data (2025-10-01 to {date_end})...")
    era5_data = extract_era5_daily(points, "2025-10-01", date_end)
    print(f"  Got {len(era5_data)} ERA5 rows")

    print(f"Extracting S2 composite ({date_start} to {date_end})...")
    s2_data = extract_s2_composite(points, date_start, date_end, hebei)
    print(f"  Got {len(s2_data)} S2 rows")

    print("Computing features...")
    features = compute_8day_features(era5_data, s2_data, args.gdd_base_c)
    print(f"  Features: {features.columns.tolist()}")

    print("Predicting Kcact...")
    results = predict_kc(args.model_path, features, [])
    print("\n=== Predicted Kcact for Hebei winter wheat (Apr 16-23, 2026) ===")
    for _, row in results.iterrows():
        print(f"  {row['point_id']} ({row['centroid_lat']:.2f}N, {row['centroid_lon']:.2f}E): "
              f"Kcact = {row['kcact_predicted']:.4f}")
    print(f"\n  Mean Kcact: {results['kcact_predicted'].mean():.4f} ± {results['kcact_predicted'].std():.4f}")


if __name__ == "__main__":
    main()
