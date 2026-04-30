#!/usr/bin/env python3
"""
Extract 8 RS indicators at 4 flux station locations for ML training.

Indicators (all MODIS-era compatible, 2003–2015):
  1. NDVI          — MODIS MOD13Q1 (16-day, 250m)
  2. EVI           — MODIS MOD13Q1
  3. LSWI          — computed from MODIS MOD09A1 (b02=NIR, b06=SWIR1)
  4. LST_Day       — MODIS MOD11A2 (8-day, 1km)
  5. LST_Night     — MODIS MOD11A2  → ΔLST computed locally
  6. Albedo_sw     — MODIS MCD43A3 (daily, 500m) → 8-day mean
  7. fPAR          — MODIS MOD15A2H (8-day, 500m)
  8. SM_surface    — ERA5-Land DAILY_AGGR (0–7cm)

Output: data/processed/station_ml_features.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBS_PATH = PROJECT_ROOT / "data" / "processed" / "station_etc_observations.csv"
STN_PATH = PROJECT_ROOT / "data" / "processed" / "station_coordinates.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "station_ml_features.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_gee():
    try:
        ee.Initialize(project="chuang-yaogan")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="chuang-yaogan")


def get_points(stn_df: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for _, r in stn_df.iterrows():
        feats.append(ee.Feature(
            ee.Geometry.Point([r["lon"], r["lat"]]),
            {"station": r["station"]},
        ))
    return ee.FeatureCollection(feats)


def fc_to_df(fc: ee.FeatureCollection) -> pd.DataFrame:
    rows = []
    for f in fc.getInfo()["features"]:
        rows.append(f["properties"].copy())
    return pd.DataFrame(rows)


def fetch_collection_batched(
    collection_id: str,
    base_bands: list[str],
    bands_out: list[str],
    points: ee.FeatureCollection,
    start_yr: int, end_yr: int,
    scale: int = 500,
    extra_bands_fn=None,
) -> pd.DataFrame:
    """Fetch a GEE ImageCollection year by year, reducing to station points.

    base_bands  — bands to select from the *original* images (must exist at source)
    bands_out  — final output column names (base + any derived bands)
    """
    frames = []
    for yr in range(start_yr, end_yr + 1):
        col = (ee.ImageCollection(collection_id)
               .filterDate(f"{yr}-01-01", f"{yr+1}-01-01")
               .select(base_bands))
        n = col.size().getInfo()
        if n == 0:
            continue

        def map_fn(img):
            if extra_bands_fn:
                img = extra_bands_fn(img)
                # Derived bands have their final names, just select them
                img_sub = img.select(bands_out)
            else:
                # Rename base_bands → bands_out
                img_sub = img.select(base_bands, bands_out)

            date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")

            # For single-band images, Reducer.mean() outputs property "mean".
            # For multi-band images, each band gets its own property.
            # Normalize: ensure all bands_out names appear as properties.
            if len(bands_out) == 1:
                reduced = img_sub.reduceRegions(
                    collection=points, reducer=ee.Reducer.mean(),
                    scale=scale, tileScale=4,
                )
                band_name = bands_out[0]
                reduced = reduced.map(
                    lambda f: ee.Feature(f).set(band_name, f.get("mean"))
                )
            else:
                reduced = img_sub.reduceRegions(
                    collection=points, reducer=ee.Reducer.mean(),
                    scale=scale, tileScale=4,
                )
            return reduced.map(lambda f: f.set({"date": date_str}))

        fc = ee.FeatureCollection(col.map(map_fn).flatten())
        df = fc_to_df(fc)
        if len(df) > 0:
            frames.append(df)
        print(f"  {collection_id.split('/')[-1]}: {yr} → {len(df)} rows")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Product-specific extractors
# ---------------------------------------------------------------------------

def extract_mod13q1(points, start_yr, end_yr):
    """NDVI, EVI from MODIS 16-day VI."""
    return fetch_collection_batched(
        "MODIS/061/MOD13Q1",
        ["NDVI", "EVI"],  # base_bands
        ["ndvi", "evi"],  # bands_out
        points, start_yr, end_yr, scale=250,
    )


def extract_mod09a1_lswi(points, start_yr, end_yr):
    """LSWI = (b02 − b06) / (b02 + b06) from MODIS 8-day surface reflectance."""
    def add_lswi(img):
        lswi = img.normalizedDifference(["sur_refl_b02", "sur_refl_b06"]).rename("lswi")
        return img.addBands(lswi)

    return fetch_collection_batched(
        "MODIS/061/MOD09A1",
        ["sur_refl_b02", "sur_refl_b06"],        # base_bands (exist in source)
        ["lswi"],                                 # bands_out (only keep derived)
        points, start_yr, end_yr, scale=500,
        extra_bands_fn=add_lswi,
    )


def extract_mod11a2(points, start_yr, end_yr):
    """LST Day/Night from MODIS 8-day LST."""
    return fetch_collection_batched(
        "MODIS/061/MOD11A2",
        ["LST_Day_1km", "LST_Night_1km"],
        ["lst_day", "lst_night"],
        points, start_yr, end_yr, scale=1000,
    )


def extract_mcd43a3_albedo(points, start_yr, end_yr):
    """Shortwave broadband albedo from MODIS daily BRDF/Albedo."""
    return fetch_collection_batched(
        "MODIS/061/MCD43A3",
        ["Albedo_WSA_shortwave"],
        ["albedo_sw"],
        points, start_yr, end_yr, scale=500,
    )


def extract_mod15a2h_fpar(points, start_yr, end_yr):
    """fPAR from MODIS 8-day LAI/fPAR."""
    return fetch_collection_batched(
        "MODIS/061/MOD15A2H",
        ["Fpar_500m"],
        ["fpar"],
        points, start_yr, end_yr, scale=500,
    )


def extract_era5_sm(points, start_yr, end_yr):
    """ERA5-Land surface soil moisture (daily)."""
    return fetch_collection_batched(
        "ECMWF/ERA5_LAND/DAILY_AGGR",
        ["volumetric_soil_water_layer_1"],
        ["sm_surface"],
        points, start_yr, end_yr, scale=11132,
    )


# ---------------------------------------------------------------------------
# Merge with observations
# ---------------------------------------------------------------------------

def build_merged_dataset(obs_df, stn_df, all_ts):
    """
    Merge all RS time series with 8-day observation windows.
    Each obs at date t represents period (t_prev, t].
    We take the mean of each indicator over that window.
    """
    obs = obs_df.copy()
    obs["date"] = pd.to_datetime(obs["date"])

    # For each station, determine window (t_prev, t_curr]
    rows = []
    for stn in obs["station"].unique():
        stn_obs = obs[obs["station"] == stn].sort_values("date")
        for i, (_, ob) in enumerate(stn_obs.iterrows()):
            t_curr = ob["date"]
            t_prev = stn_obs.iloc[i - 1]["date"] if i > 0 else t_curr - pd.Timedelta(days=8)

            row = {
                "station": stn,
                "date": t_curr.strftime("%Y-%m-%d"),
                "date_prev": t_prev.strftime("%Y-%m-%d"),
                "etc_obs_mm_d": ob["etc_8d_mm_d"],
            }

            # For each RS indicator, average values in (t_prev, t_curr]
            for col, ts_df in all_ts.items():
                if stn not in ts_df["station"].values:
                    row[col] = np.nan
                    continue
                stn_ts = ts_df[ts_df["station"] == stn].copy()
                stn_ts["date"] = pd.to_datetime(stn_ts["date"])
                window = stn_ts[
                    (stn_ts["date"] > t_prev) & (stn_ts["date"] <= t_curr)
                ][col]
                if len(window) > 0:
                    row[col] = window.mean()
                else:
                    # Fallback: nearest date within ±4 days
                    nearest = stn_ts.iloc[
                        (stn_ts["date"] - t_curr).abs().argsort()[:1]
                    ]
                    if len(nearest) > 0 and abs((nearest["date"].iloc[0] - t_curr).days) <= 4:
                        row[col] = nearest[col].iloc[0]
                    else:
                        row[col] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Extracting 8 RS indicators for station ML training")
    print("=" * 60)

    obs_df = pd.read_csv(OBS_PATH)
    stn_df = pd.read_csv(STN_PATH)
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    yr_min, yr_max = obs_df["date"].dt.year.min(), obs_df["date"].dt.year.max()
    print(f"Stations: {stn_df['station'].tolist()}")
    print(f"Date range: {yr_min}–{yr_max}")
    print(f"Observations: {len(obs_df)}")

    init_gee()
    points = get_points(stn_df)

    # Check for cached extracts
    cache_dir = PROJECT_ROOT / "data" / "processed" / "modis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_ts = {}

    # 1. NDVI, EVI
    cache = cache_dir / "mod13q1.csv"
    if cache.exists():
        print("\n[1/6] MOD13Q1 (NDVI, EVI) — cached")
        all_ts["ndvi"] = all_ts["evi"] = pd.read_csv(cache)
    else:
        print("\n[1/6] MOD13Q1 (NDVI, EVI)...")
        df_vi = extract_mod13q1(points, yr_min, yr_max)
        df_vi["ndvi"] = df_vi["ndvi"].astype(float) * 0.0001  # scale factor
        df_vi["evi"] = df_vi["evi"].astype(float) * 0.0001
        df_vi.to_csv(cache, index=False)
        all_ts["ndvi"] = all_ts["evi"] = df_vi

    # 2. LSWI (from MOD09A1)
    cache = cache_dir / "mod09a1_lswi.csv"
    if cache.exists():
        print("[2/6] MOD09A1 (LSWI) — cached")
        all_ts["lswi"] = pd.read_csv(cache)
    else:
        print("[2/6] MOD09A1 (LSWI)...")
        df_lswi = extract_mod09a1_lswi(points, yr_min, yr_max)
        df_lswi["lswi"] = df_lswi["lswi"].astype(float)
        df_lswi.to_csv(cache, index=False)
        all_ts["lswi"] = df_lswi

    # 3. LST
    cache = cache_dir / "mod11a2_lst.csv"
    if cache.exists():
        print("[3/6] MOD11A2 (LST) — cached")
        df_lst = pd.read_csv(cache)
    else:
        print("[3/6] MOD11A2 (LST Day/Night)...")
        df_lst = extract_mod11a2(points, yr_min, yr_max)
        df_lst["lst_day"] = df_lst["lst_day"].astype(float) * 0.02  # K scale
        df_lst["lst_night"] = df_lst["lst_night"].astype(float) * 0.02
        df_lst.to_csv(cache, index=False)
    all_ts["lst_day"] = all_ts["lst_night"] = df_lst

    # 4. Albedo
    cache = cache_dir / "mcd43a3_albedo.csv"
    if cache.exists():
        print("[4/6] MCD43A3 (Albedo) — cached")
        all_ts["albedo_sw"] = pd.read_csv(cache)
    else:
        print("[4/6] MCD43A3 (Albedo)...")
        df_alb = extract_mcd43a3_albedo(points, yr_min, yr_max)
        df_alb["albedo_sw"] = df_alb["albedo_sw"].astype(float)
        df_alb.to_csv(cache, index=False)
        all_ts["albedo_sw"] = df_alb

    # 5. fPAR
    cache = cache_dir / "mod15a2h_fpar.csv"
    if cache.exists():
        print("[5/6] MOD15A2H (fPAR) — cached")
        all_ts["fpar"] = pd.read_csv(cache)
    else:
        print("[5/6] MOD15A2H (fPAR)...")
        df_fpar = extract_mod15a2h_fpar(points, yr_min, yr_max)
        df_fpar["fpar"] = df_fpar["fpar"].astype(float) * 0.01  # scale factor
        df_fpar.to_csv(cache, index=False)
        all_ts["fpar"] = df_fpar

    # 6. SM (ERA5-Land)
    cache = cache_dir / "era5_sm_all.csv"
    if cache.exists():
        print("[6/6] ERA5-Land SM — cached")
        all_ts["sm_surface"] = pd.read_csv(cache)
    else:
        print("[6/6] ERA5-Land SM...")
        df_sm = extract_era5_sm(points, yr_min, yr_max)
        df_sm["sm_surface"] = df_sm["sm_surface"].astype(float)
        df_sm.to_csv(cache, index=False)
        all_ts["sm_surface"] = df_sm

    # ---- Merge all features with observations ----
    print("\n--- Merging features with observation windows ---")
    result = build_merged_dataset(obs_df, stn_df, all_ts)

    # Add derived: ΔLST
    result["delta_lst"] = result["lst_day"] - result["lst_night"]

    # Keep only the 8 target columns + meta
    feature_cols = ["ndvi", "evi", "lswi", "lst_day", "albedo_sw", "fpar", "delta_lst", "sm_surface"]
    meta_cols = ["station", "date", "date_prev", "etc_obs_mm_d"]
    result = result[meta_cols + feature_cols]

    # Load ET0 for target Kcact
    etc_et0 = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "station_etc_with_et0.csv")
    etc_et0["date"] = pd.to_datetime(etc_et0["date"])
    result["date_dt"] = pd.to_datetime(result["date"])
    result = result.merge(
        etc_et0[["station", "date", "et0_pm_mean_mm_d", "kcact"]].rename(columns={"date": "date_dt"}),
        on=["station", "date_dt"], how="left",
    )
    result = result.drop(columns=["date_dt"])

    # Drop rows missing > 4 features
    result["n_missing"] = result[feature_cols].isna().sum(axis=1)
    result_valid = result[result["n_missing"] <= 4].copy()
    result_valid = result_valid.drop(columns=["n_missing"])

    result.to_csv(OUT_PATH, index=False)
    print(f"\nFull dataset: {len(result)} rows → {OUT_PATH}")
    valid_f = result_valid.dropna(subset=["kcact"])
    print(f"Complete rows (all features + Kcact): {len(valid_f)} / {len(result)}")

    # Feature coverage stats
    print("\n--- Feature coverage ---")
    for c in feature_cols:
        n_valid = result[c].notna().sum()
        print(f"  {c}: {n_valid}/{len(result)} ({n_valid/len(result)*100:.1f}%)")

    return result_valid


if __name__ == "__main__":
    main()
