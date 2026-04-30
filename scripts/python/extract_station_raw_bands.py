#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import ee
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "modis_cache"
OUT_PATH = CACHE_DIR / "mod09a1_raw_bands.csv"

STATIONS = [
    ("yucheng", 36.829, 116.5702),
    ("weishan", 36.6493, 116.059),
    ("guantao", 36.517, 115.133),
    ("luancheng", 37.884, 114.689),
]

BANDS = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04",
         "sur_refl_b05", "sur_refl_b06", "sur_refl_b07"]
SCALE = 0.0001


def init_gee():
    try:
        ee.Initialize(project="chuang-yaogan")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="chuang-yaogan")


def get_points():
    feats = []
    for name, lat, lon in STATIONS:
        feats.append(ee.Feature(ee.Geometry.Point([lon, lat]), {"station": name}))
    return ee.FeatureCollection(feats)


def fc_to_df(fc):
    rows = []
    for f in fc.getInfo()["features"]:
        rows.append(f["properties"].copy())
    return pd.DataFrame(rows)


def fetch_collection_batched(points, start_yr, end_yr):
    frames = []
    for yr in range(start_yr, end_yr + 1):
        col = (ee.ImageCollection("MODIS/061/MOD09A1")
               .filterDate(f"{yr}-01-01", f"{yr+1}-01-01")
               .select(BANDS))
        n = col.size().getInfo()
        if n == 0:
            continue

        def map_fn(img):
            date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            reduced = img.reduceRegions(
                collection=points, reducer=ee.Reducer.mean(),
                scale=500, tileScale=4,
            )
            return reduced.map(lambda f: f.set({"date": date_str}))

        fc = ee.FeatureCollection(col.map(map_fn).flatten())
        df = fc_to_df(fc)
        if len(df) > 0:
            frames.append(df)
        print(f"  {yr}: {len(df)} rows")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    if OUT_PATH.exists():
        print(f"Cached: {OUT_PATH}")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing GEE...")
    init_gee()

    points = get_points()
    print(f"Stations: {[s[0] for s in STATIONS]}")
    print(f"Bands: {BANDS}")
    print(f"Range: 2003–2015")

    df = fetch_collection_batched(points, 2003, 2015)
    if len(df) == 0:
        print("No data extracted.")
        return

    for b in BANDS:
        if b in df.columns:
            df[b] = df[b].astype(float) * SCALE

    cols = ["station", "date"] + BANDS
    df = df[cols].sort_values(["station", "date"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
