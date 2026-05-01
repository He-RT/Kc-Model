#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import ee
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "modis_cache" / "era5_weather_all.csv"

STATIONS = [
    ("禹城", 36.829, 116.570),
    ("位山", 36.649, 116.059),
    ("馆陶", 36.517, 115.133),
    ("栾城", 37.884, 114.689),
]

RAW_BANDS = [
    "temperature_2m",
    "temperature_2m_min",
    "temperature_2m_max",
    "dewpoint_temperature_2m",
    "surface_solar_radiation_downwards_sum",
    "total_precipitation_sum",
    "surface_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]

OUT_BANDS = [
    "tmean_c",
    "tmin_c",
    "tmax_c",
    "dewpoint_c",
    "solar_rad_mj_m2_d",
    "precip_mm",
    "pressure_kpa",
    "wind_10m_m_s",
]


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


def transform_era5(img):
    tmean = img.select("temperature_2m").subtract(273.15).rename("tmean_c")
    tmin = img.select("temperature_2m_min").subtract(273.15).rename("tmin_c")
    tmax = img.select("temperature_2m_max").subtract(273.15).rename("tmax_c")
    dew = img.select("dewpoint_temperature_2m").subtract(273.15).rename("dewpoint_c")
    solar = img.select("surface_solar_radiation_downwards_sum").divide(1e6).rename("solar_rad_mj_m2_d")
    precip = img.select("total_precipitation_sum").multiply(1000).rename("precip_mm")
    press = img.select("surface_pressure").divide(1000).rename("pressure_kpa")
    u = img.select("u_component_of_wind_10m")
    v = img.select("v_component_of_wind_10m")
    wind = u.pow(2).add(v.pow(2)).sqrt().rename("wind_10m_m_s")
    return ee.Image.cat([tmean, tmin, tmax, dew, solar, precip, press, wind]).copyProperties(
        img, ["system:time_start"]
    )


def extract_era5_weather(points, start_yr, end_yr):
    frames = []
    for yr in range(start_yr, end_yr + 1):
        col = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(f"{yr}-01-01", f"{yr+1}-01-01")
            .select(RAW_BANDS)
        )
        n = col.size().getInfo()
        if n == 0:
            continue

        def map_fn(img):
            img_x = transform_era5(img)
            img_sub = img_x.select(OUT_BANDS)
            date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            reduced = img_sub.reduceRegions(
                collection=points,
                reducer=ee.Reducer.mean(),
                scale=11132,
                tileScale=4,
            )
            return reduced.map(lambda f: f.set({"date": date_str}))

        fc = ee.FeatureCollection(col.map(map_fn).flatten())
        df = fc_to_df(fc)
        if len(df) > 0:
            frames.append(df)
        print(f"  ERA5: {yr} -> {len(df)} rows")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    if OUT_PATH.exists():
        print(f"Already exists: {OUT_PATH}")
        return

    print("Extracting ERA5-Land daily weather for 4 stations (2003-2015)")
    init_gee()
    points = get_points()

    df = extract_era5_weather(points, 2003, 2015)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Done: {len(df)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
