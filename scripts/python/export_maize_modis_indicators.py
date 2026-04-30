"""Export MODIS + ERA5-SM indicators for existing maize training points via GEE."""

import argparse, json, sys
from pathlib import Path
import ee
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "processed" / "train" / "ncp_summer_maize_kcact_train_ready.parquet"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-id", default="chuang-yaogan")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--year", type=int, default=None, help="Single year (default: all 2019-2025)")
    return p.parse_args()

def init_ee(pid):
    try: ee.Initialize(project=pid)
    except: ee.Authenticate(); ee.Initialize(project=pid)

def get_maize_points():
    df = pd.read_parquet(PARQUET)
    df = df[df['qc_valid']]
    locs = df[['point_id','centroid_lat','centroid_lon']].drop_duplicates()
    print(f"Unique points: {len(locs)}")
    feats = []
    for _, r in locs.iterrows():
        pt = ee.Geometry.Point([r['centroid_lon'], r['centroid_lat']])
        feats.append(ee.Feature(pt, {"point_id": str(r['point_id'])}))
    return ee.FeatureCollection(feats), df

def build_fpar_export(points, year):
    col = (ee.ImageCollection("MODIS/061/MOD15A2H")
           .filterDate(f"{year}-06-01", f"{year}-11-01")
           .select("Fpar_500m"))
    def fn(img):
        d = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        r = img.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=4)
        return r.map(lambda f: f.set({"date": d}))
    return ee.FeatureCollection(col.map(fn).flatten())

def build_lst_export(points, year):
    col = (ee.ImageCollection("MODIS/061/MOD11A2")
           .filterDate(f"{year}-06-01", f"{year}-11-01")
           .select(["LST_Day_1km","LST_Night_1km"]))
    def fn(img):
        d = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        r = img.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=1000, tileScale=4)
        return r.map(lambda f: f.set({"date": d}))
    return ee.FeatureCollection(col.map(fn).flatten())

def build_albedo_export(points, year):
    col = (ee.ImageCollection("MODIS/061/MCD43A3")
           .filterDate(f"{year}-06-01", f"{year}-11-01")
           .select("Albedo_WSA_shortwave"))
    def fn(img):
        d = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        r = img.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=4)
        return r.map(lambda f: f.set({"date": d}))
    return ee.FeatureCollection(col.map(fn).flatten())

def build_era5_sm_export(points, year):
    col = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
           .filterDate(f"{year}-06-01", f"{year}-11-01")
           .select("volumetric_soil_water_layer_1"))
    def fn(img):
        d = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        r = img.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=11132, tileScale=4)
        return r.map(lambda f: f.set({"date": d}))
    return ee.FeatureCollection(col.map(fn).flatten())

def export_table(fc, desc, folder, prefix):
    task = ee.batch.Export.table.toDrive(
        collection=fc, description=desc, folder=folder,
        fileNamePrefix=prefix, fileFormat="CSV")
    task.start()
    return task

def main():
    args = parse_args()
    init_ee(args.project_id)
    points, _ = get_maize_points()

    years = [args.year] if args.year else list(range(2019, 2026))
    folder = "kcact_maize_modis_indicators"
    tasks = []

    for yr in years:
        print(f"Year {yr}: building exports...")
        if not args.no_export:
            t1 = export_table(build_fpar_export(points, yr),
                              f"maize_fpar_{yr}", folder, f"maize_fpar_{yr}")
            t2 = export_table(build_lst_export(points, yr),
                              f"maize_lst_{yr}", folder, f"maize_lst_{yr}")
            t3 = export_table(build_albedo_export(points, yr),
                              f"maize_albedo_{yr}", folder, f"maize_albedo_{yr}")
            t4 = export_table(build_era5_sm_export(points, yr),
                              f"maize_era5_sm_{yr}", folder, f"maize_era5_sm_{yr}")
            tasks.extend([t1, t2, t3, t4])

    if not args.no_export:
        print(f"\nStarted {len(tasks)} export tasks to Drive folder '{folder}'")
        print("Check https://code.earthengine.google.com/ -> Tasks tab")
    else:
        print("Dry run (--no-export)")

if __name__ == "__main__":
    main()
