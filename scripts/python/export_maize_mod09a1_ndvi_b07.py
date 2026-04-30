"""Export MOD09A1 b01(Red), b02(NIR), b07(SWIR2) for maize patches. One task per year."""

import argparse
from pathlib import Path
import ee
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-id", default="chuang-yaogan")
    p.add_argument("--no-export", action="store_true")
    return p.parse_args()

def init_ee(pid):
    try: ee.Initialize(project=pid)
    except: ee.Authenticate(); ee.Initialize(project=pid)

def get_points():
    df = pd.read_parquet(PARQUET)
    df = df[df['qc_valid']]
    locs = df[['point_id','centroid_lat','centroid_lon']].drop_duplicates()
    print(f"Points: {len(locs)}")
    feats = []
    for _, r in locs.iterrows():
        pt = ee.Geometry.Point([r['centroid_lon'], r['centroid_lat']])
        feats.append(ee.Feature(pt, {"point_id": str(r['point_id'])}))
    return ee.FeatureCollection(feats)

def build_export(points, year):
    col = (ee.ImageCollection("MODIS/061/MOD09A1")
           .filterDate(f"{year}-06-01", f"{year}-11-01")
           .select(["sur_refl_b01","sur_refl_b02","sur_refl_b07"]))

    def fn(img):
        d = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        ndvi = img.normalizedDifference(["sur_refl_b02","sur_refl_b01"]).rename("ndvi_m09")
        # b07 as raw scaled integer (0-10000), will scale to reflectance in Python
        out = img.addBands(ndvi).select(["ndvi_m09","sur_refl_b07"])
        r = out.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=4)
        return r.map(lambda f: f.set({"date": d}))

    return ee.FeatureCollection(col.map(fn).flatten())

def main():
    args = parse_args()
    init_ee(args.project_id)
    points = get_points()
    folder = "kcact_maize_modis_indicators"

    for yr in range(2019, 2026):
        desc = f"maize_m09vi_{yr}"
        prefix = f"maize_m09vi_{yr}"
        if args.no_export:
            print(f"  {yr}: dry-run, skipping export")
            continue
        fc = build_export(points, yr)
        task = ee.batch.Export.table.toDrive(
            collection=fc, description=desc, folder=folder,
            fileNamePrefix=prefix, fileFormat="CSV")
        task.start()
        print(f"  {yr}: task submitted")

    print(f"\n7 tasks → Drive folder '{folder}'")
    print("https://code.earthengine.google.com/ -> Tasks")

if __name__ == "__main__":
    main()
