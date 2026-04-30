"""Export S2 raw bands (B4,B8,B12) + Sentinel-1 VV/VH for maize patches."""

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
    p.add_argument("--year", type=int, default=None)
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

def export_s2_raw(points, year):
    """Export S2 B4(Red), B8(NIR), B12(SWIR2) raw reflectance (10m/20m)."""
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{year}-06-01", f"{year}-11-01")
          .filterBounds(points.geometry())
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))

    def composite_8day(start_date):
        end_date = ee.Date(start_date).advance(8, "day")
        period = s2.filterDate(start_date, end_date)
        img = period.median()
        # Scale to reflectance (S2 raw is 0-10000)
        out = img.select(["B4","B8","B12"], ["s2_b4","s2_b8","s2_b12"]).divide(10000)
        d = ee.Date(start_date).format("YYYY-MM-dd")
        r = out.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
        return r.map(lambda f: f.set({"date": d}))

    # Generate 8-day windows matching MOD16 (same as existing pipeline)
    start = ee.Date(f"{year}-06-01")
    n_windows = ee.Date(f"{year}-11-01").difference(start, "day").divide(8).ceil()
    seq = ee.List.sequence(0, n_windows.subtract(1))
    windows = seq.map(lambda i: start.advance(ee.Number(i).multiply(8), "day"))

    fc = ee.FeatureCollection(windows.map(lambda d: composite_8day(d)).flatten())
    return fc

def export_s1(points, year):
    """Export Sentinel-1 VV/VH backscatter (IW GRD, 10m)."""
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterDate(f"{year}-06-01", f"{year}-11-01")
          .filterBounds(points.geometry())
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.eq("resolution", "H")))

    def composite_8day(start_date):
        end_date = ee.Date(start_date).advance(8, "day")
        period = s1.filterDate(start_date, end_date)
        img = period.median()
        out = img.select(["VV","VH"], ["s1_vv","s1_vh"])
        d = ee.Date(start_date).format("YYYY-MM-dd")
        r = out.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
        return r.map(lambda f: f.set({"date": d}))

    start = ee.Date(f"{year}-06-01")
    n_windows = ee.Date(f"{year}-11-01").difference(start, "day").divide(8).ceil()
    seq = ee.List.sequence(0, n_windows.subtract(1))
    windows = seq.map(lambda i: start.advance(ee.Number(i).multiply(8), "day"))

    fc = ee.FeatureCollection(windows.map(lambda d: composite_8day(d)).flatten())
    return fc

def main():
    args = parse_args()
    init_ee(args.project_id)
    points = get_points()
    years = [args.year] if args.year else list(range(2019, 2026))
    folder = "kcact_maize_modis_indicators"

    for yr in years:
        if args.no_export:
            print(f"  {yr}: dry-run S2+S1")
            continue
        # S2 raw bands
        fc_s2 = export_s2_raw(points, yr)
        t1 = ee.batch.Export.table.toDrive(
            collection=fc_s2, description=f"maize_s2raw_{yr}",
            folder=folder, fileNamePrefix=f"maize_s2raw_{yr}", fileFormat="CSV")
        t1.start()
        print(f"  {yr}: S2 raw → submitted")

        # S1 SAR
        fc_s1 = export_s1(points, yr)
        t2 = ee.batch.Export.table.toDrive(
            collection=fc_s1, description=f"maize_s1_{yr}",
            folder=folder, fileNamePrefix=f"maize_s1_{yr}", fileFormat="CSV")
        t2.start()
        print(f"  {yr}: S1 SAR → submitted")

    n = 14 if args.year else 14 * len(years)
    print(f"\n~{len(years)*2} tasks → Drive '{folder}'")

if __name__ == "__main__":
    main()
