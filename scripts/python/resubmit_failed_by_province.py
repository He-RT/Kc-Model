"""Resubmit failed GEE exports split by province to avoid OOM."""

import argparse
from pathlib import Path
import ee
import pandas as pd

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

def get_points_by_province():
    df = pd.read_parquet(PARQUET)
    df = df[df['qc_valid']]
    points = {}
    for prov in sorted(df['province'].unique()):
        locs = df[df['province']==prov][['point_id','centroid_lat','centroid_lon']].drop_duplicates()
        feats = [ee.Feature(ee.Geometry.Point([r['centroid_lon'],r['centroid_lat']]),
                            {"point_id": str(r['point_id'])}) for _,r in locs.iterrows()]
        points[prov] = ee.FeatureCollection(feats)
        print(f"  {prov}: {len(locs)} points")
    return points

def export_albedo_by_province(points_by_prov, yr):
    """MCD43A3 albedo, 8-day windows, tileScale=8."""
    for prov, points in points_by_prov.items():
        col = (ee.ImageCollection("MODIS/061/MCD43A3")
               .filterDate(f"{yr}-06-01", f"{yr}-11-01")
               .select("Albedo_WSA_shortwave"))
        start = ee.Date(f"{yr}-06-01")
        n = ee.Date(f"{yr}-11-01").difference(start, "day").divide(8).ceil()
        seq = ee.List.sequence(0, n.subtract(1))
        windows = seq.map(lambda i: start.advance(ee.Number(i).multiply(8), "day"))
        def win_fn(sd):
            period = col.filterDate(sd, ee.Date(sd).advance(8,"day"))
            img = period.mean()
            d = ee.Date(sd).format("YYYY-MM-dd")
            r = img.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=8)
            return r.map(lambda f: f.set({"date": d}))
        fc = ee.FeatureCollection(windows.map(lambda d: win_fn(d)).flatten())
        desc = f"alb_{prov[:4]}_{yr}"
        task = ee.batch.Export.table.toDrive(collection=fc, description=desc,
            folder="kcact_maize_modis_indicators", fileNamePrefix=f"maize_albedo_{prov.lower()}_{yr}",
            fileFormat="CSV")
        task.start()
        print(f"    alb {prov} {yr} → submitted")

def export_s2raw_by_province(points_by_prov, yr):
    """S2 B4/B8/B12, 8-day median composite, scale=500 (not 10m to avoid OOM)."""
    for prov, points in points_by_prov.items():
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterDate(f"{yr}-06-01", f"{yr}-11-01")
              .filterBounds(points.geometry())
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
        start = ee.Date(f"{yr}-06-01")
        n = ee.Date(f"{yr}-11-01").difference(start, "day").divide(8).ceil()
        seq = ee.List.sequence(0, n.subtract(1))
        windows = seq.map(lambda i: start.advance(ee.Number(i).multiply(8), "day"))
        def win_fn(sd):
            period = s2.filterDate(sd, ee.Date(sd).advance(8,"day"))
            img = period.median()
            out = img.select(["B4","B8","B12"],["s2_b4","s2_b8","s2_b12"]).divide(10000)
            d = ee.Date(sd).format("YYYY-MM-dd")
            r = out.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=8)
            return r.map(lambda f: f.set({"date": d}))
        fc = ee.FeatureCollection(windows.map(lambda d: win_fn(d)).flatten())
        desc = f"s2r_{prov[:4]}_{yr}"
        task = ee.batch.Export.table.toDrive(collection=fc, description=desc,
            folder="kcact_maize_modis_indicators", fileNamePrefix=f"maize_s2raw_{prov.lower()}_{yr}",
            fileFormat="CSV")
        task.start()
        print(f"    s2r {prov} {yr} → submitted")

def export_s1_by_province(points_by_prov, yr):
    """S1 VV/VH, 8-day median, scale=500."""
    for prov, points in points_by_prov.items():
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterDate(f"{yr}-06-01", f"{yr}-11-01")
              .filterBounds(points.geometry())
              .filter(ee.Filter.eq("instrumentMode","IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH")))
        start = ee.Date(f"{yr}-06-01")
        n = ee.Date(f"{yr}-11-01").difference(start, "day").divide(8).ceil()
        seq = ee.List.sequence(0, n.subtract(1))
        windows = seq.map(lambda i: start.advance(ee.Number(i).multiply(8), "day"))
        def win_fn(sd):
            period = s1.filterDate(sd, ee.Date(sd).advance(8,"day"))
            img = period.select(["VV","VH"]).median()
            d = ee.Date(sd).format("YYYY-MM-dd")
            r = img.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=8)
            return r.map(lambda f: f.set({"date": d}))
        fc = ee.FeatureCollection(windows.map(lambda d: win_fn(d)).flatten())
        desc = f"s1_{prov[:4]}_{yr}"
        task = ee.batch.Export.table.toDrive(collection=fc, description=desc,
            folder="kcact_maize_modis_indicators", fileNamePrefix=f"maize_s1_{prov.lower()}_{yr}",
            fileFormat="CSV")
        task.start()
        print(f"    s1  {prov} {yr} → submitted")

def export_m09vi_by_province(points_by_prov, yr):
    """MOD09A1 ndvi+b07, 8-day, scale=500, tileScale=8."""
    for prov, points in points_by_prov.items():
        col = (ee.ImageCollection("MODIS/061/MOD09A1")
               .filterDate(f"{yr}-06-01", f"{yr}-11-01")
               .select(["sur_refl_b01","sur_refl_b02","sur_refl_b07"]))
        def fn(img):
            d = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            ndvi = img.normalizedDifference(["sur_refl_b02","sur_refl_b01"]).rename("ndvi_m09")
            out = img.addBands(ndvi).select(["ndvi_m09","sur_refl_b07"])
            r = out.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=8)
            return r.map(lambda f: f.set({"date": d}))
        fc = ee.FeatureCollection(col.map(fn).flatten())
        desc = f"m09_{prov[:4]}_{yr}"
        task = ee.batch.Export.table.toDrive(collection=fc, description=desc,
            folder="kcact_maize_modis_indicators", fileNamePrefix=f"maize_m09vi_{prov.lower()}_{yr}",
            fileFormat="CSV")
        task.start()
        print(f"    m09 {prov} {yr} → submitted")

def main():
    args = parse_args()
    init_ee(args.project_id)
    points_by_prov = get_points_by_province()
    years = [args.year] if args.year else list(range(2019, 2026))
    products = 0

    for yr in years:
        if args.no_export:
            print(f"  {yr}: dry-run")
            continue
        export_albedo_by_province(points_by_prov, yr); products += 1
        export_s2raw_by_province(points_by_prov, yr); products += 1
        export_s1_by_province(points_by_prov, yr); products += 1
        export_m09vi_by_province(points_by_prov, yr); products += 1

    provinces = len(points_by_prov)
    total = products * provinces
    print(f"\n{total} tasks submitted ({products} products × {provinces} provinces × {len(years)} years)")
    print(f"All → Drive folder 'kcact_maize_modis_indicators'")

if __name__ == "__main__":
    main()
