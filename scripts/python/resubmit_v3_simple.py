"""Resubmit v3: native image dates per province, no 8-day compositing, no geometry filter."""

import argparse
from pathlib import Path
import ee
import pandas as pd

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

def get_province_points():
    df = pd.read_parquet(PARQUET); df = df[df['qc_valid']]
    points = {}
    for prov in sorted(df['province'].unique()):
        locs = df[df['province']==prov][['point_id','centroid_lat','centroid_lon']].drop_duplicates()
        feats = [ee.Feature(ee.Geometry.Point([r['centroid_lon'],r['centroid_lat']]),
                            {"point_id": str(r['point_id'])}) for _,r in locs.iterrows()]
        points[prov] = ee.FeatureCollection(feats)
    return points

def main():
    args = parse_args()
    init_ee(args.project_id)
    points_by_prov = get_province_points()
    folder = "kcact_maize_modis_indicators"
    submitted = 0

    for yr in range(2019, 2026):
        for prov, points in points_by_prov.items():
            ptag = prov.lower()[:4]

            # Helper: 8-day windows without geometry filter
            start_d = ee.Date(f"{yr}-06-01")
            n_win = ee.Date(f"{yr}-11-01").difference(start_d, "day").divide(8).ceil()
            windows = ee.List.sequence(0, n_win.subtract(1)).map(
                lambda i: start_d.advance(ee.Number(i).multiply(8), "day"))

            # --- MCD43A3 Albedo (8-day mean) ---
            alb = (ee.ImageCollection("MODIS/061/MCD43A3")
                   .filterDate(f"{yr}-06-01", f"{yr}-11-01")
                   .select("Albedo_WSA_shortwave"))
            def alb_win(sd):
                p = alb.filterDate(sd, ee.Date(sd).advance(8,"day"))
                img = p.mean()
                d = ee.Date(sd).format("YYYY-MM-dd")
                r = img.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=8)
                return r.map(lambda f: f.set({"date": d}))
            fc_a = ee.FeatureCollection(windows.map(alb_win).flatten())
            ta = ee.batch.Export.table.toDrive(
                collection=fc_a, description=f"a3_{ptag}_{yr}",
                folder=folder, fileNamePrefix=f"maize_albedo_{prov.lower()}_{yr}", fileFormat="CSV")
            if not args.no_export: ta.start(); submitted += 1

            # --- S2 raw bands (8-day median) ---
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterDate(f"{yr}-06-01", f"{yr}-11-01")
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
            def s2_win(sd):
                p = s2.filterDate(sd, ee.Date(sd).advance(8,"day"))
                img = p.median()
                out = img.select(["B4","B8","B12"],["s2_b4","s2_b8","s2_b12"]).divide(10000)
                d = ee.Date(sd).format("YYYY-MM-dd")
                r = out.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=8)
                return r.map(lambda f: f.set({"date": d}))
            fc_s2 = ee.FeatureCollection(windows.map(s2_win).flatten())
            ts = ee.batch.Export.table.toDrive(
                collection=fc_s2, description=f"s3_{ptag}_{yr}",
                folder=folder, fileNamePrefix=f"maize_s2raw_{prov.lower()}_{yr}", fileFormat="CSV")
            if not args.no_export: ts.start(); submitted += 1

            # --- S1 VV/VH (8-day median) ---
            s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterDate(f"{yr}-06-01", f"{yr}-11-01")
                  .filter(ee.Filter.eq("instrumentMode","IW"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH")))
            def s1_win(sd):
                p = s1.filterDate(sd, ee.Date(sd).advance(8,"day"))
                img = p.median()
                out = img.select(["VV","VH"],["s1_vv","s1_vh"])
                d = ee.Date(sd).format("YYYY-MM-dd")
                r = out.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=500, tileScale=8)
                return r.map(lambda f: f.set({"date": d}))
            fc_s1 = ee.FeatureCollection(windows.map(s1_win).flatten())
            t1 = ee.batch.Export.table.toDrive(
                collection=fc_s1, description=f"s4_{ptag}_{yr}",
                folder=folder, fileNamePrefix=f"maize_s1_{prov.lower()}_{yr}", fileFormat="CSV")
            if not args.no_export: t1.start(); submitted += 1

    print(f"{submitted} tasks submitted (3 products × 4 provinces × 7 years)")

if __name__ == "__main__":
    main()
