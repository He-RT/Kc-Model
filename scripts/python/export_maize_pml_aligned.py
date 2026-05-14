#!/usr/bin/env python3
"""Export PML-V2 ET aligned to corrected maize Kcact MOD16 windows.

Default source is the current PML V2.2 asset:
  projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a
which currently overlaps the large maize sample for 2019-2024. 2025 is not
available in this asset at the time of writing.

Rows join back by coord_key + date_start + date_end + date.  Exported ET fields:
  * pml_et_mm_d: PML native ET band (V22a)
  * pml_eta_crop_mm_d: Ec + Es + Ei, i.e. transpiration + soil evaporation + interception
  * Ec, Es, Ei, Ew, PET, GPP when available
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
DEFAULT_FOLDER = "kcact_maize_modis_indicators"
DEFAULT_ASSET = "projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a"
LEGACY_ASSET = "CAS/IGSNRR/PML/V2_v018"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-id", default="chuang-yaogan")
    p.add_argument("--year", type=int, default=None, help="Single year; default exports all overlap years")
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year", type=int, default=2024, help="Inclusive. V22a currently covers through 2024; v018 through 2023.")
    p.add_argument("--folder", default=DEFAULT_FOLDER)
    p.add_argument("--asset", default=DEFAULT_ASSET, help=f"PML asset. Legacy: {LEGACY_ASSET}")
    p.add_argument("--scale", type=int, default=500)
    p.add_argument("--no-export", action="store_true")
    return p.parse_args()


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def collection_years(asset: str, start_year: int, end_year: int) -> list[int]:
    col = ee.ImageCollection(asset)
    years = []
    for y in range(start_year, end_year + 1):
        n = col.filterDate(f"{y}-01-01", f"{y+1}-01-01").size().getInfo()
        if n and n > 0:
            years.append(y)
        else:
            print(f"Skip {y}: no PML images in asset {asset}")
    return years


def load_year_points_and_windows(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "point_id",
        "coord_key",
        "centroid_lat",
        "centroid_lon",
        "date_start",
        "date_end",
        "date",
        "year",
        "qc_valid",
    ]
    df = pd.read_parquet(PARQUET, columns=cols)
    for col in ["date_start", "date_end", "date"]:
        df[col] = pd.to_datetime(df[col])
    df = df[(df["qc_valid"].astype(bool)) & (df["year"].astype(int) == int(year))].copy()
    if df.empty:
        raise ValueError(f"No valid maize rows found for year={year}")

    points = (
        df[["point_id", "coord_key", "centroid_lat", "centroid_lon"]]
        .drop_duplicates(subset=["coord_key"])
        .sort_values("coord_key")
        .reset_index(drop=True)
    )
    windows = (
        df[["date_start", "date_end", "date"]]
        .drop_duplicates()
        .sort_values("date_start")
        .reset_index(drop=True)
    )
    return points, windows


def make_points_fc(points: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for row in points.itertuples(index=False):
        geom = ee.Geometry.Point([float(row.centroid_lon), float(row.centroid_lat)])
        feats.append(
            ee.Feature(
                geom,
                {
                    "point_id": str(row.point_id),
                    "coord_key": str(row.coord_key),
                    "centroid_lon": float(row.centroid_lon),
                    "centroid_lat": float(row.centroid_lat),
                },
            )
        )
    return ee.FeatureCollection(feats)


def make_windows_fc(windows: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for row in windows.itertuples(index=False):
        feats.append(
            ee.Feature(
                None,
                {
                    "date_start": row.date_start.strftime("%Y-%m-%d"),
                    "date_end": row.date_end.strftime("%Y-%m-%d"),
                    "date": row.date.strftime("%Y-%m-%d"),
                },
            )
        )
    return ee.FeatureCollection(feats)


def pml_image_with_standard_bands(img: ee.Image, asset: str) -> ee.Image:
    names = img.bandNames()
    is_v22 = names.contains("ET")

    # V22a has ET and Ew; v018 has ET_water but no ET band.
    ec = img.select("Ec")
    es = img.select("Es")
    ei = img.select("Ei")
    crop = ec.add(es).add(ei).rename("pml_eta_crop_mm_d")

    et = ee.Image(ee.Algorithms.If(is_v22, img.select("ET"), crop)).rename("pml_et_mm_d")
    ew = ee.Image(ee.Algorithms.If(names.contains("Ew"), img.select("Ew"), img.select("ET_water"))).rename("Ew")
    pet = ee.Image(ee.Algorithms.If(names.contains("PET"), img.select("PET"), ee.Image.constant(-9999).updateMask(ee.Image.constant(0)))).rename("PET")

    out = ee.Image.cat([
        et,
        crop,
        ec.rename("Ec"),
        es.rename("Es"),
        ei.rename("Ei"),
        ew,
        pet,
        img.select("GPP").rename("GPP"),
    ])
    # V22a EE asset stores these bands with a 0.01 scale factor. Legacy v018 is already mm/d.
    return ee.Image(ee.Algorithms.If(ee.String(asset).match("V22a").length().gt(0), out.multiply(0.01), out))


def build_pml_export(points_fc: ee.FeatureCollection, windows_fc: ee.FeatureCollection, asset: str, scale: int) -> ee.FeatureCollection:
    region = points_fc.geometry()
    pml = ee.ImageCollection(asset).filterBounds(region)
    bands = ["pml_et_mm_d", "pml_eta_crop_mm_d", "Ec", "Es", "Ei", "Ew", "PET", "GPP"]

    def per_window(win):
        win = ee.Feature(win)
        start = ee.Date(win.get("date_start"))
        end = ee.Date(win.get("date_end"))
        period = pml.filterDate(start, end)
        count = period.size()
        composite = ee.Image(
            ee.Algorithms.If(
                count.gt(0),
                period.map(lambda img: pml_image_with_standard_bands(ee.Image(img), asset)).mean().rename(bands),
                ee.Image.constant([-9999] * len(bands)).rename(bands).updateMask(ee.Image.constant(0)),
            )
        )
        reduced = composite.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=scale,
            tileScale=4,
        )
        return reduced.map(
            lambda f: f.set(
                {
                    "date_start": win.get("date_start"),
                    "date_end": win.get("date_end"),
                    "date": win.get("date"),
                    "pml_obs_count_8d": count,
                    "pml_scale_m": scale,
                    "pml_asset": asset,
                }
            )
        )

    return ee.FeatureCollection(windows_fc.map(per_window).flatten())


def submit_year(year: int, folder: str, asset: str, scale: int, no_export: bool = False) -> dict:
    points, windows = load_year_points_and_windows(year)
    print(f"{year}: points={len(points):,}, windows={len(windows):,}, rows≈{len(points)*len(windows):,}")
    if no_export:
        return {"year": year, "state": "DRY_RUN", "points": len(points), "windows": len(windows), "asset": asset}
    fc = build_pml_export(make_points_fc(points), make_windows_fc(windows), asset=asset, scale=scale)
    desc = f"maize_pml_aligned_{year}"
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=desc,
        folder=folder,
        fileNamePrefix=desc,
        fileFormat="CSV",
    )
    task.start()
    status = task.status()
    return {
        "year": year,
        "description": desc,
        "id": status.get("id", ""),
        "state": status.get("state", ""),
        "points": len(points),
        "windows": len(windows),
        "asset": asset,
        "scale": scale,
    }


def main() -> None:
    args = parse_args()
    init_ee(args.project_id)
    if args.year:
        years = [args.year]
    else:
        years = collection_years(args.asset, args.start_year, args.end_year)
    statuses = [submit_year(y, args.folder, args.asset, args.scale, args.no_export) for y in years]
    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    if not args.no_export:
        print(f"\nSubmitted {len(statuses)} tasks to Drive folder: {args.folder}")
        print("Download expected files as maize_pml_aligned_YYYY.csv into data/raw/gee/kcact_maize_modis_indicators/")


if __name__ == "__main__":
    main()
