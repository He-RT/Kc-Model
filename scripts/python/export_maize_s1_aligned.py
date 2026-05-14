"""Export Sentinel-1 VV/VH aligned to corrected maize Kcact windows.

This is the data needed for the "scheme 2" SAR water-proxy experiment.

Unlike the older ``export_maize_s2raw_and_s1.py`` helper, this script reads the
corrected training parquet and uses its exact MOD16 8-day windows:

  date_start <= Sentinel-1 acquisition < date_end
  exported date = date_end

Rows therefore join back to Kcact by ``point_id``/``coord_key`` + ``date``.

Sentinel-1 reduceRegions at 10 m can OOM when all NCP points for a year are
exported as one task.  By default this script splits exports by province and
year, producing files such as ``maize_s1_aligned_2019_Henan.csv``.
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
PROVINCES = ["Anhui", "Hebei", "Henan", "Shandong"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="chuang-yaogan")
    parser.add_argument("--year", type=int, default=None, help="Single year; default exports 2019-2025")
    parser.add_argument(
        "--province",
        choices=PROVINCES,
        default=None,
        help="Single province; default exports all provinces.",
    )
    parser.add_argument(
        "--combined-year",
        action="store_true",
        help="Submit one task per year across all provinces. Not recommended; may OOM.",
    )
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def load_year_points_and_windows(
    year: int,
    province: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "point_id",
        "coord_key",
        "province",
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
    df = df[(df["qc_valid"]) & (df["year"].astype(int) == int(year))].copy()
    if province:
        df = df[df["province"].astype(str) == province].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    points = (
        df[["point_id", "coord_key", "province", "centroid_lat", "centroid_lon"]]
        .drop_duplicates()
        .sort_values(["coord_key", "point_id"])
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
                    "province": str(row.province),
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


def build_s1_export(points_fc: ee.FeatureCollection, windows_fc: ee.FeatureCollection) -> ee.FeatureCollection:
    region = points_fc.geometry()
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .select(["VV", "VH"])
    )

    def per_window(win):
        win = ee.Feature(win)
        start = ee.Date(win.get("date_start"))
        end = ee.Date(win.get("date_end"))
        date_label = ee.String(win.get("date"))
        period = s1.filterDate(start, end)
        count = period.size()
        composite = ee.Image(
            ee.Algorithms.If(
                count.gt(0),
                period.median().rename(["s1_vv", "s1_vh"]),
                ee.Image.constant([-9999, -9999]).rename(["s1_vv", "s1_vh"]).updateMask(ee.Image.constant(0)),
            )
        )
        reduced = composite.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=10,
            tileScale=4,
        )
        return reduced.map(
            lambda f: f.set(
                {
                    "date_start": win.get("date_start"),
                    "date_end": win.get("date_end"),
                    "date": date_label,
                    "s1_obs_count": count,
                }
            )
        )

    return ee.FeatureCollection(windows_fc.map(per_window).flatten())


def submit_year_province(
    year: int,
    province: str | None,
    folder: str,
    no_export: bool = False,
) -> dict:
    points, windows = load_year_points_and_windows(year, province=province)
    label = f"{year}_{province}" if province else str(year)
    if points.empty or windows.empty:
        print(f"{label}: SKIP no valid rows")
        return {
            "year": year,
            "province": province,
            "state": "SKIPPED_NO_ROWS",
            "points": 0,
            "windows": 0,
            "rows_est": 0,
        }
    print(
        f"{label}: points={len(points):,}, windows={len(windows):,}, "
        f"rows≈{len(points)*len(windows):,}"
    )
    if no_export:
        return {
            "year": year,
            "province": province,
            "state": "DRY_RUN",
            "points": len(points),
            "windows": len(windows),
            "rows_est": len(points) * len(windows),
        }

    points_fc = make_points_fc(points)
    windows_fc = make_windows_fc(windows)
    fc = build_s1_export(points_fc, windows_fc)

    desc = f"maize_s1_aligned_{label}"
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
        "province": province,
        "description": desc,
        "id": status.get("id", ""),
        "state": status.get("state", ""),
        "points": len(points),
        "windows": len(windows),
        "rows_est": len(points) * len(windows),
    }


def main() -> None:
    args = parse_args()
    init_ee(args.project_id)
    years = [args.year] if args.year else list(range(2019, 2026))
    provinces: list[str | None]
    if args.combined_year:
        provinces = [None]
    else:
        provinces = [args.province] if args.province else PROVINCES

    statuses = [
        submit_year_province(
            year,
            province=province,
            folder=args.folder,
            no_export=args.no_export,
        )
        for year in years
        for province in provinces
    ]
    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    if not args.no_export:
        print(f"\nSubmitted {len(statuses)} tasks to Drive folder: {args.folder}")
        if args.combined_year:
            print("Download expected files as maize_s1_aligned_YYYY.csv into data/raw/gee/kcact_maize_modis_indicators/")
        else:
            print(
                "Download expected files as maize_s1_aligned_YYYY_PROVINCE.csv "
                "into data/raw/gee/kcact_maize_modis_indicators/"
            )


if __name__ == "__main__":
    main()
