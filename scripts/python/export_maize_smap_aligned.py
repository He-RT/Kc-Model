"""Export SMAP L4 soil moisture aligned to corrected maize Kcact windows.

Data source:
  NASA/SMAP/SPL4SMGP/008
  SMAP L4 Global 3-hourly 9-km EASE-Grid Surface and Root Zone Soil Moisture.

The GEE catalog exposes the product after reprojection to geographic
coordinates, where the listed pixel size is ~11 km.  Scientifically this is
still the SMAP L4 9-km EASE-Grid product; we sample at the catalog/native
scale by default to avoid pretending it is a finer grid.

Unlike the older daily ERA5-SM export, this script reads the corrected Kcact
parquet and uses its exact MOD16 8-day windows:

  date_start <= SMAP acquisition < date_end
  exported date = date_end

Rows join back to Kcact by point_id/coord_key + date.  One Drive export task
is submitted per year.
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
SMAP_COLLECTION = "NASA/SMAP/SPL4SMGP/008"
SMAP_BANDS = [
    "sm_surface",
    "sm_rootzone",
    "sm_profile",
    "sm_surface_wetness",
    "sm_rootzone_wetness",
    "sm_profile_wetness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default="chuang-yaogan")
    parser.add_argument("--year", type=int, default=None, help="Single year; default exports 2019-2025")
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument(
        "--scale",
        type=int,
        default=11000,
        help="Sampling scale in meters. GEE catalog pixel size is ~11000 m for this reprojected SMAP L4 collection.",
    )
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


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
    df = df[(df["qc_valid"]) & (df["year"].astype(int) == int(year))].copy()
    if df.empty:
        raise ValueError(f"No valid maize rows found for year={year}")

    points = (
        df[["point_id", "coord_key", "centroid_lat", "centroid_lon"]]
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


def build_smap_export(
    points_fc: ee.FeatureCollection,
    windows_fc: ee.FeatureCollection,
    scale: int,
) -> ee.FeatureCollection:
    region = points_fc.geometry()
    smap = ee.ImageCollection(SMAP_COLLECTION).filterBounds(region).select(SMAP_BANDS)

    def per_window(win):
        win = ee.Feature(win)
        start = ee.Date(win.get("date_start"))
        end = ee.Date(win.get("date_end"))
        date_label = ee.String(win.get("date"))
        period = smap.filterDate(start, end)
        count = period.size()

        composite = ee.Image(
            ee.Algorithms.If(
                count.gt(0),
                period.mean().rename(SMAP_BANDS),
                ee.Image.constant([-9999] * len(SMAP_BANDS))
                .rename(SMAP_BANDS)
                .updateMask(ee.Image.constant(0)),
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
                    "date": date_label,
                    "smap_obs_count_3h": count,
                    "smap_scale_m": scale,
                }
            )
        )

    return ee.FeatureCollection(windows_fc.map(per_window).flatten())


def submit_year(year: int, folder: str, scale: int, no_export: bool = False) -> dict:
    points, windows = load_year_points_and_windows(year)
    print(f"{year}: points={len(points):,}, windows={len(windows):,}, rows≈{len(points)*len(windows):,}")
    if no_export:
        return {
            "year": year,
            "state": "DRY_RUN",
            "points": len(points),
            "windows": len(windows),
            "scale": scale,
        }

    points_fc = make_points_fc(points)
    windows_fc = make_windows_fc(windows)
    fc = build_smap_export(points_fc, windows_fc, scale=scale)

    desc = f"maize_smap_l4_aligned_{year}"
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
        "scale": scale,
    }


def main() -> None:
    args = parse_args()
    init_ee(args.project_id)
    years = [args.year] if args.year else list(range(2019, 2026))
    statuses = [
        submit_year(year, folder=args.folder, scale=args.scale, no_export=args.no_export)
        for year in years
    ]
    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    if not args.no_export:
        print(f"\nSubmitted {len(statuses)} tasks to Drive folder: {args.folder}")
        print(
            "Download expected files as maize_smap_l4_aligned_YYYY.csv into "
            "data/raw/gee/kcact_maize_modis_indicators/"
        )


if __name__ == "__main__":
    main()
