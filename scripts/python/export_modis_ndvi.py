"""Export MOD13Q1 NDVI (250m 16-day) aligned to MOD16 8-day windows.

Reuses the exact same candidate mask and point sampling as the main export
script to ensure point_id matching.

Usage:
  python scripts/python/export_modis_ndvi.py --project-id chuang-yaogan --year 2019
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ee

from export_hebei_kcact_training_data import (
    build_candidate_mask,
    sample_points_in_mask,
    init_ee,
    get_hebei_geometry,
    Config,
)


def season_dates(year):
    return ee.Date.fromYMD(year - 1, 10, 1), ee.Date.fromYMD(year, 7, 1)


def build_modis_ndvi_table(points, year, region):
    s_date, e_date = season_dates(year)

    mod16_ic = ee.ImageCollection("MODIS/061/MOD16A2GF") \
        .filterDate(s_date, e_date).filterBounds(region).sort("system:time_start")

    mod13 = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterDate(s_date.advance(-16, "day"), e_date.advance(16, "day")) \
        .filterBounds(region) \
        .select(["NDVI", "EVI", "SummaryQA"])

    image_list = mod16_ic.toList(mod16_ic.size())

    def iterate(item, acc):
        acc_fc = ee.FeatureCollection(acc)
        mod16_img = ee.Image(item)
        window_start = mod16_img.date()
        window_end = window_start.advance(8, "day")

        # MOD13Q1 images covering this MOD16 window
        mod13_matching = mod13.filterDate(window_start.advance(-8, "day"), window_end.advance(8, "day"))

        mod13_img = ee.Image(mod13_matching.sort("system:time_start").first())
        modis_ndvi = mod13_img.select("NDVI").multiply(0.0001)
        modis_evi = mod13_img.select("EVI").multiply(0.0001)
        modis_qa = mod13_img.select("SummaryQA")

        stack = modis_ndvi.rename("modis_ndvi") \
            .addBands(modis_evi.rename("modis_evi")) \
            .addBands(modis_qa.rename("modis_qa"))

        fallback = ee.Image.constant([0, 0, 255]).rename(["modis_ndvi", "modis_evi", "modis_qa"])
        final = ee.Image(ee.Algorithms.If(mod13_matching.size().gt(0), stack, fallback))

        reduced = final.reduceRegions(
            collection=points, reducer=ee.Reducer.mean(),
            scale=250, tileScale=4,
        )
        annotated = reduced.map(lambda f: f.set({
            "season_year": year,
            "date_start": window_start.format("YYYY-MM-dd"),
            "date_end": window_end.format("YYYY-MM-dd"),
            "date": window_end.format("YYYY-MM-dd"),
            "year": window_end.get("year"),
        }))
        return acc_fc.merge(annotated)

    return ee.FeatureCollection(image_list.iterate(iterate, ee.FeatureCollection([])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-id", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--drive-folder", default="kcact_hebei_training_data")
    p.add_argument("--sample-scale", type=int, default=1000)
    p.add_argument("--sample-limit", type=int, default=10000)
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args()

    init_ee(args.project_id)
    config = Config(
        year=args.year, sample_scale=args.sample_scale,
        sample_limit=args.sample_limit,
        drive_folder=args.drive_folder, export_to_drive=not args.no_export,
    )

    hebei = get_hebei_geometry(config)
    mask = build_candidate_mask(config.year, config, hebei)
    points = sample_points_in_mask(mask, hebei, config)
    fc = build_modis_ndvi_table(points, config.year, hebei)

    if args.no_export:
        print("Preview only — no export.")
        return

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"drive_hebei_kcact_modis_ndvi_{args.year}",
        folder=args.drive_folder,
        fileNamePrefix=f"hebei_kcact_modis_ndvi_{args.year}",
        fileFormat="CSV",
    )
    task.start()
    status = task.status()
    print(json.dumps({"id": status.get("id"), "state": status.get("state"),
                      "description": status.get("description")}, indent=2))


if __name__ == "__main__":
    main()
