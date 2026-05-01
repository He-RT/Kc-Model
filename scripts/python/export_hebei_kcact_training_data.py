"""Export point-based Kcact training data from GEE for Hebei winter wheat.

This script avoids the reduceToVectors bottleneck by using image.sample()
to generate point locations within the candidate mask, then extracts
point-level S2/ERA5/MOD16 time series aligned to MOD16 8-day windows.

Outputs per year (to Google Drive):
  - hebei_kcact_s2_features_{year}.csv
  - hebei_kcact_era5_daily_{year}.csv
  - hebei_kcact_mod16_etc_{year}.csv

Example:
  # Preview only (no export):
  python scripts/python/export_hebei_kcact_training_data.py \\
    --project-id chuang-yaogan --year 2025 --no-export

  # Full export:
  python scripts/python/export_hebei_kcact_training_data.py \\
    --project-id chuang-yaogan --year 2025
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import ee


@dataclass(frozen=True)
class Config:
    country_name: str = "China"
    province_name: str = "Hebei"
    boundary_province_name: str = "Hebei Sheng"
    year: int = 2025
    sample_scale: int = 1000       # grid spacing in meters for point sampling
    sample_limit: int = 10000      # max number of sample points
    cloud_probability_threshold: int = 40
    min_connected_pixels: int = 8
    export_scale_s2: int = 10
    export_scale_era5: int = 11132
    export_scale_mod16: int = 500
    max_pixels: float = 1e13
    export_to_drive: bool = True
    drive_folder: str = "kcact_hebei_training_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export point-based Kcact training data from GEE."
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--year", type=int, default=Config.year)
    parser.add_argument("--sample-scale", type=int, default=Config.sample_scale,
                        help="Grid spacing in meters for point sampling.")
    parser.add_argument("--sample-limit", type=int, default=Config.sample_limit,
                        help="Max number of sample points.")
    parser.add_argument("--drive-folder", default=Config.drive_folder)
    parser.add_argument("--no-export", action="store_true",
                        help="Preview only, skip Drive export.")
    return parser.parse_args()


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        year=args.year,
        sample_scale=args.sample_scale,
        sample_limit=args.sample_limit,
        export_to_drive=not args.no_export,
        drive_folder=args.drive_folder,
    )


# ---------------------------------------------------------------------------
# Candidate mask -- reuses the proven phenology rules
# ---------------------------------------------------------------------------

def get_hebei_geometry(config: Config) -> ee.Geometry:
    gaul_l1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    return (
        gaul_l1.filter(ee.Filter.eq("ADM0_NAME", config.country_name))
        .filter(ee.Filter.eq("ADM1_NAME", config.boundary_province_name))
        .geometry()
    )


def get_cropland_mask() -> ee.Image:
    world_cover = ee.ImageCollection("ESA/WorldCover/v200").first()
    return world_cover.select("Map").eq(40).rename("cropland")


def mask_edges(image: ee.Image) -> ee.Image:
    return image.updateMask(
        image.select("B8A").mask().updateMask(image.select("B9").mask())
    )


def build_cloud_joined_collection(
    start_date: ee.Date, end_date: ee.Date, region: ee.Geometry
) -> ee.ImageCollection:
    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    s2_clouds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    sr = (
        s2_sr.filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(mask_edges)
    )
    clouds = s2_clouds.filterBounds(region).filterDate(start_date, end_date)
    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=sr, secondary=clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined)


def add_indices(image: ee.Image, config: Config) -> ee.Image:
    scaled = image.select(
        ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    ).divide(10000)
    cloud_mask = ee.Image(image.get("cloud_mask")).select("probability")
    clean = scaled.updateMask(cloud_mask.lt(config.cloud_probability_threshold))
    ndvi = clean.normalizedDifference(["B8", "B4"]).rename("ndvi")
    lswi = clean.normalizedDifference(["B8", "B11"]).rename("lswi")
    ndre = clean.normalizedDifference(["B8", "B5"]).rename("ndre")
    evi = clean.expression(
        "2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1.0))",
        {"nir": clean.select("B8"), "red": clean.select("B4"), "blue": clean.select("B2")},
    ).rename("evi")
    return clean.addBands([ndvi, lswi, ndre, evi]).copyProperties(
        image, image.propertyNames()
    )


def get_season_windows(year: int) -> dict[str, ee.Date]:
    return {
        "autumn_start": ee.Date.fromYMD(year - 1, 10, 1),
        "autumn_end":   ee.Date.fromYMD(year - 1, 11, 30),
        "winter_start": ee.Date.fromYMD(year, 1, 1),
        "winter_end":   ee.Date.fromYMD(year, 2, 15),
        "spring_start": ee.Date.fromYMD(year, 3, 1),
        "spring_end":   ee.Date.fromYMD(year, 4, 15),
        "peak_start":   ee.Date.fromYMD(year, 4, 16),
        "peak_end":     ee.Date.fromYMD(year, 5, 20),
        "harvest_start": ee.Date.fromYMD(year, 6, 1),
        "harvest_end":  ee.Date.fromYMD(year, 6, 30),
    }


def composite_window(
    start_date: ee.Date, end_date: ee.Date, region: ee.Geometry,
    reducer_name: str, prefix: str, config: Config,
) -> ee.Image:
    collection = build_cloud_joined_collection(start_date, end_date, region).map(
        lambda image: add_indices(image, config)
    )
    fallback = ee.Image.constant([0, 0, 0, 0]).rename(
        ["ndvi", "lswi", "ndre", "evi"]
    )
    reduced = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            ee.Image(
                ee.Algorithms.If(
                    ee.String(reducer_name).compareTo("max").eq(0),
                    collection.select(["ndvi", "lswi", "ndre", "evi"]).max(),
                    collection.select(["ndvi", "lswi", "ndre", "evi"]).median(),
                )
            ),
            fallback,
        )
    )
    obs_count = ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            collection.select("ndvi").count(),
            ee.Image.constant(0),
        )
    ).rename(f"{prefix}_obs_count")
    return reduced.rename(
        [f"{prefix}_ndvi", f"{prefix}_lswi", f"{prefix}_ndre", f"{prefix}_evi"]
    ).addBands(obs_count)


def build_candidate_mask(year: int, config: Config, hebei: ee.Geometry) -> ee.Image:
    """Build the winter-wheat candidate binary mask for a given season year."""
    cropland_mask = get_cropland_mask()
    windows = get_season_windows(year)
    autumn = composite_window(
        windows["autumn_start"], windows["autumn_end"], hebei, "median", "autumn", config)
    winter = composite_window(
        windows["winter_start"], windows["winter_end"], hebei, "median", "winter", config)
    spring = composite_window(
        windows["spring_start"], windows["spring_end"], hebei, "median", "spring", config)
    peak = composite_window(
        windows["peak_start"], windows["peak_end"], hebei, "max", "peak", config)
    harvest = composite_window(
        windows["harvest_start"], windows["harvest_end"], hebei, "median", "harvest", config)

    peak_ndvi = peak.select("peak_ndvi")
    spring_ndvi = spring.select("spring_ndvi")
    winter_ndvi = winter.select("winter_ndvi")
    autumn_ndvi = autumn.select("autumn_ndvi")
    harvest_ndvi = harvest.select("harvest_ndvi")
    spring_lswi = spring.select("spring_lswi")
    peak_ndre = peak.select("peak_ndre")
    ndvi_rise = spring_ndvi.subtract(winter_ndvi).rename("ndvi_rise")
    ndvi_drop = peak_ndvi.subtract(harvest_ndvi).rename("ndvi_drop")

    enough_obs = (
        autumn.select("autumn_obs_count").gte(1)
        .And(winter.select("winter_obs_count").gte(1))
        .And(spring.select("spring_obs_count").gte(1))
        .And(peak.select("peak_obs_count").gte(1))
        .And(harvest.select("harvest_obs_count").gte(1))
    )
    candidate = (
        cropland_mask.And(enough_obs)
        .And(autumn_ndvi.gt(0.20))
        .And(winter_ndvi.gt(0.18))
        .And(spring_ndvi.gt(0.42))
        .And(peak_ndvi.gt(0.58))
        .And(ndvi_rise.gt(0.12))
        .And(ndvi_drop.gt(0.20))
        .And(spring_lswi.gt(0.05))
        .And(peak_ndre.gt(0.20))
    )
    connected = candidate.selfMask().connectedPixelCount(100, True)
    cleaned = candidate.updateMask(
        connected.gte(config.min_connected_pixels)
    ).rename("winter_wheat_candidate")
    return cleaned.clip(hebei).set({
        "season_year": year,
        "province": config.province_name,
        "crop_type": "winter_wheat_candidate",
    })


# ---------------------------------------------------------------------------
# Point sampling -- avoids reduceToVectors
# ---------------------------------------------------------------------------

def sample_points_in_mask(
    mask: ee.Image, region: ee.Geometry, config: Config
) -> ee.FeatureCollection:
    """Sample a regular grid of points within the candidate mask."""
    points = mask.selfMask().sample(
        region=region,
        scale=config.sample_scale,
        numPixels=config.sample_limit,
        seed=42,
        geometries=True,
        tileScale=4,
    )
    def annotate(feature: ee.Feature) -> ee.Feature:
        coords = feature.geometry().coordinates()
        return feature.set({
            "point_id": ee.String("pt_").cat(feature.id()),
            "centroid_lon": coords.get(0),
            "centroid_lat": coords.get(1),
        })
    return points.map(annotate)


# ---------------------------------------------------------------------------
# Time series builders -- one row per (point, time_window)
# ---------------------------------------------------------------------------

def season_date_range(year: int) -> tuple[ee.Date, ee.Date]:
    return ee.Date.fromYMD(year - 1, 10, 1), ee.Date.fromYMD(year, 7, 1)


def build_mod16_table(
    points: ee.FeatureCollection, year: int, hebei: ee.Geometry
) -> ee.FeatureCollection:
    mod16_ic = (
        ee.ImageCollection("MODIS/061/MOD16A2GF")
        .filterDate(*season_date_range(year))
        .filterBounds(hebei)
        .sort("system:time_start")
    )
    image_list = mod16_ic.toList(mod16_ic.size())

    def iterate_fn(item, acc):
        acc_fc = ee.FeatureCollection(acc)
        img = ee.Image(item)
        date_start = img.date()
        date_end = date_start.advance(8, "day")
        date_repr = date_end
        metrics = (
            img.select("ET").multiply(0.1).rename("etc_8d_mm")
            .addBands(img.select("ET_QC").rename("qc_mod16"))
        )
        reduced = metrics.reduceRegions(
            collection=points, reducer=ee.Reducer.mean(),
            scale=500, tileScale=4,
        )
        annotated = reduced.map(lambda f: f.set({
            "season_year": year,
            "date_start": date_start.format("YYYY-MM-dd"),
            "date_end": date_end.format("YYYY-MM-dd"),
            "date": date_repr.format("YYYY-MM-dd"),
            "year": date_repr.get("year"),
            "doy": date_repr.getRelative("day", "year").add(1),
        }))
        return acc_fc.merge(annotated)

    return ee.FeatureCollection(image_list.iterate(iterate_fn, ee.FeatureCollection([])))


def build_s2_table(
    points: ee.FeatureCollection, year: int, hebei: ee.Geometry, config: Config
) -> ee.FeatureCollection:
    """Build S2 8-day composites aligned to MOD16 windows."""
    mod16_ic = (
        ee.ImageCollection("MODIS/061/MOD16A2GF")
        .filterDate(*season_date_range(year))
        .filterBounds(hebei)
        .sort("system:time_start")
    )
    image_list = mod16_ic.toList(mod16_ic.size())

    def _s2_indices(image: ee.Image) -> ee.Image:
        scaled = image.select(
            ["B2", "B3", "B4", "B5", "B8", "B11"]
        ).divide(10000)
        cloud_mask = ee.Image(image.get("cloud_mask")).select("probability")
        clean = scaled.updateMask(cloud_mask.lt(config.cloud_probability_threshold))
        ndvi = clean.normalizedDifference(["B8", "B4"]).rename("ndvi")
        evi = clean.expression(
            "2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1.0))",
            {"nir": clean.select("B8"), "red": clean.select("B4"), "blue": clean.select("B2")},
        ).rename("evi")
        savi = clean.expression(
            "1.5 * ((nir - red) / (nir + red + 0.5))",
            {"nir": clean.select("B8"), "red": clean.select("B4")},
        ).rename("savi")
        gndvi = clean.normalizedDifference(["B8", "B3"]).rename("gndvi")
        lswi = clean.normalizedDifference(["B8", "B11"]).rename("lswi")
        nirv = ndvi.multiply(clean.select("B8")).rename("nirv")
        re_ndvi = clean.normalizedDifference(["B8", "B5"]).rename("re_ndvi")
        obs = clean.select("B8").mask().rename("valid_obs")
        return clean.addBands([ndvi, evi, savi, gndvi, lswi, nirv, re_ndvi, obs])

    def iterate_fn(item, acc):
        acc_fc = ee.FeatureCollection(acc)
        mod16_img = ee.Image(item)
        date_start = mod16_img.date()
        date_end = date_start.advance(8, "day")

        s2_collection = build_cloud_joined_collection(date_start, date_end, hebei).map(_s2_indices)
        band_names = ["ndvi", "evi", "savi", "gndvi", "lswi", "nirv", "re_ndvi", "valid_obs"]
        fallback = ee.Image.constant([0]*len(band_names)).rename(band_names)
        composite = ee.Image(
            ee.Algorithms.If(
                s2_collection.size().gt(0),
                s2_collection.select(band_names).mean(),
                fallback,
            )
        )
        obs_count = ee.Image(
            ee.Algorithms.If(
                s2_collection.size().gt(0),
                s2_collection.select("valid_obs").sum(),
                ee.Image.constant(0),
            )
        ).rename("obs_count_s2")
        stack = composite.addBands(obs_count)
        reduced = stack.reduceRegions(
            collection=points, reducer=ee.Reducer.mean(),
            scale=config.export_scale_s2, tileScale=4,
        )
        annotated = reduced.map(lambda f: f.set({
            "season_year": year,
            "date_start": date_start.format("YYYY-MM-dd"),
            "date_end": date_end.format("YYYY-MM-dd"),
            "date": date_end.format("YYYY-MM-dd"),
            "year": date_end.get("year"),
            "doy": date_end.getRelative("day", "year").add(1),
        }))
        return acc_fc.merge(annotated)

    return ee.FeatureCollection(image_list.iterate(iterate_fn, ee.FeatureCollection([])))


def build_era5_table(
    points: ee.FeatureCollection, year: int, hebei: ee.Geometry
) -> ee.FeatureCollection:
    era5_daily = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(*season_date_range(year))
        .filterBounds(hebei)
        .sort("system:time_start")
    )
    image_list = era5_daily.toList(era5_daily.size())

    def iterate_fn(item, acc):
        acc_fc = ee.FeatureCollection(acc)
        img = ee.Image(item)
        wind_10m = img.expression(
            "sqrt(u*u + v*v)",
            {"u": img.select("u_component_of_wind_10m"),
             "v": img.select("v_component_of_wind_10m")},
        ).rename("wind_10m_m_s")
        derived = ee.Image.cat([
            img.select("temperature_2m").subtract(273.15).rename("tmean_c"),
            img.select("temperature_2m_min").subtract(273.15).rename("tmin_c"),
            img.select("temperature_2m_max").subtract(273.15).rename("tmax_c"),
            img.select("dewpoint_temperature_2m").subtract(273.15).rename("dewpoint_c"),
            img.select("surface_solar_radiation_downwards_sum").divide(1e6).rename("solar_rad_mj_m2_d"),
            img.select("total_precipitation_sum").multiply(1000).rename("precip_mm"),
            img.select("surface_pressure").divide(1000).rename("pressure_kpa"),
            wind_10m,
        ])
        reduced = derived.reduceRegions(
            collection=points, reducer=ee.Reducer.mean(),
            scale=11132, tileScale=4,
        )
        date_str = img.date().format("YYYY-MM-dd")
        annotated = reduced.map(lambda f: f.set({"date": date_str}))
        return acc_fc.merge(annotated)

    return ee.FeatureCollection(image_list.iterate(iterate_fn, ee.FeatureCollection([])))


# ---------------------------------------------------------------------------
# Area stats
# ---------------------------------------------------------------------------

def build_area_feature(mask: ee.Image, config: Config, region: ee.Geometry, year: int) -> ee.Feature:
    area_ha = mask.selfMask().multiply(ee.Image.pixelArea()).divide(10000)
    stats = area_ha.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region,
        scale=10, maxPixels=config.max_pixels,
    )
    return ee.Feature(None, {
        "season_year": year,
        "province": config.province_name,
        "crop_type": "winter_wheat_candidate",
        "area_ha": stats.get("winter_wheat_candidate"),
    })


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_table(fc: ee.FeatureCollection, description: str,
                 file_prefix: str, drive_folder: str) -> ee.batch.Task:
    task = ee.batch.Export.table.toDrive(
        collection=fc, description=description,
        folder=drive_folder, fileNamePrefix=file_prefix,
        fileFormat="CSV",
    )
    task.start()
    return task


def start_exports(
    mod16_fc: ee.FeatureCollection,
    s2_fc: ee.FeatureCollection,
    era5_fc: ee.FeatureCollection,
    area_feature: ee.Feature,
    config: Config,
) -> list[ee.batch.Task]:
    year = config.year
    tasks = [
        export_table(mod16_fc,
                     description=f"drive_hebei_kcact_mod16_etc_{year}",
                     file_prefix=f"hebei_kcact_mod16_etc_{year}",
                     drive_folder=config.drive_folder),
        export_table(s2_fc,
                     description=f"drive_hebei_kcact_s2_features_{year}",
                     file_prefix=f"hebei_kcact_s2_features_{year}",
                     drive_folder=config.drive_folder),
        export_table(era5_fc,
                     description=f"drive_hebei_kcact_era5_daily_{year}",
                     file_prefix=f"hebei_kcact_era5_daily_{year}",
                     drive_folder=config.drive_folder),
        export_table(ee.FeatureCollection([area_feature]),
                     description=f"drive_hebei_kcact_area_{year}",
                     file_prefix=f"hebei_kcact_area_{year}",
                     drive_folder=config.drive_folder),
    ]
    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def summarize_task(task: ee.batch.Task) -> dict:
    status = task.status()
    return {
        "id": status.get("id", ""),
        "state": status.get("state", ""),
        "description": status.get("description", ""),
    }


def main() -> None:
    args = parse_args()
    init_ee(args.project_id)
    config = build_config(args)

    print(f"Building candidate mask + sample points + time series for "
          f"{config.province_name} {config.year}")
    print(f"Sample scale: {config.sample_scale}m, limit: {config.sample_limit}")
    hebei = get_hebei_geometry(config)
    mask = build_candidate_mask(config.year, config, hebei)
    points = sample_points_in_mask(mask, hebei, config)
    area_feature = build_area_feature(mask, config, hebei, config.year)

    if config.export_to_drive:
        print("Building MOD16 table...")
        mod16_fc = build_mod16_table(points, config.year, hebei)
        print("Building S2 table...")
        s2_fc = build_s2_table(points, config.year, hebei, config)
        print("Building ERA5 table...")
        era5_fc = build_era5_table(points, config.year, hebei)
        tasks = start_exports(mod16_fc, s2_fc, era5_fc, area_feature, config)
        print("Export tasks submitted:")
        print(json.dumps(
            [summarize_task(t) for t in tasks],
            ensure_ascii=False, indent=2,
        ))
    else:
        print("Preview only -- no exports. (Computation graph built, use --no-export to skip.)")


if __name__ == "__main__":
    main()
