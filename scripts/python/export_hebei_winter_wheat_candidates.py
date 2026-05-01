"""Export Hebei winter wheat candidate mask and patch list with no training samples.

This script is the no-sample fallback when the supervised RF route is blocked by
missing labels. It identifies winter wheat candidate regions in Hebei using:

- Sentinel-2 SR Harmonized
- Sentinel-2 cloud probability
- ESA WorldCover cropland prior
- phenology rules tailored to Hebei winter wheat

Outputs:
- GeoTIFF mask export to Google Drive
- patch polygons export to Google Drive
- yearly area summary export to Google Drive

Example:
    python scripts/python/export_hebei_winter_wheat_candidates.py \\
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
    cloud_probability_threshold: int = 40
    min_connected_pixels: int = 8
    export_scale: int = 10
    vector_scale: int = 30
    max_pixels: float = 1e13
    export_to_drive: bool = True
    drive_folder: str = "kcact_hebei_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Hebei winter wheat candidate mask and vector patches."
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Earth Engine project id, for example: chuang-yaogan",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=Config.year,
        help="Season year. Example: 2025 means 2024-10 to 2025-06.",
    )
    parser.add_argument(
        "--drive-folder",
        default=Config.drive_folder,
        help="Google Drive output folder.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run preview only without starting Drive export tasks.",
    )
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
        export_to_drive=not args.no_export,
        drive_folder=args.drive_folder,
    )


def get_hebei_geometry(config: Config) -> ee.Geometry:
    gaul_l1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    return (
        gaul_l1.filter(ee.Filter.eq("ADM0_NAME", config.country_name))
        .filter(ee.Filter.eq("ADM1_NAME", config.boundary_province_name))
        .geometry()
    )


def get_cropland_mask() -> ee.Image:
    world_cover_2021 = ee.ImageCollection("ESA/WorldCover/v200").first()
    return world_cover_2021.select("Map").eq(40).rename("cropland")


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
        primary=sr,
        secondary=clouds,
        condition=ee.Filter.equals(
            leftField="system:index",
            rightField="system:index",
        ),
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
        {
            "nir": clean.select("B8"),
            "red": clean.select("B4"),
            "blue": clean.select("B2"),
        },
    ).rename("evi")

    return clean.addBands([ndvi, lswi, ndre, evi]).copyProperties(
        image, image.propertyNames()
    )


def get_season_windows(year: int) -> dict[str, ee.Date]:
    return {
        "autumn_start": ee.Date.fromYMD(year - 1, 10, 1),
        "autumn_end": ee.Date.fromYMD(year - 1, 11, 30),
        "winter_start": ee.Date.fromYMD(year, 1, 1),
        "winter_end": ee.Date.fromYMD(year, 2, 15),
        "spring_start": ee.Date.fromYMD(year, 3, 1),
        "spring_end": ee.Date.fromYMD(year, 4, 15),
        "peak_start": ee.Date.fromYMD(year, 4, 16),
        "peak_end": ee.Date.fromYMD(year, 5, 20),
        "harvest_start": ee.Date.fromYMD(year, 6, 1),
        "harvest_end": ee.Date.fromYMD(year, 6, 30),
    }


def composite_window(
    start_date: ee.Date,
    end_date: ee.Date,
    region: ee.Geometry,
    reducer_name: str,
    prefix: str,
    config: Config,
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


def build_candidate_stack(
    year: int, config: Config, hebei: ee.Geometry, cropland_mask: ee.Image
) -> ee.Image:
    windows = get_season_windows(year)

    autumn = composite_window(
        windows["autumn_start"], windows["autumn_end"], hebei, "median", "autumn", config
    )
    winter = composite_window(
        windows["winter_start"], windows["winter_end"], hebei, "median", "winter", config
    )
    spring = composite_window(
        windows["spring_start"], windows["spring_end"], hebei, "median", "spring", config
    )
    peak = composite_window(
        windows["peak_start"], windows["peak_end"], hebei, "max", "peak", config
    )
    harvest = composite_window(
        windows["harvest_start"], windows["harvest_end"], hebei, "median", "harvest", config
    )

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
        autumn.select("autumn_obs_count")
        .gte(1)
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

    return ee.Image.cat(
        [
            autumn_ndvi.rename("autumn_ndvi"),
            winter_ndvi.rename("winter_ndvi"),
            spring_ndvi.rename("spring_ndvi"),
            peak_ndvi.rename("peak_ndvi"),
            harvest_ndvi.rename("harvest_ndvi"),
            ndvi_rise,
            ndvi_drop,
            spring_lswi.rename("spring_lswi"),
            peak_ndre.rename("peak_ndre"),
            cleaned,
        ]
    ).clip(hebei).set(
        {
            "season_year": year,
            "province": config.province_name,
            "crop_type": "winter_wheat_candidate",
        }
    )


def build_patch_vectors(
    mask_image: ee.Image, config: Config, region: ee.Geometry, year: int
) -> ee.FeatureCollection:
    vectors = mask_image.selfMask().reduceToVectors(
        geometry=region,
        scale=config.vector_scale,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="mask_value",
        reducer=ee.Reducer.countEvery(),
        maxPixels=config.max_pixels,
        bestEffort=True,
        tileScale=4,
    )

    def annotate(feature: ee.Feature) -> ee.Feature:
        area_ha = feature.geometry().area(1).divide(10000)
        centroid = feature.geometry().centroid(1)
        coords = ee.List(centroid.coordinates())
        bounds = ee.List(feature.geometry().bounds(1).coordinates().get(0))
        return feature.set(
            {
                "season_year": year,
                "province": config.province_name,
                "crop_type": "winter_wheat_candidate",
                "patch_id": ee.String(str(year)).cat("_").cat(feature.id()),
                "area_ha": area_ha,
                "centroid_lon": coords.get(0),
                "centroid_lat": coords.get(1),
                "bbox_ring": ee.String(bounds),
            }
        )

    return vectors.map(annotate)


def build_area_feature(mask_image: ee.Image, config: Config, region: ee.Geometry, year: int) -> ee.Feature:
    area_ha = mask_image.selfMask().multiply(ee.Image.pixelArea()).divide(10000)
    stats = area_ha.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=config.export_scale,
        maxPixels=config.max_pixels,
    )
    return ee.Feature(
        None,
        {
            "season_year": year,
            "province": config.province_name,
            "crop_type": "winter_wheat_candidate",
            "area_ha": stats.get("winter_wheat_candidate"),
        },
    )


def start_exports(
    candidate_mask: ee.Image,
    patch_vectors: ee.FeatureCollection,
    area_feature: ee.Feature,
    config: Config,
    region: ee.Geometry,
) -> list[ee.batch.Task]:
    year = config.year
    tasks: list[ee.batch.Task] = []

    mask_task = ee.batch.Export.image.toDrive(
        image=candidate_mask.toUint8(),
        description=f"drive_hebei_winter_wheat_candidate_mask_{year}",
        fileNamePrefix=f"hebei_winter_wheat_candidate_mask_{year}",
        folder=config.drive_folder,
        region=region,
        scale=config.export_scale,
        maxPixels=config.max_pixels,
        fileFormat="GeoTIFF",
    )
    mask_task.start()
    tasks.append(mask_task)

    vector_task = ee.batch.Export.table.toDrive(
        collection=patch_vectors,
        description=f"drive_hebei_winter_wheat_candidate_patches_{year}",
        fileNamePrefix=f"hebei_winter_wheat_candidate_patches_{year}",
        folder=config.drive_folder,
        fileFormat="GeoJSON",
    )
    vector_task.start()
    tasks.append(vector_task)

    area_task = ee.batch.Export.table.toDrive(
        collection=ee.FeatureCollection([area_feature]),
        description=f"drive_hebei_winter_wheat_candidate_area_{year}",
        fileNamePrefix=f"hebei_winter_wheat_candidate_area_{year}",
        folder=config.drive_folder,
        fileFormat="CSV",
    )
    area_task.start()
    tasks.append(area_task)

    return tasks


def summarize_task(task: ee.batch.Task) -> dict[str, str]:
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

    hebei = get_hebei_geometry(config)
    cropland_mask = get_cropland_mask()
    candidate_stack = build_candidate_stack(config.year, config, hebei, cropland_mask)
    candidate_mask = candidate_stack.select("winter_wheat_candidate")
    patch_vectors = build_patch_vectors(candidate_mask, config, hebei, config.year)
    area_feature = build_area_feature(candidate_mask, config, hebei, config.year)

    print("Province:", config.province_name)
    print("Season year:", config.year)

    if config.export_to_drive:
        tasks = start_exports(candidate_mask, patch_vectors, area_feature, config, hebei)
        print("Started export tasks:")
        print(json.dumps([summarize_task(task) for task in tasks], ensure_ascii=False, indent=2))
    else:
        print("Candidate area summary:")
        print(json.dumps(area_feature.getInfo(), ensure_ascii=False, indent=2))
        print("Export disabled.")


if __name__ == "__main__":
    main()
