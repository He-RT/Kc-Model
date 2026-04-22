"""Hebei winter wheat mask (2021-2025) - Random Forest production version.

This is the recommended phase-1 implementation for the project:
- multi-temporal Sentinel-2 features
- cropland prior mask
- supervised Random Forest classifier
- explicit train/validation split
- yearly winter wheat mask export

Required input asset:
- A labeled sample FeatureCollection in GEE Assets.
- See docs/phase1_rf_sample_schema.md for the required schema.

Expected label convention:
- class_id = 1: winter wheat
- class_id = 0: non winter wheat

Example:
    python scripts/python/export_hebei_winter_wheat_mask_rf.py --project-id chuang-yaogan --preview-year 2025 --no-export
    python scripts/python/export_hebei_winter_wheat_mask_rf.py --project-id chuang-yaogan
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import ee


@dataclass(frozen=True)
class Config:
    country_name: str = "China"
    province_name: str = "Hebei"
    years: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
    preview_year: int = 2025
    sample_asset: str = (
        "projects/chuang-yaogan/assets/kcact/samples/"
        "hebei_winter_wheat_samples_v1"
    )
    label_property: str = "class_id"
    sample_scale: int = 10
    train_fraction: float = 0.7
    random_seed: int = 42
    cloud_probability_threshold: int = 40
    min_connected_pixels: int = 8
    tree_count: int = 200
    variables_per_split: int | None = None
    min_leaf_population: int = 2
    bag_fraction: float = 0.6
    max_nodes: int | None = None
    export_scale: int = 10
    max_pixels: float = 1e13
    export_to_asset: bool = True
    export_to_drive: bool = True
    asset_root: str = "projects/chuang-yaogan/assets/kcact/phase1_rf"
    drive_folder: str = "kcact_hebei_phase1_rf"


GAUL_L1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
WORLD_COVER_2021 = ee.ImageCollection("ESA/WorldCover/v200").first()
S2_SR = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
S2_CLOUDS = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export Hebei winter wheat Random Forest masks in GEE."
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="GEE project id, for example: chuang-yaogan",
    )
    parser.add_argument(
        "--sample-asset",
        default=Config.sample_asset,
        help="Training sample FeatureCollection asset path.",
    )
    parser.add_argument(
        "--preview-year",
        type=int,
        default=Config.preview_year,
        help="Year used for console preview metrics.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(Config.years),
        help="Season years to process.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable both asset and drive export tasks.",
    )
    parser.add_argument(
        "--asset-root",
        default=Config.asset_root,
        help="Asset directory for raster exports.",
    )
    parser.add_argument(
        "--drive-folder",
        default=Config.drive_folder,
        help="Drive folder used for image and CSV exports.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=Config.train_fraction,
        help="Training split fraction between 0 and 1.",
    )
    parser.add_argument(
        "--tree-count",
        type=int,
        default=Config.tree_count,
        help="Random Forest tree count.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=Config.random_seed,
        help="Random seed used for split and RF.",
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
        years=tuple(args.years),
        preview_year=args.preview_year,
        sample_asset=args.sample_asset,
        train_fraction=args.train_fraction,
        tree_count=args.tree_count,
        random_seed=args.random_seed,
        export_to_asset=not args.no_export,
        export_to_drive=not args.no_export,
        asset_root=args.asset_root,
        drive_folder=args.drive_folder,
    )


def get_hebei_geometry(config: Config) -> ee.Geometry:
    return (
        GAUL_L1.filter(ee.Filter.eq("ADM0_NAME", config.country_name))
        .filter(ee.Filter.eq("ADM1_NAME", config.province_name))
        .geometry()
    )


def get_cropland_mask() -> ee.Image:
    return WORLD_COVER_2021.select("Map").eq(40).rename("cropland")


def mask_edges(image: ee.Image) -> ee.Image:
    return image.updateMask(
        image.select("B8A").mask().updateMask(image.select("B9").mask())
    )


def build_cloud_joined_collection(
    start_date: ee.Date, end_date: ee.Date, region: ee.Geometry
) -> ee.ImageCollection:
    sr = (
        S2_SR.filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(mask_edges)
    )

    clouds = S2_CLOUDS.filterBounds(region).filterDate(start_date, end_date)

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

    features = ee.Image(
        ee.Algorithms.If(
            ee.String(reducer_name).compareTo("max").eq(0),
            collection.select(["ndvi", "lswi", "ndre", "evi"]).max(),
            collection.select(["ndvi", "lswi", "ndre", "evi"]).median(),
        )
    )

    obs_count = collection.select("ndvi").count().rename(f"{prefix}_obs_count")

    return features.rename(
        [f"{prefix}_ndvi", f"{prefix}_lswi", f"{prefix}_ndre", f"{prefix}_evi"]
    ).addBands(obs_count)


def build_feature_stack(
    year: int, config: Config, hebei: ee.Geometry, cropland_mask: ee.Image
) -> ee.Image:
    windows = get_season_windows(year)

    autumn = composite_window(
        windows["autumn_start"],
        windows["autumn_end"],
        hebei,
        "median",
        "autumn",
        config,
    )
    winter = composite_window(
        windows["winter_start"],
        windows["winter_end"],
        hebei,
        "median",
        "winter",
        config,
    )
    spring = composite_window(
        windows["spring_start"],
        windows["spring_end"],
        hebei,
        "median",
        "spring",
        config,
    )
    peak = composite_window(
        windows["peak_start"],
        windows["peak_end"],
        hebei,
        "max",
        "peak",
        config,
    )
    harvest = composite_window(
        windows["harvest_start"],
        windows["harvest_end"],
        hebei,
        "median",
        "harvest",
        config,
    )

    ndvi_rise = spring.select("spring_ndvi").subtract(
        winter.select("winter_ndvi")
    ).rename("ndvi_rise")
    ndvi_drop = peak.select("peak_ndvi").subtract(
        harvest.select("harvest_ndvi")
    ).rename("ndvi_drop")
    autumn_to_peak_rise = peak.select("peak_ndvi").subtract(
        autumn.select("autumn_ndvi")
    ).rename("autumn_to_peak_rise")

    seasonal_mean_ndvi = (
        ee.Image.cat(
            [
                autumn.select("autumn_ndvi"),
                winter.select("winter_ndvi"),
                spring.select("spring_ndvi"),
                peak.select("peak_ndvi"),
                harvest.select("harvest_ndvi"),
            ]
        )
        .reduce(ee.Reducer.mean())
        .rename("seasonal_mean_ndvi")
    )

    all_features = ee.Image.cat(
        [
            autumn,
            winter,
            spring,
            peak,
            harvest,
            ndvi_rise,
            ndvi_drop,
            autumn_to_peak_rise,
            seasonal_mean_ndvi,
            cropland_mask,
        ]
    ).clip(hebei)

    valid_obs_mask = (
        autumn.select("autumn_obs_count")
        .gte(1)
        .And(winter.select("winter_obs_count").gte(1))
        .And(spring.select("spring_obs_count").gte(1))
        .And(peak.select("peak_obs_count").gte(1))
        .And(harvest.select("harvest_obs_count").gte(1))
    )

    return all_features.updateMask(valid_obs_mask).set(
        {
            "season_year": year,
            "province": config.province_name,
            "crop_type": "winter_wheat",
        }
    )


def sample_feature_stack(
    image: ee.Image, sample_asset: ee.FeatureCollection, config: Config
) -> ee.FeatureCollection:
    return image.sampleRegions(
        collection=sample_asset,
        properties=[config.label_property],
        scale=config.sample_scale,
        geometries=True,
        tileScale=4,
    )


def build_train_validation_samples(
    image: ee.Image, sample_asset: ee.FeatureCollection, config: Config
) -> tuple[ee.FeatureCollection, ee.FeatureCollection, ee.FeatureCollection]:
    sampled = (
        sample_feature_stack(image, sample_asset, config)
        .filter(ee.Filter.notNull(image.bandNames()))
        .randomColumn("split", config.random_seed)
    )
    train = sampled.filter(ee.Filter.lt("split", config.train_fraction))
    validation = sampled.filter(ee.Filter.gte("split", config.train_fraction))
    return sampled, train, validation


def train_classifier(image: ee.Image, train_samples: ee.FeatureCollection, config: Config) -> ee.Classifier:
    return ee.Classifier.smileRandomForest(
        numberOfTrees=config.tree_count,
        variablesPerSplit=config.variables_per_split,
        minLeafPopulation=config.min_leaf_population,
        bagFraction=config.bag_fraction,
        maxNodes=config.max_nodes,
        seed=config.random_seed,
    ).train(
        features=train_samples,
        classProperty=config.label_property,
        inputProperties=image.bandNames(),
    )


def classify_winter_wheat(
    year: int,
    config: Config,
    hebei: ee.Geometry,
    cropland_mask: ee.Image,
    sample_asset: ee.FeatureCollection,
) -> dict[str, object]:
    features = build_feature_stack(year, config, hebei, cropland_mask)
    sampled, train, validation = build_train_validation_samples(
        features, sample_asset, config
    )
    classifier = train_classifier(features, train, config)
    classified = features.classify(classifier).rename("class_id")
    winter_wheat = classified.eq(1).And(cropland_mask).rename("winter_wheat")
    connected = winter_wheat.selfMask().connectedPixelCount(100, True)
    cleaned = winter_wheat.updateMask(
        connected.gte(config.min_connected_pixels)
    ).rename("winter_wheat")

    train_matrix = classifier.confusionMatrix()
    validated = validation.classify(classifier)
    validation_matrix = validated.errorMatrix(config.label_property, "classification")

    return {
        "year": year,
        "features": features,
        "classifier": classifier,
        "sampled": sampled,
        "train": train,
        "validation": validation,
        "validated": validated,
        "train_matrix": train_matrix,
        "validation_matrix": validation_matrix,
        "classified": classified,
        "winter_wheat": cleaned,
    }


def build_metrics_feature(result: dict[str, object], config: Config) -> ee.Feature:
    train_matrix = result["train_matrix"]
    validation_matrix = result["validation_matrix"]
    return ee.Feature(
        None,
        {
            "season_year": result["year"],
            "province": config.province_name,
            "crop_type": "winter_wheat",
            "sample_count_all": ee.FeatureCollection(result["sampled"]).size(),
            "sample_count_train": ee.FeatureCollection(result["train"]).size(),
            "sample_count_validation": ee.FeatureCollection(result["validation"]).size(),
            "train_accuracy": train_matrix.accuracy(),
            "train_kappa": train_matrix.kappa(),
            "validation_accuracy": validation_matrix.accuracy(),
            "validation_kappa": validation_matrix.kappa(),
            "train_matrix": ee.String(train_matrix.array()),
            "validation_matrix": ee.String(validation_matrix.array()),
        },
    )


def build_area_feature(
    result: dict[str, object], config: Config, hebei: ee.Geometry
) -> ee.Feature:
    area_ha = (
        ee.Image(result["winter_wheat"])
        .selfMask()
        .multiply(ee.Image.pixelArea())
        .divide(10000)
    )
    stats = area_ha.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=hebei,
        scale=config.export_scale,
        maxPixels=config.max_pixels,
    )
    return ee.Feature(
        None,
        {
            "season_year": result["year"],
            "province": config.province_name,
            "crop_type": "winter_wheat",
            "area_ha": stats.get("winter_wheat"),
        },
    )


def start_export_tasks(
    results: Sequence[dict[str, object]],
    metrics_fc: ee.FeatureCollection,
    area_fc: ee.FeatureCollection,
    config: Config,
    hebei: ee.Geometry,
) -> list[ee.batch.Task]:
    tasks: list[ee.batch.Task] = []

    if config.export_to_drive:
        metrics_task = ee.batch.Export.table.toDrive(
            collection=metrics_fc,
            description="drive_hebei_winter_wheat_rf_metrics_2021_2025",
            folder=config.drive_folder,
            fileNamePrefix="hebei_winter_wheat_rf_metrics_2021_2025",
            fileFormat="CSV",
        )
        metrics_task.start()
        tasks.append(metrics_task)

        area_task = ee.batch.Export.table.toDrive(
            collection=area_fc,
            description="drive_hebei_winter_wheat_rf_area_summary_2021_2025",
            folder=config.drive_folder,
            fileNamePrefix="hebei_winter_wheat_rf_area_summary_2021_2025",
            fileFormat="CSV",
        )
        area_task.start()
        tasks.append(area_task)

    for result in results:
        year = result["year"]
        image = ee.Image(result["winter_wheat"]).toUint8()

        if config.export_to_asset:
            asset_task = ee.batch.Export.image.toAsset(
                image=image,
                description=f"asset_hebei_winter_wheat_rf_mask_{year}",
                assetId=f"{config.asset_root}/winter_wheat_mask_hebei_rf_{year}",
                region=hebei,
                scale=config.export_scale,
                maxPixels=config.max_pixels,
            )
            asset_task.start()
            tasks.append(asset_task)

        if config.export_to_drive:
            drive_task = ee.batch.Export.image.toDrive(
                image=image,
                description=f"drive_hebei_winter_wheat_rf_mask_{year}",
                fileNamePrefix=f"hebei_winter_wheat_rf_mask_{year}",
                folder=config.drive_folder,
                region=hebei,
                scale=config.export_scale,
                maxPixels=config.max_pixels,
                fileFormat="GeoTIFF",
            )
            drive_task.start()
            tasks.append(drive_task)

    return tasks


def summarize_task(task: ee.batch.Task) -> dict[str, str]:
    status = task.status()
    return {
        "id": status.get("id", ""),
        "state": status.get("state", ""),
        "description": status.get("description", ""),
    }


def print_preview(result: dict[str, object], sample_asset: ee.FeatureCollection) -> None:
    print("Sample asset count:", sample_asset.size().getInfo())
    print("Preview year:", result["year"])
    print("Training confusion matrix:", result["train_matrix"].getInfo())
    print("Training accuracy:", result["train_matrix"].accuracy().getInfo())
    print("Training kappa:", result["train_matrix"].kappa().getInfo())
    print("Validation confusion matrix:", result["validation_matrix"].getInfo())
    print("Validation accuracy:", result["validation_matrix"].accuracy().getInfo())
    print("Validation kappa:", result["validation_matrix"].kappa().getInfo())
    print("RF explanation:", json.dumps(result["classifier"].explain().getInfo(), ensure_ascii=False))


def main() -> None:
    args = parse_args()
    init_ee(args.project_id)
    config = build_config(args)

    hebei = get_hebei_geometry(config)
    cropland_mask = get_cropland_mask()
    sample_asset = ee.FeatureCollection(config.sample_asset).filterBounds(hebei)

    preview_result = classify_winter_wheat(
        config.preview_year, config, hebei, cropland_mask, sample_asset
    )
    print_preview(preview_result, sample_asset)

    results = [
        classify_winter_wheat(year, config, hebei, cropland_mask, sample_asset)
        for year in config.years
    ]
    metrics_fc = ee.FeatureCollection(
        [build_metrics_feature(result, config) for result in results]
    )
    area_fc = ee.FeatureCollection(
        [build_area_feature(result, config, hebei) for result in results]
    )

    print("Yearly metrics:")
    print(json.dumps(metrics_fc.getInfo(), ensure_ascii=False))
    print("Yearly area summary:")
    print(json.dumps(area_fc.getInfo(), ensure_ascii=False))

    if config.export_to_asset or config.export_to_drive:
        tasks = start_export_tasks(results, metrics_fc, area_fc, config, hebei)
        print("Started export tasks:")
        print(json.dumps([summarize_task(task) for task in tasks], ensure_ascii=False, indent=2))
    else:
        print("Export disabled: preview and metrics only.")


if __name__ == "__main__":
    main()
