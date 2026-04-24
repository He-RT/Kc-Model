"""Export patch-level S2, ERA5-Land, and MOD16 time series from GEE.

Expected patch asset:
- FeatureCollection in GEE Assets
- Includes patch geometries, ideally with patch_id, area_ha, centroid_lat, centroid_lon

This script exports:
- Sentinel-2 feature table aligned to MOD16 8-day windows
- ERA5-Land daily weather table
- MOD16 ETc 8-day table
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
    years: tuple[int, ...] = (2021, 2022, 2023)
    patch_asset: str = ""
    drive_folder: str = "kcact_hebei_patch_timeseries"
    export_scale_s2: int = 10
    export_scale_era5: int = 11132
    export_scale_mod16: int = 500
    max_pixels: float = 1e13
    cloud_probability_threshold: int = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export S2, ERA5-Land, and MOD16 patch time series from Earth Engine."
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--patch-asset", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=list(Config.years))
    parser.add_argument("--drive-folder", default=Config.drive_folder)
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
        patch_asset=args.patch_asset,
        drive_folder=args.drive_folder,
    )


def get_hebei_geometry(config: Config) -> ee.Geometry:
    gaul_l1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    return (
        gaul_l1.filter(ee.Filter.eq("ADM0_NAME", config.country_name))
        .filter(ee.Filter.eq("ADM1_NAME", config.boundary_province_name))
        .geometry()
    )


def ensure_patch_properties(patches: ee.FeatureCollection) -> ee.FeatureCollection:
    def annotate(feature: ee.Feature) -> ee.Feature:
        geom = feature.geometry()
        centroid = geom.centroid(1)
        coords = centroid.coordinates()
        patch_id = ee.Algorithms.If(feature.propertyNames().contains("patch_id"), feature.get("patch_id"), feature.id())
        area_ha = ee.Algorithms.If(feature.propertyNames().contains("area_ha"), feature.get("area_ha"), geom.area(1).divide(10000))
        return feature.set(
            {
                "patch_id": ee.String(patch_id),
                "area_ha": area_ha,
                "centroid_lon": coords.get(0),
                "centroid_lat": coords.get(1),
            }
        )

    return patches.map(annotate)


def build_cloud_joined_collection(start_date: ee.Date, end_date: ee.Date, region: ee.Geometry) -> ee.ImageCollection:
    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    s2_clouds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    sr = (
        s2_sr.filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
    )
    clouds = s2_clouds.filterBounds(region).filterDate(start_date, end_date)
    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=sr,
        secondary=clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined)


def add_s2_indices(image: ee.Image, config: Config) -> ee.Image:
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


def season_date_range(year: int) -> tuple[ee.Date, ee.Date]:
    return ee.Date.fromYMD(year - 1, 10, 1), ee.Date.fromYMD(year, 7, 1)


def build_mod16_features_for_image(img: ee.Image, patches: ee.FeatureCollection, year: int) -> ee.FeatureCollection:
    date_start = img.date()
    date_end = date_start.advance(8, "day")
    date_repr = date_end
    metrics = img.select("ET").multiply(0.1).rename("etc_8d_mm").addBands(img.select("ET_QC").rename("qc_mod16"))
    reduced = metrics.reduceRegions(
        collection=patches,
        reducer=ee.Reducer.mean(),
        scale=500,
        tileScale=4,
    )
    return reduced.map(
        lambda feature: feature.set(
            {
                "season_year": year,
                "date_start": date_start.format("YYYY-MM-dd"),
                "date_end": date_end.format("YYYY-MM-dd"),
                "date": date_repr.format("YYYY-MM-dd"),
                "year": date_repr.get("year"),
                "doy": date_repr.getRelative("day", "year").add(1),
            }
        )
    )


def build_s2_features_for_image(img: ee.Image, patches: ee.FeatureCollection, config: Config, year: int) -> ee.FeatureCollection:
    date_start = img.date()
    date_end = date_start.advance(8, "day")
    s2_collection = build_cloud_joined_collection(date_start, date_end, patches.geometry()).map(
        lambda image: add_s2_indices(image, config)
    )

    fallback = ee.Image.constant([0, 0, 0, 0, 0, 0, 0, 0]).rename(
        ["ndvi", "evi", "savi", "gndvi", "lswi", "nirv", "re_ndvi", "valid_obs"]
    )
    composite = ee.Image(
        ee.Algorithms.If(
            s2_collection.size().gt(0),
            s2_collection.select(["ndvi", "evi", "savi", "gndvi", "lswi", "nirv", "re_ndvi", "valid_obs"]).mean(),
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
    reduced = composite.addBands(obs_count).reduceRegions(
        collection=patches,
        reducer=ee.Reducer.mean(),
        scale=config.export_scale_s2,
        tileScale=4,
    )
    return reduced.map(
        lambda feature: feature.set(
            {
                "season_year": year,
                "date_start": date_start.format("YYYY-MM-dd"),
                "date_end": date_end.format("YYYY-MM-dd"),
                "date": date_end.format("YYYY-MM-dd"),
                "year": date_end.get("year"),
                "doy": date_end.getRelative("day", "year").add(1),
            }
        )
    )


def build_era5_daily_feature(img: ee.Image, patches: ee.FeatureCollection) -> ee.FeatureCollection:
    wind_10m = img.expression(
        "sqrt(u*u + v*v)",
        {"u": img.select("u_component_of_wind_10m"), "v": img.select("v_component_of_wind_10m")},
    ).rename("wind_10m_m_s")
    derived = ee.Image.cat(
        [
            img.select("temperature_2m").subtract(273.15).rename("tmean_c"),
            img.select("temperature_2m_min").subtract(273.15).rename("tmin_c"),
            img.select("temperature_2m_max").subtract(273.15).rename("tmax_c"),
            img.select("dewpoint_temperature_2m").subtract(273.15).rename("dewpoint_c"),
            img.select("surface_solar_radiation_downwards_sum").divide(1e6).rename("solar_rad_mj_m2_d"),
            img.select("total_precipitation_sum").multiply(1000).rename("precip_mm"),
            img.select("surface_pressure").divide(1000).rename("pressure_kpa"),
            wind_10m,
        ]
    )
    date = img.date()
    reduced = derived.reduceRegions(
        collection=patches,
        reducer=ee.Reducer.mean(),
        scale=11132,
        tileScale=4,
    )
    return reduced.map(
        lambda feature: feature.set({"date": date.format("YYYY-MM-dd")})
    )


def flatten_reduced_collection(
    image_list: ee.List,
    builder,
    patches: ee.FeatureCollection,
    *extra_args,
) -> ee.FeatureCollection:
    def iterate_fn(item, acc):
        acc_fc = ee.FeatureCollection(acc)
        current = builder(ee.Image(item), patches, *extra_args)
        return acc_fc.merge(current)

    return ee.FeatureCollection(image_list.iterate(iterate_fn, ee.FeatureCollection([])))


def export_table(collection: ee.FeatureCollection, description: str, file_prefix: str, drive_folder: str) -> ee.batch.Task:
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=drive_folder,
        fileNamePrefix=file_prefix,
        fileFormat="CSV",
    )
    task.start()
    return task


def main() -> None:
    args = parse_args()
    init_ee(args.project_id)
    config = build_config(args)
    hebei = get_hebei_geometry(config)
    patches = ensure_patch_properties(
        ee.FeatureCollection(config.patch_asset).filterBounds(hebei)
    )
    mod16_ic = ee.ImageCollection("MODIS/061/MOD16A2GF")
    era5_daily = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

    tasks = []
    for year in config.years:
        start_date, end_date = season_date_range(year)
        mod16_year = mod16_ic.filterDate(start_date, end_date).filterBounds(hebei).sort("system:time_start")
        era5_year = era5_daily.filterDate(start_date, end_date).filterBounds(hebei).sort("system:time_start")

        mod16_fc = flatten_reduced_collection(
            mod16_year.toList(mod16_year.size()),
            build_mod16_features_for_image,
            patches,
            year,
        )
        s2_fc = flatten_reduced_collection(
            mod16_year.toList(mod16_year.size()),
            build_s2_features_for_image,
            patches,
            config,
            year,
        )
        era5_fc = flatten_reduced_collection(
            era5_year.toList(era5_year.size()),
            build_era5_daily_feature,
            patches,
        )

        tasks.extend(
            [
                export_table(
                    mod16_fc,
                    description=f"drive_hebei_patch_mod16_etc_{year}",
                    file_prefix=f"hebei_patch_mod16_etc_{year}",
                    drive_folder=config.drive_folder,
                ),
                export_table(
                    s2_fc,
                    description=f"drive_hebei_patch_s2_features_{year}",
                    file_prefix=f"hebei_patch_s2_features_{year}",
                    drive_folder=config.drive_folder,
                ),
                export_table(
                    era5_fc,
                    description=f"drive_hebei_patch_era5_daily_{year}",
                    file_prefix=f"hebei_patch_era5_daily_{year}",
                    drive_folder=config.drive_folder,
                ),
            ]
        )

    print(
        json.dumps(
            [
                {
                    "id": task.status().get("id", ""),
                    "state": task.status().get("state", ""),
                    "description": task.status().get("description", ""),
                }
                for task in tasks
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
