"""Export point-based Kcact training data from GEE for NCP summer maize.

Reuses the same NCP provinces and 1000m grid as the winter wheat pipeline,
but for the maize growing season (June–October).

Summer maize phenology (NCP):
  - Early June: wheat harvested, bare soil / maize germination (NDVI ~0.2–0.3)
  - Late June–July: rapid vegetative growth (NDVI 0.3→0.7)
  - Late July–August: peak (NDVI > 0.65)
  - September: grain fill / senescence start (NDVI decline ≥ 0.10)
  - October: maturity / harvest (NDVI < 0.50)

Uses ESA WorldCover cropland as the sampling frame (maize is the dominant
summer crop in NCP; mixed pixels with soybean/cotton are a known limitation).

Example:
  python scripts/python/export_maize_kcact_training_data.py \\
    --project-id chuang-yaogan --province Hebei --year 2020 --no-export
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import ee

PROVINCE_REGISTRY = {
    "Hebei":    {"boundary": "Hebei Sheng",    "lat_filter_max": None},
    "Henan":    {"boundary": "Henan Sheng",    "lat_filter_max": None},
    "Shandong": {"boundary": "Shandong Sheng", "lat_filter_max": None},
    "Anhui":    {"boundary": "Anhui Sheng",    "lat_filter_max": 33.5},
}


@dataclass(frozen=True)
class Config:
    country_name: str = "China"
    province_name: str = "Hebei"
    boundary_province_name: str = "Hebei Sheng"
    year: int = 2020
    sample_scale: int = 1000
    sample_limit: int = 10000
    lat_filter_max: float | None = None
    cloud_probability_threshold: int = 40
    min_connected_pixels: int = 8
    export_scale_s2: int = 10
    export_scale_era5: int = 11132
    export_scale_mod16: int = 500
    max_pixels: float = 1e13
    export_to_drive: bool = True
    drive_folder: str = "kcact_hebei_maize_training_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export point-based Kcact training data for NCP summer maize."
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--province", default="Hebei",
                        choices=list(PROVINCE_REGISTRY.keys()))
    parser.add_argument("--year", type=int, default=Config.year)
    parser.add_argument("--sample-scale", type=int, default=Config.sample_scale)
    parser.add_argument("--sample-limit", type=int, default=Config.sample_limit)
    parser.add_argument("--lat-filter-max", type=float, default=None)
    parser.add_argument("--drive-folder", default=None)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def build_config(args: argparse.Namespace) -> Config:
    prov = PROVINCE_REGISTRY[args.province]
    lat_filter = args.lat_filter_max if args.lat_filter_max is not None else prov["lat_filter_max"]
    drive_folder = args.drive_folder or f"kcact_{args.province.lower()}_maize_training_data"
    return Config(
        province_name=args.province,
        boundary_province_name=prov["boundary"],
        year=args.year,
        sample_scale=args.sample_scale,
        sample_limit=args.sample_limit,
        lat_filter_max=lat_filter,
        export_to_drive=not args.no_export,
        drive_folder=drive_folder,
    )


# ---------------------------------------------------------------------------
# Maize candidate mask
# ---------------------------------------------------------------------------

def get_province_geometry(config: Config) -> ee.Geometry:
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
    return clean.addBands([ndvi, lswi, ndre, evi]).copyProperties(image, image.propertyNames())


def get_maize_season_windows(year: int) -> dict[str, ee.Date]:
    """Summer maize phenology windows for NCP (Jun–Oct)."""
    return {
        "early_start":    ee.Date.fromYMD(year, 6, 1),
        "early_end":      ee.Date.fromYMD(year, 6, 30),
        "growth_start":   ee.Date.fromYMD(year, 7, 1),
        "growth_end":     ee.Date.fromYMD(year, 7, 31),
        "peak_start":     ee.Date.fromYMD(year, 8, 1),
        "peak_end":       ee.Date.fromYMD(year, 8, 20),
        "senescence_start": ee.Date.fromYMD(year, 9, 1),
        "senescence_end":   ee.Date.fromYMD(year, 9, 30),
        "harvest_start":  ee.Date.fromYMD(year, 10, 1),
        "harvest_end":    ee.Date.fromYMD(year, 10, 31),
    }


def composite_window(start_date, end_date, region, reducer_name, prefix, config):
    collection = build_cloud_joined_collection(start_date, end_date, region).map(
        lambda image: add_indices(image, config)
    )
    fallback = ee.Image.constant([0, 0, 0, 0]).rename(["ndvi", "lswi", "ndre", "evi"])
    reduced = ee.Image(ee.Algorithms.If(
        collection.size().gt(0),
        ee.Image(ee.Algorithms.If(
            ee.String(reducer_name).compareTo("max").eq(0),
            collection.select(["ndvi", "lswi", "ndre", "evi"]).max(),
            collection.select(["ndvi", "lswi", "ndre", "evi"]).median(),
        )),
        fallback,
    ))
    obs_count = ee.Image(ee.Algorithms.If(
        collection.size().gt(0),
        collection.select("ndvi").count(),
        ee.Image.constant(0),
    )).rename(f"{prefix}_obs_count")
    return reduced.rename(
        [f"{prefix}_ndvi", f"{prefix}_lswi", f"{prefix}_ndre", f"{prefix}_evi"]
    ).addBands(obs_count)


def build_maize_mask(year: int, config: Config, region: ee.Geometry) -> ee.Image:
    """Build summer maize candidate mask from S2 phenology.

    Rules (NCP summer maize):
      - Cropland (WorldCover) in the province
      - Early June NDVI < 0.40 (post wheat harvest)
      - July NDVI rise > 0.15 (maize vegetative growth)
      - Late July–August peak NDVI > 0.60 (maize canopy closure)
      - September NDVI drop > 0.05 (senescence start)
      - Sufficient S2 observations in each window
    """
    cropland_mask = get_cropland_mask()
    windows = get_maize_season_windows(year)

    early = composite_window(windows["early_start"], windows["early_end"], region, "median", "early", config)
    growth = composite_window(windows["growth_start"], windows["growth_end"], region, "median", "growth", config)
    peak = composite_window(windows["peak_start"], windows["peak_end"], region, "max", "peak", config)
    senescence = composite_window(windows["senescence_start"], windows["senescence_end"], region, "median", "senescence", config)
    harvest = composite_window(windows["harvest_start"], windows["harvest_end"], region, "median", "harvest", config)

    early_ndvi = early.select("early_ndvi")
    growth_ndvi = growth.select("growth_ndvi")
    peak_ndvi = peak.select("peak_ndvi")
    senescence_ndvi = senescence.select("senescence_ndvi")
    harvest_ndvi = harvest.select("harvest_ndvi")
    peak_ndre = peak.select("peak_ndre")

    ndvi_rise = growth_ndvi.subtract(early_ndvi).rename("ndvi_rise")
    ndvi_drop = peak_ndvi.subtract(harvest_ndvi).rename("ndvi_drop")

    enough_obs = (
        early.select("early_obs_count").gte(1)
        .And(growth.select("growth_obs_count").gte(1))
        .And(peak.select("peak_obs_count").gte(1))
        .And(senescence.select("senescence_obs_count").gte(1))
        .And(harvest.select("harvest_obs_count").gte(1))
    )

    candidate = (
        cropland_mask.And(enough_obs)
        .And(early_ndvi.lt(0.40))
        .And(growth_ndvi.gt(0.40))
        .And(peak_ndvi.gt(0.60))
        .And(ndvi_rise.gt(0.15))
        .And(ndvi_drop.gt(0.05))
        .And(senescence_ndvi.gt(0.35))
        .And(peak_ndre.gt(0.20))
    )

    connected = candidate.selfMask().connectedPixelCount(100, True)
    cleaned = candidate.updateMask(
        connected.gte(config.min_connected_pixels)
    ).rename("summer_maize_candidate")
    return cleaned.clip(region).set({
        "season_year": year,
        "province": config.province_name,
        "crop_type": "summer_maize_candidate",
    })


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def sample_points_in_mask(mask, region, config):
    points = mask.selfMask().sample(
        region=region, scale=config.sample_scale,
        numPixels=config.sample_limit, seed=42,
        geometries=True, tileScale=4,
    )

    def annotate(feature):
        coords = feature.geometry().coordinates()
        return feature.set({
            "point_id": ee.String("pt_").cat(feature.id()),
            "centroid_lon": coords.get(0),
            "centroid_lat": coords.get(1),
        })

    annotated = points.map(annotate)
    if config.lat_filter_max is not None:
        annotated = annotated.filter(ee.Filter.lt("centroid_lat", config.lat_filter_max))
    return annotated


# ---------------------------------------------------------------------------
# Time series (same as wheat — just different date range)
# ---------------------------------------------------------------------------

def maize_date_range(year: int) -> tuple[ee.Date, ee.Date]:
    return ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 11, 1)


def build_mod16_table(points, year, region):
    mod16_ic = (
        ee.ImageCollection("MODIS/061/MOD16A2GF")
        .filterDate(*maize_date_range(year))
        .filterBounds(region)
        .sort("system:time_start")
    )
    image_list = mod16_ic.toList(mod16_ic.size())

    def iterate_fn(item, acc):
        acc_fc = ee.FeatureCollection(acc)
        img = ee.Image(item)
        date_start = img.date()
        date_end = date_start.advance(8, "day")
        date_repr = date_end
        metrics = img.select("ET").multiply(0.1).rename("etc_8d_mm").addBands(
            img.select("ET_QC").rename("qc_mod16"))
        reduced = metrics.reduceRegions(collection=points, reducer=ee.Reducer.mean(),
                                        scale=500, tileScale=4)
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


def build_s2_table(points, year, region, config):
    mod16_ic = (
        ee.ImageCollection("MODIS/061/MOD16A2GF")
        .filterDate(*maize_date_range(year))
        .filterBounds(region)
        .sort("system:time_start")
    )
    image_list = mod16_ic.toList(mod16_ic.size())

    def _s2_indices(image):
        scaled = image.select(["B2", "B3", "B4", "B5", "B8", "B11"]).divide(10000)
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
        s2_collection = build_cloud_joined_collection(date_start, date_end, region).map(_s2_indices)
        band_names = ["ndvi", "evi", "savi", "gndvi", "lswi", "nirv", "re_ndvi", "valid_obs"]
        fallback = ee.Image.constant([0]*len(band_names)).rename(band_names)
        composite = ee.Image(ee.Algorithms.If(
            s2_collection.size().gt(0),
            s2_collection.select(band_names).mean(),
            fallback,
        ))
        obs_count = ee.Image(ee.Algorithms.If(
            s2_collection.size().gt(0),
            s2_collection.select("valid_obs").sum(),
            ee.Image.constant(0),
        )).rename("obs_count_s2")
        stack = composite.addBands(obs_count)
        reduced = stack.reduceRegions(collection=points, reducer=ee.Reducer.mean(),
                                      scale=config.export_scale_s2, tileScale=4)
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


def build_era5_table(points, year, region):
    era5_daily = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(*maize_date_range(year))
        .filterBounds(region)
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
        reduced = derived.reduceRegions(collection=points, reducer=ee.Reducer.mean(),
                                        scale=11132, tileScale=4)
        date_str = img.date().format("YYYY-MM-dd")
        annotated = reduced.map(lambda f: f.set({"date": date_str}))
        return acc_fc.merge(annotated)

    return ee.FeatureCollection(image_list.iterate(iterate_fn, ee.FeatureCollection([])))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_table(fc, description, file_prefix, drive_folder):
    task = ee.batch.Export.table.toDrive(
        collection=fc, description=description,
        folder=drive_folder, fileNamePrefix=file_prefix,
        fileFormat="CSV",
    )
    task.start()
    return task


def start_exports(mod16_fc, s2_fc, era5_fc, config):
    year = config.year
    prov = config.province_name.lower()
    tasks = [
        export_table(mod16_fc, f"drive_{prov}_maize_mod16_etc_{year}",
                     f"{prov}_maize_mod16_etc_{year}", config.drive_folder),
        export_table(s2_fc, f"drive_{prov}_maize_s2_features_{year}",
                     f"{prov}_maize_s2_features_{year}", config.drive_folder),
        export_table(era5_fc, f"drive_{prov}_maize_era5_daily_{year}",
                     f"{prov}_maize_era5_daily_{year}", config.drive_folder),
    ]
    return tasks


def summarize_task(task):
    status = task.status()
    return {"id": status.get("id", ""), "state": status.get("state", ""),
            "description": status.get("description", "")}


def main():
    args = parse_args()
    init_ee(args.project_id)
    config = build_config(args)

    print(f"Province: {config.province_name}  |  Year: {config.year}")
    print(f"Boundary: {config.boundary_province_name}")
    if config.lat_filter_max:
        print(f"Latitude filter: centroid_lat < {config.lat_filter_max}")
    print(f"Sample: {config.sample_scale}m / limit={config.sample_limit}")
    print(f"Drive folder: {config.drive_folder}")

    region = get_province_geometry(config)
    mask = build_maize_mask(config.year, config, region)
    points = sample_points_in_mask(mask, region, config)

    if config.export_to_drive:
        print("Building MOD16 table...")
        mod16_fc = build_mod16_table(points, config.year, region)
        print("Building S2 table...")
        s2_fc = build_s2_table(points, config.year, region, config)
        print("Building ERA5 table...")
        era5_fc = build_era5_table(points, config.year, region)
        tasks = start_exports(mod16_fc, s2_fc, era5_fc, config)
        print("Export tasks submitted:")
        print(json.dumps([summarize_task(t) for t in tasks], ensure_ascii=False, indent=2))
    else:
        print("Preview only — no exports.")


if __name__ == "__main__":
    main()
