/**
 * Hebei winter wheat mask (2021-2025) - rule-based MVP for GEE Code Editor.
 *
 * Purpose:
 * 1. Produce a yearly candidate winter wheat mask for Hebei.
 * 2. Export yearly rasters to GEE Assets and/or Google Drive.
 * 3. Export a yearly area summary table.
 *
 * Why rule-based first:
 * - This version runs without a manually labeled training sample asset.
 * - It is intended as a stable first pass for phase 1 crop-mask production.
 * - Once field samples are available, this script should be upgraded to a
 *   Random Forest classifier using the same temporal features.
 *
 * Project context:
 * - GEE project id: chuang-yaogan
 * - Research scope in this file: Hebei only
 * - Season years: 2021-2025
 *
 * Data sources verified against the Earth Engine catalog:
 * - Sentinel-2 SR Harmonized:
 *   COPERNICUS/S2_SR_HARMONIZED
 * - Sentinel-2 cloud probability:
 *   COPERNICUS/S2_CLOUD_PROBABILITY
 * - ESA WorldCover v200:
 *   ESA/WorldCover/v200
 * - GAUL level1:
 *   FAO/GAUL/2015/level1
 *
 * Outputs:
 * - Asset raster per year:
 *   projects/chuang-yaogan/assets/kcact/phase1/winter_wheat_mask_hebei_<year>
 * - Drive raster per year:
 *   hebei_winter_wheat_mask_<year>.tif
 * - Drive summary CSV:
 *   hebei_winter_wheat_area_summary_2021_2025.csv
 */

var CONFIG = {
  countryName: 'China',
  provinceName: 'Hebei',
  years: [2021, 2022, 2023, 2024, 2025],
  assetRoot: 'projects/chuang-yaogan/assets/kcact/phase1',
  driveFolder: 'kcact_hebei_phase1',
  exportScale: 10,
  maxPixels: 1e13,
  cloudProbabilityThreshold: 40,
  minConnectedPixels: 8,
  exportToAsset: true,
  exportToDrive: true,
  previewYear: 2025
};

var GAUL_L1 = ee.FeatureCollection('FAO/GAUL/2015/level1');
var WORLD_COVER_2021 = ee.ImageCollection('ESA/WorldCover/v200').first();
var S2_SR = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var S2_CLOUDS = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');

var hebei = GAUL_L1
  .filter(ee.Filter.eq('ADM0_NAME', CONFIG.countryName))
  .filter(ee.Filter.eq('ADM1_NAME', CONFIG.provinceName))
  .geometry();

Map.centerObject(hebei, 7);

var croplandMask = WORLD_COVER_2021.select('Map').eq(40).rename('cropland');

function maskEdges(image) {
  return image.updateMask(
    image.select('B8A').mask().updateMask(image.select('B9').mask())
  );
}

function buildCloudJoinedCollection(startDate, endDate, region) {
  var sr = S2_SR
    .filterBounds(region)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 80))
    .map(maskEdges);

  var clouds = S2_CLOUDS
    .filterBounds(region)
    .filterDate(startDate, endDate);

  var joined = ee.Join.saveFirst('cloud_mask').apply({
    primary: sr,
    secondary: clouds,
    condition: ee.Filter.equals({
      leftField: 'system:index',
      rightField: 'system:index'
    })
  });

  return ee.ImageCollection(joined);
}

function addIndices(image) {
  var scaled = image.select([
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'
  ]).divide(10000);

  var cloudMask = ee.Image(image.get('cloud_mask')).select('probability');
  var isClear = cloudMask.lt(CONFIG.cloudProbabilityThreshold);
  var clean = scaled.updateMask(isClear);

  var ndvi = clean.normalizedDifference(['B8', 'B4']).rename('ndvi');
  var lswi = clean.normalizedDifference(['B8', 'B11']).rename('lswi');

  var evi = clean.expression(
    '2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1.0))',
    {
      nir: clean.select('B8'),
      red: clean.select('B4'),
      blue: clean.select('B2')
    }
  ).rename('evi');

  var ndre = clean.normalizedDifference(['B8', 'B5']).rename('ndre');

  return clean
    .addBands([ndvi, lswi, evi, ndre])
    .copyProperties(image, image.propertyNames());
}

function getSeasonWindows(year) {
  return {
    autumnStart: ee.Date.fromYMD(year - 1, 10, 1),
    autumnEnd: ee.Date.fromYMD(year - 1, 11, 30),
    winterStart: ee.Date.fromYMD(year, 1, 1),
    winterEnd: ee.Date.fromYMD(year, 2, 15),
    springStart: ee.Date.fromYMD(year, 3, 1),
    springEnd: ee.Date.fromYMD(year, 4, 15),
    peakStart: ee.Date.fromYMD(year, 4, 16),
    peakEnd: ee.Date.fromYMD(year, 5, 20),
    harvestStart: ee.Date.fromYMD(year, 6, 1),
    harvestEnd: ee.Date.fromYMD(year, 6, 30)
  };
}

function compositeIndex(startDate, endDate, region, reducerName) {
  var collection = buildCloudJoinedCollection(startDate, endDate, region)
    .map(addIndices);

  var composite = ee.Image(
    ee.Algorithms.If(
      reducerName === 'max',
      collection.select(['ndvi', 'lswi', 'evi', 'ndre']).max(),
      collection.select(['ndvi', 'lswi', 'evi', 'ndre']).median()
    )
  );

  var observationCount = collection.select('ndvi').count().rename('obs_count');
  return composite.addBands(observationCount).clip(region);
}

function buildWinterWheatMask(year) {
  var windows = getSeasonWindows(year);

  var autumn = compositeIndex(windows.autumnStart, windows.autumnEnd, hebei, 'median');
  var winter = compositeIndex(windows.winterStart, windows.winterEnd, hebei, 'median');
  var spring = compositeIndex(windows.springStart, windows.springEnd, hebei, 'median');
  var peak = compositeIndex(windows.peakStart, windows.peakEnd, hebei, 'max');
  var harvest = compositeIndex(windows.harvestStart, windows.harvestEnd, hebei, 'median');

  var peakNdvi = peak.select('ndvi');
  var springNdvi = spring.select('ndvi');
  var winterNdvi = winter.select('ndvi');
  var autumnNdvi = autumn.select('ndvi');
  var harvestNdvi = harvest.select('ndvi');
  var springLswi = spring.select('lswi');
  var peakNdre = peak.select('ndre');

  var ndviRise = springNdvi.subtract(winterNdvi).rename('ndvi_rise');
  var ndviDrop = peakNdvi.subtract(harvestNdvi).rename('ndvi_drop');

  var enoughObs = autumn.select('obs_count').gte(1)
    .and(winter.select('obs_count').gte(1))
    .and(spring.select('obs_count').gte(1))
    .and(peak.select('obs_count').gte(1))
    .and(harvest.select('obs_count').gte(1));

  // Hebei winter wheat phenology logic:
  // - emerged in autumn
  // - remains vegetated through winter
  // - strong green-up in March-April
  // - peak before harvest
  // - clear decline after harvest in June
  var candidate = croplandMask
    .and(enoughObs)
    .and(autumnNdvi.gt(0.20))
    .and(winterNdvi.gt(0.18))
    .and(springNdvi.gt(0.42))
    .and(peakNdvi.gt(0.58))
    .and(ndviRise.gt(0.12))
    .and(ndviDrop.gt(0.20))
    .and(springLswi.gt(0.05))
    .and(peakNdre.gt(0.20));

  // Remove isolated noise patches.
  var connected = candidate.selfMask().connectedPixelCount(100, true);
  var cleaned = candidate
    .updateMask(connected.gte(CONFIG.minConnectedPixels))
    .rename('winter_wheat');

  var featureStack = ee.Image.cat([
    autumnNdvi.rename('autumn_ndvi'),
    winterNdvi.rename('winter_ndvi'),
    springNdvi.rename('spring_ndvi'),
    peakNdvi.rename('peak_ndvi'),
    harvestNdvi.rename('harvest_ndvi'),
    ndviRise,
    ndviDrop,
    springLswi.rename('spring_lswi'),
    peakNdre.rename('peak_ndre'),
    cleaned
  ]).clip(hebei);

  return featureStack.set({
    season_year: year,
    province: CONFIG.provinceName,
    crop_type: 'winter_wheat'
  });
}

function addPreviewLayers(year) {
  var image = buildWinterWheatMask(year);

  Map.addLayer(
    croplandMask.updateMask(croplandMask).clip(hebei),
    {palette: ['#cdbb7c']},
    'Cropland mask (WorldCover 2021)'
  );

  Map.addLayer(
    image.select('peak_ndvi'),
    {min: 0.2, max: 0.9, palette: ['#f7fcf5', '#74c476', '#00441b']},
    'Peak NDVI ' + year,
    false
  );

  Map.addLayer(
    image.select('winter_wheat').selfMask(),
    {palette: ['#00a651']},
    'Winter wheat candidate mask ' + year
  );
}

function exportYear(year) {
  var image = buildWinterWheatMask(year);
  var mask = image.select('winter_wheat').toUint8();

  if (CONFIG.exportToAsset) {
    Export.image.toAsset({
      image: mask,
      description: 'asset_hebei_winter_wheat_mask_' + year,
      assetId: CONFIG.assetRoot + '/winter_wheat_mask_hebei_' + year,
      region: hebei,
      scale: CONFIG.exportScale,
      maxPixels: CONFIG.maxPixels
    });
  }

  if (CONFIG.exportToDrive) {
    Export.image.toDrive({
      image: mask,
      description: 'drive_hebei_winter_wheat_mask_' + year,
      fileNamePrefix: 'hebei_winter_wheat_mask_' + year,
      folder: CONFIG.driveFolder,
      region: hebei,
      scale: CONFIG.exportScale,
      maxPixels: CONFIG.maxPixels,
      fileFormat: 'GeoTIFF'
    });
  }
}

function buildAreaSummary(year) {
  var image = buildWinterWheatMask(year).select('winter_wheat');
  var areaHa = image.selfMask().multiply(ee.Image.pixelArea()).divide(10000);

  var stats = areaHa.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: hebei,
    scale: CONFIG.exportScale,
    maxPixels: CONFIG.maxPixels
  });

  return ee.Feature(null, {
    season_year: year,
    province: CONFIG.provinceName,
    crop_type: 'winter_wheat',
    area_ha: stats.get('winter_wheat')
  });
}

var previewImage = buildWinterWheatMask(CONFIG.previewYear);
print('AOI - Hebei', hebei);
print('Preview image', previewImage);
addPreviewLayers(CONFIG.previewYear);

var areaSummary = ee.FeatureCollection(CONFIG.years.map(buildAreaSummary));
print('Yearly winter wheat area summary (ha)', areaSummary);

if (CONFIG.exportToDrive) {
  Export.table.toDrive({
    collection: areaSummary,
    description: 'drive_hebei_winter_wheat_area_summary_2021_2025',
    folder: CONFIG.driveFolder,
    fileNamePrefix: 'hebei_winter_wheat_area_summary_2021_2025',
    fileFormat: 'CSV'
  });
}

CONFIG.years.forEach(exportYear);
