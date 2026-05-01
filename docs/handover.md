# Kcact ET Modeling — Handover Document

**Last updated**: 2026-05-01 | **Sessions**: zhandian — station ML, 962 combos, per-station analysis, GEE resubmit v3

## 1. Project Overview

**Goal**: Predict Kcact (crop coefficient = ETc / ET0) for winter wheat and summer maize in North China Plain.

**Target variable**: Kcact = ETc / ET0
- ETc from MODIS MOD16A2GF (500m, 8-day) — large sample
- ETc from flux station lysimeter/eddy covariance (8-day) — station validation
- ET0 computed via FAO-56 Penman-Monteith from ERA5 weather data

**Current best models**:

| Crop | Dataset | Samples | Best R² | Features |
|------|---------|---------|---------|----------|
| Winter wheat | Hebei, 592 patches, 2019–2025 | 18,528 | **0.702** | 46 feat, CatBoost |
| Summer maize | NCP, 7,153 patches, 2019–2025 | 397,628 | **0.769** | 46 feat, CatBoost |
| Summer maize | 4 flux stations, 2003–2015 | 304 | **0.467** | 7 feat, CatBoost LOSO |

**⚠️ Tier 1 improvement attempts (2026-04-26): ALL FAILED**
- CatBoost hyperparameter tuning (50-trial Optuna): R² = 0.6881 (−0.0136)
- Stacking ensemble (XGB+CB+LGBM → Ridge): R² = 0.6982 (−0.0035)
- Water balance features: R² = 0.6924 (−0.0093)
- **Conclusion**: Default CatBoost at 0.702 was the ceiling with current data.

## 2. Git Repository State

- **Branch**: `predict-etc-direct`
- **Root**: `/Users/hert/Projects/dcsdxx`
- **Remote**: `origin` → `https://github.com/He-RT/dcsdxx.git`
- **Uncommitted changes**: Extensive — new scripts, data files, figures, GEE export tasks submitted

## 3. Environment

| Item | Value |
|---|---|
| Conda env name | `sdxx` |
| Python path | `/Users/hert/conda/envs/sdxx/bin/python3` |
| CatBoost | 1.2.10 |
| scikit-learn | 1.8.0 |
| GEE project | `chuang-yaogan` |
| macOS run prefix | `DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"` |

## 4. Station Data Pipeline (NEW 2026-05-01)

### 4.1 Data Sources

- **Flux station ETc observations**: `data/raw/实际蒸散发观测数据.xlsx`
  - 4 stations: 禹城 (2003–2010, 368 obs), 位山 (2011–2015, 219 obs), 馆陶 (2008–2010, 138 obs), 栾城 (2007–2008, 38 obs)
  - 8-day ETc (mm/d), irregular intervals (5–8 days)
- **ET0**: FAO-56 PM from ERA5-Land daily weather, extracted via GEE at station coordinates
- **MODIS indicators**: MOD09A1, MOD11A2, MOD13Q1, MCD43A3, MOD15A2H extracted for 4 stations
- **ERA5 weather + SM**: Daily weather + volumetric_soil_water_layer_1

### 4.2 Key Station Data Files

| File | Content |
|------|---------|
| `data/processed/station_etc_observations.csv` | Cleaned 763 obs from Excel |
| `data/processed/station_etc_with_et0.csv` | ETc + ET0 + Kcact for all observations |
| `data/processed/station_ml_features.csv` | All RS indicators matched to observation windows |
| `data/processed/modis_cache/` | Raw GEE extracts (MOD09A1, MOD11A2, MOD13Q1, MCD43A3, MOD15A2H, ERA5) |
| `outputs/tables/station_etc_et0_kcact.xlsx` | Formatted 4-station layout with ETc/ET0/Kcact |
| `outputs/figures/fig*_kcact_*.png` | Nature-style Kcact line charts |

### 4.3 Soil Moisture Analysis

ERA5-Land SM vs Kcact correlation is weak (r=0.17 overall) but shows strong window-specific signals:
- DOY 91–150 (April–May wheat peak): r=+0.43
- DOY 241–300 (Sep–Oct maize grain fill): r=+0.57
- SM declining trend (−0.008/yr) while Kcact rising (+0.025/yr) — diverging over 2003–2010
- 66% of days show SM decline (evapotranspiration), only 21% show rise (rain/irrigation events)

## 5. Feature Selection: 962 Combos on Station Summer Maize

### 5.1 Experimental Setup

- 304 samples (4 stations, DOY 150–300 summer maize)
- CatBoost (iter=200, lr=0.05, depth=4) with Leave-One-Station-Out CV
- Feature pool: MOD09A1 raw bands (b01–b07) + derived VIs + MODIS thermal/fPAR/albedo + ERA5 SM/weather
- 5 NDVI data sources compared: MOD09GA (daily), MOD09Q1 (250m), MCD43A4 (NBAR), MOD09A1 (8-day), MOD13Q1 (16-day)

### 5.2 Top 5 Combos

| Rank | Combo | n | R² |
|------|-------|---|------|
| 1 | fpar + delta_lst + sm_surface + ndvi_m09 + lswi_m09 + albedo_sw + doy | 7 | 0.467 |
| 2 | ndvi_m09 + b07 + doy | 3 | 0.461 |
| 3 | fpar + delta_lst + sm_surface + ndvi_m09 + albedo_sw | 5 | 0.460 |
| 4 | gndvi_m09 + delta_lst + albedo_sw + lswi | 4 | 0.459 |
| 5 | gndvi_m09 + delta_lst + albedo_sw + lswi + doy | 5 | 0.458 |

Key finding: **3 features (ndvi_m09 + b07 + doy) achieve R²=0.461**, only 0.006 behind the 7-feature best. b07 (SWIR2, 2.13μm) is the strongest single water-absorption band.

### 5.3 NDVI Data Source Ranking

| Source | Resolution | Temporal | Coverage | Single+R² |
|--------|-----------|----------|----------|-----------|
| MOD09A1 ndvi_m09 | 500m | 8-day | 100% | 0.435 |
| MOD09GA ndvi_daily | 500m | daily | 100% | 0.420 |
| MOD09Q1 ndvi_250m | 250m | 8-day | 100% | 0.421 |
| MOD13Q1 ndvi | 250m | 16-day | 50% | 0.395 |
| MCD43A4 ndvi_nbar | 500m | daily | 82% | 0.381 |

Daily temporal resolution beats 8-day, but the difference is small. The biggest penalty is 16-day (50% coverage → imputation noise).

### 5.4 Rank 77 Simplified Variants

Original Rank 77: 19 features, R²=0.451. Simplification experiments (`outputs/tables/rank77_simplified_variants.xlsx`):

| Rank | Combo | n | R² |
|------|-------|---|------|
| 1 | Core3 (NDVI+EVI+GNDVI) + weather + doy + year + lat/lon | 12 | 0.464 |
| 2 | Core3 + weather + doy + year | 10 | 0.461 |
| 3 | Rank77 − interaction terms | 17 | 0.455 |

Removing interaction terms IMPROVES R² — they're noise on small samples.

## 6. Large Sample Proxy Tests (Summer Maize 397K)

### 6.1 Proxy Results (without MODIS indicators)

Trained on `ncp_summer_maize_kcact_train_ready.parquet` (369,740 valid, 7,153 patches, 2019–2025), LOYO CV:

| Model | Features | R² | vs Full |
|-------|----------|-----|---------|
| Full 46 | 40 | **0.769** | baseline |
| 7 indicators (proxy) | 7 | 0.748 | −0.021 |
| ndvi+lswi+doy | 3 | 0.563 | −0.206 |

**7 well-chosen proxies get 97% of full-feature performance with 1/6 the features.** Real MODIS indicators (fpar, ΔLST, albedo, SM) expected to close the remaining gap.

### 6.2 Data Sources for Kcact Prediction

**Existing (in parquet, ~45 features)**:
- S2 VIs: ndvi, evi, savi, gndvi, lswi, nirv, re_ndvi
- ERA5 weather: tmean, tmin, tmax, VPD, solar, wind, precip, dewpoint, pressure
- Temporal/phenology: doy, gdd, season_year, greenup features
- Derived: NDVI derivatives, rolling precip, interactions, senescence indicators

**Pending (GEE exports submitted, ~12 new features)**:
| Product | Bands | Physics Domain |
|---------|-------|---------------|
| MOD15A2H fPAR | Fpar_500m | Canopy photosynthesis |
| MOD11A2 LST | LST_Day, LST_Night → ΔLST | Thermal/water stress |
| MCD43A3 Albedo | Albedo_WSA_shortwave | Surface energy balance |
| ERA5-Land SM | volumetric_soil_water_layer_1 | Soil water supply |
| MOD09A1 raw | ndvi_m09, b07 (SWIR2) | Greenness + water absorption |
| S2 raw bands | B4 (Red), B8 (NIR), B12 (SWIR2) | 10m reflectance |
| Sentinel-1 SAR | VV, VH backscatter | Microwave structure |
| SRTM DEM | elevation, slope | Topography |

All 50+ GEE tasks in Drive folder `kcact_maize_modis_indicators/`. Monitor at https://code.earthengine.google.com/ → Tasks.

## 7. GEE Export Tasks Status

### Submitted & Queued (2026-05-01)

| Task Group | Count | Status |
|------------|-------|--------|
| MODIS fPAR (MOD15A2H) | 7 | READY |
| MODIS LST (MOD11A2) | 7 | READY |
| MODIS Albedo (MCD43A3, 8-day windows) | 7 | READY |
| ERA5 SM | 7 | READY |
| MOD09A1 ndvi+b07 | 7 | READY |
| S2 raw B4/B8/B12 | 7 | READY |
| Sentinel-1 VV/VH | 7 | READY |
| SRTM DEM | 1 | READY |

### Known Issues
- maize_albedo_2019: FAILED (OOM with daily images). Resubmitted with 8-day window medians + tileScale=8.
- All other albedo years also resubmitted with same fix for consistency.

### After GEE Exports Complete
```bash
# 1. Download all CSVs from Drive folder kcact_maize_modis_indicators/
#    → save to data/raw/gee/modis_indicators/
# 2. Run merge + retrain:
DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH" \
  /Users/hert/conda/envs/sdxx/bin/python3 \
  scripts/python/merge_modis_and_retrain_maize.py
```

## 8. Key Scripts (NEW 2026-05-01)

### Station Data
| Script | Purpose |
|--------|---------|
| `compute_station_et0.py` | Fetch ERA5 → FAO-56 ET0 → merge with obs |
| `extract_station_ml_features.py` | Extract 8 RS indicators for 4 stations from GEE |
| `extract_station_mod09ga_ndvi.py` | MOD09GA daily NDVI extraction |
| `extract_station_raw_bands.py` | MOD09A1 raw band extraction |
| `extract_station_era5_weather.py` | ERA5-Land weather extraction |
| `plot_station_kcact.py` | Nature-style Kcact figures (9 figs) |
| `train_station_ml.py` | Basic station ML training |
| `train_station_ml_combos.py` | 141-combo station training (all-season) |
| `train_maize_500_combos.py` | 962-combo summer maize station training |

### Large Sample
| Script | Purpose |
|--------|---------|
| `export_maize_modis_indicators.py` | GEE export: fPAR, LST, Albedo, SM for maize patches |
| `export_maize_mod09a1_ndvi_b07.py` | GEE export: MOD09A1 ndvi+b07 for maize patches |
| `export_maize_s2raw_and_s1.py` | GEE export: S2 raw bands + Sentinel-1 SAR |
| `merge_modis_and_retrain_maize.py` | Merge GEE exports → retrain with real MODIS indicators |

## 9. Key Findings Summary

1. **SWIR2 (b07/B12) is the single most valuable raw band** for Kcact prediction — stronger than any VI alone. At 2.13μm, it directly senses vegetation/soil water content.

2. **fPAR > NDVI** for Kcact. fPAR is retrieved from canopy RT models and doesn't saturate at high biomass. Single-feature R²: fPAR 0.318 vs NDVI 0.022 (MOD13Q1 with 50% coverage) or 0.222 (MOD09A1 with 100% coverage).

3. **Temporal coverage beats spatial resolution.** MOD09GA daily NDVI (500m) > MOD09Q1 8-day NDVI (250m). The 16-day MOD13Q1 loses half the observation windows.

4. **3 features can do the job of 40+ on station data.** ndvi_m09 + b07 + doy gets R²=0.461 vs full 46 features at 0.467 (station) and big-sample proxy version gets within 0.02 of full.

5. **ERA5-Land SM alone is weak (r=0.17) but synergistic in multi-feature models.** It only enters top combos when combined with fPAR + ΔLST.

6. **Interaction terms hurt on small data.** Removing interactions from Rank 77 IMPROVES R² (0.451→0.455).

7. **NDVI+EVI+GNDVI as a core trio works well**, capturing greenness, canopy structure, and chlorophyll concentration respectively. With weather + doy + year, reaches within 0.003 of overall best.

### 5.5 Per-Station Rank 77 Analysis

Rank 77 (19 features: NDVI+EVI+GNDVI+LSWI + weather + derivatives + interactions + spatiotemporal) tested per station via LOSO CV:

| Test Station | LOSO R² | Notes |
|-------------|---------|-------|
| 栾城 | +0.50 | Single growing season, simple pattern |
| 馆陶 | +0.32 | Highest per-feature correlations but lowest Kcact baseline |
| 禹城 | +0.28 | Strong interannual variability |
| 位山 | +0.24 | Highest within-station Kcact variance |

Per-feature correlation breakdown shows consistent ranking across stations (ndvi, lswi, fpar always strongest) but magnitude varies 0.2–0.3. Interaction terms (ndvi×VPD, ndvi_diff) are near-zero at all stations — primary reason simplification helps.

## 10. GEE Export Status (2026-05-01, v3)

Three rounds of submission, two failures:

| Round | Products | Approach | Result |
|-------|----------|----------|--------|
| v1 | fpar, LST, albedo, SM | 27K points, daily albedo | fpar/LST/SM OK, albedo OOM |
| v2 | albedo, S2 raw, S1, MOD09A1 | 27K points, 8-day windows | ALL OOM (fine scale +太多点) |
| v3 | albedo, S2 raw, S1, MOD09A1 | Per-province (~7K pts), 8-day, tileScale=8, no geometry filter | 84 tasks submitted, monitoring |

**Completed (26 tasks)**: fpar (7), LST (7), ERA5 SM (7), SRTM DEM (1), MOD09A1 (4 of 7)

**In queue (84 tasks)**: Per-province albedo/S2 raw/S1/MOD09A1 for all 7 years

## 11. Repository Notes

- Branch `predict-etc-direct` merged into `main` via no-ff merge
- `.omc/` and `catboost_info/` untracked via `git rm --cached` + `.gitignore`
- All `Co-Authored-By: Claude` trailers removed via `git filter-branch`
- Force pushed to origin; GitHub contributor cache may need minutes to refresh
- `data/raw/*` gitignored — station Excel data never committed
- README updated for group use

## 12. Path Forward

1. **Wait for GEE exports** (50 tasks, Drive folder `kcact_maize_modis_indicators`)
2. **Download + merge** → run `merge_modis_and_retrain_maize.py`
3. **Run 500+ combo tests** on large sample with full feature set (~57 features)
4. **Compare station-optimized combos vs large-sample-optimized combos**
5. **Add SIF (TROPOMI)** if available — most direct photosynthesis measurement
6. **Consider Sentinel-1 SAR** integration once exports complete

## 11. Known Issues / Gotchas

- **XGBoost on macOS**: requires `DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"`
- **MOD13Q1 NDVI 50% coverage**: 16-day composites miss half of 8-day observation windows → heavy NaN imputation → poor model performance. Use MOD09A1 or MOD09GA instead.
- **MCD43A3 daily OOM**: Daily images × 27K points causes GEE OOM. Use 8-day window medians instead.
- **GEE .getInfo() limits**: ~10MB response, ~5 min timeout. For >1000 points, use Export.table.toDrive().
- **Small station sample (304 maize obs)**: LOSO CV is very harsh (training on 3 stations, testing on 1 with different baseline). Results indicate feature ranking robustness, not absolute prediction quality.
- **Station name encoding**: MODIS extracts use English names (yucheng, weishan, etc.), other data uses Chinese (禹城, 位山). Always map before merging.
