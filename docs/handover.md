# Kcact ET Modeling — Handover Document

**Last updated**: 2026-05-16 | **Sessions**: zhandian, remote, mod16-compare, fc-wp-meta, spatial-alignment-fix, pml-model, midterm-report, miniprogram | **Ready for compact**

## 1. Project Overview

**Goal**: Predict Kcact (crop coefficient = ETc / ET0) for winter wheat and summer maize in North China Plain.

**Target variable**: Kcact = ETa / ET0
- Current midterm/reporting target: **PML-V2.2a ETa / ERA5 ET0** — large-sample regional product model.
- Historical comparison target: MODIS MOD16A2GF ETa / ERA5 ET0 — retained for error-source discussion, not the main product showcase.
- Station validation target: eddy-covariance / lysimeter ETa divided by station meteorological ET0 when available; station data are for applicability checks and bias explanation, not for product training.
- ET0 is computed by FAO-56 Penman-Monteith from ERA5 or station meteorological observations depending on the comparison.

**Current best models**:

| Crop | Dataset | Samples | Best R² | Features |
|------|---------|---------|---------|----------|
| Summer maize PML/ERA5 | NCP, 2019–2024, ERA5-like 0.1° grid, 8-day windows | **144,877** | **0.765402 LOYO pooled** | 7 feat: NDVI, SAVI, RDVI, GNDVI, EVI, SM, doy |
| Summer maize MOD16/ERA5 | NCP, 2019–2025, ERA5-like 0.1° grid, 8-day windows | 171,785 | ~0.750 LOYO pooled | 7 feat; historical comparison |
| Summer maize SMAP L4 | NCP, 2019–2025, SMAP L4 scale | 173,889 | 0.7208 LOYO pooled | 7 feat with `sm_surface` |
| Winter wheat | Hebei, 592 patches, 2019–2025 | 18,528 | 0.702 | 46 feat, CatBoost; older baseline |
| Summer maize station | 4 flux stations, 2003–2015 | 304 | 0.467 LOSO / 0.518 with ETc+ET0 | station validation only |

**⚠️ Tier 1 improvement attempts (2026-04-26): ALL FAILED**
- CatBoost hyperparameter tuning (50-trial Optuna): R² = 0.6881 (−0.0136)
- Stacking ensemble (XGB+CB+LGBM → Ridge): R² = 0.6982 (−0.0035)
- Water balance features: R² = 0.6924 (−0.0093)
- **Conclusion**: Default CatBoost at 0.702 was the ceiling with current data.

## 1.0 Current Midterm Baseline and Deliverables (NEW 2026-05-16)

### 1.0.1 PML/ERA5 seven-indicator model

Use this as the main midterm-report and mini-program model unless explicitly comparing alternatives.

- **Label**: `Kcact = PML-V2.2a ETa / ERA5 ET0`
- **Crop/region**: summer maize, North China Plain
- **Years**: 2019–2024
- **Scale**: ERA5-like 0.1° grid; 8-day windows
- **Features**: `NDVI, SAVI, RDVI, GNDVI, EVI, SM, doy`
- **QC**: strict numeric thresholds for Kcact, ETa, ET0, vegetation indices, SM, and 8-day windows
- **Result**: `LOYO pooled R² = 0.765402`, `RMSE = 0.101760`, `MAE = 0.075630`
- **Training table**: `data/processed/train/ncp_summer_maize_selected_indicators_pml_era5grid.parquet`
- **Script**: `scripts/python/train_maize_selected_indicators_pml_era5grid.py`

Rationale: PML-V2.2a is currently more suitable than MOD16 for the report narrative because station-location validation shows PML ETa has a much stronger relationship with measured ETa than MOD16 at flux-tower sites. The product remains feasible because PML/ERA5/Sentinel-2/SM inputs are gridded public data sources rather than station-only data.

### 1.0.2 Current report figures

| Purpose | Preferred file | Notes |
|---|---|---|
| Main seasonal Kcact comparison | `outputs/figures/pml_era5_vs_station_met_kcact_doy_summer_maize_paper.pdf` | PML/ERA5 vs station ETa/station ET0 over 06-09—10-15; in-plot R²=0.932 for 8-day mean curves |
| ERA5 ET0 bias explanation | `outputs/figures/station_era5_vs_met_et0_linear.pdf` | Overall r=0.856, R²=0.732, ERA5 mean bias +0.583 mm/d |
| PML ETa station validation | `outputs/figures/station_pml_v22a_vs_tower_eta_linear.png` | PML vs tower ETa: r=0.768, R²=0.590 |
| Technical route | `outputs/figures/technical_route_publication_bw.pdf` | Use black/white publication style; do not put title inside figure |
| Mini-program screenshots | generated from `apps/kcact-miniprogram/` | Current pages: overview, parcel, model, report, agent |

### 1.0.3 Mini-program / product demo state

- Directory: `apps/kcact-miniprogram/`
- Current display values are synchronized with the PML/ERA5 model:
  - grid cells: 3,642
  - training samples: 144,877
  - LOYO R²: 0.765
  - RMSE: 0.102
  - MAE: 0.076
- Pages:
  - `overview`: NCP monitoring dashboard
  - `parcel`: representative grid/parcel detail
  - `model`: seven-indicator model, SHAP-style contribution ranking, LOYO yearly validation
  - `report`: generated diagnostic report
  - `agent`: intelligent-assistant interaction display for midterm product vision

### 1.0.4 Reporting wording to keep consistent

- Main model wording: “PML-V2.2a ETa 与 ERA5 ET0 构建 Kcact 标签，并统一至 0.1° 尺度进行七指标建模。”
- Station data wording: “站点涡度相关/实测 ETa 用于适用性验证与误差来源分析，不作为最终产品模型的训练数据源。”
- Product feasibility wording: “最终产品应保持训练与应用输入源一致，优先使用可自动获取的公开遥感与再分析数据。”

## 1.1 Critical Data Fix — ETa/ET0 Spatial Alignment (2026-05-14)

**Problem found**: the large-sample builders used the GEE-exported `pt_*` / `feature.id()` as `patch_id`. GEE feature ids are not globally stable across export jobs, provinces, and years. As a result, rows with the same `pt_*` but different coordinates could be merged, pairing MOD16 ETa from one point with ERA5 ET0 from another point.

**Measured severity in old parquet**:

| Dataset | Old mismatched rows | Median ETa/ET0 distance | Max distance |
|---|---:|---:|---:|
| Summer maize NCP | 119,462 / 397,628 (30.04%) | 191.55 km | 563.00 km |
| Winter wheat NCP | 94,673 / 238,464 (39.70%) | 93.16 km | 459.67 km |
| Hebei winter wheat only | 0 / 18,528 | 0 km | 0 km |

**Fix implemented**:

- `src/kcact/data/kcact_builder.py`
  - adds `coord_key = round(lat, 6) + "_" + round(lon, 6)`;
  - creates stable `patch_id = crop + province + season_year + coord_key`;
  - preserves the old raw GEE id as `gee_point_id`;
  - aggregates ERA5 daily ET0 to MOD16 windows by `coord_key`, not raw `pt_*`;
  - merges MOD16/ERA5/S2 by `patch_id + coord_key + date_start + date_end + date`;
  - enforces a post-merge spatial alignment check; >10 m mismatch raises `ValueError`.
- GEE export scripts now emit stable coordinate-based `point_id`, `coord_key`, and `gee_feature_id` for future exports:
  - `scripts/python/export_maize_kcact_training_data.py`
  - `scripts/python/export_kcact_training_data.py`
  - `scripts/python/export_hebei_kcact_training_data.py`

**Rebuilt datasets after fix**:

| Dataset | Rows | Stable patch_id count | Time alignment | Spatial alignment |
|---|---:|---:|---|---|
| `ncp_summer_maize_kcact_train_ready.parquet` | 512,149 | 42,693 | all 8-day windows, `date=date_end` | max ETa/ET0 coord diff < 1 mm |
| `ncp_winter_wheat_kcact_train_ready.parquet` | 289,690 | 15,899 | all 8-day windows, `date=date_end` | max ETa/ET0 coord diff < 1 mm |

**Important**: all large-sample model scores before 2026-05-14 should be treated as historical/pre-fix. Retrain the large-sample models with the rebuilt parquet before reporting final performance.

## 1.2 Direct RDVI Export + Strict Numeric QC (2026-05-14)

**Problem found**: the 7-indicator summer-maize run used RDVI reconstructed from aligned NDVI/SAVI because the old S2 export did not include Red/NIR-derived RDVI. The reconstruction is useful for continuity but is not ideal for reporting because S2 window composites store mean indices, not necessarily mean reflectances. EVI also had a tiny number of denominator blow-ups (e.g. EVI outside physical crop ranges).

**Fix implemented**:

- Updated S2 GEE export code in:
  - `scripts/python/export_maize_kcact_training_data.py`
  - `scripts/python/export_kcact_training_data.py`
  - `scripts/python/export_hebei_kcact_training_data.py`
- New S2 tables include:
  - direct `rdvi = (NIR - Red) / sqrt(max(NIR + Red, 1e-6))`
  - `s2_red`, `s2_nir` audit columns
- Updated `scripts/python/train_maize_selected_indicators.py`:
  - `--rdvi-source auto|direct|fallback`; final/reported runs should use `--rdvi-source direct` after S2 re-export;
  - strict numeric QC before training:
    - `0.02 <= Kcact <= 1.60`
    - `0.10 <= ET0_daily <= 10.0 mm/d`
    - `0.0 <= ETa_daily <= 10.0 mm/d`
    - `-1.0 <= EVI <= 1.5`
    - `0.02 <= SM <= 0.60`
    - NDVI/GNDVI/LSWI in `[-1, 1]`, SAVI in `[-1, 1.5]`

**Current rerun on legacy parquet** (RDVI fallback, not final direct-RDVI result):

| Metric | Value |
|---|---:|
| QC-valid rows before strict QC | 512,149 |
| Rows after strict QC before complete-case | 471,916 |
| Complete 7-indicator rows | 466,381 |
| Pooled LOYO R² | 0.7119 |
| RMSE | 0.1935 |
| MAE | 0.1435 |

**Next required step for final reporting**: re-export/rebuild S2 features with direct RDVI, then rerun:

```bash
python scripts/python/train_maize_selected_indicators.py --rdvi-source direct
```

## 1.3 SMAP L4 Grid-Scale Seven-Indicator Run (2026-05-14)

**Data added**:

- Copied from Google Drive into `data/raw/gee/kcact_maize_modis_indicators/`:
  - `maize_smap_l4_aligned_2019.csv` … `maize_smap_l4_aligned_2025.csv`
- Export source: `NASA/SMAP/SPL4SMGP/008`, 8-day windows aligned to corrected maize Kcact rows.

**Script added**:

- `scripts/python/train_maize_selected_indicators_smapgrid.py`

**Method**:

- Unifies the seven selected indicators to the actual SMAP L4 sampled cell/window scale.
- Infers practical SMAP cell-window groups from identical SMAP sampled values in the same window, rather than guessing EASE-Grid cell boundaries from lat/lon.
- Aggregates target as `mean(MOD16 ETa) / mean(ERA5 ET0)`.
- Applies strict QC before aggregation:
  - 8-day window only
  - `0.10 <= ET0_daily <= 10.0 mm/d`
  - `0.0 <= ETa_daily <= 10.0 mm/d`
  - `0.02 <= Kcact <= 1.60`
  - NDVI/GNDVI in `[-1, 1]`, SAVI in `[-1, 1.5]`, RDVI in `[-2, 2]`, EVI in `[-1, 1.5]`
  - SM in `[0.02, 0.60]`

**Results**:

| SMAP SM source | Rows | Pooled LOYO R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| `sm_surface` | 173,889 | **0.7208** | 0.1818 | 0.1370 |
| `sm_rootzone` | 173,855 | 0.6866 | 0.1926 | 0.1460 |
| `sm_profile` | 173,850 | 0.6824 | 0.1939 | 0.1473 |

**Takeaway**: SMAP surface SM is best among SMAP L4 variables in this seven-indicator setup. It improves product realism/availability but does not beat the earlier ERA5-like 0.1° aggregation run (R²≈0.746).

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

## 11. Desktop (mlpc) Status

- **SSH**: `ssh mlpc` (192.168.1.118, key auth, no sshpass needed)
- **Code sync**: git push/pull
- **Data sync**: `rsync -avz ./data/ mlpc:~/dcsdxx/data/`
- **Monitor**: `http://192.168.1.118:8765/mlpc_monitor.html`
- **GPU**: RTX 5060, CUDA 13.1, CatBoost `task_type='GPU'`
- **Speed**: ~5-10x faster than M5 for CatBoost training

## 13. Repository Notes

- Branch `predict-etc-direct` merged into `main` via no-ff merge
- `.omc/` and `catboost_info/` untracked via `git rm --cached` + `.gitignore`
- All `Co-Authored-By: Claude` trailers removed via `git filter-branch`
- Force pushed to origin; GitHub contributor cache may need minutes to refresh
- `data/raw/*` gitignored — station Excel data never committed
- README updated for group use

## 12. Latest Findings (2026-05-02 ~ 05-06)

### 859-Combo Exhaustive Sweep (Large Sample)
- Best: **9 features R²=0.758** (fpar+SM+NDVI+LSWI+DOY+VPD+precip+solar+tmean)
- Full: 51 features R²=0.765 — gap only 0.007
- Top 20 all cluster at 0.756-0.758
- No-weather best: 8 features R²=0.684 (fpar+ΔLST+SM+NDVI+SWIR2+LSWI+Albedo+DOY)
- Weather variables contribute ~0.07 R² independently

### Kc Indirect vs ETc Direct
- Large sample (370K): ETc direct ≈ Kc indirect (R² diff < 0.01)
- Station (304): ETc+ET0 > Kc indirect (0.518 vs 0.486)
- Without weather: ETc direct >> Kc indirect (0.746 vs 0.666) because Kc needs ET0 context

### SHAP Feature Importance
- Station: fPAR, ΔLST, Albedo are strongest 3 features
- Large sample: fPAR + NDVI + SWIR2(b07) + LSWI + DOY is minimal sufficient set
- DOY appears in 6/10 station top combos — codes both phenology and seasonal climate

### ET0 Calculation
- FAO-56 Penman-Monteith via `src/kcact/features/et0.py`
- Uses ERA5-Land dewpoint temperature when available (most accurate ea calculation)
- Without dewpoint: RHmean simplification (≠ textbook RHmax/RHmin weighting)
- Difference from textbook method: ~2.5%

### VI Permutation & RDVI (2026-05-08)
- 255 combos with RDVI added: R² ceiling 0.661, RDVI adds 0.001 over baseline
- 5 VIs (NDVI/EVI/GNDVI/SAVI/RDVI) are highly collinear — any 1-2 + SM + DOY ≈ any 5
- MOD09A1 7-band full set (b01-b07) now available locally (56 new CSV files)

### Growth Stage Split (2026-05-08)
- 5 stages tested: Rapid growth (180-210) R²=0.51 best, Peak/tasseling (210-240) R²=0.10 worst
- Full-season R²=0.66 is 83% DOY-driven — stage-split exposes NDVI saturation ceiling
- SHAP confirms DOY dominates full-season prediction

### Corrected ET0 & Station Figures (2026-05-09)
- ET0 code fully aligned with FAO-56: Tmean=(Tmax+Tmin)/2 (Eq.9), T+273 (Eq.6 denominator), T+273.16 (Eq.39)
- UTC+8 timezone correction for station observation matching
- Station elevation from SRTM applied (禹城20.6m, 位山34m, 馆陶41.9m, 栾城53.2m)
- 5 summer maize Kcact figures regenerated: `outputs/figures/corrected/maize_fig*.png`
- Corrected xlsx: `outputs/tables/station_etc_et0_kcact.xlsx`

### Satellite SM Attempts (2026-05-09)
- ESA CCI SM: not on CDS, not on GEE (requires separate auth) — not yet downloaded
- GLDAS SM: 13 GEE export tasks submitted (Drive folder `kcact_maize_modis_indicators`)
  - GLDAS is model-based (NASA Noah), not true satellite retrieval — similar to ERA5-Land
- ERA5-Land SM: already extracted (29M rows), used in station 962 combos
- Station top-20 already uses real ERA5 SM (sm_surface), not proxy

### Remote Training (mlpc)
- SSH: `ssh mlpc` (192.168.1.118, key auth in ~/.ssh/config)
- Code: git push/pull
- Data: `rsync -avz ./data/ mlpc:~/dcsdxx/data/`
- Python: `/home/hert/miniforge3/envs/sdxx/bin/python`
- GPU training: CatBoost `task_type='GPU', devices='0'`
- Long tasks: `ssh mlpc "nohup python script.py > log 2>&1 &"`
- Monitor: `http://192.168.1.118:8765/mlpc_monitor.html`

### Data Source Lessons Learned
- MODIS NDVI coverage determines ranking: 8-day (100%) >> 16-day (50%)
- SWIR2(b07, 2.13μm) is single strongest raw band for Kcact — better than any VI alone
- GEE exports: split by province to avoid OOM (27K points per task = memory overflow)
- Albedo/S2 raw/S1 SAR: 3 export rounds, 109+84 failures — use v3 per-province approach

## 14. Path Forward

1. ~~FAO-56 ET0 audit~~ → compliant, equation-numbered code in `src/kcact/features/et0.py`
2. ~~ETa/ET0 spatial alignment fix~~ → coordinate-stable `patch_id` and 10 m post-merge assertion implemented
3. ~~Strict QC and seven-indicator pipeline~~ → Kcact/ETa/ET0/VI/SM thresholds implemented
4. ~~SMAP L4 comparison~~ → `sm_surface` best but below ERA5-like 0.1° result
5. ~~PML-V2.2a large-sample export and model~~ → current midterm main result: R²=0.765, RMSE=0.102, MAE=0.076
6. ~~Station validation figures~~ → ERA5-vs-station ET0, PML-vs-tower ETa, and PML/ERA5-vs-station/met Kcact seasonal curves generated
7. ~~Mini-program midterm demo~~ → PML model values and agent-style interaction page added
8. **Direct RDVI finalization** → re-export/rebuild S2 with direct Red/NIR-derived RDVI and rerun PML/ERA5 model with `--rdvi-source direct` if final paper needs a completely direct RDVI provenance
9. **Report polishing** → use black/white technical route, captions outside figures, superscript citations, Chinese full names for indicators on first mention
10. **Product-side preprocessing spec** → document how one-click farmland query reproduces training features: PML/ERA5/S2/SM windows, 0.1° aggregation, QC, and Kcact prediction
11. **SM downscaling research path** → if time permits, test RF/CatBoost downscaling of ERA5/SMAP SM using S1/S2/DEM/DOY auxiliary variables; do not use station-only data for product model
12. **Post-midterm model update** → add 2025 PML if available and repeat LOYO/temporal holdout; keep training/application data sources consistent

## 15. MOD16 vs Tower Kcact Comparison (NEW 2026-05-13)

### 15.1 Motivation

大样本 Kcact 的 ETa 来自 MOD16A2GF (500m, 8天)，站点 Kcact 的 ETa 来自通量塔实测。两者使用相同的 ERA5 ET0 分母。在通量塔坐标上同时提取 MOD16 ET，直接对比可量化 MOD16 产品的点位偏差。

### 15.2 Method

- 提取 `MODIS/061/MOD16A2GF` 在 4 站坐标的 8 天总 ET (mm)，除以 8 转日均
- 匹配站点观测窗口（重叠 MOD16 8 天时段），取重合 MOD16 窗口的均值
- 同一 ET0 分母下计算 Kc_mod16 和 Kc_tower，对比
- 脚本：`scripts/python/compare_kc_mod16_vs_tower.py`
- 输出：`outputs/tables/kc_mod16_vs_tower.csv`

### 15.3 Results

| 指标 | 值 |
|------|-----|
| r(Kc_tower, Kc_mod16) | **0.14** |
| r(ETc_tower, ETc_mod16) | **0.47** |
| RMSE(Kc) | 0.39 |
| Kc_tower 均值 | 0.55 |
| Kc_mod16 均值 | 0.40 (**−29%** 系统性低估) |

**Per-station r(ETc)**:

| 站 | r(ETc) | tower 均值 | MOD16 均值 | Kc 偏差 |
|----|--------|----------|-----------|--------|
| 馆陶 | **0.63** | 1.39 mm/d | 1.15 mm/d | +0.02 |
| 禹城 | 0.51 | 1.86 mm/d | 1.15 mm/d | −0.14 |
| 位山 | 0.41 | 2.13 mm/d | 0.93 mm/d | −0.29 |
| 栾城 | N/A | — | — | MOD16 无数据 |

**Seasonal r(ETc)**:

| 季节 | r(ETc) |
|------|--------|
| 夏季 (6-9月) | **0.24** — 生长期最差 |
| 春季 (3-5月) | 0.48 |
| 冬季 (11-2月) | 0.06 |

### 15.4 Implications

1. **MOD16 500m ET 在点位尺度上不可靠。** r=0.47 的原始 ETc 相关性说明 500m 像素平均化和通量足迹的信号差异是根本性的。
2. **大样本 R²=0.77 学到的是 MOD16 PM 算法的内部规律，不是真正的田间作物蒸散。** 模型预测的是 MOD16 会输出什么，而非作物实际蒸腾了多少。
3. **站点 R²=0.47 被低估了。** 通量塔实测 Kc 是更真实的作物信号——它更难预测因为噪声更少，不是模型更差。
4. **栾城 MOD16 500 景全 NaN** — MOD16A2GF 在太行山前平原（37.88°N, 114.69°E）无有效数据，原因待查（可能被土地覆盖掩膜排除）。
5. **馆陶偏差异常小（+0.02）**，位山异常大（−0.29）。站间差异说明 MOD16 的偏差不是全局常数，与局地土地覆盖异质性有关。

## 16. Spatial Scale Mismatch Analysis (NEW 2026-05-13)

### 16.1 The Problem

站点 Kcact = ETa / ET0，但分子分母不在同一空间尺度：

| 变量 | 来源 | 原生分辨率 | 面积 | vs 通量足迹 (0.1km²) |
|------|------|-----------|------|---------------------|
| ETa | 涡度相关/蒸渗仪 | ~100-500m 足迹 | ~0.1 km² | 基准 |
| ET0 | ERA5 → FAO-56 PM | 0.1° (~11km) | ~100 km² | **100-3000x** |
| NDVI/EVI | MOD13Q1 | 250m | 0.06 km² | 0.6-25x |
| SM_surface | ERA5-Land | 0.1° (~11km) | ~100 km² | 100-3000x |
| fPAR | MOD15A2H | 500m | 0.25 km² | 1-25x |
| LST | MOD11A2 | 1km | 1 km² | 3-100x |

### 16.2 Impact by Pipeline

- **大样本**：ETa (MOD16 500m) + ET0 (ERA5 11km) — 都是网格产品，尺度更一致。模型学到网格到网格映射。但目标本身（MOD16 Kc）已偏离真实作物蒸散。
- **站点**：ETa (通量塔点) + ET0 (ERA5 11km) — 尺度错配最大。Kc 目标更真实但输入特征无法匹配通量塔的微气象。

### 16.3 Possible Fixes

1. **站级气象数据**（最优）：用通量塔同步气象观测算 ET0，与 ETa 同尺度的分子分母
2. **CMFD**（中国区域强迫数据）：仍 0.1° 但基于 740 站实测插值，比 ERA5 准
3. **DEM 降尺度**：气温递减率校正 ERA5 气温到 30m 网格，物理合理、可批量

## 17. Soil Hydraulic Parameters (NEW 2026-05-13)

### 17.1 Dai et al. 2013 Dataset

- TPDC: `https://data.tpdc.ac.cn/zh-hans/data/205da4ae-63cd-48e1-994e-0b5d8830812a`
- 30 arc-second (~1km), 7 layers depth=[4.5, 9.1, 16.6, 28.9, 49.3, 82.9, 138.3] cm
- TH33=FC (33kPa), TH1500=WP (1500kPa), units cm³/cm³
- 多 PTF 集成，作者自述 "accuracy is unknown for lack of in-situ and regional measurements"
- **Luancheng verification**: 数据集的 FC/WP 系统性低于实测 ~50%，非读取错误，是 PTF 的区域偏差
- 10 个参数 NC 文件嵌套 zip 中（ALPHA, N, LAMBDA 等），仅解压了 TH33 和 TH1500

### 17.2 Guantao FC/WP from Soil Texture Image

- 方法：Saxton-Rawls 2006 PTF（修正版）
- 修正：论文系数 `0.0452` → `0.452`，加缺失的第二步调整
- 3 层：0-25cm (clay 41.7%), 26-45cm (clay 56.6%), 45-100cm (clay 30.1%)
- 0-100cm 加权：FC=38.0% (380mm), WP=23.1% (231mm), PAW=14.9% (149mm)
- 假设：OM=1.5%（无实测值）

### 17.3 Meta-Analysis

`outputs/tables/meta_fc_wp.csv` — 17 条记录，覆盖 4 通量站 + 泰安、封丘参考站。含方法、来源、可靠度标记。

## 18. Soil Moisture Data Inventory (NEW 2026-05-13)

| 数据集 | 时空 | 层次 | 站点 | 文件 |
|--------|------|------|------|------|
| FLDAS (Noah01) | 月, 0.1° | 0-10,10-40,40-100,100-200cm | 4 站已提取 | `data/processed/fldas_station_sm.csv` |
| ERA5-Land SM | 月/日, 0.1° | 0-7,7-28,28-100,100-289cm | 4 站已提取，SM_root_mm 已写入 Excel | `data/processed/era5_station_sm.csv` |
| GLDAS SM | 月, 0.25° | 4 层 | 13 GEE export 待下载 | `kcact_maize_modis_indicators/` |

**SM_root 计算** (0-100cm): `SM_root_mm = θL1×70 + θL2×210 + θL3×720` (ERA5 层次厚度加权)，基于标准土壤物理学 (Hillel) 和 FAO-56 框架。

**SM vs Kc 关系**：ERA5 SM 与 Kcact 总体相关弱 (r=0.17)，但季节性窗口强——DOY 91-150 r=+0.43 (小麦), DOY 241-300 r=+0.57 (玉米灌浆)。

## 19. Summer Maize Phenology & Varieties (NEW 2026-05-13)

### 19.1 Phenological Dates

**Project convention for future figures/statistics (updated 2026-05-16):**

- Core summer-maize stages used in the report:
  - Initial stage: `06-15—07-06`
  - Developing stage: `07-07—08-08`
  - Mid stage: `08-09—09-12`
  - End stage: `09-13—10-11`
- For all later summer-maize growth-season line charts, stage summaries, and station-vs-regional comparisons, keep the full 8-day window coverage as **`06-09—10-15`**.
- Interpretation: `06-09` is the leading 8-day window retained before the first core stage, and `10-15` is the trailing 8-day window retained after the last core stage. Stage shading/labels should still correspond to `06-15—10-11`; the extra margins are only to avoid truncating 8-day ETa/ET0/PML/remote-sensing windows.
- Current canonical example figure:
  - `outputs/figures/pml_era5_vs_station_met_kcact_doy_summer_maize_paper.png`
  - `outputs/figures/pml_era5_vs_station_met_kcact_doy_summer_maize_paper.pdf`
  - script: `scripts/python/plot_pml_era5_vs_station_met_kc_doy.py`

| 站 | 播种 | 抽雄 | 收获 | 全生育期 |
|----|------|------|------|---------|
| 栾城 (37.88°N) | 6/15-6/20 | 7/28-8/05 | 9/25-10/05 | ~105d |
| 禹城 (36.83°N) | 6/12-6/18 | 7/25-8/02 | 9/22-10/01 | ~103d |
| 位山 (36.65°N) | 6/12-6/18 | 7/25-8/02 | 9/22-10/01 | ~103d |
| 馆陶 (36.52°N) | 6/10-6/15 | 7/22-7/30 | 9/20-9/28 | ~100d |

南北纬度差 1.5° → 积温差 ~100°C·d → 日期差 5-7 天。

### 19.2 Varieties

- **未找到站级品种记录。** CERN/ScienceDB 公开数据库含通量数据但无田间管理记录。
- 品种年会报告在 CERN 内部受限制数据库中，非公开可获取。
- 文献推断主导品种：**郑单 958** (2004+)，此前为农大 108 (pre-2004)
- 站间品种差异附加影响约 2-5 天，小于纬度梯度的 5-7 天

## 20. Midterm Report and Product Demo Notes (NEW 2026-05-16)

### 20.1 Main narrative

The report should not present the work as “a station model”. The scientifically safer story is:

1. Build a regional gridded Kcact label from public, automatable data (`PML-V2.2a ETa / ERA5 ET0`).
2. Align ETa, ET0, Sentinel-2 vegetation indices, SM, and DOY in the same 8-day windows and 0.1° grid.
3. Train and validate with Leave-One-Year-Out to avoid random temporal leakage.
4. Use station eddy-covariance / meteorological observations as an independent scale-difference and applicability check.
5. Package outputs into a mini-program style decision-support product: monitoring, parcel diagnosis, model explanation, report generation, and intelligent assistant.

### 20.2 Figure rules learned from teacher feedback

- Do not put the figure title inside the image; use Word caption text instead.
- Use black/white or one-accent publication style for technical-route figures; avoid colorful dashboard-style route charts in the report body.
- Use Chinese full names plus acronyms on first mention, e.g. “归一化植被指数（NDVI）”.
- If a linear/correlation plot already has R², avoid also displaying Pearson r unless necessary.
- For the summer-maize seasonal line chart, use `06-09—10-15` as the plotted window and explain `06-15—10-11` as the core stage span.

### 20.3 Current product demo state

Mini-program directory: `apps/kcact-miniprogram/`.

Current pages:

- `overview`: regional water-demand/risk dashboard
- `parcel`: representative grid/parcel detail and seasonal curve
- `model`: seven-indicator model explanation, contribution ranking, yearly validation
- `report`: generated diagnostic report
- `agent`: intelligent assistant interaction display

For midterm presentation, use product wording such as “智能交互模块”“地块诊断报告”“模型解释模块”.

## 11. Known Issues / Gotchas

- **XGBoost on macOS**: requires `DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"`
- **MOD13Q1 NDVI 50% coverage**: 16-day composites miss half of 8-day observation windows → heavy NaN imputation → poor model performance. Use MOD09A1 or MOD09GA instead.
- **MCD43A3 daily OOM**: Daily images × 27K points causes GEE OOM. Use 8-day window medians instead.
- **GEE .getInfo() limits**: ~10MB response, ~5 min timeout. For >1000 points, use Export.table.toDrive().
- **Small station sample (304 maize obs)**: LOSO CV is very harsh (training on 3 stations, testing on 1 with different baseline). Results indicate feature ranking robustness, not absolute prediction quality.
- **Station name encoding**: MODIS extracts use English names (yucheng, weishan, etc.), other data uses Chinese (禹城, 位山). Always map before merging.
- **MOD16A2GF at Luancheng**: 500 MOD16 windows all NaN at Luancheng (37.88°N, 114.69°E) — MOD16 land cover mask likely excludes this pixel. Other 3 stations OK.
- **MOD16 8-day total vs daily**: MOD16A2GF ET band is 8-day total (mm) with scale factor 0.1. Must divide by 8 to compare with daily mean ETc from tower.
- **Spatial scale gap**: ERA5-Land weather+SM at 11km vs tower ETa at 100-500m footprint creates a 100-3000x area mismatch in station Kcact denominator. Large-sample pipeline has both ETa (MOD16 500m) and ET0 (ERA5 11km) as gridded products and now uses coordinate-stable joins, but native resolution mismatch and MOD16 PM algorithm bias (~29% low vs tower) remain.
- **Do not join large-sample tables by raw GEE `pt_*`**: fixed on 2026-05-14. Always use `coord_key`/stable `patch_id`; raw `pt_*` survives only as `gee_point_id`.
- **Temporal coverage beats spatial resolution** for NDVI: MOD09GA daily 500m outperforms MOD09Q1 8-day 250m, and MOD13Q1 16-day is the worst despite having 250m resolution.
