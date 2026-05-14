# Kcact ET Modeling — Handover Document

**Last updated**: 2026-05-13 | **Sessions**: zhandian, remote, mod16-compare, fc-wp-meta | **Ready for compact**

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

1. ~~FAO-56 ET0 audit~~ → fully compliant, code annotated with equation numbers
2. ~~GEE exports~~ → 110 CSVs in `data/raw/gee/kcact_maize_modis_indicators/`, including MOD09A1 7-band full set
3. ~~500+ combo tests~~ → 859 combos (exhaustive) + 255 combos (RDVI) complete
4. ~~VI permutation~~ → NDVI+EVI+GNDVI+SAVI+RDVI all tested, ceiling at 0.66
5. ~~Growth stage split~~ → 5-stage analysis complete, DOY dominance documented
6. **Merge ERA5-Land SM into large-sample parquet** (29M rows, year-by-year to avoid OOM)
7. **GLDAS SM** — 13 GEE export tasks pending, download when ready
8. **Direct ETc prediction** with final feature set
9. **Paper writeup** — core results ready
10. **Spatial scale unification** — investigate station meteorological observations or DEM downscaling for station-consistent ET0
11. **MOD16-tower calibration** — build per-station MOD16 bias correction using Guantao (r=0.63, bias minimal) as anchor
12. **Sentinel-2 for station era** — temporal mismatch (S2 2017+ vs stations 2003-2015); Landsat 5/7/8 (30m, 1984+) is the viable high-res alternative for station period

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

## 11. Known Issues / Gotchas

- **XGBoost on macOS**: requires `DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"`
- **MOD13Q1 NDVI 50% coverage**: 16-day composites miss half of 8-day observation windows → heavy NaN imputation → poor model performance. Use MOD09A1 or MOD09GA instead.
- **MCD43A3 daily OOM**: Daily images × 27K points causes GEE OOM. Use 8-day window medians instead.
- **GEE .getInfo() limits**: ~10MB response, ~5 min timeout. For >1000 points, use Export.table.toDrive().
- **Small station sample (304 maize obs)**: LOSO CV is very harsh (training on 3 stations, testing on 1 with different baseline). Results indicate feature ranking robustness, not absolute prediction quality.
- **Station name encoding**: MODIS extracts use English names (yucheng, weishan, etc.), other data uses Chinese (禹城, 位山). Always map before merging.
- **MOD16A2GF at Luancheng**: 500 MOD16 windows all NaN at Luancheng (37.88°N, 114.69°E) — MOD16 land cover mask likely excludes this pixel. Other 3 stations OK.
- **MOD16 8-day total vs daily**: MOD16A2GF ET band is 8-day total (mm) with scale factor 0.1. Must divide by 8 to compare with daily mean ETc from tower.
- **Spatial scale gap**: ERA5-Land weather+SM at 11km vs tower ETa at 100-500m footprint creates a 100-3000x area mismatch in station Kcact denominator. Large-sample pipeline avoids this by having both ETa (MOD16 500m) and ET0 (ERA5 11km) as gridded products, but MOD16 PM algorithm introduces its own ~29% low bias vs tower.
- **Temporal coverage beats spatial resolution** for NDVI: MOD09GA daily 500m outperforms MOD09Q1 8-day 250m, and MOD13Q1 16-day is the worst despite having 250m resolution.
