# Kcact Winter Wheat ET Modeling — Handover Document

**Last updated**: 2026-04-26 | **Session**: Tier 1 experiments completed → transferring to desktop

## 1. Project Overview

**Goal**: Predict Kcact (crop coefficient = ETc / ET0) for winter wheat in Hebei Province, North China Plain.

**Target variable**: Kcact = ETc / ET0
- ETc from MODIS MOD16A2GF (500m, 8-day)
- ET0 computed via FAO-56 Penman-Monteith from ERA5 weather data

**Current best model**: CatBoost (default), LOYO CV pooled R² = **0.7017**
- 18,528 samples, 592 patches, 7 years (2019–2025), 46 features
- Features: S2 vegetation indices + ERA5 weather + phenology + interactions
- VPD and day-of-year dominate feature importance

**⚠️ Tier 1 improvement attempts (2026-04-26): ALL FAILED**
- CatBoost hyperparameter tuning (50-trial Optuna): R² = 0.6881 (−0.0136)
- Stacking ensemble (XGB+CB+LGBM → Ridge): R² = 0.6982 (−0.0035)
- Water balance features (water_balance_cum, aridity_index, etc.): R² = 0.6924 (−0.0093)
- Simple average ensemble: R² = 0.6983 (−0.0034)
- **Conclusion**: Default CatBoost at 0.7017 is the ceiling with current data.

## 2. Git Repository State

- **Branch**: `predict-etc-direct`
- **Root**: `/Users/hert/Projects/dcsdxx`
- **Remote**: `origin` → `https://github.com/He-RT/dcsdxx.git`
- **Uncommitted changes**:
  - `scripts/python/cross_validate_kcact_v2.py` (+132 lines — added LightGBM, TunedCatBoost, Ensemble_Avg)
  - `catboost_info/` — tuning logs
  - New scripts: `tune_kcact_catboost.py`, `train_kcact_stacking.py` (untracked)
  - New docs: `docs/handover.md`, `docs/plans/` (untracked)

## 3. Environment

| Item | Value |
|---|---|
| Conda env name | `sdxx` |
| Python path | `/Users/hert/conda/envs/sdxx/bin/python3` |
| XGBoost | 3.2.0 |
| CatBoost | 1.2.10 |
| LightGBM | 4.6.0 (newly installed) |
| Optuna | 4.8.0 |
| scikit-learn | 1.8.0 |
| GEE project | `chuang-yaogan` |

**macOS run prefix** (for M5 MacBook Air):
```bash
DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH" /Users/hert/conda/envs/sdxx/bin/python3 scripts/python/<script_name>.py
```

**Desktop (7500F + RTX 5060)**: After cloning repo and creating `sdxx` env, run WITHOUT the `DYLD_LIBRARY_PATH` prefix. Install GPU-enabled versions:
```bash
pip install xgboost catboost lightgbm optuna scikit-learn pandas numpy pyarrow
# CatBoost + XGBoost + LightGBM auto-detect CUDA on RTX 5060
```

## 4. Directory Structure (key files)

```
dcsdxx/
├── src/kcact/
│   └── data/kcact_builder.py            # Feature engineering — WATER BALANCE REVERTED
├── scripts/python/
│   ├── cross_validate_kcact_v2.py       # ★ Main CV script (updated with 7 models)
│   ├── tune_kcact_catboost.py           # ★ NEW: CatBoost Optuna tuning
│   ├── train_kcact_stacking.py          # ★ NEW: nested LOYO stacking ensemble
│   ├── tune_kcact_xgb.py                # XGBoost Optuna tuning (existing)
│   ├── run_cleaning_experiments.py      # Data cleaning experiments
│   └── ... (other GEE export/training scripts)
├── outputs/
│   ├── models/
│   │   └── kcact_catboost_tuned.cbm     # ★ NEW: 50-trial tuned (but worse than default)
│   └── tables/
│       ├── kcact_loyo_v2_summary.csv    # Latest CV results
│       ├── kcact_catboost_tuned_params.csv
│       └── kcact_stacking_results.csv
└── docs/
    ├── handover.md                      # This file
    └── plans/
        └── kcact_data_scaling_plan.md
```

## 5. Training Data

| Property | Value |
|---|---|
| Source | `data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet` |
| Samples (qc_valid) | 18,528 |
| Patches (sequences) | 592 |
| Years | 2019–2025 (7 years) |
| Features | 46 numeric |
| Excluded cols | patch_id, point_id, date, date_start, date_end, province, crop_type, qc_valid, kcact, etc_8d_mm, et0_pm_8d_mm, qc_mod16, .geo, system:index, valid_obs |

### 46 Feature Columns

| Category | Count | Columns |
|---|---|---|
| S2 indices | 7 | ndvi, evi, savi, gndvi, lswi, nirv, re_ndvi |
| Weather | 7 | tmean_c, tmin_c, tmax_c, vpd_kpa_mean_8d, solar_rad_mj_m2_d_sum_8d, wind_2m_m_s_mean_8d, precip_mm_8d |
| Temporal | 3 | doy, gdd_8d, season_year |
| Rolling precip | 3 | precip_7d, precip_15d, precip_30d |
| NDVI derivatives | 5 | ndvi_lag1, ndvi_diff, ndvi_accel, ndvi_mean_prev_3win |
| Phenology | 4 | doy_season, gdd_cum, gdd_frac, ndvi_rel |
| Interactions | 5 | ndvi_vpd, lswi_vpd, vpd_sq, vpd_doy_season, vpd_gdd_frac |
| Senescence | 3 | ndvi_peak_dist, ndvi_decline, lswi_diff |
| Ancillary | 9 | centroid_lat/lon, dewpoint, pressure, obs_count_s2, winter_wheat_candidate, wind, elev variants |

Also available: `hebei_winter_wheat_kcact_with_modis.parquet` (+3 cols: modis_ndvi, modis_evi, modis_qa) — **proven NO-GO**.

**Water balance features attempted and REVERTED** (degraded R² by 0.009):
- `water_balance_8d` = precip − ET0
- `water_balance_cum`, `water_balance_30d` (rolling)
- `aridity_index_8d` = precip / ET0

## 6. Model Results Summary

### Latest CV (2026-04-26, 46 features — clean baseline)

| Model | Pooled R² | RMSE | vs. Baseline Δ |
|---|---|---|---|
| **CatBoost (default)** | **0.7017** | 0.1119 | baseline |
| Ensemble_Avg (XGB/CB/LGBM) | 0.6983 | 0.1125 | −0.0034 |
| TunedCatBoost (50-trial) | 0.6881 | 0.1144 | −0.0136 |
| XGB_raw | 0.6797 | 0.1160 | −0.0220 |
| LightGBM | 0.6791 | 0.1161 | −0.0226 |
| XGB_log | 0.6764 | 0.1165 | −0.0253 |

### Stacking results (from separate script — uses `year` as feature)

| Model | Pooled R² |
|---|---|
| Ridge stacking | 0.6982 |
| Simple average | 0.6971 |

### Earlier experiments (from original handover)

| Experiment | R² | Verdict |
|---|---|---|
| Strict MOD16 QC | 0.7024 | Equivalent, cleaner data |
| + MODIS NDVI | 0.6992 | NO-GO |
| Kcact smoothing | 0.5751 | NO-GO |
| Stage-split CatBoost | ~0.696 | NO-GO |
| ETc-direct | ~0.697 | NO-GO |
| Water balance feat. | 0.6924 | **NO-GO** |
| CatBoost tuning | 0.6881 | **NO-GO** |
| Stacking ensemble | 0.6982 | **NO-GO** |

### Per-stage R² ceiling (XGB, 46feat)

| NDVI stage | R² | n |
|---|---|---|
| Early (< 0.35) | 0.80 | 8,687 |
| Mid (0.35–0.60) | 0.75 | 3,972 |
| Late (≥ 0.60) | **0.43** | 5,869 |

### Per-year CatBoost R²

| Year | R² |
|---|---|
| 2019 | 0.8851 |
| 2020 | 0.8206 |
| 2021 | 0.6723 |
| 2022 | 0.6339 |
| 2023 | 0.6643 |
| 2024 | 0.4859 |
| 2025 | 0.3427 |

## 7. Tier 1 Experiments Log (2026-04-26)

### a. Water balance features → NO-GO
Added `water_balance_8d/_cum/_30d`, `aridity_index_8d` in `kcact_builder.py`. Rebuilt parquet.
- Result: CatBoost dropped from 0.7017 → 0.6924 (−0.0093)
- **Action**: Reverted changes in `kcact_builder.py`. Parquet rebuilt to 46-feature clean state.

### b. CatBoost Optuna tuning → NO-GO
Created `scripts/python/tune_kcact_catboost.py` with 8-param search space + early stopping.
- 50 trials, ~21 min on M5 MacBook Air
- Best CV mean R²: 0.5835
- Best params: iterations=700, depth=6, lr=0.0398, l2_leaf_reg=16.43, subsample=0.782
- Actual pooled R² when applied: **0.6881** (worse than default 0.7017)
- **Action**: Script kept for reference. Tuned model saved to `outputs/models/kcact_catboost_tuned.cbm`.

### c. Stacking ensemble → NO-GO
Created `scripts/python/train_kcact_stacking.py` with nested LOYO + Ridge meta-learner.
- Uses `year` as a feature column (51 features vs 46 in CV)
- Simple average pooled R²: 0.6971
- Ridge stacking pooled R²: 0.6982
- Neither beats CatBoost 0.7017
- Script timed out on M5 (nest LOYO + 3 base models = slow). Needs GPU.
- Per-year: stacking helps 2019/2022/2023, hurts 2021/2024/2025.

### d. CV script updates
Updated `cross_validate_kcact_v2.py` (+132 lines):
- Added LightGBM import and baseline
- Added TunedCatBoost (loads params from CSV, graceful fallback)
- Added simple average ensemble (XGB + CB + LGBM)
- Preserved all original variants (XGB_raw, XGB_log, CatBoost, stage-split)

## 8. Path Forward: Geographic Expansion

**Only viable path to break the 0.70 ceiling.** All feature/model optimizations exhausted.

### Implementation plan
1. Add Henan, Shandong, Anhui provinces to GEE export
2. Same S2+ERA5+MOD16 pipeline, same 2019–2025 window
3. Expected: 2–3x sequence count (~1,200–1,800 patches)
4. Enables LSTM/sequence models + more robust CatBoost

### GEE export for new provinces
Modify `export_hebei_kcact_training_data.py` or create province-specific copies:
- Change the geographic bounding box / admin filter
- Keep same S2 indices, ERA5 variables, MOD16 product
- Export at 1,000m spacing initially

### Desktop advantages
With GPU (RTX 5060), CatBoost tuning drops from ~21 min → ~2–3 min. Nest LOYO stacking becomes practical. LSTM training becomes viable with expanded data.

## 9. Known Issues / Gotchas

- **XGBoost on macOS**: requires `DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"`
- **2024/2025 data**: mid-season only → no early/late NDVI stages → anomalously low R²
- **Late-stage NDVI**: R² ≈ 0.43 ceiling — inherent physiological variability, not modeling issue
- **CatBoost tuning overfits to CV mean**: Optuna objective = mean fold R², but best params generalize worse than defaults
- **Water balance features**: precip − ET0 is arithmetically sound but statistically pure noise for this task
- **Training parquet rebuilt**: `build_hebei_kcact_table.py` with 2019–2025 CSVs → 46 clean features

## 10. Key Commands

```bash
# On macOS (M5):
DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH" /Users/hert/conda/envs/sdxx/bin/python3 scripts/python/cross_validate_kcact_v2.py
DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH" /Users/hert/conda/envs/sdxx/bin/python3 scripts/python/tune_kcact_catboost.py --n-trials 50

# On desktop (Linux/Windows, no DYLD_ prefix):
python scripts/python/cross_validate_kcact_v2.py
python scripts/python/tune_kcact_catboost.py --n-trials 50
python scripts/python/train_kcact_stacking.py

# Rebuild training parquet:
python scripts/python/build_hebei_kcact_table.py \
  --s2-csv data/raw/gee/hebei_kcact_s2_features_*.csv \
  --era5-csv data/raw/gee/hebei_kcact_era5_daily_*.csv \
  --mod16-csv data/raw/gee/hebei_kcact_mod16_etc_*.csv

# Trigger new GEE export:
python scripts/python/export_hebei_kcact_training_data.py --project-id chuang-yaogan --year 2025
```
