# Kcact Data Scaling Plan

**Created**: 2026-04-25 | **Branch**: `predict-etc-direct`

## Objective

Scale the Kcact training dataset to enable LSTM and other sequence models, while improving overall model accuracy by adding MODIS NDVI as a supplementary feature and expanding temporal coverage to 2000–2025 (26 years).

## Current State

| Metric | Value |
|---|---|
| Patches (sequences) | 592 |
| Total samples | 18,528 |
| Years covered | 2019–2025 (7 years) |
| Features | 46 (S2 indices + weather + phenology + interactions) |
| Best model | CatBoost, pooled LOYO R² = 0.702 |
| LSTM R² (2023 test) | 0.497 (592 sequences, insufficient) |

## Phase Plan

### Phase 1: Validate MODIS NDVI (COMPLETE — NO-GO)

**Goal**: Determine if MODIS NDVI adds predictive value over S2-only features.

- Export MOD13Q1 NDVI for 2019–2025 at 1,000m spacing (7 tasks submitted)
- Download CSVs from Google Drive
- Merge `modis_ndvi`, `modis_evi` into training table
- Run CatBoost LOYO CV with MODIS NDVI features
- **Result: ΔR² = -0.0025 (Baseline 0.7017 → +MODIS 0.6992) → NO-GO**

```
GEE Tasks: 7 (ALL COMPLETED)
Output: data/raw/gee/hebei_kcact_modis_ndvi_{year}.csv
Model: CatBoost (500 iters, depth=6, lr=0.05) LOYO CV
Decision: ΔR² = -0.0025 < 0.005 threshold → NO-GO
```

### Phase 2: Multi-Year MODIS (2000–2018)

**Goal**: Add 19 years of MODIS-only data to massively increase sequence count.

- Export MOD16 ET, MOD13Q1 NDVI, ERA5 for 2000–2018
- Sampling: 500m spacing (~2,400 patches/year)
- No S2 features available (pre-Sentinel-2 era)
- Merge with Phase 1 data → unified 2000–2025 MODIS dataset
- Train LSTM + CatBoost on MODIS-only features
- **Expected**: ~35,000 sequences, LSTM becomes viable

```
GEE Tasks: 19 years × 3 products = 57 tasks
Sampling: 500m grid within MODIS land cover cropland mask
Features: modis_ndvi, modis_evi, weather (tmean, VPD, etc.), ET0
Output: data/raw/gee/hebei_kcact_*_2000.csv ... 2018.csv
Model: LSTM + CatBoost LOYO CV
Target R²: 0.60+ (MODIS-only, lower ceiling than S2+MODIS)
```

### Phase 3: High-Res S2 + MODIS (2019–2025)

**Goal**: Re-export S2 features at 250m spacing for maximum sequence density.

- Re-export S2 features for 2019–2025 at 250m spacing
- Merge with Phase 2 MODIS data for 2019–2025
- Train LSTM with full feature set on 2000–2025 data
- **Expected**: ~234,000 total sequences, LSTM should outperform CatBoost
- Train final production models

```
GEE Tasks: 7 years × 4 products = 28 tasks
Sampling: 250m grid within full phenology-based winter wheat mask
Features: S2 (ndvi, evi, savi, gndvi, lswi, nirv, re_ndvi) + MODIS (modis_ndvi, modis_evi) + weather + phenology
Output: data/raw/gee/hebei_kcact_s2_features_{year}_250m.csv etc.
Model: LSTM + CatBoost LOYO CV
Target R²: 0.72+ (combined S2+MODIS+large sequences)
```

## Sequence Count Projection

| Phase | Years | Spacing | Products | Est. Sequences | Est. Total Samples |
|---|---|---|---|---|---|
| Current | 2019–2025 | 1,000m | S2+ERA5+MOD16 | 4,144 | 18,528 |
| Phase 1 | 2019–2025 | 1,000m | +MODIS NDVI | 4,144 | 18,528 |
| Phase 2 | 2000–2018 | 500m | MOD16+MOD13+ERA5 | ~35,000 | ~500,000 |
| Phase 2+3 | 2000–2025 | 250m/500m | All products | ~234,000 | ~2,500,000 |

## File Paths

```
Project root: /Users/hert/Projects/dcsdxx

Raw GEE exports:
  data/raw/gee/hebei_kcact_modis_ndvi_{year}.csv          (Phase 1)
  data/raw/gee/hebei_kcact_mod16_etc_{year}.csv           (Phase 2)
  data/raw/gee/hebei_kcact_modis_ndvi_{year}.csv          (Phase 2)
  data/raw/gee/hebei_kcact_era5_daily_{year}.csv          (Phase 2)
  data/raw/gee/hebei_kcact_s2_features_{year}_250m.csv    (Phase 3)

Training tables:
  data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet  (current)
  data/processed/train/hebei_winter_wheat_kcact_multi_year.parquet   (Phase 2+)

Scripts:
  scripts/python/export_modis_ndvi.py                 (Phase 1 export)
  scripts/python/cross_validate_kcact_v2.py           (Phase 1+2 CV)
  scripts/python/train_kcact_lstm.py                  (Phase 2+3 LSTM)
  scripts/python/build_hebei_kcact_table.py           (Phase 2+3 builder)

Models saved:
  outputs/models/kcact_catboost_v2_46feat.cbm/.pkl    (current best)
  outputs/models/kcact_catboost_modis_ndvi.cbm/.pkl   (Phase 1)
  outputs/models/kcact_lstm_multi_year.pt             (Phase 2+)

Reports:
  outputs/reports/kcact_data_cleaning_experiments.md  (strict QC + smoothing)
```

## Go/No-Go Decision Log

| Gate | Date | Decision | Rationale |
|---|---|---|---|
| Phase 1 → Phase 2 | 2026-04-25 | NO-GO | MODIS NDVI ΔR² = -0.0025 (degradation) |

## Revised Strategy (Post Phase 1 NO-GO)

MODIS NDVI at 250m pixel resolution does not carry predictive signal for Kcact
beyond what S2 indices already provide. MOD13Q1's 250m pixel scale likely does
not represent the 500m MOD16 ET footprint well enough, and the 16-day compositing
adds temporal noise.

**Viable alternatives to scale data:**

1. **Geographic expansion** — add Henan, Shandong, Anhui provinces. Same
   S2+ERA5+MOD16 pipeline, same 2019–2025 window. Could 2-3x the sequence count
   without new feature engineering.

2. **Higher-resolution resampling (Phase 3 only)** — 250m spacing within Hebei
   for 2019–2025. More sequences from denser grid, same feature set.

3. **ERA5-only pre-2019** — skip MODIS entirely. Use weather-only features
   (ERA5 reanalysis goes back decades). Lower accuracy ceiling but larger sample
   count for sequence models.
