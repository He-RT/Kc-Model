# Phase 2 Runbook: Candidate Patches to Kcact Training Table

## 1. 目标

在河北冬小麦候选斑块出来后，直接产出：

- `ET0`
- `ETc`
- `Kcact`
- `NDVI / EVI / SAVI / GNDVI / LSWI / NIRV / RE-NDVI`
- 可直接用于深度学习的序列数组

## 2. 脚本清单

### 2.1 GEE 斑块时间序列导出

- [export_hebei_patch_timeseries.py](/Users/hert/Projects/dcsdxx/scripts/python/export_hebei_patch_timeseries.py:1)

输入：

- 河北冬小麦候选斑块资产 `FeatureCollection`

输出到 Drive：

- `hebei_patch_s2_features_<year>.csv`
- `hebei_patch_era5_daily_<year>.csv`
- `hebei_patch_mod16_etc_<year>.csv`

### 2.2 本地训练表构建

- [build_hebei_kcact_table.py](/Users/hert/Projects/dcsdxx/scripts/python/build_hebei_kcact_table.py:1)

输入：

- 上面 3 类 CSV

输出：

- 全量表
- 质控后的训练表

### 2.3 深度学习序列样本生成

- [make_hebei_dl_sequences.py](/Users/hert/Projects/dcsdxx/scripts/python/make_hebei_dl_sequences.py:1)

输出：

- `npz` 数组
- `meta.csv`
- `features.json`

## 3. 推荐执行顺序

### Step 1. 把候选斑块上传为 GEE 资产

假设资产路径为：

```text
projects/chuang-yaogan/assets/kcact/patches/hebei_winter_wheat_candidate_patches_2025
```

### Step 2. 在 GEE 端导出斑块时间序列表

```bash
conda run -n sdxx python /Users/hert/Projects/dcsdxx/scripts/python/export_hebei_patch_timeseries.py \
  --project-id chuang-yaogan \
  --patch-asset projects/chuang-yaogan/assets/kcact/patches/hebei_winter_wheat_candidate_patches_2025 \
  --years 2025 \
  --drive-folder kcact_hebei_patch_timeseries
```

### Step 3. 下载 CSV 到本地后构建 Kcact 表

```bash
conda run -n sdxx python /Users/hert/Projects/dcsdxx/scripts/python/build_hebei_kcact_table.py \
  --s2-csv "/Users/hert/Projects/dcsdxx/data/raw/gee/hebei_patch_s2_features_2025*.csv" \
  --era5-csv "/Users/hert/Projects/dcsdxx/data/raw/gee/hebei_patch_era5_daily_2025*.csv" \
  --mod16-csv "/Users/hert/Projects/dcsdxx/data/raw/gee/hebei_patch_mod16_etc_2025*.csv"
```

### Step 4. 生成深度学习序列样本

```bash
conda run -n sdxx python /Users/hert/Projects/dcsdxx/scripts/python/make_hebei_dl_sequences.py \
  --input-table /Users/hert/Projects/dcsdxx/data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet \
  --seq-len 6 \
  --train-years 2025 \
  --test-years 2025 \
  --output-prefix /Users/hert/Projects/dcsdxx/data/processed/train/hebei_winter_wheat_kcact_seq_2025
```

## 4. 训练表关键字段

输出训练表里会包含这些核心字段：

- `patch_id`
- `date`
- `date_start`
- `date_end`
- `ndvi`
- `evi`
- `savi`
- `gndvi`
- `lswi`
- `nirv`
- `re_ndvi`
- `tmean_c`
- `tmax_c`
- `tmin_c`
- `precip_mm_8d`
- `wind_2m_m_s_mean_8d`
- `solar_rad_mj_m2_d_sum_8d`
- `vpd_kpa_mean_8d`
- `et0_pm_8d_mm`
- `etc_8d_mm`
- `kcact`

## 5. 说明

这条程序链的前提是：

- 你已经拿到了候选斑块
- 或者后面替换成了监督分类后的斑块

这样结果一落地，就不需要再重新设计程序，直接继续做 `Kcact` 和深度学习即可。
