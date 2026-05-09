# 华北平原作物 Kcact 数据集生产与建模

基于遥感与气象数据的作物系数（Kcact = ETc / ET0）预测，覆盖冬小麦和夏玉米。

**当前最优模型**：

| 作物 | 数据规模 | 样本数 | LOYO R² | 特征数 | 备注 |
|------|---------|--------|---------|--------|------|
| 冬小麦 | 河北 592 patches, 2019–2025 | 18,528 | 0.702 | 46 | |
| 夏玉米 | 华北四省 7,153 patches, 2019–2025 | 397,628 | **0.758** (9 feat) / **0.765** (51 feat) | 9 / 51 | 859组合全量实验 |
| 夏玉米 | 同上，无天气版 | 同上 | **0.684** (8 feat) | 8 | 纯遥感+DOY |
| 夏玉米 | 同上，ETc直接预测 | 同上 | **0.762** (9 feat) | 9 | 直接法≈间接法 |
| 夏玉米 | 同上，VI+SM+DOY 全排列 | 同上 | **0.661** (5 feat) | 5 | 255组合，RDVI零贡献 |
| 夏玉米（站点） | 4 通量站, 2003–2015 | 304 | **0.467** (7 feat) / **0.518** (ETc+ET0) | 5-7 |

## 数据流

```
GEE 导出                                 本地处理
─────────                              ─────────
S2 地表反射率 ──→ VI 计算              ┌─→ 合并训练表
ERA5-Land 气象 ──→ FAO-56 ET0 ────────┤
MOD16A2 ETc ────→ 8天窗口聚合 ────────┤
MODIS 热红外/fPAR/反照率 ─────────────┤
ERA5-Land 土壤水分 ───────────────────┘
                                       └─→ CatBoost LOYO CV
```

## 目录结构

```
.
├── configs/                  # YAML 配置
├── data/
│   ├── external/             # 外部输入（地块边界、土壤等）
│   ├── raw/                  # GEE 导出 CSV + 站点实测 Excel
│   │   └── gee/              # GEE Drive 导出文件
│   ├── interim/              # 中间结果
│   ├── processed/            # 训练 parquet + 站点衍生 CSV
│   │   └── modis_cache/      # GEE 提取缓存（gitignored）
│   └── metadata/             # 字段字典、区域列表
├── docs/                     # handover、数据流、候选指标清单
├── gee/                      # GEE JavaScript 脚本
├── scripts/python/           # 入口脚本
├── src/kcact/                # Python 业务模块
│   ├── data/                 # kcact_builder, io
│   └── features/             # et0 (FAO-56 PM)
├── outputs/
│   ├── figures/              # Nature-style 图表
│   ├── models/               # 训练好的模型
│   └── tables/               # CV 结果、组合对比
└── tests/                    # 单元测试
```

## 环境

**macOS (M5)**：
```bash
conda activate sdxx
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```

**Desktop (RTX 5060, CUDA)**：
```bash
conda create -n sdxx python=3.11 -y && conda activate sdxx
pip install catboost xgboost lightgbm scikit-learn pandas numpy pyarrow openpyxl earthengine-api
# GPU 自动检测
```

核心依赖：`catboost scikit-learn pandas numpy pyarrow openpyxl earthengine-api shap`

### 远程训练 (mlpc)

台式机 7500F + RTX 5060 (CUDA 13.1)，GPU 加速训练，M5 的 5-10x 速度。
```bash
ssh mlpc  # 已配密钥
# 代码: git push/pull
# 数据: rsync -avz ./data/ mlpc:~/dcsdxx/data/
# 训练: ssh mlpc "cd ~/dcsdxx && nohup python scripts/xxx.py > ~/log 2>&1 &"
# 进度: ssh mlpc "tail -20 ~/log"
# 监控: http://192.168.1.118:8765/mlpc_monitor.html
```

## 快速开始（桌面版）

```bash
# 1. 从 Mac 复制数据
scp -r mac:Projects/dcsdxx/data/ ./

# 2. 站点分析
python scripts/python/compute_station_et0.py      # Excel → ET0 + Kcact
python scripts/python/train_maize_500_combos.py   # 962 组合

# 3. 大样本（已有 MODIS 指标）
python scripts/python/merge_modis_yearly.py        # 合并 + 训练

# 4. 画图
python scripts/python/plot_station_kcact.py        # 9 张 Nature 风格图
```

## 关键脚本

### 站点实测数据

| 脚本 | 作用 |
|------|------|
| `compute_station_et0.py` | Excel → ERA5 逐日 ET0 → 8天窗口聚合 → Kcact |
| `extract_station_ml_features.py` | GEE 提取 8 个 MODIS/ERA5 遥感指标 |
| `extract_station_raw_bands.py` | GEE 提取 MOD09A1 全部 7 个原始波段 |
| `extract_station_era5_weather.py` | GEE 提取 ERA5-Land 气象变量 |
| `extract_station_mod09ga_ndvi.py` | GEE 提取 MOD09GA 逐日 NDVI |
| `train_maize_500_combos.py` | 962 组合 LOSO CV，夏玉米站点特征选择 |
| `train_station_ml_combos.py` | 141 组合全季节站点训练 |
| `plot_station_kcact.py` | Nature-style 宋体 Kcact 折线图（9 张） |
| `desktop_merge_and_train.py` | 合并 MODIS 指标 + 训练 Top 50 站点组合 |
| `desktop_exhaustive_combos.py` | 全量组合（859个）LOYO CV |
| `stats_collector.py` | mlpc CPU/GPU 状态收集 |
| `mlpc_monitor.html` | 远程训练实时监控仪表盘 |

### 大样本训练

| 脚本 | 作用 |
|------|------|
| `cross_validate_kcact_v2.py` | 主 CV 脚本（CatBoost/XGB/LGBM/Ensemble） |
| `build_maize_kcact_table.py` | 从 GEE 导出 CSV 构建夏玉米训练 parquet |
| `export_maize_kcact_training_data.py` | GEE 导出 S2/ERA5/MOD16 数据 |
| `export_maize_modis_indicators.py` | GEE 导出 MODIS fPAR/LST/Albedo/SM |
| `export_maize_s2raw_and_s1.py` | GEE 导出 S2 原始波段 + Sentinel-1 SAR |
| `merge_modis_and_retrain_maize.py` | 合并 MODIS 指标 → 大样本重训练 |

## 站点验证关键发现

### 8 个最优遥感指标

1. **fPAR** — 光合有效辐射吸收比（MOD15A2H, 8天, 500m）
2. **ΔLST** — 昼夜地表温差（MOD11A2, 8天, 1km）
3. **SM surface** — 表层土壤水分（ERA5-Land, 逐日, 9km）
4. **NDVI** — 归一化植被指数（MOD09A1, 8天, 500m）
5. **LSWI** — 地表水分指数（MOD09A1, 8天, 500m）
6. **Albedo** — 短波宽波段反照率（MCD43A3, 逐日, 500m）
7. **DOY** — 日序（物候阶段编码）
8. **b07 (SWIR2)** — 2.13μm 短波红外反射率，最强水分吸收波段

### Kc 间接法 vs ETc 直接法

在站点数据（304样本）上测试，直接法+ET0 超过间接法：

| 方法 | 最优组合 | R² |
|------|---------|-----|
| 间接法 (Kc) | fpar+dLST+SM+ndvi+lswi+alb+doy | 0.486 |
| 直接法 (ETc) | gndvi+dLST+alb+doy | 0.481 |
| 直接法+ET0 | gndvi+dLST+alb+doy+ET0 | **0.518** |

间接法在小样本上稳定，直接法+ET0最优但需要大气背景值。详见 `outputs/tables/kcact_vs_etc_direct_top20.csv`。

### 核心结论

- **3 个特征接近 7 个的效果**：ndvi + b07(SWIR2) + doy → R²=0.461（vs 7 特征 0.467）
- **覆盖率 > 分辨率**：逐日 MOD09GA NDVI (500m) 优于 16 天 MOD13Q1 (250m)
- **单指标最强**：fPAR (R²=0.318) > NDVI (R²=0.222) > LST (R²=−0.016)
- **站间差异显著**：馆陶的指标相关性系统性最强，位山最弱。站间 Kcact 基线差 0.05–0.10
- **大样本 7 指标代理版**：R²=0.748 vs 全特征 46 个 R²=0.769（差仅 0.021）

### 站间差异

同一指标在不同站点的相关性差异可达 0.2–0.3。albedo_sw 在栾城正相关（+0.16）但在位山负相关（−0.54）。LOSO CV 跨站外推时模型的 R² 因此受限。加经纬度或站点哑变量可部分缓解。

## GEE 导出状态

| 产品 | 状态 | 备注 |
|------|------|------|
| fPAR (MOD15A2H) | 7/7 已完成 | 已下载并合并 |
| LST (MOD11A2) | 7/7 已完成 | ΔLST 已计算 |
| SM (ERA5-Land) | 7/7 已完成 | 29M行，待按窗口聚合 |
| MOD09A1 ndvi+b07 | 32/32 已完成 | 含按省导出，已去重合并 |
| SRTM DEM | 1/1 已完成 | 高程已合并 |
| Albedo (MCD43A3) | 0/7 | 三次尝试全失败（OOM） |
| S2 raw / S1 SAR | 0/14 | 三次尝试全失败 |

文件位置：`data/raw/gee/kcact_maize_modis_indicators/`（54个CSV）

合并后 parquet：`data/processed/train/ncp_summer_maize_kcact_with_modis.parquet`

## 关键发现 (2026-05)

### Kc 间接 vs ETc 直接
大样本（37万）上直接预测 ETc 与间接法（Kc×ET0）效果等价（R² 差异 <0.01）。小样本上间接法更稳定。详见 §Kc 间接法 vs ETc 直接法。

### 天气变量的贡献
大样本上天气四变量（VPD + 降水 + 太阳辐射 + 气温）单独贡献 R² 约 0.07。无天气时遥感和 DOY 最优 R²=0.684。DOY 同时编码了物候和季节平均气候，是小样本的关键特征。

### SHAP 特征重要性
- **站点**：fPAR、ΔLST、Albedo 是最强三特征，DOY 在不含天气组合中频繁出现
- **大样本**：fPAR + NDVI + SWIR2(b07) + LSWI + DOY 是最小充分集
- **DOY 主导**：全季模型 80% 预测力来自 DOY，分阶段后 RS 指标 R² 从 0.66 断崖降至 0.10-0.51

### 生长阶段分拆
夏玉米分 5 阶段训练（初期→快速→抽雄→灌浆→成熟），快速生长期 R²=0.51 最好，抽雄期 0.10 最差。全季 R²=0.66 高是因为 DOY 编码了物候——排除 DOY 后纯遥感预测能力暴露了 NDVI 饱和的物理天花板。详见 §分阶段分析。

### VI 排列组合
NDVI+EVI+GNDVI+SAVI+RDVI 五个 VI 在 127 和 255 组合中对 R² 的增量 <0.002，高度共线。DOY+SM 是不可替代的核心，风速始终减分。

### GEE 数据
MOD09A1 全套 7 波段 (b01-b07) 已下载 (56 新文件 + 原有 = 110 文件)，RDVI 等任意 VI 可本地计算。

### 站间差异
同一指标在不同站点的相关性差异可达 0.2-0.3。LOSO CV 跨站外推时 R² 因此受限。

## ET0 计算说明 (FAO-56 完全审计通过)

- 公式: FAO-56 Penman-Monteith Eq.6, 所有参数标注公式编号（参见 `src/kcact/features/et0.py`）
- Tmean = (Tmax+Tmin)/2 (Eq.9), T+273 (Eq.6 分母), T+273.16 (Eq.39 Stefan-Boltzmann)
- ea 从露点温度直接计算 (Eq.14, 最准确方法)
- 站点观测 UTC+8 → ERA5 UTC 已做8小时时区对齐
- 站点高程已应用 (禹城20.6m, 位山34m, 馆陶41.9m, 栾城53.2m)
- 修正后图表: `outputs/figures/corrected/maize_fig*.png`
- 修正后数据: `outputs/tables/station_etc_et0_kcact.xlsx`

## 约束

- 单作物建模，不混作物
- 验证用 Leave-One-Year-Out（LOYO）或 Leave-One-Station-Out（LOSO），禁止随机打散
- 站点实测数据不外传（`data/raw/*` 已 gitignore）
- ETc 与 ET0 来源和方法统一
- 不直接在 README 里粘贴密码、Token 或 API Key
