# 华北平原作物 Kcact 数据集生产与建模

基于遥感与气象数据的作物系数（Kcact = ETc / ET0）预测，覆盖冬小麦和夏玉米。

**当前最优模型**：

| 作物 | 数据规模 | 样本数 | LOYO R² | 特征数 |
|------|---------|--------|---------|--------|
| 冬小麦 | 河北 592 patches, 2019–2025 | 18,528 | 0.702 | 46 |
| 夏玉米 | 华北四省 7,153 patches, 2019–2025 | 397,628 | 0.769 | 46 |
| 夏玉米（站点验证） | 4 通量站, 2003–2015 | 304 | 0.467 | 7 |

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

```bash
conda activate sdxx
# macOS 需要：
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```

核心依赖：`catboost scikit-learn pandas numpy pyarrow openpyxl earthengine-api`

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

### 核心结论

- **3 个特征接近 7 个的效果**：ndvi + b07(SWIR2) + doy → R²=0.461（vs 7 特征 0.467）
- **覆盖率 > 分辨率**：逐日 MOD09GA NDVI (500m) 优于 16 天 MOD13Q1 (250m)
- **单指标最强**：fPAR (R²=0.318) > NDVI (R²=0.222) > LST (R²=−0.016)
- **站间差异显著**：馆陶的指标相关性系统性最强，位山最弱。站间 Kcact 基线差 0.05–0.10
- **大样本 7 指标代理版**：R²=0.748 vs 全特征 46 个 R²=0.769（差仅 0.021）

### 站间差异

同一指标在不同站点的相关性差异可达 0.2–0.3。albedo_sw 在栾城正相关（+0.16）但在位山负相关（−0.54）。LOSO CV 跨站外推时模型的 R² 因此受限。加经纬度或站点哑变量可部分缓解。

## GEE 导出任务

50+ 任务已提交至 Drive 文件夹 `kcact_maize_modis_indicators/`，等待完成后合并。详见 `docs/handover.md` §7。

## 约束

- 单作物建模，不混作物
- 验证用 Leave-One-Year-Out（LOYO）或 Leave-One-Station-Out（LOSO），禁止随机打散
- 站点实测数据不外传（`data/raw/*` 已 gitignore）
- ETc 与 ET0 来源和方法统一
- 不直接在 README 里粘贴密码、Token 或 API Key
