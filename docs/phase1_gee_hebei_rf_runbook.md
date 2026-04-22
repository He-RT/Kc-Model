# Phase 1 GEE Runbook: Hebei Winter Wheat Random Forest

这份文档保留为方法说明。当前正式可运行版本已经改为 Python：

- [export_hebei_winter_wheat_mask_rf.py](/Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py:1)
- [phase1_python_gee_hebei_rf_runbook.md](/Users/hert/Projects/dcsdxx/docs/phase1_python_gee_hebei_rf_runbook.md:1)

## 1. 目标

运行河北省冬小麦 `Random Forest` 监督分类版，并导出：

- 年度冬小麦掩膜
- 年度精度评估表
- 年度面积统计表

当前方法对应的正式脚本位置：

- [export_hebei_winter_wheat_mask_rf.py](/Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py:1)

## 2. 为什么这版是当前最合适的

相较于论文中的 `SVM` 和当前无样本规则版，这版更适合作为你的正式第一部分：

- 比规则阈值法更稳
- 比 `SVM` 更适合大范围批处理
- 更容易扩展到河南、山东、安徽北部、江苏北部
- 可以自然接入后续样本迭代

## 3. 输入

### 3.1 GEE 数据集

- `COPERNICUS/S2_SR_HARMONIZED`
- `COPERNICUS/S2_CLOUD_PROBABILITY`
- `ESA/WorldCover/v200`
- `FAO/GAUL/2015/level1`

### 3.2 训练样本资产

默认资产路径：

```text
projects/chuang-yaogan/assets/kcact/samples/hebei_winter_wheat_samples_v1
```

样本字段规范见：

- [phase1_rf_sample_schema.md](/Users/hert/Projects/dcsdxx/docs/phase1_rf_sample_schema.md:1)

## 4. 特征设计

当前脚本使用这些多时相特征：

- 秋季 `NDVI / LSWI / NDRE / EVI`
- 冬季 `NDVI / LSWI / NDRE / EVI`
- 春季 `NDVI / LSWI / NDRE / EVI`
- 峰值期 `NDVI / LSWI / NDRE / EVI`
- 收获期 `NDVI / LSWI / NDRE / EVI`
- `ndvi_rise`
- `ndvi_drop`
- `autumn_to_peak_rise`
- `seasonal_mean_ndvi`
- `cropland`

## 5. 运行前检查

### 5.1 样本资产是否存在

如果这个资产不存在，脚本不会成功：

```text
projects/chuang-yaogan/assets/kcact/samples/hebei_winter_wheat_samples_v1
```

### 5.2 导出目录是否存在

脚本默认导出到：

```text
projects/chuang-yaogan/assets/kcact/phase1_rf
```

如果目录不存在，需要先在 GEE Assets 面板创建。

### 5.3 仅做预览时的开关

如果你先只想看结果，不想启动导出，把：

```javascript
exportToAsset: true,
exportToDrive: true
```

改成：

```javascript
exportToAsset: false,
exportToDrive: false
```

## 6. 运行步骤

1. 打开 [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. 新建脚本
3. 运行 [export_hebei_winter_wheat_mask_rf.py](/Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py:1)
4. 确认当前项目是 `chuang-yaogan`
5. 先关闭导出，只看 `2025` 年预览
6. 检查训练/验证样本、分类图层和精度输出
7. 确认结果可接受后，再开启导出

## 7. 输出

### 7.1 栅格

每年输出一个冬小麦掩膜：

- `winter_wheat_mask_hebei_rf_2021`
- `winter_wheat_mask_hebei_rf_2022`
- `winter_wheat_mask_hebei_rf_2023`
- `winter_wheat_mask_hebei_rf_2024`
- `winter_wheat_mask_hebei_rf_2025`

### 7.2 表格

精度表：

- `hebei_winter_wheat_rf_metrics_2021_2025.csv`

面积表：

- `hebei_winter_wheat_rf_area_summary_2021_2025.csv`

## 8. 当前评估指标

脚本会输出：

- 训练混淆矩阵
- 训练精度
- 训练 `kappa`
- 验证混淆矩阵
- 验证精度
- 验证 `kappa`

## 9. 推荐使用顺序

最稳的推进顺序：

1. 先补训练样本资产
2. 跑 `2025` 年预览
3. 看误分区
4. 补负类样本
5. 再跑 `2021-2025`

## 10. 当前脚本与规则版的关系

规则版脚本：

- [export_hebei_winter_wheat_mask_2021_2025.js](/Users/hert/Projects/dcsdxx/gee/export_hebei_winter_wheat_mask_2021_2025.js:1)

定位：

- 规则版：无样本过渡版
- RF 版：正式推荐版
