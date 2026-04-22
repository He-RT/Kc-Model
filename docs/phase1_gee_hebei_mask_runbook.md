# Phase 1 GEE Runbook: Hebei Winter Wheat Mask (2021-2025)

## 1. 目标

运行一版不依赖人工训练样本的首版冬小麦掩膜脚本，先在河北省上把第一部分跑通。

脚本位置：

- [export_hebei_winter_wheat_mask_2021_2025.js](/Users/hert/Projects/dcsdxx/gee/export_hebei_winter_wheat_mask_2021_2025.js:1)

## 2. 当前实现方式

这是一个“物候规则版候选掩膜”，不是最终监督分类版。

逻辑是：

1. 用 `GAUL` 取河北省行政边界
2. 用 `ESA WorldCover 2021` 的 `Cropland` 作为耕地先验约束
3. 用 `Sentinel-2 SR Harmonized + S2 cloud probability` 做去云
4. 计算冬小麦典型时序特征：
   - 秋播后已出苗
   - 冬季仍保持一定植被覆盖
   - 春季快速返青拔节
   - 4-5 月出现峰值
   - 6 月收获后明显回落
5. 输出每年的候选冬小麦掩膜

## 3. GEE 数据源

已核实的 Earth Engine 数据集：

- `COPERNICUS/S2_SR_HARMONIZED`
- `COPERNICUS/S2_CLOUD_PROBABILITY`
- `ESA/WorldCover/v200`
- `FAO/GAUL/2015/level1`

## 4. 运行前需要确认

### 4.1 Asset 目录

脚本默认导出到：

```text
projects/chuang-yaogan/assets/kcact/phase1
```

如果这个目录不存在，你需要先在 GEE Assets 面板创建对应目录。

### 4.2 导出开关

脚本顶部 `CONFIG` 里有两个开关：

- `exportToAsset`
- `exportToDrive`

如果你只想先测试显示结果，可以先把这两个改成：

```javascript
exportToAsset: false,
exportToDrive: false
```

## 5. 运行步骤

### 方式一：GEE Code Editor

1. 打开 [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. 新建脚本
3. 粘贴 [export_hebei_winter_wheat_mask_2021_2025.js](/Users/hert/Projects/dcsdxx/gee/export_hebei_winter_wheat_mask_2021_2025.js:1) 内容
4. 确认使用的项目是 `chuang-yaogan`
5. 点击 `Run`
6. 在右侧 `Tasks` 面板启动导出任务

### 方式二：先仅做可视化检查

把：

```javascript
exportToAsset: true,
exportToDrive: true
```

改成：

```javascript
exportToAsset: false,
exportToDrive: false
```

这样先只看地图图层：

- `Cropland mask (WorldCover 2021)`
- `Peak NDVI <year>`
- `Winter wheat candidate mask <year>`

## 6. 输出内容

### 6.1 影像输出

每年 1 个二值栅格：

- `winter_wheat_mask_hebei_2021`
- `winter_wheat_mask_hebei_2022`
- `winter_wheat_mask_hebei_2023`
- `winter_wheat_mask_hebei_2024`
- `winter_wheat_mask_hebei_2025`

像元值含义：

- `1`：候选冬小麦
- `masked / no data`：非冬小麦

### 6.2 统计表输出

1 个按年面积统计表：

- `hebei_winter_wheat_area_summary_2021_2025.csv`

字段：

- `season_year`
- `province`
- `crop_type`
- `area_ha`

## 7. 这版脚本的已知限制

这版先求稳，不追求一步到位：

- 用的是 `2021` 年耕地先验掩膜，不是逐年耕地变化图
- 用的是规则筛选，不是训练样本驱动的监督分类
- 安徽北部、江苏北部这些复杂子区还没纳入
- 阈值是“河北先验阈值”，后续需要用样本校准

## 8. 下一步怎么接

第一部分建议按这个顺序推进：

1. 先跑 `Hebei 2025` 单年预览
2. 看结果是否明显漏检或误检
3. 调阈值
4. 再导出 `2021-2025`
5. 结果可用后，再升级成监督分类版

## 9. 下一步最值得做的升级

当你准备好训练样本后，我建议下一步直接新增一版：

- `scripts/python/export_hebei_winter_wheat_mask_rf.py`

其输入将变成：

- 当前脚本里的多时相特征
- 你提供的冬小麦 / 非冬小麦训练样本

然后用 `Random Forest` 取代规则阈值版。
