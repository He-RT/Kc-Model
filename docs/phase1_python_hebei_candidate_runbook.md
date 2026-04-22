# Phase 1 Python Runbook: Hebei Winter Wheat Candidate Patches

## 1. 用途

当训练样本资产还没准备好时，先直接提取河北省冬小麦候选区域和候选斑块。

脚本位置：

- [export_hebei_winter_wheat_candidates.py](/Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_candidates.py:1)

## 2. 方法

这版不依赖训练样本，使用：

- `Sentinel-2 SR Harmonized`
- `Sentinel-2 cloud probability`
- `ESA WorldCover` 耕地先验
- 河北冬小麦物候阈值规则

输出：

- 候选掩膜 GeoTIFF
- 候选斑块 GeoJSON
- 总面积 CSV

## 3. 运行

只看预览，不导出：

```bash
conda run -n sdxx python /Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_candidates.py \
  --project-id chuang-yaogan \
  --year 2025 \
  --no-export
```

正式导出到 Drive：

```bash
conda run -n sdxx python /Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_candidates.py \
  --project-id chuang-yaogan \
  --year 2025
```

## 4. 输出文件

默认 Drive 目录：

```text
kcact_hebei_candidates
```

默认文件名：

- `hebei_winter_wheat_candidate_mask_2025.tif`
- `hebei_winter_wheat_candidate_patches_2025.geojson`
- `hebei_winter_wheat_candidate_area_2025.csv`

## 5. 注意

这版输出的是“候选冬小麦斑块”，不是严格意义上的真实承包地块边界。

它适合：

- 先把河北冬小麦大体范围提出来
- 后续人工检查
- 后续升级到样本驱动分类
