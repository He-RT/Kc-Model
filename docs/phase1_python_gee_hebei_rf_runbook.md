# Phase 1 Python GEE Runbook: Hebei Winter Wheat Random Forest

## 1. 目标

使用 Python + Earth Engine API 运行河北省冬小麦 `Random Forest` 监督分类版，并导出：

- 年度冬小麦掩膜
- 年度精度评估表
- 年度面积统计表

脚本位置：

- [export_hebei_winter_wheat_mask_rf.py](/Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py:1)

## 2. 依赖

最少需要：

```bash
python3 -m pip install earthengine-api
```

首次使用 Earth Engine Python API 时，需要完成认证：

```bash
earthengine authenticate
```

## 3. 训练样本资产

默认资产路径：

```text
projects/chuang-yaogan/assets/kcact/samples/hebei_winter_wheat_samples_v1
```

字段规范见：

- [phase1_rf_sample_schema.md](/Users/hert/Projects/dcsdxx/docs/phase1_rf_sample_schema.md:1)

## 4. 快速预览

先不导出，只看 `2025` 年结果和精度：

```bash
python3 /Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py \
  --project-id chuang-yaogan \
  --preview-year 2025 \
  --no-export
```

## 5. 正式运行

运行 `2021-2025` 并启动导出任务：

```bash
python3 /Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py \
  --project-id chuang-yaogan
```

## 6. 可选参数

指定年份：

```bash
python3 /Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py \
  --project-id chuang-yaogan \
  --years 2023 2024 2025
```

指定样本资产：

```bash
python3 /Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py \
  --project-id chuang-yaogan \
  --sample-asset projects/chuang-yaogan/assets/kcact/samples/hebei_winter_wheat_samples_v2
```

指定树数量：

```bash
python3 /Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py \
  --project-id chuang-yaogan \
  --tree-count 300
```

## 7. 输出

默认资产目录：

```text
projects/chuang-yaogan/assets/kcact/phase1_rf
```

默认 Drive 目录：

```text
kcact_hebei_phase1_rf
```

## 8. 结果解释

脚本会在控制台打印：

- 样本数量
- 训练混淆矩阵
- 验证混淆矩阵
- 训练精度
- 验证精度
- 训练 `kappa`
- 验证 `kappa`
- RF 特征解释信息

## 9. 推荐执行顺序

1. 先补样本资产
2. 用 `--no-export` 跑 `2025`
3. 检查验证精度和误分区
4. 补负类样本
5. 再跑 `2021-2025`
