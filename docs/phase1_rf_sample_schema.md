# Phase 1 RF Sample Schema

## 1. 用途

这个文档定义 [export_hebei_winter_wheat_mask_rf.py](/Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py:1) 所需的训练样本资产格式。

如果样本资产字段不符合这里的定义，脚本无法直接运行。

## 2. 推荐资产路径

```text
projects/chuang-yaogan/assets/kcact/samples/hebei_winter_wheat_samples_v1
```

## 3. 几何类型

推荐两种之一：

- `Point`
- `Polygon`

更推荐 `Point`，因为：

- 更轻
- 更容易人工检查
- 在 `sampleRegions` 中更稳定

如果使用 `Polygon`，请确保：

- 面积不要过大
- 尽量是纯净样区
- 不要跨多种地类

## 4. 必需字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `class_id` | int | 类别标签，`1=冬小麦`，`0=非冬小麦` |
| `sample_id` | string | 样本唯一标识 |

## 5. 建议字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `class_name` | string | `winter_wheat` 或 `non_winter_wheat` |
| `source` | string | 样本来源，如 `manual_digitize`、`field_survey` |
| `season_year` | int | 样本对应年份，可选 |
| `province` | string | 建议填 `Hebei` |
| `notes` | string | 备注 |

## 6. 类别定义

### 6.1 正类

`class_id = 1`

表示：

- 冬小麦稳定种植区
- 与脚本识别年份的物候相一致

### 6.2 负类

`class_id = 0`

建议覆盖这些典型混淆对象：

- 裸地
- 其他冬季绿植
- 林地
- 居民地
- 水体
- 设施农业
- 其他作物

## 7. 样本数量建议

河北首版建议最低配置：

- 冬小麦：`200-500`
- 非冬小麦：`200-500`

更稳妥的配置：

- 冬小麦：`500+`
- 非冬小麦：`500+`

不要只堆正类样本，负类覆盖面同样重要。

## 8. 样本分布建议

样本不要只放在一个县。

河北首版至少覆盖：

- 冀南
- 冀中
- 冀东平原耕地

尽量让样本同时覆盖：

- 高覆盖冬小麦
- 稀疏冬小麦
- 地块边缘
- 易混地类

## 9. 当前脚本如何使用这些样本

脚本会：

1. 读取样本资产
2. 从多时相 Sentinel-2 特征栈里抽样
3. 按 `trainFraction` 做随机训练/验证拆分
4. 用 `class_id` 训练 `Random Forest`
5. 输出分类结果和精度评估

## 10. 最小可用例子

最少字段只要这两个就能跑：

| class_id | sample_id |
|---|---|
| 1 | ww_0001 |
| 0 | nonww_0001 |

但正式生产时，不建议只保留最小字段。
