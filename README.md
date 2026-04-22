# 华北平原单作物 Kcact 数据集生产与建模

本仓库用于构建“区域尺度、多年份、单作物”的 `Kcact` 数据集与建模流程。第一阶段先聚焦河北省冬小麦 `2021-2023` 的最小可行版本（MVP），目标是把数据生产线、质量控制和基线模型先跑通。

核心标签定义：

```text
Kcact = ETc / ET0
```

当前首版实现路线采用：

- `ETc`：统一使用 `MODIS/061/MOD16A2GF` 8 天蒸散发产品
- `ET0`：统一使用 FAO Penman-Monteith 方法计算
- 样本单元：`field-date`，其中 `date` 先采用与 MOD16A2GF 对齐的 8 天窗口结束日期
- 研究对象：河北省冬小麦

## 当前目录结构

```text
.
├── configs/                  # YAML 配置：区域、年份、数据源、路径、QC 阈值
├── data/
│   ├── external/             # 需要人工准备的外部输入，如地块边界、土壤数据
│   ├── raw/                  # 直接来自 GEE / 原始下载的数据
│   ├── interim/              # 中间结果，如 ET0 日尺度表、特征聚合表
│   ├── processed/            # 最终可训练样本表
│   └── metadata/             # 字段字典、版本说明、样本统计
├── docs/                     # 项目说明、数据流、字段规范
├── gee/                      # GEE JavaScript 脚本
├── logs/                     # 运行日志
├── notebooks/                # 仅用于分析和可视化，不承载正式流水线
├── outputs/
│   ├── figures/              # 图件
│   ├── models/               # 训练好的模型与特征重要性
│   ├── reports/              # 评估报告
│   └── tables/               # 评估表、分组统计表
├── scripts/
│   └── python/               # 可直接执行的 Python 入口脚本
├── src/
│   └── kcact/                # Python 业务模块
│       ├── config/
│       ├── data/
│       ├── features/
│       ├── modeling/
│       └── utils/
└── tests/                    # 单元测试与数据契约测试
```

## MVP 数据流

完整设计见 [docs/mvp-dataflow.md](/Users/hert/Projects/dcsdxx/docs/mvp-dataflow.md)。

MVP 的稳妥路径是：

1. 准备河北省冬小麦地块边界或等价采样单元
2. 在 GEE 中导出 Sentinel-2 地块统计特征
3. 在 GEE 中导出 ERA5-Land 日尺度气象变量
4. 在 GEE 中导出 MOD16A2GF 地块 ETc
5. 在本地计算 ET0
6. 合并成统一的 `field-date` 训练表
7. 做 `Kcact` 标签和质量控制
8. 训练跨年份验证的 XGBoost baseline

## 当前已固定的关键约束

- 先做单作物，不混作物
- 先做表格特征模型，不直接上端到端深度影像模型
- 训练验证优先采用跨年份切分，禁止随机打散泄漏
- `ETc` 与 `ET0` 的来源和计算方法必须统一

## 需要你先准备的外部数据

MVP 不默认这些数据已存在。以下输入需要你后续补齐到 `data/external/`：

- 河北省冬小麦地块边界：
  `data/external/field_boundaries/hebei_winter_wheat_fields.geojson`
- 如果要加静态特征，还需要：
  - 土壤类型图
  - DEM / 坡度
  - 灌溉区掩膜（可选）

如果当前没有真实地块边界，也能先用网格采样单元做技术验证，但那是备选，不是默认路线。
