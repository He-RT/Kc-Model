# 外部数据目录

这个目录存放不直接纳入版本控制的大体量输入数据。

MVP 至少需要补齐：

- `field_boundaries/hebei_winter_wheat_fields.geojson`
- 可选的土壤、DEM、灌溉区等静态数据

如果你后续补充更细分的目录，可以按下面扩展：

```text
data/external/
├── field_boundaries/
├── soil/
├── dem/
└── irrigation/
```
