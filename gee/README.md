# GEE 脚本目录

后续这里会放：

- `export_hebei_winter_wheat_mask_2021_2025.js`
- `export_s2_field_features.js`
- `export_era5_daily.js`
- `export_mod16_etc.js`

这些脚本统一从 `configs/mvp_hebei_winter_wheat_2021_2023.yaml` 的约定读取输入契约，但 GEE 运行时仍需手动填写资产路径和导出目录。

当前已落地的脚本：

- [export_hebei_winter_wheat_mask_2021_2025.js](/Users/hert/Projects/dcsdxx/gee/export_hebei_winter_wheat_mask_2021_2025.js:1)

对应运行说明：

- [phase1_gee_hebei_mask_runbook.md](/Users/hert/Projects/dcsdxx/docs/phase1_gee_hebei_mask_runbook.md:1)
- [phase1_gee_hebei_rf_runbook.md](/Users/hert/Projects/dcsdxx/docs/phase1_gee_hebei_rf_runbook.md:1)

当前正式推荐版已经改为 Python：

- [export_hebei_winter_wheat_mask_rf.py](/Users/hert/Projects/dcsdxx/scripts/python/export_hebei_winter_wheat_mask_rf.py:1)
- [phase1_python_gee_hebei_rf_runbook.md](/Users/hert/Projects/dcsdxx/docs/phase1_python_gee_hebei_rf_runbook.md:1)
