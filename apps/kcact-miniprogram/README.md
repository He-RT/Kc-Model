# Kcact Vision 微信小程序

中期说明书和现场扫码演示用的原生微信小程序。

## 页面

- `总览`：华北平原 Kcact 风险热力图、核心指标、生育期进度
- `地块`：地块编号、Kcact/ERA5 ET0/PML ETa、趋势柱图、选用指标
- `模型`：NDVI、SAVI、RDVI、GNDVI、EVI、SM、doy 7 指标组合和贡献排序
- `报告`：地块诊断报告样例
- `助手`：Agent 对话诊断，融合地块、遥感、气象与模型结果

## 打开方式

1. 安装微信开发者工具。
2. 导入项目目录：

```text
/Users/hert/Projects/dcsdxx/apps/kcact-miniprogram
```

3. 如果没有正式小程序 AppID，使用测试号或游客模式预览。
4. 如果有 AppID，可在微信公众平台添加体验成员，然后上传体验版给组员/老师扫码。

## 说明

当前版本数值已替换为 PML-V2.2a ETa / ERA5 ET0 新标签模型的展示结果。后续可把页面 JS 中的数组替换为模型产出的实时地块结果，并把地块上下文传给 Agent 生成解释。
