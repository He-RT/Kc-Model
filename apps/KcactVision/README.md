# Kcact Vision iOS Prototype

中期说明书用的静态 SwiftUI 产品原型，定位为“华北平原作物耗水与水分风险监测平台”。

## 页面

- 总览：华北平原热力图、监测地块数、Kcact、ET0/ETc 风险提示、生育期进度
- 地块：单地块 Kcact/ET0/ETc 指标、趋势曲线、遥感与土壤指标
- 模型：NDVI/SAVI/RDVI/GNDVI/EVI/SM/doy 7 指标版表现和特征贡献排序
- 报告：地块诊断报告样例，可直接截图放进中期说明书

## 运行

当前仓库新增独立 Xcode 工程：

```bash
open apps/KcactVision/KcactVision.xcodeproj
```

本机需要完整 Xcode，并安装 iOS 平台和模拟器运行时。当前机器已有 `/Applications/Xcode.app`，但系统 `xcode-select` 指向 Command Line Tools；如需命令行构建，可先切换：

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

或者临时使用：

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild -project apps/KcactVision/KcactVision.xcodeproj \
  -scheme KcactVision \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  build
```

## 备注

界面数值用于中期产品形态展示，混合了项目现有实验结论和静态样例值；正式版本再接入 parquet/CSV 处理结果和真实地块数据。
