import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            OverviewScreen()
                .tabItem { Label("总览", systemImage: "map") }

            ParcelScreen()
                .tabItem { Label("地块", systemImage: "leaf") }

            ModelScreen()
                .tabItem { Label("模型", systemImage: "chart.bar.xaxis") }

            ReportScreen()
                .tabItem { Label("报告", systemImage: "doc.text") }
        }
        .tint(.brandGreen)
    }
}

private struct OverviewScreen: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    HeroMapPanel()

                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                        MetricCard(title: "监测地块", value: "7,153", unit: "patches", icon: "square.grid.3x3")
                        MetricCard(title: "当前作物", value: "夏玉米", unit: "2019-2025", icon: "leaf.fill")
                        MetricCard(title: "平均 Kcact", value: "0.62", unit: "8日窗口", icon: "drop.fill")
                        MetricCard(title: "模型 R²", value: "0.661", unit: "VI+SM+doy", icon: "speedometer")
                    }

                    RiskStrip()
                    PhenologyPanel()
                }
                .padding(16)
                .padding(.bottom, 96)
            }
            .background(Color.appBackground)
            .navigationTitle("Kcact Vision")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

private struct HeroMapPanel: View {
    private let spots: [MapSpot] = [
        .init(x: 0.42, y: 0.22, size: 34, color: .riskMedium, label: "河北"),
        .init(x: 0.48, y: 0.44, size: 52, color: .riskHigh, label: "山东"),
        .init(x: 0.36, y: 0.56, size: 42, color: .riskMedium, label: "河南"),
        .init(x: 0.56, y: 0.64, size: 30, color: .riskLow, label: "安徽")
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("华北平原作物耗水监测")
                        .font(.title2.bold())
                    Text("遥感 + 气象 + 作物系数模型")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                StatusPill(text: "中期原型", color: .brandGreen)
            }

            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(
                        LinearGradient(
                            colors: [.mapBase, .mapBase.opacity(0.78)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

                HeatMapGrid()
                    .opacity(0.96)

                ForEach(spots) { spot in
                    GeometryReader { proxy in
                        VStack(spacing: 4) {
                            Circle()
                                .fill(spot.color.opacity(0.82))
                                .frame(width: spot.size, height: spot.size)
                                .overlay(Circle().stroke(.white.opacity(0.9), lineWidth: 1))
                                .shadow(color: spot.color.opacity(0.35), radius: 10)
                            Text(spot.label)
                                .font(.caption2.weight(.medium))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 3)
                                .background(.black.opacity(0.34), in: Capsule())
                        }
                        .position(x: proxy.size.width * spot.x, y: proxy.size.height * spot.y)
                    }
                }

                VStack {
                    Spacer()
                    HStack {
                        LegendItem(color: .riskLow, text: "低")
                        LegendItem(color: .riskMedium, text: "中")
                        LegendItem(color: .riskHigh, text: "高")
                        Spacer()
                        Text("Kcact / ETc 风险")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.white.opacity(0.86))
                    }
                    .padding(12)
                    .background(.black.opacity(0.22), in: RoundedRectangle(cornerRadius: 6))
                    .padding(12)
                }
            }
            .frame(height: 260)
        }
        .panelStyle()
    }
}

private struct HeatMapGrid: View {
    private let values: [Double] = [
        0.30, 0.36, 0.42, 0.50, 0.58, 0.54, 0.46,
        0.34, 0.43, 0.56, 0.66, 0.74, 0.62, 0.48,
        0.40, 0.52, 0.68, 0.86, 0.78, 0.63, 0.45,
        0.32, 0.49, 0.61, 0.72, 0.69, 0.52, 0.38,
        0.28, 0.37, 0.45, 0.55, 0.50, 0.42, 0.33
    ]

    var body: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 7), count: 7), spacing: 7) {
            ForEach(values.indices, id: \.self) { index in
                RoundedRectangle(cornerRadius: 6)
                    .fill(color(for: values[index]))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.white.opacity(0.08), lineWidth: 1))
                    .aspectRatio(1.0, contentMode: .fit)
            }
        }
        .padding(18)
    }

    private func color(for value: Double) -> Color {
        if value > 0.72 { return .riskHigh.opacity(0.82) }
        if value > 0.52 { return .riskMedium.opacity(0.76) }
        return .riskLow.opacity(0.72)
    }
}

private struct RiskStrip: View {
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.title3)
                .foregroundStyle(Color.riskHigh)

            VStack(alignment: .leading, spacing: 4) {
                Text("抽雄-灌浆窗口水分风险偏高")
                    .font(.headline)
                Text("山东西部与河南北部 8 日 ETc 较常年高 11.8%，建议作为重点核查区域。")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(16)
        .background(Color.warningBackground, in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct PhenologyPanel: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("生育期进度", systemImage: "calendar")
                    .font(.headline)
                Spacer()
                Text("DOY 226")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                StageCapsule(name: "出苗", progress: 1.0, active: false)
                StageCapsule(name: "拔节", progress: 1.0, active: false)
                StageCapsule(name: "抽雄", progress: 0.85, active: true)
                StageCapsule(name: "灌浆", progress: 0.30, active: false)
                StageCapsule(name: "成熟", progress: 0.0, active: false)
            }

            Text("本页为中期展示静态样例，数值采用项目现有结果和合理示例组合，用于说明产品形态。")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .panelStyle()
    }
}

private struct ParcelScreen: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    ParcelHeader()

                    HStack(spacing: 12) {
                        GaugeTile(title: "Kcact", value: "0.71", status: "偏高", color: .riskHigh)
                        GaugeTile(title: "ET0", value: "5.2", status: "mm/d", color: .sky)
                        GaugeTile(title: "ETc", value: "3.7", status: "mm/d", color: .brandGreen)
                    }

                    LineChartPanel()
                    IndicatorPanel()
                }
                .padding(16)
                .padding(.bottom, 96)
            }
            .background(Color.appBackground)
            .navigationTitle("地块详情")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

private struct ParcelHeader: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("NCP-MZ-2024-0831")
                        .font(.title3.bold())
                    Text("夏玉米 · 山东德州 · 抽雄期")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                StatusPill(text: "重点关注", color: .riskHigh)
            }

            HStack(spacing: 10) {
                InfoChip(icon: "location.fill", text: "36.83°N, 116.57°E")
                InfoChip(icon: "square.dashed", text: "500m 网格")
            }
        }
        .panelStyle()
    }
}

private struct LineChartPanel: View {
    private let kcact = [0.31, 0.36, 0.42, 0.55, 0.63, 0.71, 0.69, 0.62, 0.50]
    private let et0 = [0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.64, 0.56, 0.48]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("近 72 天耗水趋势")
                    .font(.headline)
                Spacer()
                LegendItem(color: .brandGreen, text: "Kcact")
                LegendItem(color: .sky, text: "ET0")
            }

            ZStack {
                LineChart(values: et0, color: .sky)
                LineChart(values: kcact, color: .brandGreen)
            }
            .frame(height: 180)
            .padding(.vertical, 8)

            HStack {
                Text("6/10")
                Spacer()
                Text("7/18")
                Spacer()
                Text("8/20")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .panelStyle()
    }
}

private struct IndicatorPanel: View {
    private let indicators = [
        Indicator(name: "NDVI", value: 0.76, note: "植被覆盖"),
        Indicator(name: "SAVI", value: 0.68, note: "土壤校正"),
        Indicator(name: "RDVI", value: 0.62, note: "冠层结构"),
        Indicator(name: "GNDVI", value: 0.72, note: "叶绿素"),
        Indicator(name: "EVI", value: 0.70, note: "高 biomass"),
        Indicator(name: "SM", value: 0.54, note: "土壤水分"),
        Indicator(name: "doy", value: 0.82, note: "物候阶段")
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("关键遥感与土壤指标")
                .font(.headline)

            ForEach(indicators) { item in
                VStack(alignment: .leading, spacing: 7) {
                    HStack {
                        Text(item.name)
                            .font(.subheadline.weight(.semibold))
                        Spacer()
                        Text(item.note)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    ProgressView(value: item.value)
                        .tint(item.value > 0.66 ? .riskHigh : .brandGreen)
                }
            }
        }
        .panelStyle()
    }
}

private struct ModelScreen: View {
    private let features = [
        FeatureScore(name: "doy", score: 0.92, color: .ink),
        FeatureScore(name: "NDVI", score: 0.78, color: .leaf),
        FeatureScore(name: "SM", score: 0.72, color: .soil),
        FeatureScore(name: "EVI", score: 0.66, color: .brandGreen),
        FeatureScore(name: "GNDVI", score: 0.61, color: .sky),
        FeatureScore(name: "SAVI", score: 0.54, color: .riskMedium),
        FeatureScore(name: "RDVI", score: 0.48, color: .riskHigh)
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("模型解释")
                            .font(.title2.bold())
                        Text("CatBoost VI+SM+doy 7 指标版，突出植被指数、土壤水分与物候阶段。")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .panelStyle()

                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Text("特征贡献排序")
                                .font(.headline)
                            Spacer()
                            Text("R² 0.661")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }

                        ForEach(features) { feature in
                            HStack(spacing: 12) {
                                Text(feature.name)
                                    .font(.subheadline.weight(.semibold))
                                    .frame(width: 54, alignment: .leading)
                                GeometryReader { proxy in
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(feature.color.opacity(0.78))
                                        .frame(width: proxy.size.width * feature.score)
                                }
                                .frame(height: 18)
                                Text("\(Int(feature.score * 100))")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                                    .frame(width: 28, alignment: .trailing)
                            }
                        }
                    }
                    .panelStyle()

                    ModelNotePanel()
                }
                .padding(16)
                .padding(.bottom, 96)
            }
            .background(Color.appBackground)
            .navigationTitle("模型")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

private struct ModelNotePanel: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("中期说明建议", systemImage: "lightbulb")
                .font(.headline)

            Text("当前原型强调地块级 Kcact 与 ETc 的可视化诊断。站点尺度与 MOD16 产品存在差异，后续将通过站点气象数据、MOD16 偏差校正和根区土壤水分约束继续完善。")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .panelStyle()
    }
}

private struct ReportScreen: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("地块诊断报告")
                                    .font(.title2.bold())
                                Text("2026-05-13 · 样例导出页")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Image(systemName: "doc.text.magnifyingglass")
                                .font(.title2)
                                .foregroundStyle(Color.brandGreen)
                        }

                        Divider()

                        ReportRow(title: "作物", value: "夏玉米")
                        ReportRow(title: "生育期", value: "抽雄-灌浆过渡")
                        ReportRow(title: "耗水状态", value: "ETc 较常年偏高")
                        ReportRow(title: "水分风险", value: "中高风险")
                        ReportRow(title: "主要因子", value: "NDVI、SAVI、RDVI、GNDVI、EVI、SM、doy")
                    }
                    .panelStyle()

                    VStack(alignment: .leading, spacing: 12) {
                        Text("自动生成结论")
                            .font(.headline)
                        Text("该地块近期冠层活跃度较高，地表昼夜温差扩大，根区土壤水分处于偏低区间。建议在实地墒情核查基础上，将抽雄至灌浆阶段作为灌溉调度重点。")
                            .font(.body)
                            .foregroundStyle(.primary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .panelStyle()

                    VStack(alignment: .leading, spacing: 12) {
                        Text("说明书配图用途")
                            .font(.headline)
                        Text("这页适合放在中期说明书的“产品输出形式”小节，表达系统最终可以从模型结果生成面向管理者的地块报告。")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .panelStyle()
                }
                .padding(16)
                .padding(.bottom, 96)
            }
            .background(Color.appBackground)
            .navigationTitle("报告")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

private struct MetricCard: View {
    let title: String
    let value: String
    let unit: String
    let icon: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundStyle(Color.brandGreen)
                Spacer()
            }
            Text(value)
                .font(.title2.bold())
                .monospacedDigit()
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption.weight(.semibold))
                Text(unit)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color.panelBackground, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.black.opacity(0.05), lineWidth: 1))
    }
}

private struct GaugeTile: View {
    let title: String
    let value: String
    let status: String
    let color: Color

    var body: some View {
        VStack(spacing: 7) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title3.bold())
                .monospacedDigit()
            Text(status)
                .font(.caption2.weight(.medium))
                .foregroundStyle(color)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(Color.panelBackground, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(color.opacity(0.22), lineWidth: 1))
    }
}

private struct StageCapsule: View {
    let name: String
    let progress: Double
    let active: Bool

    var body: some View {
        VStack(spacing: 7) {
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.black.opacity(0.08))
                    Capsule()
                        .fill(active ? Color.riskHigh : Color.brandGreen)
                        .frame(width: proxy.size.width * progress)
                }
            }
            .frame(height: 8)

            Text(name)
                .font(.caption2.weight(active ? .bold : .regular))
                .foregroundStyle(active ? .primary : .secondary)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity)
    }
}

private struct LineChart: View {
    let values: [Double]
    let color: Color

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                ChartGrid()
                Path { path in
                    let points = chartPoints(in: proxy.size)
                    guard let first = points.first else { return }
                    path.move(to: first)
                    for point in points.dropFirst() {
                        path.addLine(to: point)
                    }
                }
                .stroke(color, style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))

                ForEach(chartPoints(in: proxy.size).indices, id: \.self) { index in
                    Circle()
                        .fill(color)
                        .frame(width: 7, height: 7)
                        .position(chartPoints(in: proxy.size)[index])
                }
            }
        }
    }

    private func chartPoints(in size: CGSize) -> [CGPoint] {
        guard values.count > 1 else { return [] }
        let minValue = values.min() ?? 0
        let maxValue = values.max() ?? 1
        let span = max(maxValue - minValue, 0.01)
        return values.enumerated().map { index, value in
            let x = size.width * CGFloat(index) / CGFloat(values.count - 1)
            let yRatio = CGFloat((value - minValue) / span)
            let y = size.height - (size.height * yRatio)
            return CGPoint(x: x, y: y)
        }
    }
}

private struct ChartGrid: View {
    var body: some View {
        VStack {
            ForEach(0..<4, id: \.self) { _ in
                Divider()
                Spacer()
            }
            Divider()
        }
        .foregroundStyle(Color.black.opacity(0.08))
    }
}

private struct StatusPill: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(color.opacity(0.12), in: Capsule())
    }
}

private struct InfoChip: View {
    let icon: String
    let text: String

    var body: some View {
        Label(text, systemImage: icon)
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(Color.black.opacity(0.05), in: Capsule())
    }
}

private struct LegendItem: View {
    let color: Color
    let text: String

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(text)
                .font(.caption2.weight(.medium))
        }
        .foregroundStyle(.secondary)
    }
}

private struct ReportRow: View {
    let title: String
    let value: String

    var body: some View {
        HStack(alignment: .top) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(width: 82, alignment: .leading)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct MapSpot: Identifiable {
    let id = UUID()
    let x: CGFloat
    let y: CGFloat
    let size: CGFloat
    let color: Color
    let label: String
}

private struct Indicator: Identifiable {
    let id = UUID()
    let name: String
    let value: Double
    let note: String
}

private struct FeatureScore: Identifiable {
    let id = UUID()
    let name: String
    let score: Double
    let color: Color
}

private extension View {
    func panelStyle() -> some View {
        self
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.panelBackground, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.black.opacity(0.05), lineWidth: 1))
    }
}

private extension Color {
    static let appBackground = Color(hex: 0xF4F6F1)
    static let panelBackground = Color(hex: 0xFFFFFF)
    static let brandGreen = Color(hex: 0x1F7A4D)
    static let leaf = Color(hex: 0x65A30D)
    static let sky = Color(hex: 0x0E7490)
    static let soil = Color(hex: 0x8B5E34)
    static let ink = Color(hex: 0x25323A)
    static let mapBase = Color(hex: 0x203B3A)
    static let riskLow = Color(hex: 0x2F9E44)
    static let riskMedium = Color(hex: 0xD99A21)
    static let riskHigh = Color(hex: 0xC2410C)
    static let warningBackground = Color(hex: 0xFFF4E6)

    init(hex: UInt, alpha: Double = 1.0) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xff) / 255,
            green: Double((hex >> 8) & 0xff) / 255,
            blue: Double(hex & 0xff) / 255,
            opacity: alpha
        )
    }
}

#Preview {
    ContentView()
}
