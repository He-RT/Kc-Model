Page({
  data: {
    contextCards: [
      { label: '地块', value: 'NCP-MZ-PML-225' },
      { label: '阶段', value: 'DOY 225 · 抽雄吐丝' },
      { label: 'Kcact', value: '0.91' },
      { label: '风险', value: '偏高' }
    ],
    messages: [
      { role: 'user', text: '这块夏玉米地现在需不需要重点灌溉？' },
      { role: 'agent', text: '当前 Kcact 为 0.91，处在多年生长季峰值窗口，说明作物耗水需求较高。建议把它列为重点巡查地块。' },
      { role: 'user', text: '为什么判断风险偏高？' },
      { role: 'agent', text: '模型看到三个信号：DOY 225 已进入抽雄吐丝期；NDVI/EVI 较高，冠层蒸腾强；PML ETa 约 4.2 mm/d，接近 ERA5 ET0 4.6 mm/d。' },
      { role: 'agent', text: '我会结合地块边界、最新遥感与气象指标、历史生长曲线和模型输出，持续生成灌溉建议与风险解释。' }
    ],
    suggestions: ['生成地块报告', '解释模型指标', '查看近8日耗水', '给出灌溉建议']
  }
})
