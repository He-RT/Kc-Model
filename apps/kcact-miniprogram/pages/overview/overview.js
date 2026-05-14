Page({
  data: {
    metrics: [
      { label: '监测地块', value: '7,153', unit: 'patches', mark: 'Grid' },
      { label: '当前作物', value: '夏玉米', unit: '2019-2025', mark: 'Crop' },
      { label: '平均 Kcact', value: '0.62', unit: '8日窗口', mark: 'Kc' },
      { label: '模型 R²', value: '0.661', unit: 'VI+SM+doy', mark: 'R2' }
    ],
    heatCells: [
      'low', 'low', 'low', 'low', 'mid', 'mid', 'low',
      'low', 'low', 'mid', 'mid', 'high', 'mid', 'low',
      'low', 'low', 'mid', 'high', 'high', 'mid', 'low',
      'low', 'low', 'mid', 'mid', 'mid', 'low', 'low',
      'low', 'low', 'low', 'mid', 'low', 'low', 'low'
    ],
    spots: [
      { name: '河北', x: 38, y: 14, size: 'spot-sm spot-mid' },
      { name: '山东', x: 47, y: 35, size: 'spot-lg spot-high' },
      { name: '河南', x: 30, y: 50, size: 'spot-md spot-mid' },
      { name: '安徽', x: 55, y: 58, size: 'spot-sm spot-low' }
    ],
    stages: [
      { name: '出苗', progress: 100, active: false },
      { name: '拔节', progress: 100, active: false },
      { name: '抽雄', progress: 85, active: true },
      { name: '灌浆', progress: 30, active: false },
      { name: '成熟', progress: 0, active: false }
    ]
  }
})
