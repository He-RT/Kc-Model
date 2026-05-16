Page({
  data: {
    metrics: [
      { label: '监测格网', value: '3,642', unit: '0.1° cells', mark: 'Grid' },
      { label: '训练样本', value: '144,877', unit: '2019-2024 · 8日', mark: 'N' },
      { label: '平均 Kcact', value: '0.67', unit: 'PML/ERA5', mark: 'Kc' },
      { label: '模型 R²', value: '0.765', unit: 'LOYO pooled', mark: 'R2' }
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
      { name: '苗期', progress: 100, active: false },
      { name: '拔节', progress: 100, active: false },
      { name: '抽雄吐丝', progress: 100, active: true },
      { name: '灌浆', progress: 0, active: false },
      { name: '成熟', progress: 0, active: false }
    ]
  }
})
