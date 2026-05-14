Page({
  data: {
    gauges: [
      { label: 'Kcact', value: '0.71', unit: '偏高', tone: 'warn' },
      { label: 'ET0', value: '5.2', unit: 'mm/d', tone: 'green' },
      { label: 'ETc', value: '3.7', unit: 'mm/d', tone: 'soil' }
    ],
    bars: [
      { day: '6/10', kc: 36, et: 42 },
      { day: '6/18', kc: 44, et: 48 },
      { day: '6/26', kc: 52, et: 54 },
      { day: '7/04', kc: 66, et: 61 },
      { day: '7/12', kc: 76, et: 68 },
      { day: '7/20', kc: 84, et: 72 },
      { day: '7/28', kc: 79, et: 66 },
      { day: '8/05', kc: 68, et: 58 }
    ],
    indicators: [
      { name: 'NDVI', value: 76, note: '植被覆盖' },
      { name: 'SAVI', value: 68, note: '土壤校正' },
      { name: 'RDVI', value: 62, note: '冠层结构' },
      { name: 'GNDVI', value: 72, note: '叶绿素' },
      { name: 'EVI', value: 70, note: '高 biomass' },
      { name: 'SM', value: 54, note: '土壤水分' },
      { name: 'doy', value: 82, note: '物候阶段' }
    ]
  }
})
