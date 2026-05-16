Page({
  data: {
    gauges: [
      { label: 'Kcact', value: '0.91', unit: '峰值窗口', tone: 'warn' },
      { label: 'ERA5 ET0', value: '4.6', unit: 'mm/d', tone: 'green' },
      { label: 'PML ETa', value: '4.2', unit: 'mm/d', tone: 'soil' }
    ],
    bars: [
      { day: '6/09', kc: 49, et: 59 },
      { day: '6/25', kc: 53, et: 64 },
      { day: '7/11', kc: 76, et: 77 },
      { day: '7/27', kc: 86, et: 82 },
      { day: '8/12', kc: 91, et: 90 },
      { day: '8/28', kc: 88, et: 88 },
      { day: '9/13', kc: 80, et: 78 },
      { day: '9/29', kc: 61, et: 46 },
      { day: '10/15', kc: 50, et: 27 },
      { day: '10/31', kc: 37, et: 19 }
    ],
    indicators: [
      { name: 'NDVI', value: 73, note: '峰值窗口均值 0.73' },
      { name: 'SAVI', value: 51, note: '峰值窗口均值 0.51' },
      { name: 'RDVI', value: 48, note: '峰值窗口均值 0.48' },
      { name: 'GNDVI', value: 66, note: '峰值窗口均值 0.66' },
      { name: 'EVI', value: 60, note: '峰值窗口均值 0.60' },
      { name: 'SM', value: 50, note: '体积含水量约 0.30' },
      { name: 'doy', value: 75, note: 'DOY 225' }
    ]
  }
})
