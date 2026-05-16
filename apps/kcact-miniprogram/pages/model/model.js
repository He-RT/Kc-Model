Page({
  data: {
    chips: ['NDVI', 'SAVI', 'RDVI', 'GNDVI', 'EVI', 'SM', 'doy'],
    modelCards: [
      { label: 'LOYO R²', value: '0.765', unit: 'pooled' },
      { label: 'RMSE', value: '0.102', unit: 'Kcact' },
      { label: 'MAE', value: '0.076', unit: 'Kcact' }
    ],
    yearly: [
      { year: '2019', r2: '0.790', score: 79 },
      { year: '2020', r2: '0.686', score: 69 },
      { year: '2021', r2: '0.689', score: 69 },
      { year: '2022', r2: '0.771', score: 77 },
      { year: '2023', r2: '0.780', score: 78 },
      { year: '2024', r2: '0.803', score: 80 }
    ],
    features: [
      { name: 'doy', score: 100, tone: 'ink' },
      { name: 'SM', score: 58, tone: 'soilbar' },
      { name: 'NDVI', score: 32, tone: 'greenbar' },
      { name: 'EVI', score: 29, tone: 'greenbar' },
      { name: 'GNDVI', score: 10, tone: 'bluebar' },
      { name: 'SAVI', score: 4, tone: 'midbar' },
      { name: 'RDVI', score: 2, tone: 'warnbar' }
    ]
  }
})
