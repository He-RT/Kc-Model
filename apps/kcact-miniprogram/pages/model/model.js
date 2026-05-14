Page({
  data: {
    chips: ['NDVI', 'SAVI', 'RDVI', 'GNDVI', 'EVI', 'SM', 'doy'],
    features: [
      { name: 'doy', score: 92, tone: 'ink' },
      { name: 'NDVI', score: 78, tone: 'greenbar' },
      { name: 'SM', score: 72, tone: 'soilbar' },
      { name: 'EVI', score: 66, tone: 'greenbar' },
      { name: 'GNDVI', score: 61, tone: 'bluebar' },
      { name: 'SAVI', score: 54, tone: 'midbar' },
      { name: 'RDVI', score: 48, tone: 'warnbar' }
    ]
  }
})
