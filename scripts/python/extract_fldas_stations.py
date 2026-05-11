"""Extract FLDAS monthly soil moisture at 4 flux station coordinates via GEE.
Efficient: reduceRegions once per image for all 4 stations."""

import ee
import pandas as pd
from pathlib import Path

ee.Initialize(project='chuang-yaogan')

STATIONS = [
    ("禹城", 36.829, 116.5702),
    ("位山", 36.6493, 116.059),
    ("馆陶", 36.517, 115.133),
    ("栾城", 37.884, 114.689),
]

OUT = Path("/Users/hert/Projects/dcsdxx/data/processed/fldas_station_sm.csv")

if OUT.exists():
    df = pd.read_csv(OUT)
    print(f"Cached: {OUT} ({len(df)} rows)")
    for name in df['station'].unique():
        s = df[df['station'] == name]
        print(f"  {name}: {len(s)} months, "
              f"SM00-10 mean={s['SoilMoi00_10cm_tavg'].mean():.4f}, "
              f"SM40-100 mean={s['SoilMoi40_100cm_tavg'].mean():.4f}")
    print(df.head(10))
    import sys; sys.exit(0)

bands = ['SoilMoi00_10cm_tavg', 'SoilMoi10_40cm_tavg',
         'SoilMoi40_100cm_tavg', 'SoilMoi100_200cm_tavg']

# Create FeatureCollection of 4 station points
pts = [ee.Feature(ee.Geometry.Point(lon, lat), {'station': name})
       for name, lat, lon in STATIONS]
fc = ee.FeatureCollection(pts)

# Get all images as a list, then map reduceRegions
col = ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001")
col = col.select(bands).filterDate('2003-01-01', '2016-01-01')

img_list = col.toList(200)
n = img_list.length().getInfo()
print(f"Extracting {n} months × {len(STATIONS)} stations...")

rows = []
for i in range(n):
    img = ee.Image(img_list.get(i))
    d = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

    # reduceRegions for all 4 stations in one call
    result = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=11132,
    ).getInfo()

    for feat in result['features']:
        props = feat['properties']
        row = {'date': d, 'station': props['station']}
        for b in bands:
            row[b] = props.get(b)
        rows.append(row)

    if (i + 1) % 24 == 0:
        print(f"  {i+1}/{n} done")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nSaved {OUT}: {len(df)} rows")
for name in df['station'].unique():
    s = df[df['station'] == name]
    print(f"  {name}: {len(s)} months, "
          f"SM00-10 mean={s['SoilMoi00_10cm_tavg'].mean():.4f}, "
          f"SM40-100 mean={s['SoilMoi40_100cm_tavg'].mean():.4f}")
print(df.head(10))
