"""Extract MOD16A2GF ET at 4 flux station coordinates via GEE."""
from __future__ import annotations

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

OUT = Path("/Users/hert/Projects/dcsdxx/data/processed/mod16_station_et.csv")

if OUT.exists():
    df = pd.read_csv(OUT)
    print(f"Cached: {OUT} ({len(df)} rows)")
    for name in df['station'].unique():
        s = df[df['station'] == name]
        print(f"  {name}: {len(s)} obs, MOD16 ET mean={s['et_500m'].mean():.3f} mm/d")
    import sys; sys.exit(0)

pts = [ee.Feature(ee.Geometry.Point(lon, lat), {'station': name})
       for name, lat, lon in STATIONS]
fc = ee.FeatureCollection(pts)

col = ee.ImageCollection("MODIS/061/MOD16A2GF")
col = col.select(["ET", "ET_QC"]).filterDate('2003-01-01', '2016-01-01')

img_list = col.toList(500)
n = img_list.length().getInfo()
print(f"Extracting MOD16 windows: {n}")

rows = []
for i in range(n):
    img = ee.Image(img_list.get(i))
    d = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    result = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                               scale=500).getInfo()
    for feat in result['features']:
        props = feat['properties']
        rows.append({
            'date': d,
            'station': props['station'],
            'et_500m': props['ET'] * 0.1 if props.get('ET') is not None else None,
            'et_qc': props['ET_QC'],
        })
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{n} done")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nSaved {OUT}: {len(df)} rows")
for name in df['station'].unique():
    s = df[df['station'] == name]
    valid = s['et_500m'].notna().sum()
    print(f"  {name}: {len(s)} windows, {valid} valid, ET mean={s['et_500m'].mean():.3f} mm/d")
