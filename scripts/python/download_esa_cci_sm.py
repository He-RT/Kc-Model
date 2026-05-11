"""Download ESA CCI SM daily data from CDS for 4 flux stations."""

import cdsapi
import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr

STATIONS = [
    ("禹城", 36.829, 116.5702),
    ("位山", 36.6493, 116.059),
    ("馆陶", 36.517, 115.133),
    ("栾城", 37.884, 114.689),
]

OUT_DIR = Path("/Users/hert/Projects/dcsdxx/data/raw/sat_sm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / "esa_cci_sm_all_stations.csv"

if CACHE.exists():
    print(f"Cached: {CACHE}")
    import sys; sys.exit(0)

# Define bounding box covering all 4 stations with some margin
lats = [s[1] for s in STATIONS]; lons = [s[2] for s in STATIONS]
bbox = [max(lats)+0.5, min(lons)-0.5, min(lats)-0.5, max(lons)+0.5]  # N,W,S,E

# Request annual files to avoid CDS limits
all_frames = []
for yr in range(2003, 2016):
    nc_file = OUT_DIR / f"esa_cci_sm_{yr}.nc"

    if not nc_file.exists():
        print(f"Downloading {yr}...")
        client = cdsapi.Client()
        client.retrieve(
            "satellite-soil-moisture",
            {
                "variable": "combined_daily_sm",
                "type_of_sensor": "combined_product",
                "type_of_record": "cdr",
                "year": str(yr),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "area": bbox,  # N,W,S,E
                "format": "netcdf",
            },
            str(nc_file),
        )

    # Read NetCDF and extract station values
    ds = xr.open_dataset(nc_file)
    sm = ds["sm"].values
    times = pd.to_datetime(ds["time"].values)
    lats_nc = ds["lat"].values
    lons_nc = ds["lon"].values

    for name, lat, lon in STATIONS:
        # Find nearest grid cell
        i_lat = np.abs(lats_nc - lat).argmin()
        i_lon = np.abs(lons_nc - lon).argmin()
        vals = sm[:, i_lat, i_lon]

        df = pd.DataFrame({
            "date": pd.to_datetime(times),
            "station": name,
            "sm_esa": vals,
        })
        all_frames.append(df)

    ds.close()
    print(f"  {yr}: extracted {len(df)} rows")

result = pd.concat(all_frames, ignore_index=True)
result = result.dropna(subset=["sm_esa"])
result.to_csv(CACHE, index=False)
print(f"Saved {CACHE}: {len(result)} rows")
for name, _, _ in STATIONS:
    s = result[result["station"] == name]
    print(f"  {name}: {len(s)} obs, SM range {s['sm_esa'].min():.3f} – {s['sm_esa'].max():.3f}")
