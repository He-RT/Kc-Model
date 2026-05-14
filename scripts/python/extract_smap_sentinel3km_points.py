"""Extract downloaded SMAP/Sentinel-1 3 km HDF5 files to Kcact point-windows.

This is the second step after ``query_smap_sentinel3km.py``:

1. Download SPL2SMAP_S V003 HDF5 files listed in the manifest.
2. Put them under ``data/raw/nsidc/smap_sentinel3km/h5``.
3. Run this extractor for a year/province.

The product is a swath/scene product, not a daily global grid.  Each file only
contains cells covered by the Sentinel-1 scene.  We read the validated 3 km
group, nearest-neighbor match each corrected maize point to the scene cells,
then aggregate all valid observations within each MOD16 8-day window.

Required optional dependency: h5py.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
H5_DIR = ROOT / "data/raw/nsidc/smap_sentinel3km/h5"
OUT_DIR = ROOT / "data/raw/nsidc/smap_sentinel3km/extracted"
PROVINCES = ["Anhui", "Hebei", "Henan", "Shandong"]

GROUP = "Soil_Moisture_Retrieval_Data_3km"
FIELDS = {
    "smap_s1_sm_3km": "soil_moisture_3km",
    "smap_s1_sm_apm_3km": "soil_moisture_apm_3km",
    "smap_s1_sm_std_3km": "soil_moisture_std_dev_3km",
    "smap_s1_retrieval_qual_flag_3km": "retrieval_qual_flag_3km",
    "smap_s1_overpass_timediff_hr_3km": "SMAP_Sentinel-1_overpass_timediff_hr_3km",
    "smap_s1_sigma0_vv_3km": "sigma0_vv_aggregated_3km",
    "smap_s1_sigma0_vh_3km": "sigma0_vh_aggregated_3km",
    "smap_s1_vwc_3km": "vegetation_water_content_3km",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--province", choices=PROVINCES, default=None)
    parser.add_argument("--h5-dir", default=str(H5_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=3.0,
        help="Reject point-to-cell nearest matches farther than this distance.",
    )
    return parser.parse_args()


def require_h5py():
    try:
        import h5py  # type: ignore

        return h5py
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency h5py. Install in the project env, e.g. "
            "`uv pip install h5py` or `pip install h5py`."
        ) from exc


def load_points_and_windows(year: int, province: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "point_id",
        "coord_key",
        "province",
        "centroid_lat",
        "centroid_lon",
        "date_start",
        "date_end",
        "date",
        "year",
        "qc_valid",
    ]
    df = pd.read_parquet(PARQUET, columns=cols)
    for c in ["date_start", "date_end", "date"]:
        df[c] = pd.to_datetime(df[c])
    df = df[(df["qc_valid"]) & (df["year"].astype(int) == int(year))].copy()
    if province:
        df = df[df["province"].astype(str) == province].copy()
    if df.empty:
        raise SystemExit(f"No corrected maize rows for year={year}, province={province}")

    points = (
        df[["point_id", "coord_key", "province", "centroid_lat", "centroid_lon"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    windows = df[["date_start", "date_end", "date"]].drop_duplicates().sort_values("date_start")
    return points, windows


def granule_time(path: Path) -> pd.Timestamp:
    # SMAP_L2_SM_SP_..._20190601T224104_...
    m = re.search(r"_(\d{8}T\d{6})_", path.name)
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S", utc=True).tz_convert(None)


def assign_window(time: pd.Timestamp, windows: pd.DataFrame) -> pd.Timestamp | None:
    if pd.isna(time):
        return None
    match = windows[(windows["date_start"] <= time) & (time < windows["date_end"])]
    if match.empty:
        return None
    return pd.Timestamp(match.iloc[0]["date"])


def approx_xy(lat: np.ndarray, lon: np.ndarray, lat0: float) -> np.ndarray:
    """Approximate lon/lat to km coordinates for nearest-neighbor matching."""
    x = lon * 111.32 * np.cos(np.deg2rad(lat0))
    y = lat * 110.57
    return np.column_stack([x, y])


def read_dataset(group, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    arr = group[name][()]
    arr = np.asarray(arr)
    arr = arr.astype("float64", copy=False)
    arr[arr <= -9990] = np.nan
    return arr


def extract_file(path: Path, points: pd.DataFrame, date_label: pd.Timestamp, max_distance_km: float) -> pd.DataFrame:
    h5py = require_h5py()
    with h5py.File(path, "r") as h5:
        if GROUP not in h5:
            return pd.DataFrame()
        g = h5[GROUP]
        lat = read_dataset(g, "latitude_3km")
        lon = read_dataset(g, "longitude_3km")
        if lat is None or lon is None:
            return pd.DataFrame()

        valid = np.isfinite(lat) & np.isfinite(lon)
        if not valid.any():
            return pd.DataFrame()

        lat0 = float(np.nanmean(points["centroid_lat"]))
        grid_xy = approx_xy(lat[valid].ravel(), lon[valid].ravel(), lat0)
        tree = cKDTree(grid_xy)
        pt_xy = approx_xy(points["centroid_lat"].to_numpy(float), points["centroid_lon"].to_numpy(float), lat0)
        dist, idx = tree.query(pt_xy, k=1)
        keep_pt = dist <= max_distance_km
        if not keep_pt.any():
            return pd.DataFrame()

        flat_index = np.flatnonzero(valid.ravel())[idx[keep_pt]]
        out = points.loc[keep_pt, ["point_id", "coord_key", "province", "centroid_lat", "centroid_lon"]].copy()
        out["date"] = date_label
        out["source_file"] = path.name
        out["smap_s1_match_distance_km"] = dist[keep_pt]

        for out_name, h5_name in FIELDS.items():
            arr = read_dataset(g, h5_name)
            if arr is not None:
                out[out_name] = arr.ravel()[flat_index]

        return out


def main() -> None:
    args = parse_args()
    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points, windows = load_points_and_windows(args.year, args.province)
    files = sorted(h5_dir.glob(f"*{args.year}*.h5"))
    if not files:
        raise SystemExit(f"No HDF5 files found for {args.year} in {h5_dir}")

    frames = []
    for i, path in enumerate(files, 1):
        t = granule_time(path)
        date_label = assign_window(t, windows)
        if date_label is None:
            continue
        part = extract_file(path, points, date_label, max_distance_km=args.max_distance_km)
        if not part.empty:
            frames.append(part)
        if i % 50 == 0:
            print(f"{i}/{len(files)} files, extracted parts={len(frames)}")

    if not frames:
        raise SystemExit("No point matches extracted.")
    raw = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in raw.columns if c.startswith("smap_s1_") and c not in {"smap_s1_retrieval_qual_flag_3km"}]
    agg_cols = {c: (c, "mean") for c in value_cols}
    agg_cols["smap_s1_obs_count"] = ("source_file", "nunique")
    agg = raw.groupby(["point_id", "coord_key", "province", "date"], as_index=False).agg(**agg_cols)

    suffix = f"{args.year}_{args.province}" if args.province else str(args.year)
    raw.to_csv(out_dir / f"smap_sentinel3km_points_raw_{suffix}.csv", index=False)
    agg.to_csv(out_dir / f"smap_sentinel3km_points_8d_{suffix}.csv", index=False)
    print(f"raw rows={len(raw):,}; aggregated rows={len(agg):,}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
