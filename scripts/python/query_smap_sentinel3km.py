"""Query NASA NSIDC SMAP/Sentinel-1 3 km soil-moisture granules.

Product:
  SPL2SMAP_S V003
  SMAP/Sentinel-1 L2 Radiometer/Radar 30-Second Scene 3 km EASE-Grid
  Soil Moisture

This script does **not** download data.  It writes a manifest of HTTPS/S3/OPeNDAP
links returned by NASA CMR for the NCP maize time range/region.  Downloading the
HDF5 files requires a NASA Earthdata account (.netrc, earthaccess, wget cookie,
or Earthdata Download).

Example:
  python scripts/python/query_smap_sentinel3km.py --year 2019
  python scripts/python/query_smap_sentinel3km.py --year 2019 --province Henan
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
OUT_DIR = ROOT / "data/raw/nsidc/smap_sentinel3km/manifests"

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION_CONCEPT_ID = "C2938663471-NSIDC_CPRD"  # SPL2SMAP_S V003
PROVINCES = ["Anhui", "Hebei", "Henan", "Shandong"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None, help="Single year; default queries 2019-2025.")
    parser.add_argument("--province", choices=PROVINCES, default=None)
    parser.add_argument("--split-province", action="store_true", help="Query each province separately.")
    parser.add_argument(
        "--bbox",
        default=None,
        help="Override bounding box as W,S,E,N. Default is computed from corrected parquet.",
    )
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    return parser.parse_args()


def load_extent_and_time(year: int, province: str | None = None) -> tuple[str, str, str]:
    cols = ["province", "centroid_lat", "centroid_lon", "date_start", "date_end", "year", "qc_valid"]
    df = pd.read_parquet(PARQUET, columns=cols)
    for col in ["date_start", "date_end"]:
        df[col] = pd.to_datetime(df[col])
    df = df[(df["qc_valid"]) & (df["year"].astype(int) == int(year))].copy()
    if province:
        df = df[df["province"].astype(str) == province].copy()
    if df.empty:
        raise SystemExit(f"No valid rows in corrected maize parquet for year={year}, province={province}")

    # Add a small margin so CMR spatial filtering does not miss edge scenes.
    west = float(df["centroid_lon"].min()) - 0.2
    east = float(df["centroid_lon"].max()) + 0.2
    south = float(df["centroid_lat"].min()) - 0.2
    north = float(df["centroid_lat"].max()) + 0.2
    bbox = f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"

    start = df["date_start"].min().strftime("%Y-%m-%dT00:00:00Z")
    # CMR temporal end is inclusive enough; keep one day after max date_end.
    end = (df["date_end"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    return bbox, start, end


def cmr_get(params: dict) -> tuple[list[dict], int]:
    url = CMR_GRANULES + "?" + urlencode(params)
    req = Request(url, headers={"Client-Id": "dcsdxx-kcact"})
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        hits = int(resp.headers.get("CMR-Hits", "0"))
    return data.get("feed", {}).get("entry", []), hits


def data_links(entry: dict) -> dict[str, str]:
    out = {"https_url": "", "s3_url": "", "opendap_url": ""}
    for link in entry.get("links", []):
        href = link.get("href", "")
        rel = link.get("rel", "")
        if href.endswith(".h5") and href.startswith("https://") and "/protected/" in href:
            out["https_url"] = href
        elif href.endswith(".h5") and href.startswith("s3://"):
            out["s3_url"] = href
        elif "opendap.earthdata.nasa.gov" in href:
            out["opendap_url"] = href
    if not out["https_url"] and out["s3_url"]:
        out["https_url"] = out["s3_url"].replace(
            "s3://nsidc-cumulus-prod-protected/",
            "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/",
        )
    return out


def query_manifest(year: int, province: str | None, bbox_override: str | None, page_size: int) -> pd.DataFrame:
    bbox, temporal_start, temporal_end = load_extent_and_time(year, province)
    if bbox_override:
        bbox = bbox_override

    records = []
    page = 1
    hits = None
    while True:
        params = {
            "collection_concept_id": COLLECTION_CONCEPT_ID,
            "bounding_box": bbox,
            "temporal": f"{temporal_start},{temporal_end}",
            "page_size": page_size,
            "page_num": page,
            "sort_key": "start_date",
        }
        entries, this_hits = cmr_get(params)
        hits = this_hits if hits is None else hits
        if not entries:
            break
        for e in entries:
            links = data_links(e)
            records.append(
                {
                    "year": year,
                    "province": province or "",
                    "title": e.get("title", ""),
                    "time_start": e.get("time_start", ""),
                    "time_end": e.get("time_end", ""),
                    "bbox": bbox,
                    **links,
                }
            )
        print(f"{year} {province or 'all'}: page {page}, got {len(entries)}, total {len(records)}/{hits}")
        if len(entries) < page_size or len(records) >= hits:
            break
        page += 1

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    years = [args.year] if args.year else list(range(2019, 2026))
    provinces: list[str | None]
    if args.split_province:
        provinces = [args.province] if args.province else PROVINCES
    else:
        provinces = [args.province]

    frames = []
    for year in years:
        for province in provinces:
            df = query_manifest(year, province, args.bbox, args.page_size)
            frames.append(df)
            suffix = f"{year}_{province}" if province else str(year)
            out = out_dir / f"smap_sentinel3km_manifest_{suffix}.csv"
            df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"saved {len(df):,} rows -> {out}")

    if len(frames) > 1:
        all_df = pd.concat(frames, ignore_index=True)
        out = out_dir / "smap_sentinel3km_manifest_all.csv"
        all_df.to_csv(out, index=False)
        print(f"saved combined {len(all_df):,} rows -> {out}")


if __name__ == "__main__":
    main()
