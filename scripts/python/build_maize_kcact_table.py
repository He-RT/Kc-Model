"""Build summer maize Kcact training table from GEE exports.

Example:
  python scripts/python/build_maize_kcact_table.py \\
    --provinces Hebei Henan Shandong Anhui \\
    --data-dir data/raw/gee
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from kcact.data.io import read_many_csv, write_table
from kcact.data.kcact_builder import build_training_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build summer maize Kcact training table."
    )
    parser.add_argument("--provinces", nargs="+", required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "gee"))
    parser.add_argument("--output-valid",
                        default=str(ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"))
    parser.add_argument("--temp-dir", default=str(ROOT / "data" / "interim"))
    return parser.parse_args()


def make_globs(data_dir: str, province: str) -> dict[str, list[str]]:
    prov = province.lower()
    base = str(Path(data_dir))
    return {
        "s2":    [f"{base}/{prov}_maize_s2_features_*.csv"],
        "era5":  [f"{base}/{prov}_maize_era5_daily_*.csv"],
        "mod16": [f"{base}/{prov}_maize_mod16_etc_*.csv"],
    }


def main():
    args = parse_args()
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_files = []
    province_counts = {}

    for province in args.provinces:
        print(f"\n{'='*50}")
        print(f"Processing: {province} (summer maize)")
        print(f"{'='*50}")
        globs = make_globs(args.data_dir, province)
        try:
            s2_df = read_many_csv(globs["s2"], parse_dates=["date_start", "date_end", "date"])
            era5_df = read_many_csv(globs["era5"], parse_dates=["date"])
            mod16_df = read_many_csv(globs["mod16"], parse_dates=["date_start", "date_end", "date"])
        except FileNotFoundError as e:
            print(f"  [SKIP] {province}: {e}")
            continue

        _all_rows, valid_rows = build_training_table(
            s2_df, era5_df, mod16_df,
            crop_type="summer_maize_candidate",
            province=province,
        )
        print(f"  Valid rows: {len(valid_rows)}  ({len(_all_rows)} total)")

        del s2_df, era5_df, mod16_df, _all_rows

        temp_path = temp_dir / f"maize_{province.lower()}_valid.parquet"
        write_table(valid_rows, str(temp_path))
        temp_files.append(temp_path)
        province_counts[province] = len(valid_rows)
        del valid_rows

    if not temp_files:
        print("No data loaded.")
        return

    print(f"\n{'='*50}")
    print("Merging provinces...")
    print(f"{'='*50}")
    frames = [pd.read_parquet(tf) for tf in temp_files]
    merged = pd.concat(frames, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=["patch_id", "date"], keep="first")
    print(f"  Dedupe: {n_before:,} -> {len(merged):,}")

    write_table(merged, args.output_valid)
    print(f"Saved: {args.output_valid}")
    for prov, count in province_counts.items():
        print(f"  {prov}: {count:,}")


if __name__ == "__main__":
    main()
