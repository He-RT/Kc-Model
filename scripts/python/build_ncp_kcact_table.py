"""Build a merged NCP (North China Plain) Kcact training table from multi-province GEE exports.

Processes provinces one-at-a-time to avoid OOM, writing intermediate parquets.

Example:
  python scripts/python/build_ncp_kcact_table.py \\
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

import numpy as np
import pandas as pd

from kcact.data.io import read_many_csv, write_table
from kcact.data.kcact_builder import build_training_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build merged NCP patch-date Kcact tables from multi-province GEE exports."
    )
    parser.add_argument(
        "--provinces", nargs="+", required=True,
        help="Province names (e.g. Hebei Henan Shandong Anhui).",
    )
    parser.add_argument(
        "--data-dir", default=str(ROOT / "data" / "raw" / "gee"),
        help="Directory containing per-province GEE CSV exports.",
    )
    parser.add_argument(
        "--output-valid",
        default=str(ROOT / "data" / "processed" / "train" / "ncp_winter_wheat_kcact_train_ready.parquet"),
    )
    parser.add_argument(
        "--temp-dir",
        default=str(ROOT / "data" / "interim"),
        help="Directory for intermediate per-province parquets.",
    )
    return parser.parse_args()


def make_globs(data_dir: str, province: str) -> dict[str, list[str]]:
    prov = province.lower()
    base = str(Path(data_dir))
    return {
        "s2":    [f"{base}/{prov}_kcact_s2_features_*.csv"],
        "era5":  [f"{base}/{prov}_kcact_era5_daily_*.csv"],
        "mod16": [f"{base}/{prov}_kcact_mod16_etc_*.csv"],
    }


def main() -> None:
    args = parse_args()
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_files = []
    province_counts = {}

    for province in args.provinces:
        print(f"\n{'='*50}")
        print(f"Processing: {province}")
        print(f"{'='*50}")

        globs = make_globs(args.data_dir, province)
        try:
            s2_df = read_many_csv(globs["s2"], parse_dates=["date_start", "date_end", "date"])
            era5_df = read_many_csv(globs["era5"], parse_dates=["date"])
            mod16_df = read_many_csv(globs["mod16"], parse_dates=["date_start", "date_end", "date"])
        except FileNotFoundError as e:
            print(f"  [SKIP] {province}: {e}")
            continue

        print(f"  S2 rows: {len(s2_df)}, ERA5 rows: {len(era5_df)}, MOD16 rows: {len(mod16_df)}")

        _all_rows, valid_rows = build_training_table(s2_df, era5_df, mod16_df,
                                                      province=province)
        print(f"  Valid rows: {len(valid_rows)}  ({len(_all_rows)} total, "
              f"{len(valid_rows)/max(len(_all_rows),1)*100:.1f}% pass QC)")

        # Free input CSVs immediately
        del s2_df, era5_df, mod16_df, _all_rows

        # Save intermediate parquet
        temp_path = temp_dir / f"ncp_{province.lower()}_valid.parquet"
        write_table(valid_rows, str(temp_path))
        temp_files.append(temp_path)
        province_counts[province] = len(valid_rows)
        del valid_rows

    if not temp_files:
        print("No data loaded. Check --data-dir and --provinces.")
        return

    # Concat all temp parquets (each is small, concat is cheap)
    print(f"\n{'='*50}")
    print("Merging provinces...")
    print(f"{'='*50}")
    frames = []
    for tf in temp_files:
        print(f"  Reading {tf.name}: {tf.stat().st_size / 1e6:.1f} MB")
        frames.append(pd.read_parquet(tf, dtype_backend="pyarrow"))

    merged = pd.concat(frames, ignore_index=True)
    del frames

    # Deduplicate after province merge.  patch_id is crop+province+season_year
    # + coordinate, not the unstable GEE pt_* id, so it is globally stable.
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=["patch_id", "date"], keep="first")
    if len(merged) < n_before:
        print(f"  Removed {n_before - len(merged)} duplicate (patch_id, date) rows")

    valid_path = write_table(merged, args.output_valid)
    print(f"\n  Total valid rows: {len(merged)}")
    for prov, count in province_counts.items():
        print(f"    {prov}: {count}")
    print(f"\nSaved to: {valid_path}")

    # Feature summary
    num_cols = [c for c in merged.columns
                if c not in {"qc_valid", "kcact", "etc_8d_mm", "et0_pm_8d_mm"}
                and merged[c].dtype in ("float64", "float32", "int64", "int32",
                                        "Float64", "Float32", "Int64", "Int32")]
    print(f"  Numeric features: {len(num_cols)}")


if __name__ == "__main__":
    main()
