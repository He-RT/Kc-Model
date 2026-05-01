"""Build a patch-date Kcact training table from exported GEE CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kcact.data.io import read_many_csv, write_table
from kcact.data.kcact_builder import build_training_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Hebei patch-date Kcact tables from raw GEE exports."
    )
    parser.add_argument("--s2-csv", nargs="+", required=True, help="One or more S2 feature CSV paths or globs.")
    parser.add_argument("--era5-csv", nargs="+", required=True, help="One or more ERA5 daily CSV paths or globs.")
    parser.add_argument("--mod16-csv", nargs="+", required=True, help="One or more MOD16 CSV paths or globs.")
    parser.add_argument(
        "--output-all",
        default="/Users/hert/Projects/dcsdxx/data/interim/hebei_kcact_all_rows.parquet",
    )
    parser.add_argument(
        "--output-valid",
        default="/Users/hert/Projects/dcsdxx/data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s2_df = read_many_csv(args.s2_csv, parse_dates=["date_start", "date_end", "date"])
    era5_df = read_many_csv(args.era5_csv, parse_dates=["date"])
    mod16_df = read_many_csv(args.mod16_csv, parse_dates=["date_start", "date_end", "date"])

    all_rows, valid_rows = build_training_table(s2_df, era5_df, mod16_df)

    all_path = write_table(all_rows, args.output_all)
    valid_path = write_table(valid_rows, args.output_valid)

    print(f"Saved all rows to: {all_path}")
    print(f"Saved valid rows to: {valid_path}")
    print(f"All rows: {len(all_rows)}")
    print(f"Valid rows: {len(valid_rows)}")
    if not valid_rows.empty:
        print("Feature columns available for modeling:")
        print(", ".join(sorted(valid_rows.columns)))


if __name__ == "__main__":
    main()
