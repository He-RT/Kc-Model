"""Build a merged NCP (North China Plain) Kcact training table from multi-province GEE exports.

Reads per-province S2/ERA5/MOD16 CSVs, builds training tables via kcact_builder,
concatenates all valid rows, and outputs a single NCP-wide parquet file.

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
        "--output-all",
        default=str(ROOT / "data" / "interim" / "ncp_kcact_all_rows.parquet"),
    )
    parser.add_argument(
        "--output-valid",
        default=str(ROOT / "data" / "processed" / "train" / "ncp_winter_wheat_kcact_train_ready.parquet"),
    )
    return parser.parse_args()


def make_globs(data_dir: str, province: str) -> dict[str, list[str]]:
    """Build glob patterns for a province following the standard naming convention."""
    prov = province.lower()
    base = str(Path(data_dir))
    return {
        "s2":    [f"{base}/{prov}_kcact_s2_features_*.csv"],
        "era5":  [f"{base}/{prov}_kcact_era5_daily_*.csv"],
        "mod16": [f"{base}/{prov}_kcact_mod16_etc_*.csv"],
    }


def main() -> None:
    args = parse_args()
    all_valid_frames = []
    province_counts = {}

    for province in args.provinces:
        globs = make_globs(args.data_dir, province)
        try:
            s2_df = read_many_csv(globs["s2"], parse_dates=["date_start", "date_end", "date"])
            era5_df = read_many_csv(globs["era5"], parse_dates=["date"])
            mod16_df = read_many_csv(globs["mod16"], parse_dates=["date_start", "date_end", "date"])
        except FileNotFoundError as e:
            print(f"  [SKIP] {province}: {e}")
            continue

        _all_rows, valid_rows = build_training_table(s2_df, era5_df, mod16_df,
                                                      province=province)
        print(f"  {province}: {len(valid_rows)} valid rows  ({len(_all_rows)} total)")
        all_valid_frames.append(valid_rows)
        province_counts[province] = len(valid_rows)

    if not all_valid_frames:
        print("No data loaded. Check --data-dir and --provinces.")
        return

    merged = all_valid_frames[0]
    for frame in all_valid_frames[1:]:
        merged = merged.merge(
            frame, how="outer",
            on=list(merged.columns.intersection(frame.columns)),
        ) if set(merged.columns) != set(frame.columns) else merged

    # Actually, use pd.concat since each frame has same columns but different rows
    import pandas as pd
    merged = pd.concat(all_valid_frames, ignore_index=True)

    all_path = write_table(merged, args.output_all)
    valid_path = write_table(merged, args.output_valid)

    print(f"\nProvince breakdown:")
    for prov, count in province_counts.items():
        print(f"  {prov}: {count}")
    print(f"  Total: {len(merged)} valid rows")
    print(f"\nSaved all rows to: {all_path}")
    print(f"Saved valid rows to: {valid_path}")
    if not merged.empty:
        print(f"Feature columns ({len([c for c in merged.columns if c != 'qc_valid'])}):")
        print(", ".join(sorted(
            c for c in merged.columns
            if c not in {"qc_valid", "kcact", "etc_8d_mm", "et0_pm_8d_mm"}
            and merged[c].dtype in ("float64", "float32", "int64", "int32")
        )))


if __name__ == "__main__":
    main()
