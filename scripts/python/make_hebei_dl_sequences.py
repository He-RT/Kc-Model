"""Convert the patch-date Kcact table into train/test sequence arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from kcact.modeling.sequences import build_sequences, default_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deep-learning-ready sequences from the Kcact training table."
    )
    parser.add_argument(
        "--input-table",
        default="/Users/hert/Projects/dcsdxx/data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet",
    )
    parser.add_argument("--seq-len", type=int, default=6, help="Number of 8-day steps per sequence.")
    parser.add_argument("--train-years", nargs="+", type=int, default=[2021, 2022])
    parser.add_argument("--test-years", nargs="+", type=int, default=[2023])
    parser.add_argument(
        "--output-prefix",
        default="/Users/hert/Projects/dcsdxx/data/processed/train/hebei_winter_wheat_kcact_seq",
    )
    return parser.parse_args()


def read_table(path: str) -> pd.DataFrame:
    resolved = Path(path)
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved, parse_dates=["date", "date_start", "date_end"])


def main() -> None:
    args = parse_args()
    df = read_table(args.input_table)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["qc_valid"]].copy()

    feature_columns = default_feature_columns(df)
    sequence_data = build_sequences(
        df,
        seq_len=args.seq_len,
        feature_columns=feature_columns,
        target_column="kcact",
    )

    meta = sequence_data.meta.copy()
    train_mask = meta["target_year"].isin(args.train_years).to_numpy()
    test_mask = meta["target_year"].isin(args.test_years).to_numpy()

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        f"{prefix}.npz",
        x_train=sequence_data.x[train_mask],
        y_train=sequence_data.y[train_mask],
        x_test=sequence_data.x[test_mask],
        y_test=sequence_data.y[test_mask],
    )
    meta.to_csv(f"{prefix}_meta.csv", index=False)
    Path(f"{prefix}_features.json").write_text(
        json.dumps(sequence_data.feature_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved sequence arrays to: {prefix}.npz")
    print(f"Saved sequence metadata to: {prefix}_meta.csv")
    print(f"Saved feature list to: {prefix}_features.json")
    print(f"Train sequences: {int(train_mask.sum())}")
    print(f"Test sequences: {int(test_mask.sum())}")


if __name__ == "__main__":
    main()
