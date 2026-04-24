"""Data IO helpers for the Kcact pipeline."""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable

import pandas as pd


def expand_input_paths(paths: Iterable[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        matches = sorted(glob(raw))
        if matches:
            expanded.extend(Path(item) for item in matches)
        else:
            expanded.append(Path(raw))
    unique = []
    seen: set[Path] = set()
    for path in expanded:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def read_many_csv(paths: Iterable[str], parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    resolved = expand_input_paths(paths)
    if not resolved:
        raise FileNotFoundError("No input CSV files matched")
    frames = [pd.read_csv(path, parse_dates=list(parse_dates or [])) for path in resolved]
    return pd.concat(frames, ignore_index=True)


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_table(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = ensure_parent_dir(output_path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
        return path
    df.to_csv(path, index=False)
    return path
