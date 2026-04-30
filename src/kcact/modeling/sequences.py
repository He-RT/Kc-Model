"""Sequence dataset helpers for deep learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceBuildResult:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    feature_columns: list[str]


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "patch_id",
        "date",
        "date_start",
        "date_end",
        "province",
        "crop_type",
        "qc_valid",
        "kcact",
        "year",
    }
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_columns: list[str] | None = None,
    target_column: str = "kcact",
    expected_step_days: int = 8,
    max_gap_days: int = 10,
) -> SequenceBuildResult:
    if feature_columns is None:
        feature_columns = default_feature_columns(df)

    working = df.sort_values(["patch_id", "date"]).copy()
    working["date"] = pd.to_datetime(working["date"])

    xs: list[np.ndarray] = []
    ys: list[float] = []
    metas: list[dict[str, object]] = []

    for patch_id, group in working.groupby("patch_id"):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < seq_len:
            continue
        date_diffs = group["date"].diff().dt.days
        for end_idx in range(seq_len - 1, len(group)):
            window = group.iloc[end_idx - seq_len + 1 : end_idx + 1].copy()
            if len(window) != seq_len:
                continue
            diffs = window["date"].diff().dt.days.iloc[1:]
            if diffs.notna().any() and diffs.max() > max_gap_days:
                continue
            x = window[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
            y = float(window.iloc[-1][target_column])
            xs.append(x)
            ys.append(y)
            metas.append(
                {
                    "patch_id": patch_id,
                    "target_date": window.iloc[-1]["date"],
                    "target_year": int(window.iloc[-1]["year"]),
                    "seq_len": seq_len,
                    "expected_step_days": expected_step_days,
                }
            )

    if not xs:
        raise ValueError("No valid sequences were generated")

    return SequenceBuildResult(
        x=np.stack(xs),
        y=np.asarray(ys, dtype=np.float32),
        meta=pd.DataFrame(metas),
        feature_columns=feature_columns,
    )
