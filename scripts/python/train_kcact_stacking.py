"""Nested LOYO-safe stacking ensemble for Kcact prediction.

Compares a simple average ensemble against manual Ridge stacking with:
  - XGBoost
  - CatBoost
  - LightGBM

Protocol:
1. Outer LOYO for final evaluation.
2. Inner LOYO on the outer-train split to generate OOF base predictions.
3. Ridge meta-learner trained only on OOF base predictions.

Results are saved to outputs/tables/kcact_stacking_results.csv.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from kcact.utils.gpu import get_gpu_config, make_xgb_params, make_catboost_params, make_lgbm_params


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INPUT_TABLE = ROOT / "data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet"
OUTPUT_TABLE = ROOT / "outputs/tables/kcact_stacking_results.csv"

EXCLUDE_COLS = {
    "patch_id", "point_id", "date", "date_start", "date_end",
    "province", "crop_type", "qc_valid", "kcact",
    "etc_8d_mm", "et0_pm_8d_mm", "qc_mod16",
    ".geo", "system:index", "valid_obs",
}

BASE_MODEL_NAMES = ["xgb", "catboost", "lgbm"]


def load_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.select_dtypes(include=["number", "bool"]).columns
        if col not in EXCLUDE_COLS
    ]


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": len(y_true),
    }


def make_base_models() -> dict[str, object]:
    return {
        "xgb": xgb.XGBRegressor(
            **make_xgb_params(extra={
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
            })
        ),
        "catboost": CatBoostRegressor(
            **make_catboost_params(extra={
                "iterations": 500, "depth": 6, "learning_rate": 0.05,
                "random_seed": 42, "verbose": 0,
            })
        ),
        "lgbm": LGBMRegressor(
            **make_lgbm_params(extra={
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "random_state": 42, "verbose": -1,
            })
        ),
    }


def fit_predict_base_models(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["kcact"].to_numpy()
    X_pred = pred_df[feature_cols].fillna(0.0)

    predictions: dict[str, np.ndarray] = {}
    for model_name, model in make_base_models().items():
        model.fit(X_train, y_train)
        predictions[model_name] = model.predict(X_pred)
    return predictions


def generate_inner_oof_predictions(
    outer_train: pd.DataFrame,
    feature_cols: list[str],
    train_years: list[int],
) -> np.ndarray:
    oof_preds = np.full((len(outer_train), len(BASE_MODEL_NAMES)), np.nan, dtype=float)

    for inner_year in train_years:
        inner_train = outer_train[outer_train["year"] != inner_year]
        inner_val = outer_train[outer_train["year"] == inner_year]

        if inner_train.empty or inner_val.empty:
            continue

        fold_preds = fit_predict_base_models(inner_train, inner_val, feature_cols)
        val_positions = outer_train.index.get_indexer(inner_val.index)

        for idx, model_name in enumerate(BASE_MODEL_NAMES):
            oof_preds[val_positions, idx] = fold_preds[model_name]

    if np.isnan(oof_preds).any():
        missing_rows = int(np.isnan(oof_preds).any(axis=1).sum())
        raise RuntimeError(
            f"Inner OOF generation left {missing_rows} rows without predictions."
        )

    return oof_preds


def print_comparison_table(results_df: pd.DataFrame) -> None:
    per_year = results_df[results_df["test_year"] != "pooled"].copy()
    if per_year.empty:
        return

    pivot = per_year.pivot(index="test_year", columns="model", values=["r2", "rmse", "mae", "n_samples"])
    pivot = pivot.sort_index()
    print("\n=== Per-Year Comparison ===")
    print(pivot.round(4).to_string())


def main() -> None:
    df = pd.read_parquet(INPUT_TABLE)
    df = df[df["qc_valid"]].copy()
    df = df.sort_values(["year"]).reset_index(drop=True)

    feature_cols = load_feature_cols(df)
    years = sorted(df["year"].dropna().unique().astype(int))

    print(f"Years: {years}")
    print(f"Features ({len(feature_cols)}): {', '.join(sorted(feature_cols))}")
    print(f"Total valid samples: {len(df)}")

    cfg = get_gpu_config()
    print(cfg.summary())

    all_results: list[dict] = []
    pooled_true: list[float] = []
    pooled_avg_pred: list[float] = []
    pooled_stack_pred: list[float] = []

    for test_year in years:
        outer_train = df[df["year"] != test_year].copy()
        outer_test = df[df["year"] == test_year].copy()
        train_years = sorted(outer_train["year"].dropna().unique().astype(int))

        print(
            f"\n--- Test year: {test_year} | Outer train: {len(outer_train)}, "
            f"Outer test: {len(outer_test)} | Inner years: {train_years} ---"
        )

        oof_base_preds = generate_inner_oof_predictions(outer_train, feature_cols, train_years)
        y_outer_train = outer_train["kcact"].to_numpy()

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(oof_base_preds, y_outer_train)

        test_base_preds = fit_predict_base_models(outer_train, outer_test, feature_cols)
        test_base_matrix = np.column_stack([test_base_preds[name] for name in BASE_MODEL_NAMES])

        avg_pred = test_base_matrix.mean(axis=1)
        stack_pred = meta_model.predict(test_base_matrix)
        y_test = outer_test["kcact"].to_numpy()

        avg_metrics = evaluate("simple_average", y_test, avg_pred)
        avg_metrics["test_year"] = test_year
        all_results.append(avg_metrics)

        stack_metrics = evaluate("ridge_stacking", y_test, stack_pred)
        stack_metrics["test_year"] = test_year
        all_results.append(stack_metrics)

        print(
            f"  simple_average  R²={avg_metrics['r2']:.4f} "
            f"RMSE={avg_metrics['rmse']:.4f} MAE={avg_metrics['mae']:.4f} "
            f"n={avg_metrics['n_samples']}"
        )
        print(
            f"  ridge_stacking  R²={stack_metrics['r2']:.4f} "
            f"RMSE={stack_metrics['rmse']:.4f} MAE={stack_metrics['mae']:.4f} "
            f"n={stack_metrics['n_samples']}"
        )

        pooled_true.extend(y_test.tolist())
        pooled_avg_pred.extend(avg_pred.tolist())
        pooled_stack_pred.extend(stack_pred.tolist())

    pooled_true_arr = np.asarray(pooled_true)
    pooled_avg_arr = np.asarray(pooled_avg_pred)
    pooled_stack_arr = np.asarray(pooled_stack_pred)

    pooled_avg = evaluate("simple_average", pooled_true_arr, pooled_avg_arr)
    pooled_avg["test_year"] = "pooled"
    all_results.append(pooled_avg)

    pooled_stack = evaluate("ridge_stacking", pooled_true_arr, pooled_stack_arr)
    pooled_stack["test_year"] = "pooled"
    all_results.append(pooled_stack)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[["model", "test_year", "r2", "rmse", "mae", "n_samples"]]

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_TABLE, index=False)

    print_comparison_table(results_df)

    print("\n=== Pooled Summary ===")
    pooled_rows = results_df[results_df["test_year"] == "pooled"]
    print(pooled_rows.round(4).to_string(index=False))
    print(f"\nResults saved to {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
