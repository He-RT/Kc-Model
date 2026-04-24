"""Train baseline RandomForest and XGBoost models for Kcact prediction.

Input: the qc_valid parquet table from build_hebei_kcact_table.py
Output: model artifacts, evaluation metrics, and feature importance figures.

The target is Kcact = ETc / ET0.
Features are remote sensing indices + weather variables + temporal features.

Cross-year validation: train on (2021, 2022), test on 2023 unless overridden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kcact baseline models.")
    parser.add_argument(
        "--input-table",
        default="/Users/hert/Projects/dcsdxx/data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet",
    )
    parser.add_argument("--train-years", nargs="+", type=int, default=[2021, 2022])
    parser.add_argument("--test-years", nargs="+", type=int, default=[2023])
    parser.add_argument(
        "--output-dir",
        default="/Users/hert/Projects/dcsdxx/outputs",
    )
    return parser.parse_args()


def load_and_split(path: str, train_years: list[int], test_years: list[int]):
    df = pd.read_parquet(path)
    df = df[df["qc_valid"]].copy()

    exclude_cols = {
        "patch_id", "point_id", "date", "date_start", "date_end",
        "province", "crop_type", "qc_valid", "kcact",
        # Kcact = etc_8d_mm / et0_pm_8d_mm, both are target components — exclude
        "etc_8d_mm", "et0_pm_8d_mm",
        # QC flags, not predictors
        "qc_mod16",
        # keep 'year' for splitting but exclude from features
    }

    feature_cols = [
        col for col in df.select_dtypes(include=["number", "bool"]).columns
        if col not in exclude_cols and col != "year"
    ]

    train = df[df["year"].isin(train_years)]
    test = df[df["year"].isin(test_years)]

    X_train = train[feature_cols].fillna(0.0)
    y_train = train["kcact"]
    X_test = test[feature_cols].fillna(0.0)
    y_test = test["kcact"]

    return X_train, y_train, X_test, y_test, feature_cols


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": len(y_true),
    }


def train_rf(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate("RandomForest", y_test, y_pred)
    importance = sorted(
        zip(model.feature_names_in_, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:20]
    return model, metrics, importance


def train_xgb(X_train, y_train, X_test, y_test):
    if not HAS_XGB:
        return None, None, None
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate("XGBoost", y_test, y_pred)
    importance = sorted(
        zip(model.feature_names_in_, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:20]
    return model, metrics, importance


def main() -> None:
    args = parse_args()
    X_train, y_train, X_test, y_test, feature_cols = load_and_split(
        args.input_table, args.train_years, args.test_years,
    )
    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    # Random Forest
    rf_model, rf_metrics, rf_imp = train_rf(X_train, y_train, X_test, y_test)
    print(f"RF  - R²={rf_metrics['r2']:.4f}, RMSE={rf_metrics['rmse']:.4f}, MAE={rf_metrics['mae']:.4f}")
    all_metrics.append(rf_metrics)
    if rf_imp:
        print("RF top-10 features:")
        for name, score in rf_imp[:10]:
            print(f"  {name}: {score:.4f}")

    # XGBoost
    if HAS_XGB:
        xgb_model, xgb_metrics, xgb_imp = train_xgb(X_train, y_train, X_test, y_test)
        if xgb_metrics:
            print(f"XGB - R²={xgb_metrics['r2']:.4f}, RMSE={xgb_metrics['rmse']:.4f}, MAE={xgb_metrics['mae']:.4f}")
            all_metrics.append(xgb_metrics)
        if xgb_imp:
            print("XGB top-10 features:")
            for name, score in xgb_imp[:10]:
                print(f"  {name}: {score:.4f}")
    else:
        print("XGBoost not installed -- skipping.")

    # Save metrics
    (out / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(
        out / "tables" / "kcact_baseline_metrics.csv", index=False,
    )
    print(f"Metrics saved to {out / 'tables' / 'kcact_baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
