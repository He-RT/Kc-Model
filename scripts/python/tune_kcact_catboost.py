"""Hyperparameter tuning for CatBoost Kcact model using Optuna LOYO CV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INPUT_TABLE = ROOT / "data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet"
OUTPUT_DIR = ROOT / "outputs"

EXCLUDE_COLS = {
    "patch_id", "point_id", "date", "date_start", "date_end",
    "province", "crop_type", "qc_valid", "kcact",
    "etc_8d_mm", "et0_pm_8d_mm", "qc_mod16",
    ".geo", "system:index", "valid_obs",
}

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def load_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.select_dtypes(include=["number", "bool"]).columns
        if col not in EXCLUDE_COLS and col != "year"
    ]


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def build_model(params: dict) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        thread_count=-1,
        verbose=0,
        **params,
    )


def loyo_objective(trial, df: pd.DataFrame, feature_cols: list[str], years: list[int]) -> float:
    """Optuna objective: mean LOYO R² across all years."""
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 15.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }

    folds_r2: list[float] = []
    for test_year in years:
        test = df[df["year"] == test_year]
        if len(test) < 50:
            continue

        train = df[df["year"] != test_year].copy()
        fit_years = sorted(train["year"].dropna().unique().astype(int))
        has_val = len(fit_years) >= 2

        if has_val:
            val_year = fit_years[-1]
            fit = train[train["year"] != val_year]
            val = train[train["year"] == val_year]
        else:
            fit = train
            val = None

        X_fit = fit[feature_cols].fillna(0.0)
        y_fit = fit["kcact"]
        X_test = test[feature_cols].fillna(0.0)
        y_test = test["kcact"].to_numpy()

        model = build_model(params)
        fit_kwargs = {}
        if has_val and val is not None:
            X_val = val[feature_cols].fillna(0.0)
            y_val = val["kcact"]
            fit_kwargs = {
                "eval_set": (X_val, y_val),
                "early_stopping_rounds": 50,
                "use_best_model": True,
            }

        model.fit(X_fit, y_fit, **fit_kwargs)
        y_pred = model.predict(X_test)
        folds_r2.append(r2_manual(y_test, y_pred))

    return float(np.mean(folds_r2)) if folds_r2 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick smoke test with 3 Optuna trials.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not HAS_CATBOOST:
        print("CatBoost not installed.")
        return
    if not HAS_OPTUNA:
        print("Optuna not installed.")
        return

    n_trials = 3 if args.smoke else args.n_trials

    df = pd.read_parquet(INPUT_TABLE)
    df = df[df["qc_valid"]].copy()
    feature_cols = load_feature_cols(df)
    years = sorted(df["year"].dropna().unique().astype(int))

    print(f"Features: {len(feature_cols)}")
    print(f"Years: {years}")
    print(f"Samples: {len(df)}")
    print(f"Trials: {n_trials}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: loyo_objective(trial, df, feature_cols, years),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\nBest LOYO R²: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(best, indent=2)}")

    tables_dir = OUTPUT_DIR / "tables"
    models_dir = OUTPUT_DIR / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{**best, "loyo_r2": study.best_value}]).to_csv(
        tables_dir / "kcact_catboost_tuned_params.csv",
        index=False,
    )

    model = build_model(best)
    X_all = df[feature_cols].fillna(0.0)
    y_all = df["kcact"]
    model.fit(X_all, y_all)
    model_path = models_dir / "kcact_catboost_tuned.cbm"
    model.save_model(str(model_path))
    print(f"Saved tuned params to {tables_dir / 'kcact_catboost_tuned_params.csv'}")
    print(f"Saved tuned model to {model_path}")


if __name__ == "__main__":
    main()
