"""Hyperparameter tuning for XGBoost Kcact model using Optuna or randomized search."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

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
}

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def load_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.select_dtypes(include=["number", "bool"]).columns
            if col not in EXCLUDE_COLS and col != "year"]


def loyo_objective(trial, df, feature_cols, years):
    """Optuna objective: mean LOYO R² across all years."""
    folds_r2 = []
    for test_year in years:
        train = df[df["year"] != test_year]
        test = df[df["year"] == test_year]
        if len(test) < 50:
            continue

        X_train = train[feature_cols].fillna(0.0)
        y_train = train["kcact"]
        X_test = test[feature_cols].fillna(0.0)
        y_test = test["kcact"]

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "random_state": 42, "n_jobs": -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        ss_res = ((y_test.values - y_pred) ** 2).sum()
        ss_tot = ((y_test.values - y_test.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        folds_r2.append(r2)

    return float(np.mean(folds_r2)) if folds_r2 else 0.0


def main():
    if not HAS_XGB:
        print("XGBoost not installed.")
        return

    df = pd.read_parquet(INPUT_TABLE)
    df = df[df["qc_valid"]].copy()
    feature_cols = load_feature_cols(df)
    years = sorted(df["year"].dropna().unique().astype(int))

    print(f"Features: {len(feature_cols)}")
    print(f"Years: {years}")
    print(f"Samples: {len(df)}")

    if HAS_OPTUNA:
        print("\nRunning Optuna hyperparameter optimization (50 trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: loyo_objective(trial, df, feature_cols, years),
            n_trials=50, show_progress_bar=True,
        )
        best = study.best_params
        print(f"\nBest LOYO R²: {study.best_value:.4f}")
        print(f"Best params: {json.dumps(best, indent=2)}")

        # Save
        out = OUTPUT_DIR / "tables"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{**best, "loyo_r2": study.best_value}]).to_csv(
            out / "kcact_xgb_tuned_params.csv", index=False)
    else:
        print("Optuna not installed. Running randomized search...")
        from sklearn.model_selection import RandomizedSearchCV

        param_dist = {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [4, 5, 6, 7, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 1.0],
            "reg_lambda": [0, 0.1, 1.0],
        }

        # Use a representative train/test split for search
        train = df[df["year"] != years[-1]]
        test = df[df["year"] == years[-1]]
        X_train = train[feature_cols].fillna(0.0)
        y_train = train["kcact"]

        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            model, param_dist, n_iter=50, cv=3, scoring="r2",
            random_state=42, n_jobs=-1, verbose=1,
        )
        search.fit(X_train, y_train)
        print(f"\nBest R²: {search.best_score_:.4f}")
        print(f"Best params: {json.dumps(search.best_params_, indent=2)}")


if __name__ == "__main__":
    main()
