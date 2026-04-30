#!/usr/bin/env python3
"""Train CatBoost model on station Kcact with 8 RS indicators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "station_ml_features.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"

FEATURES_RS = ["ndvi", "evi", "lswi", "lst_day", "albedo_sw", "fpar", "delta_lst", "sm_surface"]
FEATURES_TIME = ["doy", "year"]
FEATURES_ALL = FEATURES_RS + FEATURES_TIME
TARGET = "kcact"

def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year

    # Drop rows with NaN in target or too many NaN features
    model_df = df.dropna(subset=[TARGET]).copy()
    # Fill remaining feature NaNs with per-station median
    for col in FEATURES_RS:
        if model_df[col].isna().any():
            model_df[col] = model_df.groupby("station")[col].transform(
                lambda x: x.fillna(x.median())
            )
    model_df = model_df.dropna(subset=FEATURES_RS)
    print(f"Training samples: {len(model_df)}")

    X = model_df[FEATURES_ALL].values
    y = model_df[TARGET].values
    stations = model_df["station"].values

    # ---- 1. Leave-One-Station-Out CV ----
    print("\n--- Leave-One-Station-Out CV ---")
    station_folds = []
    for stn in np.unique(stations):
        train_idx = stations != stn
        test_idx = stations == stn
        station_folds.append((np.where(train_idx)[0], np.where(test_idx)[0]))

    loso_preds = np.zeros(len(y))
    loso_actuals = y.copy()

    for train_idx, test_idx in station_folds:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
        model.fit(X_tr, y_tr)
        loso_preds[test_idx] = model.predict(X_te)

    r2_loso = r2_score(loso_actuals, loso_preds)
    rmse_loso = np.sqrt(mean_squared_error(loso_actuals, loso_preds))
    print(f"  LOSO pooled R²  = {r2_loso:.4f}")
    print(f"  LOSO pooled RMSE = {rmse_loso:.4f}")

    # Per-station LOSO R²
    print("  Per-station:")
    for stn in np.unique(stations):
        mask = stations == stn
        if mask.sum() > 5:
            r2_stn = r2_score(y[mask], loso_preds[mask])
            rmse_stn = np.sqrt(mean_squared_error(y[mask], loso_preds[mask]))
            print(f"    {stn}: R²={r2_stn:.4f}, RMSE={rmse_stn:.4f}, n={mask.sum()}")

    # ---- 2. 4-Fold CV (shuffle split by station) ----
    print("\n--- 4-Fold CV ---")
    model_cv = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=5, l2_leaf_reg=3,
        loss_function="RMSE", random_seed=42, verbose=False,
    )
    cv_scores = cross_val_score(model_cv, X, y, cv=4, scoring="r2")
    print(f"  4-Fold R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ---- 3. Feature importance ----
    print("\n--- Feature Importance ---")
    final_model = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=5, l2_leaf_reg=3,
        loss_function="RMSE", random_seed=42, verbose=False,
    )
    final_model.fit(X, y)
    importance = final_model.get_feature_importance()
    feat_imp = sorted(zip(FEATURES_ALL, importance), key=lambda x: -x[1])
    for name, imp in feat_imp:
        pct = imp / importance.sum() * 100
        bar = "█" * int(pct)
        print(f"  {name:12s}  {pct:5.1f}%  {bar}")

    # ---- 4. Naive baseline (just mean Kcact) ----
    y_mean = np.full_like(y, y.mean())
    r2_naive = r2_score(y, y_mean)
    print(f"\n  Naive (mean) R² = {r2_naive:.4f}")

    # Save results
    results = pd.DataFrame({
        "metric": ["LOSO_R2", "LOSO_RMSE", "4Fold_R2_mean", "4Fold_R2_std", "Naive_R2"],
        "value": [r2_loso, rmse_loso, cv_scores.mean(), cv_scores.std(), r2_naive],
    })
    out = OUT_DIR / "station_ml_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nResults saved to {out}")

    # Per-station results
    stn_results = []
    for stn in np.unique(stations):
        mask = stations == stn
        if mask.sum() > 5:
            stn_results.append({
                "station": stn,
                "n": mask.sum(),
                "R2": r2_score(y[mask], loso_preds[mask]),
                "RMSE": np.sqrt(mean_squared_error(y[mask], loso_preds[mask])),
            })
    stn_df = pd.DataFrame(stn_results)
    stn_out = OUT_DIR / "station_ml_per_station.csv"
    stn_df.to_csv(stn_out, index=False)
    print(f"Per-station results saved to {stn_out}")


if __name__ == "__main__":
    main()
