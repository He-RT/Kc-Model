"""Leave-one-year-out cross-validation for Kcact models.

Runs both RF and XGBoost, each year held out as test set once.
Reports per-year metrics + summary stats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import sys
import json

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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


def load_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.select_dtypes(include=["number", "bool"]).columns
            if col not in EXCLUDE_COLS and col != "year"]


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": len(y_true),
    }


def main():
    df = pd.read_parquet(INPUT_TABLE)
    df = df[df["qc_valid"]].copy()
    feature_cols = load_feature_cols(df)
    years = sorted(df["year"].dropna().unique().astype(int))
    print(f"Years: {years}")
    print(f"Features: {len(feature_cols)}")
    print(f"Total valid samples: {len(df)}")

    rf_results, xgb_results = [], []
    y_all_true, y_all_pred_rf, y_all_pred_xgb = [], [], []

    for test_year in years:
        train = df[df["year"] != test_year]
        test = df[df["year"] == test_year]

        X_train = train[feature_cols].fillna(0.0)
        y_train = train["kcact"]
        X_test = test[feature_cols].fillna(0.0)
        y_test = test["kcact"]

        print(f"\n--- Test year: {test_year} | Train: {len(X_train)}, Test: {len(X_test)} ---")

        # RF
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        m_rf = evaluate("RF", y_test, y_pred_rf)
        m_rf["test_year"] = test_year
        rf_results.append(m_rf)
        print(f"  RF   R²={m_rf['r2']:.4f}  RMSE={m_rf['rmse']:.4f}  MAE={m_rf['mae']:.4f}")
        importances = sorted(zip(rf.feature_names_in_, rf.feature_importances_),
                             key=lambda x: x[1], reverse=True)[:5]
        print(f"       Top-5: {', '.join(f'{n}({v:.3f})' for n,v in importances)}")

        # XGB
        if HAS_XGB:
            xgb_m = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            )
            xgb_m.fit(X_train, y_train)
            y_pred_xgb = xgb_m.predict(X_test)
            m_xgb = evaluate("XGB", y_test, y_pred_xgb)
            m_xgb["test_year"] = test_year
            xgb_results.append(m_xgb)
            print(f"  XGB  R²={m_xgb['r2']:.4f}  RMSE={m_xgb['rmse']:.4f}  MAE={m_xgb['mae']:.4f}")
            importances_x = sorted(zip(xgb_m.feature_names_in_, xgb_m.feature_importances_),
                                   key=lambda x: x[1], reverse=True)[:5]
            print(f"       Top-5: {', '.join(f'{n}({v:.3f})' for n,v in importances_x)}")

            y_all_true.extend(y_test.values)
            y_all_pred_rf.extend(y_pred_rf)
            y_all_pred_xgb.extend(y_pred_xgb)

    # Overall pooled
    yt = np.array(y_all_true)
    print(f"\n=== Overall (all years pooled) ===")
    over_rf = evaluate("RF", yt, np.array(y_all_pred_rf))
    print(f"  RF   R²={over_rf['r2']:.4f}  RMSE={over_rf['rmse']:.4f}  MAE={over_rf['mae']:.4f}")
    if HAS_XGB:
        over_xgb = evaluate("XGB", yt, np.array(y_all_pred_xgb))
        print(f"  XGB  R²={over_xgb['r2']:.4f}  RMSE={over_xgb['rmse']:.4f}  MAE={over_xgb['mae']:.4f}")

    # Save
    out = OUTPUT_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rf_results).to_csv(out / "kcact_loyo_rf.csv", index=False)
    if HAS_XGB:
        pd.DataFrame(xgb_results).to_csv(out / "kcact_loyo_xgb.csv", index=False)
    print(f"\nPer-year results saved to {out}/kcact_loyo_*.csv")


if __name__ == "__main__":
    main()
