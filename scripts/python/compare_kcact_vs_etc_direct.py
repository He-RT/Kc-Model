"""Compare Kcact-direct vs ETc-direct modeling approaches.

Kcact-direct:  model learns features → Kcact = ETc/ET0
ETc-direct:    model learns features → ETc, then Kc = ETc_pred / ET0

Hypothesis: ETc-direct removes ET0 denominator noise from training,
producing more stable Kc estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import sys

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INPUT_TABLE = ROOT / "data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet"
OUTPUT_DIR = ROOT / "outputs"

# Features: exclude both target columns + metadata
EXCLUDE_COLS = {
    "patch_id", "point_id", "date", "date_start", "date_end",
    "province", "crop_type", "qc_valid", "kcact",
    "etc_8d_mm", "et0_pm_8d_mm", "qc_mod16",
    ".geo", "system:index", "valid_obs",
}


def load_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.select_dtypes(include=["number", "bool"]).columns
            if col not in EXCLUDE_COLS and col != "year"]


def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": len(y_true),
    }


def make_catboost():
    return CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        random_seed=42, verbose=0, thread_count=-1)


def main():
    df = pd.read_parquet(INPUT_TABLE)
    df = df[df["qc_valid"]].copy()
    feature_cols = load_feature_cols(df)
    years = sorted(df["year"].dropna().unique().astype(int))
    print(f"Features: {len(feature_cols)}")
    print(f"Years: {years}, Samples: {len(df)}")

    # ---- Approach 1: Kcact-direct (baseline) ----
    print("\n=== Approach 1: Kcact-direct (CatBoost) ===")
    kc_results = []
    y_all_true_kc, y_all_pred_kc = [], []

    for test_year in years:
        train = df[df["year"] != test_year]
        test = df[df["year"] == test_year]

        X_train = train[feature_cols].fillna(0.0)
        y_train = train["kcact"].values
        X_test = test[feature_cols].fillna(0.0)
        y_test = test["kcact"].values

        m = make_catboost()
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        e = evaluate("KC_direct", y_test, y_pred)
        e["test_year"] = test_year
        kc_results.append(e)
        print(f"  {test_year}: R²={e['r2']:.4f} RMSE={e['rmse']:.4f} MAE={e['mae']:.4f}  n={len(y_test)}")

        y_all_true_kc.extend(y_test)
        y_all_pred_kc.extend(y_pred)

    yt_kc = np.array(y_all_true_kc)
    yp_kc = np.array(y_all_pred_kc)
    kc_pooled = evaluate("KC_direct", yt_kc, yp_kc)
    print(f"  POOLED: R²={kc_pooled['r2']:.4f} RMSE={kc_pooled['rmse']:.4f} MAE={kc_pooled['mae']:.4f}")

    # ---- Approach 2: ETc-direct ----
    print("\n=== Approach 2: ETc-direct → Kc = ETc_pred / ET0 ===")
    etc_results = []
    y_all_true_etc, y_all_pred_etc = [], []
    y_all_et0 = []

    for test_year in years:
        train = df[df["year"] != test_year]
        test = df[df["year"] == test_year]

        X_train = train[feature_cols].fillna(0.0)
        y_train_etc = train["etc_8d_mm"].values  # target: raw ETc
        X_test = test[feature_cols].fillna(0.0)
        y_test_etc = test["etc_8d_mm"].values
        et0_test = test["et0_pm_8d_mm"].values

        m = make_catboost()
        m.fit(X_train, y_train_etc)
        etc_pred = m.predict(X_test)

        # Convert to Kc space for evaluation
        kc_true = y_test_etc / et0_test
        kc_pred = etc_pred / et0_test

        e = evaluate("ETC_direct", kc_true, kc_pred)
        e["test_year"] = test_year
        etc_results.append(e)
        print(f"  {test_year}: R²={e['r2']:.4f} RMSE={e['rmse']:.4f} MAE={e['mae']:.4f}  n={len(kc_true)}")

        y_all_true_etc.extend(kc_true)
        y_all_pred_etc.extend(kc_pred)
        y_all_et0.extend(et0_test)

    yt_etc = np.array(y_all_true_etc)
    yp_etc = np.array(y_all_pred_etc)
    etc_pooled = evaluate("ETC_direct", yt_etc, yp_etc)
    print(f"  POOLED: R²={etc_pooled['r2']:.4f} RMSE={etc_pooled['rmse']:.4f} MAE={etc_pooled['mae']:.4f}")

    # ---- Per-stage comparison ----
    print("\n=== Per-Stage Comparison ===")
    ndvi_arr = df["ndvi"].values

    for stage_name, lo, hi in [("early", 0, 0.35), ("mid", 0.35, 0.60), ("late", 0.60, 1.0)]:
        mask = (ndvi_arr >= lo) & (ndvi_arr < hi)
        mask_indices = np.where(mask)[0]

        kc_direct_r2 = r2_score(yt_kc[mask_indices], yp_kc[mask_indices])
        etc_direct_r2 = r2_score(yt_etc[mask_indices], yp_etc[mask_indices])
        delta = etc_direct_r2 - kc_direct_r2
        n = mask.sum()
        print(f"  {stage_name:5s} (n={n:4d}): KC_direct={kc_direct_r2:.4f}  "
              f"ETC_direct={etc_direct_r2:.4f}  Δ={delta:+.4f}")

    # ---- Final comparison ----
    print(f"\n{'='*60}")
    print("=== FINAL COMPARISON (Kc space) ===")
    print(f"  Kcact-direct:  R²={kc_pooled['r2']:.4f}  RMSE={kc_pooled['rmse']:.4f}  MAE={kc_pooled['mae']:.4f}")
    print(f"  ETc-direct:    R²={etc_pooled['r2']:.4f}  RMSE={etc_pooled['rmse']:.4f}  MAE={etc_pooled['mae']:.4f}")
    delta = etc_pooled['r2'] - kc_pooled['r2']
    print(f"  Δ:             R²={delta:+.4f}  RMSE={etc_pooled['rmse']-kc_pooled['rmse']:+.4f}  MAE={etc_pooled['mae']-kc_pooled['mae']:+.4f}")

    if delta > 0.001:
        print("\n  *** ETc-direct WINS ***")
    elif delta < -0.001:
        print("\n  *** Kcact-direct WINS ***")
    else:
        print("\n  *** TIED — no meaningful difference ***")

    # Save
    out = OUTPUT_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kc_results + etc_results).to_csv(out / "kcact_etc_direct_comparison.csv", index=False)
    print(f"\nResults saved to {out}/kcact_etc_direct_comparison.csv")


if __name__ == "__main__":
    main()
