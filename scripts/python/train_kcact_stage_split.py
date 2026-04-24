"""Stage-split Kcact models: train separate XGB/CatBoost per NDVI stage.

Splits data into 3 phenology stages by NDVI:
  early: NDVI < 0.35 (emergence, early vegetative)
  mid:   0.35 <= NDVI < 0.60 (peak vegetative)
  late:  NDVI >= 0.60 (senescence, harvest)

Each stage gets its own model trained on LOYO CV.
"""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import sys

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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

EXCLUDE_COLS = {
    "patch_id", "point_id", "date", "date_start", "date_end",
    "province", "crop_type", "qc_valid", "kcact",
    "etc_8d_mm", "et0_pm_8d_mm", "qc_mod16",
    ".geo", "system:index", "valid_obs",
}


def load_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.select_dtypes(include=["number", "bool"]).columns
            if col not in EXCLUDE_COLS and col != "year"]


def ndvi_stage_mask(df, stage):
    ndvi = df["ndvi"].values
    if stage == "early":
        return ndvi < 0.35
    elif stage == "mid":
        return (ndvi >= 0.35) & (ndvi < 0.60)
    else:
        return ndvi >= 0.60


def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": len(y_true),
    }


def make_model(model_type):
    if model_type == "xgb":
        return xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1)
    elif model_type == "catboost":
        return CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=42, verbose=0, thread_count=-1)


STAGES = ["early", "mid", "late"]


def main():
    df = pd.read_parquet(INPUT_TABLE)
    df = df[df["qc_valid"]].copy()
    feature_cols = load_feature_cols(df)
    years = sorted(df["year"].dropna().unique().astype(int))
    print(f"Features: {len(feature_cols)}")
    print(f"Years: {years}, Total valid: {len(df)}")

    model_types = []
    if HAS_XGB:
        model_types.append("xgb")
    if HAS_CATBOOST:
        model_types.append("catboost")

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"=== Stage-Split {model_type.upper()} ===")
        print(f"{'='*60}")

        # Collect all predictions
        stage_preds = {s: {"true": [], "pred": []} for s in STAGES}
        year_results = []

        for test_year in years:
            test = df[df["year"] == test_year]
            train = df[df["year"] != test_year]

            y_test_all = test["kcact"].values
            y_pred_all = np.full(len(y_test_all), np.nan)

            for stage in STAGES:
                train_mask = ndvi_stage_mask(train, stage)
                test_mask = ndvi_stage_mask(test, stage)

                if test_mask.sum() == 0 or train_mask.sum() < 10:
                    continue

                X_train_s = train.loc[train_mask, feature_cols].fillna(0.0)
                y_train_s = train.loc[train_mask, "kcact"].values
                X_test_s = test.loc[test_mask, feature_cols].fillna(0.0)
                y_test_s = test.loc[test_mask, "kcact"].values

                model_s = make_model(model_type)
                model_s.fit(X_train_s, y_train_s)
                y_pred_s = model_s.predict(X_test_s)

                y_pred_all[test_mask] = y_pred_s
                stage_preds[stage]["true"].extend(y_test_s.tolist())
                stage_preds[stage]["pred"].extend(y_pred_s.tolist())

            valid_mask = ~np.isnan(y_pred_all)
            if valid_mask.sum() == 0:
                print(f"  {test_year}: SKIP (no valid stage predictions)")
                continue
            yr_res = evaluate(model_type.upper(), y_test_all[valid_mask], y_pred_all[valid_mask])
            yr_res["test_year"] = test_year
            year_results.append(yr_res)
            print(f"  {test_year}: R²={yr_res['r2']:.4f} RMSE={yr_res['rmse']:.4f} MAE={yr_res['mae']:.4f}  n={yr_res['n_samples']}")

        # Per-stage metrics
        print(f"\n  --- Per-Stage ---")
        for stage in STAGES:
            yt = np.array(stage_preds[stage]["true"])
            yp = np.array(stage_preds[stage]["pred"])
            if len(yt) > 0:
                print(f"  {stage:5s}: n={len(yt):4d}  R²={r2_score(yt,yp):.4f}  "
                      f"RMSE={np.sqrt(mean_squared_error(yt,yp)):.4f}  "
                      f"MAE={mean_absolute_error(yt,yp):.4f}  mean_Kc={yt.mean():.3f}")

        # Overall pooled
        yt_all = np.concatenate([np.array(stage_preds[s]["true"]) for s in STAGES])
        yp_all = np.concatenate([np.array(stage_preds[s]["pred"]) for s in STAGES])
        overall = evaluate(f"Stage-{model_type.upper()}", yt_all, yp_all)
        print(f"\n  POOLED: R²={overall['r2']:.4f} RMSE={overall['rmse']:.4f} MAE={overall['mae']:.4f}")

        # Save per-year results
        out = OUTPUT_DIR / "tables"
        out.mkdir(parents=True, exist_ok=True)
        yr_df = pd.DataFrame(year_results)
        yr_df.to_csv(out / f"kcact_stage_split_{model_type}_loyo.csv", index=False)

    # ---- Comparison: single model vs stage-split ----
    print(f"\n{'='*60}")
    print("=== STAGE-SPLIT vs SINGLE MODEL ===")
    print(f"{'='*60}")
    # Read v2 results for comparison
    v2_summary = pd.read_csv(out / "kcact_loyo_v2_summary.csv")
    single_best = v2_summary[v2_summary["model"].str.contains("CatBoost|XGB_raw|XGB_log")].copy()
    for _, row in single_best.iterrows():
        print(f"  {row['model']:20s}  R²={row['r2']:.4f}  RMSE={row['rmse']:.4f}  MAE={row['mae']:.4f}")

    print("\nStage-split results saved to outputs/tables/kcact_stage_split_*_loyo.csv")


if __name__ == "__main__":
    main()
