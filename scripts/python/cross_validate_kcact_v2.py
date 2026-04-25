"""Enhanced LOYO cross-validation with log-transform, CatBoost, and stage-split.

Runs multiple model variants and compares:
1. XGBoost baseline (raw Kcact)
2. XGBoost log-transform (train on log Kcact, predict exp)
3. CatBoost (raw Kcact)
4. Stage-split analysis: per-NDVI-stage metrics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

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


def run_loyo(df, feature_cols, years, model_name, make_model_fn, transform_y_fn=None):
    """Run LOYO CV with a given model factory.

    If transform_y_fn is provided, it maps (y_train, y_test) to transformed
    versions and the inverse is untransform_y_fn applied to predictions.
    """
    results = []
    y_all_true, y_all_pred = [], []

    for test_year in years:
        train = df[df["year"] != test_year]
        test = df[df["year"] == test_year]

        X_train = train[feature_cols].fillna(0.0)
        y_train_raw = train["kcact"].values
        X_test = test[feature_cols].fillna(0.0)
        y_test_raw = test["kcact"].values

        if transform_y_fn == "log":
            y_train = np.log(np.clip(y_train_raw, 0.01, None))
            y_test = y_test_raw  # metrics computed on raw scale
        else:
            y_train = y_train_raw
            y_test = y_test_raw

        model = make_model_fn()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if transform_y_fn == "log":
            y_pred = np.exp(np.clip(y_pred, -2.0, 2.0))

        m = evaluate(model_name, y_test_raw, y_pred)
        m["test_year"] = test_year
        results.append(m)

        y_all_true.extend(y_test_raw)
        y_all_pred.extend(y_pred)

    yt = np.array(y_all_true)
    yp = np.array(y_all_pred)
    overall = evaluate(model_name, yt, yp)
    overall["test_year"] = "pooled"
    return results, overall


def ndvi_stage(ndvi: float) -> str:
    if ndvi < 0.35:
        return "early"
    elif ndvi < 0.60:
        return "mid"
    else:
        return "late"


def main():
    df = pd.read_parquet(INPUT_TABLE)
    df = df[df["qc_valid"]].copy()
    feature_cols = load_feature_cols(df)
    years = sorted(df["year"].dropna().unique().astype(int))
    print(f"Years: {years}")
    print(f"Features ({len(feature_cols)}): {', '.join(sorted(feature_cols))}")
    print(f"Total valid samples: {len(df)}")

    all_per_year = []
    all_overall = []

    # ---- 1. XGBoost baseline (raw) ----
    if HAS_XGB:
        print("\n=== XGBoost Baseline ===")
        per, overall = run_loyo(df, feature_cols, years, "XGB_raw",
                               lambda: xgb.XGBRegressor(
                                   n_estimators=200, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, n_jobs=-1))
        for r in per:
            print(f"  {r['test_year']}: R²={r['r2']:.4f} RMSE={r['rmse']:.4f} MAE={r['mae']:.4f}")
        print(f"  POOLED: R²={overall['r2']:.4f} RMSE={overall['rmse']:.4f} MAE={overall['mae']:.4f}")
        all_per_year.extend(per)
        all_overall.append(overall)

    # ---- 2. XGBoost log-transform ----
    if HAS_XGB:
        print("\n=== XGBoost Log-Transform ===")
        per, overall = run_loyo(df, feature_cols, years, "XGB_log",
                               lambda: xgb.XGBRegressor(
                                   n_estimators=200, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, n_jobs=-1),
                               transform_y_fn="log")
        for r in per:
            print(f"  {r['test_year']}: R²={r['r2']:.4f} RMSE={r['rmse']:.4f} MAE={r['mae']:.4f}")
        print(f"  POOLED: R²={overall['r2']:.4f} RMSE={overall['rmse']:.4f} MAE={overall['mae']:.4f}")
        all_per_year.extend(per)
        all_overall.append(overall)

    # ---- 3. CatBoost ----
    if HAS_CATBOOST:
        print("\n=== CatBoost ===")
        per, overall = run_loyo(df, feature_cols, years, "CatBoost",
                               lambda: CatBoostRegressor(
                                   iterations=500, depth=6, learning_rate=0.05,
                                   random_seed=42, verbose=0, thread_count=-1))
        for r in per:
            print(f"  {r['test_year']}: R²={r['r2']:.4f} RMSE={r['rmse']:.4f} MAE={r['mae']:.4f}")
        print(f"  POOLED: R²={overall['r2']:.4f} RMSE={overall['rmse']:.4f} MAE={overall['mae']:.4f}")
        all_per_year.extend(per)
        all_overall.append(overall)

    # ---- 4. Stage-split analysis (XGB) ----
    if HAS_XGB:
        print("\n=== Stage-Split Analysis (XGB) ===")
        stages = {"early": [], "mid": [], "late": []}
        stages_y = {"early": [], "mid": [], "late": []}

        for test_year in years:
            train = df[df["year"] != test_year]
            test = df[df["year"] == test_year]

            X_train = train[feature_cols].fillna(0.0)
            y_train = train["kcact"].values
            X_test = test[feature_cols].fillna(0.0)
            y_test = test["kcact"].values

            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            for i in range(len(y_test)):
                stage = ndvi_stage(test["ndvi"].values[i])
                stages[stage].append(y_pred[i])
                stages_y[stage].append(y_test[i])

        for stage_name in ["early", "mid", "late"]:
            if stages[stage_name]:
                yt_s = np.array(stages_y[stage_name])
                yp_s = np.array(stages[stage_name])
                print(f"  {stage_name}: n={len(yt_s):4d}  "
                      f"R²={r2_score(yt_s, yp_s):.4f}  "
                      f"RMSE={np.sqrt(mean_squared_error(yt_s, yp_s)):.4f}  "
                      f"MAE={mean_absolute_error(yt_s, yp_s):.4f}  "
                      f"mean Kc={yt_s.mean():.3f}")

    # ---- 5. Tuned CatBoost (from params file) ----
    if HAS_CATBOOST:
        print("\n=== Tuned CatBoost ===")
        # Load tuned params if available
        tuned_params = {}
        tuned_params_path = ROOT / "outputs/tables/kcact_catboost_tuned_params.csv"
        if tuned_params_path.exists():
            tuned_df = pd.read_csv(tuned_params_path)
            if not tuned_df.empty:
                tuned_params = tuned_df.iloc[0].to_dict()
                # Remove non-CatBoost keys
                for k in ["loyo_r2", "Unnamed: 0"]:
                    tuned_params.pop(k, None)
                print(f"  Loaded tuned params: {tuned_params}")
            else:
                print("  WARNING: tuned params file exists but is empty, using defaults")
        else:
            print("  WARNING: tuned params file not found, using defaults")
            tuned_params = {}

        def make_tuned_catboost():
            params = dict(iterations=500, depth=6, learning_rate=0.05,
                         random_seed=42, verbose=0, thread_count=-1)
            # Override with tuned params if available
            for k, v in tuned_params.items():
                # Convert Optuna suggest_* keys to CatBoost constructor keys
                if k == "n_estimators":
                    params["iterations"] = int(v)
                elif k == "max_depth":
                    params["depth"] = int(v)
                elif k == "learning_rate":
                    params["learning_rate"] = float(v)
                elif k == "subsample":
                    params["subsample"] = float(v)
                elif k == "l2_leaf_reg":
                    params["l2_leaf_reg"] = float(v)
                elif k == "random_strength":
                    params["random_strength"] = float(v)
                elif k == "min_data_in_leaf":
                    params["min_data_in_leaf"] = int(v)
                elif k == "border_count":
                    params["border_count"] = int(v)
                elif k == "min_child_weight":
                    params["min_data_in_leaf"] = int(v)
                elif k == "reg_alpha" or k == "reg_lambda":
                    pass  # CatBoost uses l2_leaf_reg instead
            return CatBoostRegressor(**params)

        per, overall = run_loyo(df, feature_cols, years, "TunedCatBoost",
                               make_tuned_catboost)
        for r in per:
            print(f"  {r['test_year']}: R²={r['r2']:.4f} RMSE={r['rmse']:.4f} MAE={r['mae']:.4f}")
        print(f"  POOLED: R²={overall['r2']:.4f} RMSE={overall['rmse']:.4f} MAE={overall['mae']:.4f}")
        all_per_year.extend(per)
        all_overall.append(overall)

    # ---- 6. LightGBM Baseline ----
    if HAS_LGBM:
        print("\n=== LightGBM ===")
        per, overall = run_loyo(df, feature_cols, years, "LightGBM",
                               lambda: LGBMRegressor(
                                   n_estimators=300, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, n_jobs=-1, verbose=-1))
        for r in per:
            print(f"  {r['test_year']}: R²={r['r2']:.4f} RMSE={r['rmse']:.4f} MAE={r['mae']:.4f}")
        print(f"  POOLED: R²={overall['r2']:.4f} RMSE={overall['rmse']:.4f} MAE={overall['mae']:.4f}")
        all_per_year.extend(per)
        all_overall.append(overall)
    else:
        print("\n=== LightGBM not available ===")

    # ---- 7. Simple Average Ensemble ----
    if HAS_XGB or HAS_CATBOOST or HAS_LGBM:
        print("\n=== Simple Average Ensemble ===")
        models_for_ensemble = []
        names_for_ensemble = []
        if HAS_XGB:
            models_for_ensemble.append(lambda: xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1))
            names_for_ensemble.append("XGB")
        if HAS_CATBOOST:
            models_for_ensemble.append(lambda: CatBoostRegressor(
                iterations=500, depth=6, learning_rate=0.05,
                random_seed=42, verbose=0, thread_count=-1))
            names_for_ensemble.append("CB")
        if HAS_LGBM:
            models_for_ensemble.append(lambda: LGBMRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1))
            names_for_ensemble.append("LGBM")

        y_all_true, y_all_pred = [], []
        for test_year in years:
            train = df[df["year"] != test_year]
            test = df[df["year"] == test_year]
            X_train = train[feature_cols].fillna(0.0)
            y_train = train["kcact"].values
            X_test = test[feature_cols].fillna(0.0)
            y_test = test["kcact"].values

            predictions = []
            for mk_model in models_for_ensemble:
                m = mk_model()
                m.fit(X_train, y_train)
                predictions.append(m.predict(X_test))
            y_pred_avg = np.mean(predictions, axis=0)

            m = evaluate(f"Ensemble_Avg({'/'.join(names_for_ensemble)})",
                        y_test, y_pred_avg)
            m["test_year"] = test_year
            all_per_year.append(m)
            y_all_true.extend(y_test)
            y_all_pred.extend(y_pred_avg)

        yt = np.array(y_all_true)
        yp_all = np.array(y_all_pred)
        overall = evaluate(f"Ensemble_Avg({'/'.join(names_for_ensemble)})",
                          yt, yp_all)
        overall["test_year"] = "pooled"
        all_overall.append(overall)
        print(f"  POOLED: R²={overall['r2']:.4f} RMSE={overall['rmse']:.4f} MAE={overall['mae']:.4f}")

    # ---- Summary ----
    print("\n=== Overall Summary ===")
    for ov in all_overall:
        print(f"  {ov['model']:12s}  R²={ov['r2']:.4f}  RMSE={ov['rmse']:.4f}  MAE={ov['mae']:.4f}")

    # Save
    out = OUTPUT_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_per_year).to_csv(out / "kcact_loyo_v2_detail.csv", index=False)
    pd.DataFrame(all_overall).to_csv(out / "kcact_loyo_v2_summary.csv", index=False)
    print(f"\nResults saved to {out}/kcact_loyo_v2_*.csv")


if __name__ == "__main__":
    main()
