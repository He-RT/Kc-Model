"""Run feature-subset experiments on ERA5-like 0.1° maize grid table.

Driven by SHAP results for the saved 7-feature CatBoost model.  Main search:
keep DOY+SM and enumerate all non-empty subsets of the five vegetation indices.
Also add a few diagnostic controls without DOY or without SM.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/train/ncp_summer_maize_selected_indicators_era5grid.parquet"
OUT = ROOT / "outputs/tables/maize_era5grid_feature_combos.csv"

VI = ["ndvi", "savi", "rdvi", "gndvi", "evi"]
CORE = ["sm", "doy"]


def make_model(task_type: str | None = "GPU") -> CatBoostRegressor:
    params = dict(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
    if task_type:
        params.update(task_type=task_type, devices="0")
    return CatBoostRegressor(**params)


def eval_combo(df: pd.DataFrame, features: list[str], combo_name: str) -> dict:
    sub = df.dropna(subset=features + ["kcact"]).copy()
    years = sorted(sub["year"].astype(int).unique())
    all_y, all_p = [], []
    rows = []
    task_type: str | None = "GPU"
    for yr in years:
        tr = sub[sub["year"] != yr]
        te = sub[sub["year"] == yr]
        if len(te) < 10 or len(tr) < 10:
            continue
        model = make_model(task_type)
        try:
            model.fit(tr[features], tr["kcact"])
        except Exception as exc:
            if task_type == "GPU":
                print(f"GPU failed ({exc}); retrying CPU")
                task_type = None
                model = make_model(task_type)
                model.fit(tr[features], tr["kcact"])
            else:
                raise
        pred = model.predict(te[features])
        y = te["kcact"].to_numpy()
        rows.append({
            "test_year": int(yr),
            "r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "n_test": int(len(te)),
        })
        all_y.extend(y)
        all_p.extend(pred)
    y = np.asarray(all_y)
    p = np.asarray(all_p)
    return {
        "combo": combo_name,
        "features": "+".join(features),
        "n_features": len(features),
        "n_samples": int(len(sub)),
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "task_type": task_type or "CPU",
        "per_year": rows,
    }


def build_combos() -> list[tuple[str, list[str]]]:
    combos: list[tuple[str, list[str]]] = []
    # Main SHAP-driven search: always keep SM + DOY, vary VI set.
    for k in range(1, len(VI) + 1):
        for subset in itertools.combinations(VI, k):
            feats = list(subset) + CORE
            combos.append(("core_" + "+".join(feats), feats))
    # Add a no-VI baseline and a few controls to verify indispensability.
    controls = [
        ("core_sm_doy_only", ["sm", "doy"]),
        ("best_shap_ndvi_evi_sm_doy", ["ndvi", "evi", "sm", "doy"]),
        ("ndvi_sm_doy", ["ndvi", "sm", "doy"]),
        ("evi_sm_doy", ["evi", "sm", "doy"]),
        ("ndvi_evi_gndvi_sm_doy", ["ndvi", "evi", "gndvi", "sm", "doy"]),
        ("all7_no_savi", ["ndvi", "rdvi", "gndvi", "evi", "sm", "doy"]),
        ("all7_no_rdvi", ["ndvi", "savi", "gndvi", "evi", "sm", "doy"]),
        ("all7_no_gndvi", ["ndvi", "savi", "rdvi", "evi", "sm", "doy"]),
        ("all7_no_low3", ["ndvi", "evi", "sm", "doy"]),
        ("vi_only_all", VI),
        ("vi_plus_doy_no_sm", VI + ["doy"]),
        ("vi_plus_sm_no_doy", VI + ["sm"]),
    ]
    seen = {tuple(f) for _, f in combos}
    for name, feats in controls:
        t = tuple(feats)
        if t not in seen:
            combos.append((name, feats))
            seen.add(t)
    return combos


def main() -> None:
    df = pd.read_parquet(DATA)
    print(f"Loaded {DATA}: {len(df):,} rows")
    combos = build_combos()
    print(f"Running {len(combos)} combos")
    results = []
    for i, (name, feats) in enumerate(combos, 1):
        print(f"[{i:02d}/{len(combos)}] {name}: {feats}")
        res = eval_combo(df, feats, name)
        print(f"  R²={res['r2']:.5f} RMSE={res['rmse']:.5f} MAE={res['mae']:.5f} n={res['n_samples']:,}")
        # Keep per-year compact fields for sorting CSV friendliness.
        per_year = res.pop("per_year")
        for yr in per_year:
            res[f"r2_{yr['test_year']}"] = yr["r2"]
        results.append(res)
    out = pd.DataFrame(results).sort_values(["r2", "n_features"], ascending=[False, True])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("\n=== Top 15 ===")
    print(out.head(15).to_string(index=False))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
