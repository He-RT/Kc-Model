"""Aggregate selected maize indicators to ERA5-Land-like 0.1° grid and train.

This is an experimental "unify to the coarsest spatial resolution" run.
With the currently available point-sampled GEE exports, we cannot reconstruct
full raster area-weighted 0.1° cells.  Instead, we aggregate all crop sample
points that fall in the same 0.1° lat/lon cell and MOD16 8-day window:

* features: mean NDVI/SAVI/RDVI/GNDVI/EVI/SM/DOY within the cell-window
* target: mean(MOD16 ETa 8d) / mean(ERA5 ET0 8d) within the cell-window

This tests whether a coarser, ERA5-resolution representation changes model
performance relative to the point-aligned table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
SELECTED = ROOT / "data/processed/train/ncp_summer_maize_selected_indicators.parquet"
OUT_PARQUET = ROOT / "data/processed/train/ncp_summer_maize_selected_indicators_era5grid.parquet"
OUT_CSV = ROOT / "outputs/tables/maize_selected_indicators_era5grid_loyo.csv"

FEATURES = ["ndvi", "savi", "rdvi", "gndvi", "evi", "sm", "doy"]


def build_grid_table() -> pd.DataFrame:
    selected_cols = ["patch_id", "date", "rdvi", "sm"]
    selected = pd.read_parquet(SELECTED, columns=selected_cols)
    selected["date"] = pd.to_datetime(selected["date"])

    base_cols = [
        "patch_id",
        "coord_key",
        "province",
        "date",
        "date_start",
        "date_end",
        "year",
        "centroid_lat",
        "centroid_lon",
        "ndvi",
        "savi",
        "gndvi",
        "evi",
        "doy",
        "etc_8d_mm",
        "et0_pm_8d_mm",
        "qc_valid",
    ]
    base = pd.read_parquet(BASE, columns=base_cols)
    for col in ["date", "date_start", "date_end"]:
        base[col] = pd.to_datetime(base[col])
    base = base[base["qc_valid"]].copy()

    df = base.merge(selected, on=["patch_id", "date"], how="left", validate="one_to_one")
    df = df.dropna(subset=FEATURES + ["etc_8d_mm", "et0_pm_8d_mm"]).copy()

    # Approximate ERA5-Land 0.1° cells. Use floor bins so every crop point maps
    # into one stable coarse cell. Store cell center for inspection.
    df["era5_lat_bin"] = np.floor(pd.to_numeric(df["centroid_lat"]) * 10.0) / 10.0
    df["era5_lon_bin"] = np.floor(pd.to_numeric(df["centroid_lon"]) * 10.0) / 10.0
    df["era5_cell"] = (
        df["era5_lat_bin"].map(lambda v: f"{v:.1f}")
        + "_"
        + df["era5_lon_bin"].map(lambda v: f"{v:.1f}")
    )

    group_cols = [
        "era5_cell",
        "era5_lat_bin",
        "era5_lon_bin",
        "date",
        "date_start",
        "date_end",
        "year",
    ]
    agg = {
        "province": ("province", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        "n_points": ("patch_id", "nunique"),
        "n_rows": ("patch_id", "size"),
        "centroid_lat": ("centroid_lat", "mean"),
        "centroid_lon": ("centroid_lon", "mean"),
        "etc_8d_mm": ("etc_8d_mm", "mean"),
        "et0_pm_8d_mm": ("et0_pm_8d_mm", "mean"),
    }
    for feature in FEATURES:
        agg[feature] = (feature, "mean")

    grid = df.groupby(group_cols, as_index=False).agg(**agg)
    grid["kcact"] = grid["etc_8d_mm"] / grid["et0_pm_8d_mm"]
    grid["grid_patch_id"] = grid["era5_cell"]
    grid["window_days"] = (grid["date_end"] - grid["date_start"]).dt.days
    grid.to_parquet(OUT_PARQUET, index=False)
    return grid


def train_loyo(grid: pd.DataFrame) -> pd.DataFrame:
    train = grid.dropna(subset=FEATURES + ["kcact"]).copy()
    years = sorted(train["year"].astype(int).unique())
    rows = []
    all_true, all_pred = [], []

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

    task_type: str | None = "GPU"
    for year in years:
        tr = train[train["year"] != year]
        te = train[train["year"] == year]
        model = make_model(task_type)
        try:
            model.fit(tr[FEATURES], tr["kcact"])
        except Exception as exc:
            if task_type == "GPU":
                print(f"GPU failed ({exc}); retrying CPU")
                task_type = None
                model = make_model(task_type)
                model.fit(tr[FEATURES], tr["kcact"])
            else:
                raise
        pred = model.predict(te[FEATURES])
        y = te["kcact"].to_numpy()
        rows.append(
            {
                "test_year": int(year),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "r2": float(r2_score(y, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "mae": float(mean_absolute_error(y, pred)),
            }
        )
        all_true.extend(y)
        all_pred.extend(pred)
        print(f"{year}: n={len(te):,} R²={rows[-1]['r2']:.5f} RMSE={rows[-1]['rmse']:.5f}")

    y_all = np.asarray(all_true)
    p_all = np.asarray(all_pred)
    rows.append(
        {
            "test_year": "pooled",
            "n_train": None,
            "n_test": int(len(y_all)),
            "r2": float(r2_score(y_all, p_all)),
            "rmse": float(np.sqrt(mean_squared_error(y_all, p_all))),
            "mae": float(mean_absolute_error(y_all, p_all)),
            "features": "+".join(FEATURES),
            "task_type": task_type or "CPU",
        }
    )
    result = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    return result


def main() -> None:
    grid = build_grid_table()
    print("=== ERA5-like 0.1° grid table ===")
    print(f"rows: {len(grid):,}")
    print(f"cells: {grid['era5_cell'].nunique():,}")
    print(f"years: {sorted(grid['year'].astype(int).unique().tolist())}")
    print(f"window_days: {grid['window_days'].value_counts().sort_index().to_dict()}")
    print(f"n_points per cell-window median: {grid['n_points'].median():.1f}")
    print(f"n_points per cell-window p90: {grid['n_points'].quantile(0.9):.1f}")
    print(f"saved: {OUT_PARQUET}")

    print("\n=== LOYO CatBoost on ERA5-like grid ===")
    result = train_loyo(grid)
    print(result.to_string(index=False))
    print(f"saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
