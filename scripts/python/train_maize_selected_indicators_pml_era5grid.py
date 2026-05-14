#!/usr/bin/env python3
"""Train selected 7 indicators against PML-derived Kcact at ERA5-like scale.

Target:
  kcact_pml = PML crop ETa daily / ERA5 ET0 daily
where PML crop ETa = Ec + Es + Ei exported as pml_eta_crop_mm_d.

Safeguards:
  * corrected base parquet with stable coord_key/date windows;
  * PML joined by coord_key + date_start + date_end + date;
  * seven indicators only: NDVI/SAVI/RDVI/GNDVI/EVI/SM/DOY;
  * no future/statistical full-season features;
  * strict numeric thresholds;
  * unified to ERA5-like 0.1 degree grid before training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
SELECTED = ROOT / "data/processed/train/ncp_summer_maize_selected_indicators.parquet"
PML_DIR = ROOT / "data/raw/gee/kcact_maize_modis_indicators"
OUT_PARQUET = ROOT / "data/processed/train/ncp_summer_maize_selected_indicators_pml_era5grid.parquet"
OUT_CSV = ROOT / "outputs/tables/maize_selected_indicators_pml_era5grid_loyo.csv"
OUT_META = ROOT / "outputs/tables/maize_selected_indicators_pml_era5grid_meta.json"

FEATURES = ["ndvi", "savi", "rdvi", "gndvi", "evi", "sm", "doy"]
TARGET = "kcact_pml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cell-deg", type=float, default=0.1)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year", type=int, default=2024)
    return p.parse_args()


def load_pml(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for y in range(start_year, end_year + 1):
        path = PML_DIR / f"maize_pml_aligned_{y}.csv"
        if not path.exists():
            print(f"WARNING missing {path}")
            continue
        usecols = [
            "coord_key", "date_start", "date_end", "date",
            "pml_et_mm_d", "pml_eta_crop_mm_d", "Ec", "Es", "Ei", "PET", "GPP", "pml_obs_count_8d"
        ]
        d = pd.read_csv(path, usecols=lambda c: c in usecols)
        for c in ["date_start", "date_end", "date"]:
            d[c] = pd.to_datetime(d[c])
        d["coord_key"] = d["coord_key"].astype(str)
        for c in ["pml_et_mm_d", "pml_eta_crop_mm_d", "Ec", "Es", "Ei", "PET", "GPP"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        # Safety: if an unscaled V22a CSV sneaks in, values will be O(100) mm/d.
        if d["pml_eta_crop_mm_d"].median(skipna=True) > 50:
            print(f"WARNING {path.name}: appears unscaled; multiplying PML fluxes by 0.01")
            for c in ["pml_et_mm_d", "pml_eta_crop_mm_d", "Ec", "Es", "Ei", "PET", "GPP"]:
                if c in d.columns:
                    d[c] = d[c] * 0.01
        frames.append(d)
    if not frames:
        raise FileNotFoundError("No maize_pml_aligned_YYYY.csv files found")
    pml = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in ["pml_et_mm_d", "pml_eta_crop_mm_d", "Ec", "Es", "Ei", "PET", "GPP", "pml_obs_count_8d"] if c in pml.columns]
    pml = pml.groupby(["coord_key", "date_start", "date_end", "date"], as_index=False)[value_cols].mean()
    return pml


def build_point_table(start_year: int, end_year: int) -> tuple[pd.DataFrame, dict]:
    base_cols = [
        "patch_id", "coord_key", "province", "date", "date_start", "date_end", "year",
        "centroid_lat", "centroid_lon", "ndvi", "savi", "gndvi", "evi", "doy",
        "et0_pm_8d_mm", "qc_valid",
    ]
    base = pd.read_parquet(BASE, columns=base_cols)
    for c in ["date", "date_start", "date_end"]:
        base[c] = pd.to_datetime(base[c])
    base = base[base["qc_valid"].astype(bool)].copy()
    base = base[(base["year"].astype(int) >= start_year) & (base["year"].astype(int) <= end_year)].copy()
    base["coord_key"] = base["coord_key"].astype(str)

    sel_cols = ["patch_id", "coord_key", "date_start", "date_end", "date", "rdvi", "sm"]
    selected = pd.read_parquet(SELECTED, columns=sel_cols)
    for c in ["date", "date_start", "date_end"]:
        selected[c] = pd.to_datetime(selected[c])
    selected["coord_key"] = selected["coord_key"].astype(str)

    pml = load_pml(start_year, end_year)

    df = base.merge(
        selected,
        on=["patch_id", "coord_key", "date_start", "date_end", "date"],
        how="left",
        validate="one_to_one",
    )
    df = df.merge(
        pml,
        on=["coord_key", "date_start", "date_end", "date"],
        how="left",
        validate="many_to_one",
    )

    df["window_days"] = (df["date_end"] - df["date_start"]).dt.days
    df["et0_pm_daily_mm"] = df["et0_pm_8d_mm"] / df["window_days"]
    df["kcact_pml"] = df["pml_eta_crop_mm_d"] / df["et0_pm_daily_mm"]

    meta = {
        "rows_base": int(len(base)),
        "rows_after_selected_merge": int(len(df)),
        "pml_rows": int(len(pml)),
        "pml_years": sorted(pd.to_datetime(pml["date"]).dt.year.unique().astype(int).tolist()),
        "coverage": {c: float(df[c].notna().mean()) for c in [*FEATURES, "pml_eta_crop_mm_d", "kcact_pml"]},
        "pml_eta_median": float(df["pml_eta_crop_mm_d"].median(skipna=True)),
        "pml_eta_max": float(df["pml_eta_crop_mm_d"].max(skipna=True)),
    }
    return df, meta


def apply_strict_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = pd.Series(True, index=df.index)
    drops = {"rows_before_qc": int(len(df))}

    def rule(name: str, cond: pd.Series) -> None:
        nonlocal mask
        cond = cond.fillna(False)
        drops[f"drop_{name}"] = int((mask & ~cond).sum())
        mask &= cond

    rule("complete_selected_features_and_target", df[FEATURES + [TARGET]].notna().all(axis=1))
    rule("window_8d", df["window_days"].eq(8))
    rule("et0_daily_range", df["et0_pm_daily_mm"].between(0.10, 10.0))
    rule("pml_eta_daily_range", df["pml_eta_crop_mm_d"].between(0.0, 10.0))
    rule("kcact_pml_range", df["kcact_pml"].between(0.02, 1.60))
    for c in ["ndvi", "gndvi"]:
        rule(f"{c}_range", df[c].between(-1.0, 1.0))
    rule("savi_range", df["savi"].between(-1.0, 1.5))
    rule("rdvi_range", df["rdvi"].between(-2.0, 2.0))
    rule("evi_range", df["evi"].between(-1.0, 1.5))
    rule("sm_range", df["sm"].between(0.02, 0.60))

    out = df[mask].copy()
    drops["rows_after_qc"] = int(len(out))
    drops["rows_dropped_total"] = int(len(df) - len(out))
    return out, drops


def aggregate_era5_grid(df: pd.DataFrame, cell_deg: float) -> pd.DataFrame:
    d = df.copy()
    d["era5_lat_bin"] = (np.floor(d["centroid_lat"].astype(float) / cell_deg) * cell_deg).round(4)
    d["era5_lon_bin"] = (np.floor(d["centroid_lon"].astype(float) / cell_deg) * cell_deg).round(4)
    d["era5_cell"] = d["era5_lat_bin"].map(lambda v: f"{v:.1f}") + "_" + d["era5_lon_bin"].map(lambda v: f"{v:.1f}")
    keys = ["era5_cell", "era5_lat_bin", "era5_lon_bin", "date", "date_start", "date_end", "year"]
    agg = {
        "province": ("province", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        "n_points": ("patch_id", "nunique"),
        "n_rows": ("patch_id", "size"),
        "centroid_lat": ("centroid_lat", "mean"),
        "centroid_lon": ("centroid_lon", "mean"),
        "et0_pm_daily_mm": ("et0_pm_daily_mm", "mean"),
        "pml_eta_crop_mm_d": ("pml_eta_crop_mm_d", "mean"),
        "pml_et_mm_d": ("pml_et_mm_d", "mean"),
    }
    for f in FEATURES:
        agg[f] = (f, "mean")
    grid = d.groupby(keys, as_index=False).agg(**agg)
    grid["kcact_pml"] = grid["pml_eta_crop_mm_d"] / grid["et0_pm_daily_mm"]
    grid["window_days"] = (grid["date_end"] - grid["date_start"]).dt.days
    grid["grid_patch_id"] = grid["era5_cell"]
    return grid


def train_loyo(grid: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    train = grid.dropna(subset=FEATURES + [TARGET]).copy()
    years = sorted(train["year"].astype(int).unique().tolist())
    params = dict(iterations=args.iterations, learning_rate=0.03, depth=6, l2_leaf_reg=3, loss_function="RMSE", random_seed=42, verbose=False)
    if args.task_type == "GPU":
        params.update(task_type="GPU", devices=args.devices)
    rows, all_y, all_p = [], [], []
    task_type = args.task_type
    for y in years:
        tr = train[train["year"].astype(int) != y]
        te = train[train["year"].astype(int) == y]
        model = CatBoostRegressor(**params)
        try:
            model.fit(tr[FEATURES], tr[TARGET])
        except Exception as e:
            if task_type == "GPU":
                print(f"GPU failed ({e}); retry CPU")
                params.pop("task_type", None); params.pop("devices", None); task_type = "CPU"
                model = CatBoostRegressor(**params); model.fit(tr[FEATURES], tr[TARGET])
            else:
                raise
        pred = model.predict(te[FEATURES])
        ytrue = te[TARGET].to_numpy()
        rows.append({"test_year": y, "n_train": len(tr), "n_test": len(te), "r2": r2_score(ytrue, pred), "rmse": math.sqrt(mean_squared_error(ytrue, pred)), "mae": mean_absolute_error(ytrue, pred)})
        all_y.extend(ytrue); all_p.extend(pred)
        print(f"{y}: n={len(te):,} R²={rows[-1]['r2']:.5f} RMSE={rows[-1]['rmse']:.5f}")
    all_y = np.asarray(all_y); all_p = np.asarray(all_p)
    pooled = {"test_year": "pooled", "n_train": None, "n_test": len(all_y), "r2": r2_score(all_y, all_p), "rmse": math.sqrt(mean_squared_error(all_y, all_p)), "mae": mean_absolute_error(all_y, all_p), "features": "+".join(FEATURES), "task_type": task_type}
    rows.append(pooled)
    return pd.DataFrame(rows), pooled


def main() -> None:
    args = parse_args()
    point, meta = build_point_table(args.start_year, args.end_year)
    point_qc, qc = apply_strict_qc(point)
    grid = aggregate_era5_grid(point_qc, args.cell_deg)
    # Re-apply target range after aggregation.
    grid = grid[
        grid["window_days"].eq(8)
        & grid["kcact_pml"].between(0.02, 1.60)
        & grid["pml_eta_crop_mm_d"].between(0.0, 10.0)
        & grid["et0_pm_daily_mm"].between(0.10, 10.0)
    ].dropna(subset=FEATURES + [TARGET]).copy()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(OUT_PARQUET, index=False)
    print(f"Grid rows={len(grid):,}, cells={grid['era5_cell'].nunique():,}, years={sorted(grid['year'].astype(int).unique().tolist())}")
    print(f"Saved {OUT_PARQUET}")
    result, pooled = train_loyo(grid, args)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    meta.update({"qc": qc, "grid_rows": int(len(grid)), "grid_cells": int(grid["era5_cell"].nunique()), "features": FEATURES, "target": TARGET, "pooled": pooled, "cell_deg": args.cell_deg, "notes": ["Target uses PML V22a pml_eta_crop_mm_d=Ec+Es+Ei divided by ERA5 ET0 daily.", "PML CSVs were checked for 0.01 scaling; median ETa is in physical mm/day range.", "No full-season/greenup/future statistical features are used; only same-window seven indicators and DOY.", "All rows are joined by stable coord_key + exact 8-day window, then aggregated to 0.1 degree ERA5-like cells."]})
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Saved {OUT_CSV}\nSaved {OUT_META}")


if __name__ == "__main__":
    main()
