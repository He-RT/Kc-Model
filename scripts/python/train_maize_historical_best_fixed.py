#!/usr/bin/env python3
"""Rerun the historical best maize combo after spatial-fix, strict QC, no leakage, and scale unification.

Historical best pre-fix combo:
  core_fpar_sm_pr_ndvi__lswi_doy+W

Expanded features used here:
  fpar, sm, precip_mm_8d, ndvi_m09, lswi, doy,
  vpd_kpa_mean_8d, solar_rad_mj_m2_d_sum_8d, tmean_c

Important fixes vs the historical run:
  * starts from corrected ncp_summer_maize_kcact_train_ready.parquet;
  * joins MODIS by stable coord_key/date, not GEE pt_*;
  * replaces old sm_proxy with real ERA5-Land SM 8-day mean;
  * applies strict numeric QC thresholds;
  * avoids future-leakage engineered features (the selected combo uses only same-window inputs);
  * unifies all variables and target to ERA5-like 0.1 degree cells before LOYO training.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
CSV_DIR = ROOT / "data/raw/gee/kcact_maize_modis_indicators"
OUT_TABLE = ROOT / "outputs/tables/maize_historical_best_spatialfix_qc_era5grid_loyo.csv"
OUT_PARQUET = ROOT / "data/processed/train/ncp_summer_maize_historical_best_era5grid.parquet"
OUT_META = ROOT / "outputs/tables/maize_historical_best_spatialfix_qc_era5grid_meta.json"

FEATURES = [
    "fpar",
    "sm",
    "precip_mm_8d",
    "ndvi_m09",
    "lswi",
    "doy",
    "vpd_kpa_mean_8d",
    "solar_rad_mj_m2_d_sum_8d",
    "tmean_c",
]
TARGET = "kcact"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--csv-dir", default=str(CSV_DIR))
    p.add_argument("--out-table", default=str(OUT_TABLE))
    p.add_argument("--out-parquet", default=str(OUT_PARQUET))
    p.add_argument("--out-meta", default=str(OUT_META))
    p.add_argument("--cell-deg", type=float, default=0.1, help="ERA5-like grid size in degrees")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--sample-frac", type=float, default=1.0, help="Debug only")
    return p.parse_args()


def parse_geo_lonlat(s: object) -> tuple[float, float]:
    try:
        g = json.loads(s)
        lon, lat = g["coordinates"][:2]
        return float(lon), float(lat)
    except Exception:
        return (np.nan, np.nan)


def add_coord_key_from_geo(df: pd.DataFrame) -> pd.DataFrame:
    if "coord_key" in df.columns:
        return df
    if "centroid_lat" not in df.columns or "centroid_lon" not in df.columns:
        lonlat = df[".geo"].map(parse_geo_lonlat)
        df["centroid_lon"] = lonlat.map(lambda x: x[0])
        df["centroid_lat"] = lonlat.map(lambda x: x[1])
    df["coord_key"] = df.apply(
        lambda r: f"{float(r['centroid_lat']):.6f}_{float(r['centroid_lon']):.6f}", axis=1
    )
    return df


def read_csv_with_coord(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=usecols)
    df = add_coord_key_from_geo(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_fpar(csv_dir: Path, years: list[int]) -> pd.DataFrame:
    frames = []
    for yr in years:
        path = csv_dir / f"maize_fpar_{yr}.csv"
        if not path.exists():
            print(f"WARNING missing {path}")
            continue
        df = read_csv_with_coord(path, usecols=["date", "mean", ".geo"])
        df["fpar"] = pd.to_numeric(df["mean"], errors="coerce") * 0.01
        frames.append(df[["coord_key", "date", "fpar"]])
    if not frames:
        return pd.DataFrame(columns=["coord_key", "date", "fpar"])
    out = pd.concat(frames, ignore_index=True)
    return out.groupby(["coord_key", "date"], as_index=False)["fpar"].mean()


def load_m09(csv_dir: Path, years: list[int]) -> pd.DataFrame:
    frames = []
    for yr in years:
        files = sorted(glob.glob(str(csv_dir / f"maize_m09vi*{yr}*.csv")))
        if not files:
            print(f"WARNING no m09 files for {yr}")
            continue
        for f in files:
            df = read_csv_with_coord(f, usecols=["date", "ndvi_m09", "sur_refl_b07", ".geo"])
            df["ndvi_m09"] = pd.to_numeric(df["ndvi_m09"], errors="coerce")
            # b07 not in this combo, but keep for diagnostics/future extension.
            df["b07"] = pd.to_numeric(df["sur_refl_b07"], errors="coerce") * 0.0001
            frames.append(df[["coord_key", "date", "ndvi_m09", "b07"]])
    if not frames:
        return pd.DataFrame(columns=["coord_key", "date", "ndvi_m09", "b07"])
    out = pd.concat(frames, ignore_index=True)
    return out.groupby(["coord_key", "date"], as_index=False)[["ndvi_m09", "b07"]].mean()


def load_sm_8d(csv_dir: Path, windows: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Load daily ERA5-Land SM and aggregate to exact MOD16 8-day windows.

    Joined by stable coordinate key and exact date_start/date_end/date labels.
    """
    frames = []
    for yr in years:
        path = csv_dir / f"maize_era5_sm_{yr}.csv"
        if not path.exists():
            print(f"WARNING missing {path}")
            continue
        df = read_csv_with_coord(path, usecols=["date", "mean", ".geo"])
        df["sm_daily"] = pd.to_numeric(df["mean"], errors="coerce")
        df = df[["coord_key", "date", "sm_daily"]].dropna(subset=["sm_daily"])
        w = windows[windows["year"].astype(int) == int(yr)].copy()
        # Number of maize windows is small (<= 20); loop is memory-safe and avoids range-join pitfalls.
        parts = []
        for row in w.itertuples(index=False):
            m = df[(df["date"] >= row.date_start) & (df["date"] < row.date_end)]
            if m.empty:
                continue
            g = m.groupby("coord_key", as_index=False)["sm_daily"].mean().rename(columns={"sm_daily": "sm"})
            g["date_start"] = row.date_start
            g["date_end"] = row.date_end
            g["date"] = row.date
            parts.append(g)
        if parts:
            frames.append(pd.concat(parts, ignore_index=True))
    if not frames:
        return pd.DataFrame(columns=["coord_key", "date_start", "date_end", "date", "sm"])
    out = pd.concat(frames, ignore_index=True)
    return out.groupby(["coord_key", "date_start", "date_end", "date"], as_index=False)["sm"].mean()


def strict_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = len(df)
    work = df.copy()
    work["date_start"] = pd.to_datetime(work["date_start"])
    work["date_end"] = pd.to_datetime(work["date_end"])
    work["window_days"] = (work["date_end"] - work["date_start"]).dt.days
    work["et0_pm_daily_mm"] = work["et0_pm_8d_mm"] / work["window_days"]
    work["eta_daily_mm"] = work["etc_8d_mm"] / work["window_days"]

    mask = pd.Series(True, index=work.index)
    mask &= work.get("qc_valid", True).astype(bool)
    mask &= work["window_days"].eq(8)
    mask &= work["kcact"].between(0.02, 1.60)
    mask &= work["et0_pm_daily_mm"].between(0.10, 10.0)
    mask &= work["eta_daily_mm"].between(0.0, 10.0)
    for c in ["ndvi", "gndvi", "lswi", "ndvi_m09"]:
        if c in work.columns:
            mask &= work[c].between(-1.0, 1.0) | work[c].isna()
    if "savi" in work.columns:
        mask &= work["savi"].between(-1.0, 1.5) | work["savi"].isna()
    if "evi" in work.columns:
        mask &= work["evi"].between(-1.0, 1.5) | work["evi"].isna()
    if "fpar" in work.columns:
        mask &= work["fpar"].between(0.0, 1.0) | work["fpar"].isna()
    if "sm" in work.columns:
        mask &= work["sm"].between(0.02, 0.60) | work["sm"].isna()
    if "vpd_kpa_mean_8d" in work.columns:
        mask &= work["vpd_kpa_mean_8d"].between(0.0, 10.0) | work["vpd_kpa_mean_8d"].isna()
    if "tmean_c" in work.columns:
        mask &= work["tmean_c"].between(-20.0, 45.0) | work["tmean_c"].isna()
    if "precip_mm_8d" in work.columns:
        mask &= work["precip_mm_8d"].between(0.0, 500.0) | work["precip_mm_8d"].isna()
    if "solar_rad_mj_m2_d_sum_8d" in work.columns:
        # This column is the sum of daily radiation over an 8-day window.
        mask &= work["solar_rad_mj_m2_d_sum_8d"].between(0.0, 280.0) | work["solar_rad_mj_m2_d_sum_8d"].isna()

    work = work[mask].copy()
    return work, {"rows_before_qc": int(before), "rows_after_qc": int(len(work)), "rows_removed_qc": int(before - len(work))}


def aggregate_era5_grid(df: pd.DataFrame, cell_deg: float) -> pd.DataFrame:
    work = df.copy()
    work["era5_lat_bin"] = np.floor(work["centroid_lat"].astype(float) / cell_deg) * cell_deg
    work["era5_lon_bin"] = np.floor(work["centroid_lon"].astype(float) / cell_deg) * cell_deg
    work["era5_lat_bin"] = work["era5_lat_bin"].round(4)
    work["era5_lon_bin"] = work["era5_lon_bin"].round(4)
    work["era5_cell"] = work["era5_lat_bin"].map(lambda x: f"{x:.1f}") + "_" + work["era5_lon_bin"].map(lambda x: f"{x:.1f}")

    keys = ["era5_cell", "era5_lat_bin", "era5_lon_bin", "date", "date_start", "date_end", "year", "province"]
    agg = {
        "patch_id": pd.Series.nunique,
        "centroid_lat": "mean",
        "centroid_lon": "mean",
        "etc_8d_mm": "mean",
        "et0_pm_8d_mm": "mean",
    }
    for c in FEATURES:
        agg[c] = "mean"
    out = work.groupby(keys, as_index=False).agg(agg).rename(columns={"patch_id": "n_points"})
    out["n_rows"] = work.groupby(keys).size().values
    out["kcact"] = out["etc_8d_mm"] / out["et0_pm_8d_mm"]
    out["window_days"] = (pd.to_datetime(out["date_end"]) - pd.to_datetime(out["date_start"])).dt.days
    out["grid_patch_id"] = out["era5_cell"]
    return out


def train_loyo(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    sub = df.dropna(subset=FEATURES + [TARGET, "year"]).copy()
    years = sorted(int(y) for y in sub["year"].dropna().unique())
    all_pred = []
    rows = []
    params = dict(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=3,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
    if args.task_type == "GPU":
        params.update(task_type="GPU", devices=args.devices)

    for yr in years:
        tr = sub[sub["year"].astype(int) != yr]
        te = sub[sub["year"].astype(int) == yr]
        if len(te) < 10 or len(tr) < 100:
            continue
        model = CatBoostRegressor(**params)
        model.fit(tr[FEATURES], tr[TARGET])
        pred = model.predict(te[FEATURES])
        fold = te[["year", "era5_cell", "date", TARGET]].copy()
        fold["pred"] = pred
        all_pred.append(fold)
        rows.append({
            "test_year": yr,
            "r2": r2_score(te[TARGET], pred),
            "rmse": math.sqrt(mean_squared_error(te[TARGET], pred)),
            "mae": mean_absolute_error(te[TARGET], pred),
            "n": len(te),
        })
        print(f"  {yr}: R²={rows[-1]['r2']:.5f} RMSE={rows[-1]['rmse']:.5f} n={len(te):,}")

    pred_df = pd.concat(all_pred, ignore_index=True)
    summary = {
        "pooled_r2": float(r2_score(pred_df[TARGET], pred_df["pred"])),
        "pooled_rmse": float(math.sqrt(mean_squared_error(pred_df[TARGET], pred_df["pred"]))),
        "pooled_mae": float(mean_absolute_error(pred_df[TARGET], pred_df["pred"])),
        "n_trainable_rows": int(len(sub)),
        "years": years,
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    csv_dir = Path(args.csv_dir)
    print(f"Loading corrected base: {in_path}", flush=True)
    base = pd.read_parquet(in_path)
    for c in ["date", "date_start", "date_end"]:
        base[c] = pd.to_datetime(base[c])
    base = base[base["qc_valid"].astype(bool)].copy()
    if "coord_key" not in base.columns:
        base["coord_key"] = base.apply(
            lambda r: f"{float(r['centroid_lat']):.6f}_{float(r['centroid_lon']):.6f}", axis=1
        )
    base["coord_key"] = base["coord_key"].astype(str)
    if args.sample_frac < 1.0:
        base = base.sample(frac=args.sample_frac, random_state=42).copy()
    years = sorted(int(y) for y in base["year"].dropna().unique())
    print(f"Base rows={len(base):,}, points={base['coord_key'].nunique():,}, years={years}", flush=True)

    windows = base[["year", "date_start", "date_end", "date"]].drop_duplicates().sort_values(["year", "date_start"])

    print("Loading MODIS fPAR by coord/date...", flush=True)
    fpar = load_fpar(csv_dir, years)
    print(f"  fpar rows={len(fpar):,}", flush=True)
    print("Loading MOD09A1 NDVI by coord/date...", flush=True)
    m09 = load_m09(csv_dir, years)
    print(f"  m09 rows={len(m09):,}", flush=True)
    print("Loading ERA5-Land SM daily and aggregating to exact 8-day windows...", flush=True)
    sm = load_sm_8d(csv_dir, windows, years)
    print(f"  sm-window rows={len(sm):,}", flush=True)

    # MODIS 8-day products are labeled by system:time_start.
    # Align them to the target window start, not date_end, to avoid pulling the next composite.
    fpar = fpar.rename(columns={"date": "modis_date_start"})
    m09 = m09.rename(columns={"date": "modis_date_start"})
    df = base.merge(
        fpar,
        left_on=["coord_key", "date_start"],
        right_on=["coord_key", "modis_date_start"],
        how="left",
    ).drop(columns=["modis_date_start"], errors="ignore")
    df = df.merge(
        m09,
        left_on=["coord_key", "date_start"],
        right_on=["coord_key", "modis_date_start"],
        how="left",
    ).drop(columns=["modis_date_start"], errors="ignore")
    df = df.merge(sm, on=["coord_key", "date_start", "date_end", "date"], how="left")

    print("Coverage after stable coord/date merge:", flush=True)
    for c in ["fpar", "ndvi_m09", "b07", "sm", "lswi"]:
        print(f"  {c:10s}: {df[c].notna().sum():,}/{len(df):,} ({df[c].notna().mean()*100:.1f}%)")

    print("Applying strict thresholds...", flush=True)
    df_qc, qc_meta = strict_qc(df)
    print(f"  rows: {qc_meta['rows_before_qc']:,} -> {qc_meta['rows_after_qc']:,}", flush=True)

    print(f"Aggregating to ERA5-like grid, cell_deg={args.cell_deg}...", flush=True)
    grid = aggregate_era5_grid(df_qc, args.cell_deg)
    # Re-apply target/feature ranges after aggregation and require complete selected combo.
    before_grid = len(grid)
    grid = grid[grid["window_days"].eq(8)].copy()
    grid = grid[grid["kcact"].between(0.02, 1.60)].copy()
    grid = grid.dropna(subset=FEATURES + [TARGET])
    grid = grid[grid["sm"].between(0.02, 0.60)]
    grid = grid[grid["fpar"].between(0, 1)]
    grid = grid[grid["ndvi_m09"].between(-1, 1)]
    grid = grid[grid["lswi"].between(-1, 1)]
    print(f"  grid rows: {before_grid:,} -> complete/QC {len(grid):,}; cells={grid['era5_cell'].nunique():,}")

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(out_parquet, index=False)
    print(f"Saved grid table: {out_parquet}")

    print("Training LOYO CatBoost...", flush=True)
    folds, summary = train_loyo(grid, args)
    pooled_row = {"test_year": "pooled", "r2": summary["pooled_r2"], "rmse": summary["pooled_rmse"], "mae": summary["pooled_mae"], "n": summary["n_trainable_rows"]}
    out = pd.concat([folds, pd.DataFrame([pooled_row])], ignore_index=True)
    out_table = Path(args.out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_table, index=False)
    print(f"Saved CV table: {out_table}")

    meta = {
        "historical_combo": "core_fpar_sm_pr_ndvi__lswi_doy+W",
        "features": FEATURES,
        "target": TARGET,
        "input": str(in_path),
        "csv_dir": str(csv_dir),
        "out_parquet": str(out_parquet),
        "out_table": str(out_table),
        "cell_deg": args.cell_deg,
        "qc": qc_meta,
        "rows_after_grid_complete_qc": int(len(grid)),
        "cells": int(grid["era5_cell"].nunique()),
        "summary": summary,
        "notes": [
            "Corrected base parquet uses stable coord_key joins for ETa/ET0/S2.",
            "MODIS 8-day CSVs are re-joined by coord_key + date_start; SM is joined by coord_key + exact window; no pt_* joins.",
            "No future-window features are included; MODIS composites use the target window start, not the next composite at date_end.",
            "All variables and target are unified to ERA5-like 0.1 degree grid before training.",
            "SM is real ERA5-Land 8-day mean, replacing old precip_30d sm_proxy.",
        ],
    }
    out_meta = Path(args.out_meta)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved meta: {out_meta}")
    print("\nPooled:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
