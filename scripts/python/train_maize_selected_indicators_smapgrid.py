"""Aggregate selected maize indicators to SMAP L4 grid scale and train.

This run unifies the seven selected indicators to the SMAP L4 catalog sampling
scale used by ``export_maize_smap_aligned.py``.  The exported SMAP table is
point-sampled, so we infer practical SMAP cells by grouping points that share
identical SMAP values within the same MOD16 8-day window.  This avoids guessing
EASE-Grid cell boundaries from latitude/longitude and preserves the actual GEE
SMAP sampling result used for the query product.

For each SMAP cell-window:
  * features: mean NDVI/SAVI/RDVI/GNDVI/EVI/DOY + SMAP SM
  * target: mean(MOD16 ETa 8d) / mean(ERA5 ET0 8d)

SM source is configurable: sm_surface, sm_rootzone, sm_profile, or wetness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
SELECTED = ROOT / "data/processed/train/ncp_summer_maize_selected_indicators.parquet"
SMAP_DIR = ROOT / "data/raw/gee/kcact_maize_modis_indicators"
OUT_DIR = ROOT / "outputs/tables"
TRAIN_DIR = ROOT / "data/processed/train"

VI_FEATURES = ["ndvi", "savi", "rdvi", "gndvi", "evi", "doy"]
FEATURES = VI_FEATURES + ["sm"]
SMAP_COLUMNS = [
    "sm_surface",
    "sm_rootzone",
    "sm_profile",
    "sm_surface_wetness",
    "sm_rootzone_wetness",
    "sm_profile_wetness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sm-column",
        choices=SMAP_COLUMNS,
        default="sm_rootzone",
        help="SMAP variable to use as the selected SM feature.",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_smap(sm_column: str) -> pd.DataFrame:
    frames = []
    for path in sorted(SMAP_DIR.glob("maize_smap_l4_aligned_*.csv")):
        usecols = [
            "point_id",
            "coord_key",
            "centroid_lat",
            "centroid_lon",
            "date",
            "date_start",
            "date_end",
            "smap_obs_count_3h",
            "smap_scale_m",
            *SMAP_COLUMNS,
        ]
        d = pd.read_csv(path, usecols=lambda c: c in usecols)
        for c in ["date", "date_start", "date_end"]:
            if c in d.columns:
                d[c] = pd.to_datetime(d[c])
        for c in SMAP_COLUMNS:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d["sm"] = d[sm_column]
        frames.append(d)
        print(f"SMAP {path.name}: {len(d):,} rows, {d['coord_key'].nunique():,} coords")
    if not frames:
        raise FileNotFoundError(f"No maize_smap_l4_aligned_*.csv in {SMAP_DIR}")
    smap = pd.concat(frames, ignore_index=True)
    # Stable inferred cell id: identical SMAP values in a window correspond to the
    # same sampled SMAP grid cell for our point set. Include all SMAP variables to
    # reduce accidental merging of unrelated cells with identical one-band values.
    for c in SMAP_COLUMNS:
        smap[f"{c}_key"] = pd.to_numeric(smap[c], errors="coerce").round(6)
    key_cols = [f"{c}_key" for c in SMAP_COLUMNS]
    smap["smap_cell"] = (
        smap["date"].dt.strftime("%Y%m%d")
        + "_"
        + smap[key_cols].astype(str).agg("_".join, axis=1)
    )
    smap = smap.drop_duplicates(subset=["point_id", "coord_key", "date"], keep="first")
    return smap[[
        "point_id", "coord_key", "date", "sm", "smap_cell", "smap_obs_count_3h", "smap_scale_m",
        *SMAP_COLUMNS,
    ]]


def build_smap_grid(sm_column: str) -> pd.DataFrame:
    selected = pd.read_parquet(SELECTED, columns=["patch_id", "date", "rdvi"])
    selected["date"] = pd.to_datetime(selected["date"])

    base_cols = [
        "patch_id",
        "point_id",
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
    for c in ["date", "date_start", "date_end"]:
        base[c] = pd.to_datetime(base[c])
    base = base[base["qc_valid"]].copy()

    df = base.merge(selected, on=["patch_id", "date"], how="left", validate="one_to_one")
    smap = load_smap(sm_column)
    df = df.merge(
        smap,
        on=["point_id", "coord_key", "date"],
        how="inner",
        validate="one_to_one",
    )
    df = df.dropna(subset=FEATURES + ["etc_8d_mm", "et0_pm_8d_mm", "smap_cell"]).copy()

    # Conservative physical/numeric filters aligned with
    # train_maize_selected_indicators.py strict QC.
    window_days = (df["date_end"] - df["date_start"]).dt.days
    et0_daily = df["et0_pm_8d_mm"] / window_days
    eta_daily = df["etc_8d_mm"] / window_days
    kc_point = df["etc_8d_mm"] / df["et0_pm_8d_mm"]
    qc_mask = (
        (window_days == 8)
        & et0_daily.between(0.10, 10.0)
        & eta_daily.between(0.0, 10.0)
        & kc_point.between(0.02, 1.60)
        & df["ndvi"].between(-1.0, 1.0)
        & df["gndvi"].between(-1.0, 1.0)
        & df["savi"].between(-1.0, 1.5)
        & df["rdvi"].between(-2.0, 2.0)
        & df["evi"].between(-1.0, 1.5)
        & df["sm"].between(0.02, 0.60)
    )
    print(
        f"Strict QC drop: {int((~qc_mask).sum()):,}/{len(df):,} "
        f"({(~qc_mask).mean()*100:.2f}%)"
    )
    df = df[qc_mask].copy()
    df["et0_pm_daily_mm"] = et0_daily.loc[df.index]
    df["eta_daily_mm"] = eta_daily.loc[df.index]

    group_cols = ["smap_cell", "date", "date_start", "date_end", "year"]
    agg = {
        "province": ("province", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        "n_points": ("patch_id", "nunique"),
        "n_rows": ("patch_id", "size"),
        "centroid_lat": ("centroid_lat", "mean"),
        "centroid_lon": ("centroid_lon", "mean"),
        "etc_8d_mm": ("etc_8d_mm", "mean"),
        "et0_pm_8d_mm": ("et0_pm_8d_mm", "mean"),
        "et0_pm_daily_mm": ("et0_pm_daily_mm", "mean"),
        "eta_daily_mm": ("eta_daily_mm", "mean"),
        "smap_obs_count_3h": ("smap_obs_count_3h", "mean"),
        "smap_scale_m": ("smap_scale_m", "first"),
    }
    for feature in FEATURES:
        agg[feature] = (feature, "mean")
    for c in SMAP_COLUMNS:
        agg[c] = (c, "mean")

    grid = df.groupby(group_cols, as_index=False).agg(**agg)
    grid["kcact"] = grid["etc_8d_mm"] / grid["et0_pm_8d_mm"]
    grid["window_days"] = (grid["date_end"] - grid["date_start"]).dt.days
    grid["sm_source"] = sm_column
    return grid


def train_loyo(grid: pd.DataFrame, sm_column: str, use_gpu: bool) -> pd.DataFrame:
    train = grid.dropna(subset=FEATURES + ["kcact"]).copy()
    years = sorted(train["year"].astype(int).unique())
    rows = []
    all_true, all_pred = [], []
    task_type: str | None = "GPU" if use_gpu else None

    def make_model() -> CatBoostRegressor:
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

    for year in years:
        tr = train[train["year"] != year]
        te = train[train["year"] == year]
        model = make_model()
        try:
            model.fit(tr[FEATURES], tr["kcact"])
        except Exception as exc:
            if task_type == "GPU":
                print(f"GPU failed ({exc}); retrying CPU")
                task_type = None
                model = make_model()
                model.fit(tr[FEATURES], tr["kcact"])
            else:
                raise
        pred = model.predict(te[FEATURES])
        y = te["kcact"].to_numpy()
        rows.append({
            "test_year": int(year),
            "sm_source": sm_column,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
        })
        all_true.extend(y)
        all_pred.extend(pred)
        print(f"{sm_column} {year}: n={len(te):,} R²={rows[-1]['r2']:.5f} RMSE={rows[-1]['rmse']:.5f}")

    y_all = np.asarray(all_true)
    p_all = np.asarray(all_pred)
    rows.append({
        "test_year": "pooled",
        "sm_source": sm_column,
        "n_train": None,
        "n_test": int(len(y_all)),
        "r2": float(r2_score(y_all, p_all)),
        "rmse": float(np.sqrt(mean_squared_error(y_all, p_all))),
        "mae": float(mean_absolute_error(y_all, p_all)),
        "features": "+".join(FEATURES),
        "task_type": task_type or "CPU",
    })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    grid = build_smap_grid(args.sm_column)
    safe = args.sm_column.replace("sm_", "")
    out_parquet = TRAIN_DIR / f"ncp_summer_maize_selected_indicators_smapgrid_{safe}.parquet"
    out_csv = OUT_DIR / f"maize_selected_indicators_smapgrid_{safe}_loyo.csv"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(out_parquet, index=False)

    print("=== SMAP L4 grid table ===")
    print(f"sm_source: {args.sm_column}")
    print(f"rows: {len(grid):,}")
    print(f"cells/windows: {grid['smap_cell'].nunique():,}")
    print(f"years: {sorted(grid['year'].astype(int).unique().tolist())}")
    print(f"window_days: {grid['window_days'].value_counts().sort_index().to_dict()}")
    print(f"n_points median: {grid['n_points'].median():.1f}; p90: {grid['n_points'].quantile(0.9):.1f}")
    print(f"SM range: {grid['sm'].min():.4f}..{grid['sm'].max():.4f}")
    print(f"saved: {out_parquet}")

    print("\n=== LOYO CatBoost on SMAP grid ===")
    res = train_loyo(grid, args.sm_column, use_gpu=not args.cpu)
    res.to_csv(out_csv, index=False)
    print(res.to_string(index=False))
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
