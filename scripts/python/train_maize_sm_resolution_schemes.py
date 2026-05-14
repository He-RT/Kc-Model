"""Compare SM-resolution schemes at the coarsest non-ET0 target scale.

Requested schemes:
1. S2 water proxy: replace coarse ERA5 SM with S2 LSWI/MSI water proxies.
2. Sentinel-1 SAR proxy: use S1 VV/VH/RVI if maize_s1_*.csv exists.
3. Conservation downscaling: redistribute coarse ERA5 SM inside each 0.1° cell
   using S2 water-proxy weights, preserving the cell-window mean.

ET0 is deliberately not downscaled.  The target still uses Kcact, but features
are evaluated on an approximate MOD16/ETa target scale by aggregating points to
0.005° cells (~500 m; close to MOD16 500 m and coarser than S2/S1 10-20 m).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
SELECTED = ROOT / "data/processed/train/ncp_summer_maize_selected_indicators.parquet"
S1_DIR = ROOT / "data/raw/gee/kcact_maize_modis_indicators"
OUT_CSV = ROOT / "outputs/tables/maize_sm_resolution_schemes_500m.csv"
OUT_PARQUET = ROOT / "data/processed/train/ncp_summer_maize_sm_resolution_schemes_500m.parquet"

VI_BASE = ["ndvi", "savi", "rdvi", "gndvi", "evi", "doy"]


def msi_from_lswi(lswi: pd.Series) -> pd.Series:
    return (1.0 - lswi) / (1.0 + lswi.replace(-1.0, np.nan))


def add_downscaled_sm(df: pd.DataFrame, alpha: float = 0.35) -> pd.DataFrame:
    """Conservation-style sampled downscaling of ERA5 SM with S2 LSWI.

    For each 0.1° coarse cell and date:
      sm_cell = mean(sm)
      weight_i = exp(alpha * z(lswi_i))
      sm_down_i = sm_cell * weight_i / mean(weight_i)

    Thus mean(sm_down_i) within the sampled cell-window equals sm_cell.
    """
    out = df.copy()
    coarse = ["coarse_cell", "date"]
    grp = out.groupby(coarse, observed=True)
    mean_lswi = grp["lswi"].transform("mean")
    std_lswi = grp["lswi"].transform("std").replace(0, np.nan).fillna(1.0)
    z = ((out["lswi"] - mean_lswi) / std_lswi).clip(-3, 3)
    weight = np.exp(alpha * z)
    weight_mean = weight.groupby([out["coarse_cell"], out["date"]], observed=True).transform("mean")
    sm_cell = grp["sm"].transform("mean")
    out["sm_downscaled"] = sm_cell * weight / weight_mean
    # Keep physical-ish range for volumetric water content.
    out["sm_downscaled"] = out["sm_downscaled"].clip(0.02, 0.60)
    return out


def load_s1_if_available() -> pd.DataFrame | None:
    files = sorted(glob.glob(str(S1_DIR / "maize_s1_*.csv")))
    if not files:
        return None
    frames = []
    for path in files:
        d = pd.read_csv(path)
        if ".geo" not in d.columns or "date" not in d.columns:
            continue
        d["date"] = pd.to_datetime(d["date"])
        # Older export_s1 used window start as date; the new
        # maize_s1_aligned_YYYY export already writes date=date_end.
        if "aligned" not in Path(path).stem:
            d["date"] = d["date"] + pd.Timedelta(days=8)
        if "coord_key" not in d.columns:
            coords = d[".geo"].astype(str).str.extract(r"\[([\-0-9.]+),([\-0-9.]+)\]")
            lon = pd.to_numeric(coords[0], errors="coerce").round(6)
            lat = pd.to_numeric(coords[1], errors="coerce").round(6)
            d["coord_key"] = lat.map(lambda v: f"{v:.6f}") + "_" + lon.map(lambda v: f"{v:.6f}")
        keep = ["coord_key", "date"]
        for c in ["s1_vv", "s1_vh", "VV", "VH"]:
            if c in d.columns:
                keep.append(c)
        d = d[keep].copy()
        if "VV" in d.columns and "s1_vv" not in d.columns:
            d = d.rename(columns={"VV": "s1_vv"})
        if "VH" in d.columns and "s1_vh" not in d.columns:
            d = d.rename(columns={"VH": "s1_vh"})
        frames.append(d)
    if not frames:
        return None
    s1 = pd.concat(frames, ignore_index=True)
    if "s1_vv" not in s1.columns or "s1_vh" not in s1.columns:
        return None
    s1 = s1.groupby(["coord_key", "date"], as_index=False)[["s1_vv", "s1_vh"]].mean()
    # dB difference and radar vegetation index-like proxy. S1 GRD is dB in GEE.
    s1["s1_vv_vh_diff"] = s1["s1_vv"] - s1["s1_vh"]
    vv_lin = 10 ** (s1["s1_vv"] / 10.0)
    vh_lin = 10 ** (s1["s1_vh"] / 10.0)
    s1["s1_rvi"] = 4 * vh_lin / (vv_lin + vh_lin)
    return s1


def build_point_table() -> pd.DataFrame:
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
        "lswi",
        "doy",
        "etc_8d_mm",
        "et0_pm_8d_mm",
        "qc_valid",
    ]
    base = pd.read_parquet(BASE, columns=base_cols)
    for c in ["date", "date_start", "date_end"]:
        base[c] = pd.to_datetime(base[c])
    base = base[base["qc_valid"]].copy()

    selected = pd.read_parquet(SELECTED, columns=["patch_id", "date", "rdvi", "sm"])
    selected["date"] = pd.to_datetime(selected["date"])
    df = base.merge(selected, on=["patch_id", "date"], how="left", validate="one_to_one")
    df["msi"] = msi_from_lswi(df["lswi"])

    # Coarse ERA5-ish cell for conservation downscaling, 500m-ish cell for
    # the requested "lowest non-ET0 resolution" aggregation.
    df["coarse_lat_bin"] = np.floor(pd.to_numeric(df["centroid_lat"]) * 10.0) / 10.0
    df["coarse_lon_bin"] = np.floor(pd.to_numeric(df["centroid_lon"]) * 10.0) / 10.0
    df["coarse_cell"] = (
        df["coarse_lat_bin"].map(lambda v: f"{v:.1f}")
        + "_"
        + df["coarse_lon_bin"].map(lambda v: f"{v:.1f}")
    )
    df["cell500_lat_bin"] = np.floor(pd.to_numeric(df["centroid_lat"]) / 0.005) * 0.005
    df["cell500_lon_bin"] = np.floor(pd.to_numeric(df["centroid_lon"]) / 0.005) * 0.005
    df["cell500"] = (
        df["cell500_lat_bin"].map(lambda v: f"{v:.3f}")
        + "_"
        + df["cell500_lon_bin"].map(lambda v: f"{v:.3f}")
    )
    df = add_downscaled_sm(df)

    s1 = load_s1_if_available()
    if s1 is not None:
        df = df.merge(s1, on=["coord_key", "date"], how="left")
        print(f"S1 loaded: {len(s1):,} coord-window rows")
    else:
        print("S1 not available: no maize_s1_*.csv found or missing VV/VH columns.")
    return df


def aggregate_500m(df: pd.DataFrame, feature_cols: list[str], scheme: str) -> pd.DataFrame:
    sub = df.dropna(subset=feature_cols + ["etc_8d_mm", "et0_pm_8d_mm"]).copy()
    group_cols = ["cell500", "date", "date_start", "date_end", "year"]
    agg = {
        "province": ("province", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        "n_points": ("patch_id", "nunique"),
        "etc_8d_mm": ("etc_8d_mm", "mean"),
        "et0_pm_8d_mm": ("et0_pm_8d_mm", "mean"),
    }
    for c in feature_cols:
        agg[c] = (c, "mean")
    grid = sub.groupby(group_cols, as_index=False).agg(**agg)
    grid["kcact"] = grid["etc_8d_mm"] / grid["et0_pm_8d_mm"]
    grid["scheme"] = scheme
    grid["window_days"] = (grid["date_end"] - grid["date_start"]).dt.days
    return grid


def train_scheme(grid: pd.DataFrame, feature_cols: list[str], scheme: str) -> list[dict]:
    years = sorted(grid["year"].astype(int).unique())
    all_y, all_p = [], []
    rows = []
    task_type: str | None = "GPU"

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

    for yr in years:
        tr = grid[grid["year"] != yr]
        te = grid[grid["year"] == yr]
        model = make_model()
        try:
            model.fit(tr[feature_cols], tr["kcact"])
        except Exception as exc:
            if task_type == "GPU":
                print(f"{scheme}: GPU failed ({exc}); retrying CPU")
                task_type = None
                model = make_model()
                model.fit(tr[feature_cols], tr["kcact"])
            else:
                raise
        pred = model.predict(te[feature_cols])
        y = te["kcact"].to_numpy()
        rows.append(
            {
                "scheme": scheme,
                "test_year": int(yr),
                "features": "+".join(feature_cols),
                "n_samples": int(len(te)),
                "n_cells": int(te["cell500"].nunique()),
                "r2": float(r2_score(y, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "mae": float(mean_absolute_error(y, pred)),
                "task_type": task_type or "CPU",
            }
        )
        all_y.extend(y)
        all_p.extend(pred)
        print(f"{scheme} {yr}: n={len(te):,} R²={rows[-1]['r2']:.5f}")

    y = np.asarray(all_y)
    p = np.asarray(all_p)
    rows.append(
        {
            "scheme": scheme,
            "test_year": "pooled",
            "features": "+".join(feature_cols),
            "n_samples": int(len(grid)),
            "n_cells": int(grid["cell500"].nunique()),
            "r2": float(r2_score(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "task_type": task_type or "CPU",
        }
    )
    return rows


def main() -> None:
    df = build_point_table()
    schemes = {
        "s2_water_proxy": VI_BASE + ["lswi", "msi"],
        "sm_conservation_downscaled": VI_BASE + ["sm_downscaled"],
    }
    if {"s1_vv", "s1_vh", "s1_vv_vh_diff", "s1_rvi"}.issubset(df.columns):
        schemes["s1_sar_proxy"] = VI_BASE + ["s1_vv", "s1_vh", "s1_vv_vh_diff", "s1_rvi"]
    else:
        print("Skipping s1_sar_proxy until maize_s1_*.csv exports are available.")

    all_grids = []
    results = []
    for scheme, feats in schemes.items():
        grid = aggregate_500m(df, feats, scheme)
        print("\n===", scheme, "===")
        print(f"features: {feats}")
        print(f"rows: {len(grid):,}; cells: {grid['cell500'].nunique():,}; "
              f"median points/cell-window: {grid['n_points'].median():.1f}")
        print(f"window_days: {grid['window_days'].value_counts().sort_index().to_dict()}")
        all_grids.append(grid)
        results.extend(train_scheme(grid, feats, scheme))

    out_grid = pd.concat(all_grids, ignore_index=True)
    out_grid.to_parquet(OUT_PARQUET, index=False)
    res = pd.DataFrame(results)

    # Add explicit skipped row for scheme 2 if no S1.
    if "s1_sar_proxy" not in schemes:
        res = pd.concat(
            [
                res,
                pd.DataFrame(
                    [
                        {
                            "scheme": "s1_sar_proxy",
                            "test_year": "skipped",
                            "features": "VV+VH+VV/VH/RVI",
                            "n_samples": 0,
                            "n_cells": 0,
                            "r2": np.nan,
                            "rmse": np.nan,
                            "mae": np.nan,
                            "task_type": "missing maize_s1_*.csv",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    print("\n=== Summary ===")
    print(res[res["test_year"].astype(str).isin(["pooled", "skipped"])].to_string(index=False))
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_PARQUET}")


if __name__ == "__main__":
    main()
