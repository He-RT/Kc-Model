"""Train summer-maize Kcact with NDVI/SAVI/RDVI/GNDVI/EVI/SM/DOY.

This script is intentionally strict about alignment:

* Kcact, NDVI, SAVI, GNDVI, EVI, DOY come from the corrected
  ``ncp_summer_maize_kcact_train_ready.parquet`` table.  In that table S2,
  MOD16 ETa and ERA5 ET0 are already joined by stable coordinate key and the
  same MOD16 8-day window.
* RDVI is expected to come directly from the aligned S2 Red/NIR bands exported
  for the same MOD16 window.  For old parquet files that predate the direct
  RDVI export, ``--rdvi-source auto`` falls back to the old NDVI/SAVI
  reconstruction but reports this explicitly; use ``--rdvi-source direct`` for
  final/reported runs after re-exporting S2.
* SM is aggregated from daily ERA5-Land soil moisture exports to the exact
  MOD16 window [date_start, date_end), joined by the coordinate key parsed from
  the exported point geometry.
* Strict numeric QC is applied before training: implausible ET0/ETa per-day
  rates, Kcact tails, and unstable VI values (especially EVI denominator
  blow-ups) are removed rather than silently clipped into the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from catboost import CatBoostRegressor
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("catboost is required: pip/uv install catboost scikit-learn") from exc


ROOT = Path(__file__).resolve().parents[2]
BASE_PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
SM_DIR = ROOT / "data/raw/gee/kcact_maize_modis_indicators"
OUT_PARQUET = ROOT / "data/processed/train/ncp_summer_maize_selected_indicators.parquet"
OUT_CSV = ROOT / "outputs/tables/maize_selected_indicators_loyo.csv"

FEATURES = ["ndvi", "savi", "rdvi", "gndvi", "evi", "sm", "doy"]


DEFAULT_QC = {
    "kc_min": 0.02,
    "kc_max": 1.60,
    "et0_daily_min": 0.10,
    "et0_daily_max": 10.0,
    "eta_daily_min": 0.0,
    "eta_daily_max": 10.0,
    "evi_min": -1.0,
    "evi_max": 1.5,
    "sm_min": 0.02,
    "sm_max": 0.60,
}


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    radius_m = 6371008.8
    lat1 = np.radians(pd.to_numeric(lat1, errors="raise").astype(float))
    lon1 = np.radians(pd.to_numeric(lon1, errors="raise").astype(float))
    lat2 = np.radians(pd.to_numeric(lat2, errors="raise").astype(float))
    lon2 = np.radians(pd.to_numeric(lon2, errors="raise").astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * np.arcsin(np.sqrt(a))


def compute_rdvi_from_aligned_s2(df: pd.DataFrame) -> pd.Series:
    """Approximate RDVI from aligned S2 NDVI/SAVI.

    With NDVI = D/S and SAVI = 1.5D/(S+0.5), solve S=NIR+Red then compute
    RDVI = D/sqrt(S) = NDVI*sqrt(S).  This keeps RDVI on the same S2/MOD16
    window as the other VIs.  Invalid near-zero NDVI cases are left as NaN and
    excluded from the final training subset.
    """
    ndvi = pd.to_numeric(df["ndvi"], errors="coerce")
    savi = pd.to_numeric(df["savi"], errors="coerce")
    ratio = savi / ndvi.where(ndvi.abs() > 1e-9)
    sum_reflectance = (0.5 * ratio) / (1.5 - ratio)
    valid = np.isfinite(sum_reflectance) & (sum_reflectance > 0)
    return ndvi * np.sqrt(sum_reflectance.where(valid))


def apply_strict_numeric_qc(df: pd.DataFrame, qc: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """Drop physically implausible or numerically unstable rows.

    Thresholds are intentionally generous for NCP summer maize.  They remove
    clear data/algorithm artifacts (e.g. EVI denominator blow-ups, Kcact tails)
    while avoiding overfitting QC to the current model score.
    """
    out = df.copy()
    n0 = len(out)
    mask = pd.Series(True, index=out.index)
    drops: dict[str, int | float] = {"rows_before_qc": int(n0)}

    def add_rule(name: str, rule: pd.Series) -> None:
        nonlocal mask
        rule = rule.fillna(False)
        drops[f"drop_{name}"] = int((mask & ~rule).sum())
        mask &= rule

    window_days = (out["date_end"] - out["date_start"]).dt.days
    et0_daily = pd.to_numeric(out["et0_pm_8d_mm"], errors="coerce") / window_days
    eta_daily = pd.to_numeric(out["etc_8d_mm"], errors="coerce") / window_days
    kc = pd.to_numeric(out["kcact"], errors="coerce")

    out["et0_pm_daily_mm"] = et0_daily
    out["eta_daily_mm"] = eta_daily

    add_rule("window_not_8d", window_days == 8)
    add_rule("et0_daily_range", et0_daily.between(qc["et0_daily_min"], qc["et0_daily_max"], inclusive="both"))
    add_rule("eta_daily_range", eta_daily.between(qc["eta_daily_min"], qc["eta_daily_max"], inclusive="both"))
    add_rule("kcact_range", kc.between(qc["kc_min"], qc["kc_max"], inclusive="both"))

    for col in ["ndvi", "gndvi", "lswi"]:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            add_rule(f"{col}_range", s.between(-1.0, 1.0, inclusive="both"))
    if "savi" in out.columns:
        s = pd.to_numeric(out["savi"], errors="coerce")
        add_rule("savi_range", s.between(-1.0, 1.5, inclusive="both"))
    if "evi" in out.columns:
        s = pd.to_numeric(out["evi"], errors="coerce")
        add_rule("evi_range", s.between(qc["evi_min"], qc["evi_max"], inclusive="both"))
    if "rdvi" in out.columns:
        s = pd.to_numeric(out["rdvi"], errors="coerce")
        add_rule("rdvi_range", s.between(-2.0, 2.0, inclusive="both"))
    if "sm" in out.columns:
        s = pd.to_numeric(out["sm"], errors="coerce")
        add_rule("sm_range", s.between(qc["sm_min"], qc["sm_max"], inclusive="both"))

    out = out[mask].copy()
    drops["rows_after_qc"] = int(len(out))
    drops["rows_dropped_total"] = int(n0 - len(out))
    drops["drop_fraction"] = float((n0 - len(out)) / n0) if n0 else 0.0
    return out, drops


def date_to_window_map(windows: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map daily ERA5 dates to the MOD16 window-end date."""
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for row in windows[["date_start", "date_end", "date"]].drop_duplicates().itertuples(index=False):
        for day in pd.date_range(row.date_start, row.date_end - pd.Timedelta(days=1), freq="D"):
            mapping[pd.Timestamp(day)] = pd.Timestamp(row.date)
    return mapping


def parse_geo_to_coord_key(geo: pd.Series) -> pd.Series:
    coords = geo.astype(str).str.extract(r"\[([\-0-9.]+),([\-0-9.]+)\]")
    lon = pd.to_numeric(coords[0], errors="coerce").round(6)
    lat = pd.to_numeric(coords[1], errors="coerce").round(6)
    return lat.map(lambda v: f"{v:.6f}") + "_" + lon.map(lambda v: f"{v:.6f}")


def aggregate_sm_for_year(base_year: pd.DataFrame, sm_csv: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Aggregate daily ERA5-Land SM to each MOD16 8-day window."""
    wanted_coords = set(base_year["coord_key"].unique())
    date_map = date_to_window_map(base_year[["date_start", "date_end", "date"]])

    partials: list[pd.DataFrame] = []
    usecols = ["date", "mean", ".geo"]
    for chunk in pd.read_csv(sm_csv, usecols=usecols, chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["date"] = chunk["date"].map(date_map)
        chunk = chunk[chunk["date"].notna()]
        if chunk.empty:
            continue
        chunk["coord_key"] = parse_geo_to_coord_key(chunk[".geo"])
        chunk = chunk[chunk["coord_key"].isin(wanted_coords)]
        if chunk.empty:
            continue
        chunk["sm_sum"] = pd.to_numeric(chunk["mean"], errors="coerce")
        chunk = chunk[chunk["sm_sum"].notna()]
        chunk["sm_count"] = 1
        partials.append(
            chunk.groupby(["coord_key", "date"], as_index=False)[["sm_sum", "sm_count"]].sum()
        )

    if not partials:
        return pd.DataFrame(columns=["coord_key", "date", "sm"])

    sm = pd.concat(partials, ignore_index=True)
    sm = sm.groupby(["coord_key", "date"], as_index=False)[["sm_sum", "sm_count"]].sum()
    sm["sm"] = sm["sm_sum"] / sm["sm_count"]
    return sm[["coord_key", "date", "sm"]]


def prepare_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    cols = [
        "patch_id",
        "coord_key",
        "province",
        "date_start",
        "date_end",
        "date",
        "year",
        "centroid_lat",
        "centroid_lon",
        "centroid_lat_weather",
        "centroid_lon_weather",
        "centroid_lat_s2",
        "centroid_lon_s2",
        "ndvi",
        "savi",
        "gndvi",
        "evi",
        "doy",
        "kcact",
        "qc_valid",
    ]
    # Read all columns because new direct-RDVI exports add optional fields
    # (rdvi, s2_red, s2_nir), while older parquet files do not have them.
    df = pd.read_parquet(BASE_PARQUET)
    missing_required = sorted(set(cols) - set(df.columns))
    if missing_required:
        raise ValueError(f"Base parquet missing required columns: {missing_required}")
    keep = cols + [c for c in ["rdvi", "s2_red", "s2_nir", "etc_8d_mm", "et0_pm_8d_mm", "lswi"] if c in df.columns]
    df = df[dict.fromkeys(keep).keys()].copy()
    for col in ["date_start", "date_end", "date"]:
        df[col] = pd.to_datetime(df[col])
    df = df[df["qc_valid"]].copy()
    rows_qc_valid = int(len(df))

    # Verify Kcact target spatial/time alignment already present in base table.
    weather_dist = haversine_m(
        df["centroid_lat"],
        df["centroid_lon"],
        df["centroid_lat_weather"],
        df["centroid_lon_weather"],
    )
    s2_dist = haversine_m(
        df["centroid_lat"],
        df["centroid_lon"],
        df["centroid_lat_s2"],
        df["centroid_lon_s2"],
    )
    window_days = (df["date_end"] - df["date_start"]).dt.days
    if window_days.nunique() != 1 or int(window_days.iloc[0]) != 8:
        raise ValueError("Expected all MOD16 windows to be 8 days.")
    if not (df["date"] == df["date_end"]).all():
        raise ValueError("Expected date to equal date_end.")
    if float(np.nanmax(weather_dist)) > 0.01 or float(np.nanmax(s2_dist)) > 0.01:
        raise ValueError("Base table spatial alignment check failed.")

    has_direct_rdvi = "rdvi" in df.columns and df["rdvi"].notna().any()
    if args.rdvi_source == "direct" and not has_direct_rdvi:
        raise ValueError(
            "Direct RDVI is not present in the base parquet. Re-export S2 with "
            "the updated export_maize_kcact_training_data.py, rebuild the table, "
            "then rerun with --rdvi-source direct."
        )
    if args.rdvi_source == "fallback" or (args.rdvi_source == "auto" and not has_direct_rdvi):
        df["rdvi"] = compute_rdvi_from_aligned_s2(df)
        rdvi_source = "fallback_ndvi_savi_reconstruction"
        print(
            "WARNING: direct RDVI not found; using NDVI/SAVI reconstruction. "
            "For reported final results, re-export S2 and run --rdvi-source direct."
        )
    else:
        rdvi_source = "direct_s2_red_nir_export"

    qc = DEFAULT_QC | {
        "kc_min": args.kc_min,
        "kc_max": args.kc_max,
        "et0_daily_min": args.et0_daily_min,
        "et0_daily_max": args.et0_daily_max,
        "eta_daily_min": args.eta_daily_min,
        "eta_daily_max": args.eta_daily_max,
        "evi_min": args.evi_min,
        "evi_max": args.evi_max,
    }
    df, qc_report = apply_strict_numeric_qc(df, qc)

    sm_frames = []
    for year, base_year in df.groupby("year", sort=True):
        sm_csv = SM_DIR / f"maize_era5_sm_{int(year)}.csv"
        if not sm_csv.exists():
            print(f"WARNING: missing {sm_csv}")
            continue
        sm = aggregate_sm_for_year(base_year, sm_csv)
        sm_frames.append(sm)
        print(
            f"SM {int(year)}: {len(sm):,} coord-window rows, "
            f"{sm['coord_key'].nunique():,} coords"
        )

    if not sm_frames:
        raise FileNotFoundError(f"No maize_era5_sm_*.csv files found in {SM_DIR}")
    sm_all = pd.concat(sm_frames, ignore_index=True)
    df = df.merge(sm_all, on=["coord_key", "date"], how="left")
    sm_present = df["sm"].notna()
    sm_in_range = pd.to_numeric(df["sm"], errors="coerce").between(
        qc["sm_min"], qc["sm_max"], inclusive="both"
    )
    qc_report["drop_sm_range"] = int((sm_present & ~sm_in_range).sum())
    df = df[(~sm_present) | sm_in_range].copy()

    coverage = {feature: float(df[feature].notna().mean()) for feature in FEATURES}
    aligned = {
        "rows_qc_valid_before_strict_qc": rows_qc_valid,
        "rows_after_strict_qc_before_complete_case": int(len(df)),
        "rdvi_source": rdvi_source,
        "strict_qc": qc,
        "strict_qc_report": qc_report,
        "weather_max_distance_m": float(np.nanmax(weather_dist)),
        "s2_max_distance_m": float(np.nanmax(s2_dist)),
        "window_days_unique": sorted(window_days.unique().astype(int).tolist()),
        "date_equals_date_end": bool((df["date"] == df["date_end"]).all()),
        "feature_coverage": coverage,
    }

    train = df.dropna(subset=FEATURES + ["kcact"]).copy()
    train.to_parquet(OUT_PARQUET, index=False)
    aligned["rows_train_complete"] = int(len(train))
    aligned["patches_train_complete"] = int(train["patch_id"].nunique())
    return train, aligned


def fit_catboost(train: pd.DataFrame, use_gpu: bool = True) -> tuple[pd.DataFrame, dict]:
    years = sorted(train["year"].dropna().astype(int).unique())
    all_true, all_pred = [], []
    rows = []

    def make_model(task_type: str | None):
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

    task_type: str | None = "GPU" if use_gpu else None
    for year in years:
        tr = train[train["year"] != year]
        te = train[train["year"] == year]
        if len(te) < 10 or len(tr) < 10:
            continue
        model = make_model(task_type)
        try:
            model.fit(tr[FEATURES], tr["kcact"])
        except Exception as exc:
            if task_type == "GPU":
                print(f"GPU CatBoost failed ({exc}); retrying CPU.")
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
        print(
            f"{year}: n={len(te):,} R²={rows[-1]['r2']:.5f} "
            f"RMSE={rows[-1]['rmse']:.5f} MAE={rows[-1]['mae']:.5f}"
        )

    y_all = np.asarray(all_true)
    p_all = np.asarray(all_pred)
    overall = {
        "test_year": "pooled",
        "n_train": None,
        "n_test": int(len(y_all)),
        "r2": float(r2_score(y_all, p_all)),
        "rmse": float(np.sqrt(mean_squared_error(y_all, p_all))),
        "mae": float(mean_absolute_error(y_all, p_all)),
        "features": "+".join(FEATURES),
        "task_type": task_type or "CPU",
    }
    result = pd.concat([pd.DataFrame(rows), pd.DataFrame([overall])], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    return result, overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU CatBoost")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--rdvi-source",
        choices=["auto", "direct", "fallback"],
        default="auto",
        help=(
            "RDVI source. 'direct' requires an S2-exported rdvi column; "
            "'fallback' reconstructs from NDVI/SAVI for legacy data; "
            "'auto' prefers direct and falls back with a warning."
        ),
    )
    parser.add_argument("--kc-min", type=float, default=DEFAULT_QC["kc_min"])
    parser.add_argument("--kc-max", type=float, default=DEFAULT_QC["kc_max"])
    parser.add_argument("--et0-daily-min", type=float, default=DEFAULT_QC["et0_daily_min"])
    parser.add_argument("--et0-daily-max", type=float, default=DEFAULT_QC["et0_daily_max"])
    parser.add_argument("--eta-daily-min", type=float, default=DEFAULT_QC["eta_daily_min"])
    parser.add_argument("--eta-daily-max", type=float, default=DEFAULT_QC["eta_daily_max"])
    parser.add_argument("--evi-min", type=float, default=DEFAULT_QC["evi_min"])
    parser.add_argument("--evi-max", type=float, default=DEFAULT_QC["evi_max"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, alignment = prepare_dataset(args)
    print("\n=== Alignment / coverage ===")
    for key, value in alignment.items():
        print(f"{key}: {value}")
    print(f"Selected feature parquet: {OUT_PARQUET}")

    if args.prepare_only:
        return

    print("\n=== LOYO CatBoost: NDVI/SAVI/RDVI/GNDVI/EVI/SM/DOY ===")
    result, overall = fit_catboost(train, use_gpu=not args.cpu)
    print("\n=== Result ===")
    print(result.to_string(index=False))
    print(f"Saved: {OUT_CSV}")
    print(f"POOLED R²={overall['r2']:.5f} RMSE={overall['rmse']:.5f} MAE={overall['mae']:.5f}")


if __name__ == "__main__":
    main()
