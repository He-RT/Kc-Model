"""Merge MODIS NDVI into training table and evaluate via CatBoost LOYO CV.

Usage:
  python scripts/python/merge_validate_modis_ndvi.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import glob
import sys

ROOT = Path("/Users/hert/Projects/dcsdxx")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kcact.utils.gpu import get_gpu_config, make_catboost_params
INPUT_TABLE = ROOT / "data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet"
MODIS_CSV_DIR = ROOT / "data/raw/gee"
OUTPUT_TABLE = ROOT / "data/processed/train/hebei_winter_wheat_kcact_with_modis.parquet"
REPORT_MD = ROOT / "outputs/reports/kcact_modis_ndvi_results.md"

EXCLUDE_COLS = {
    "patch_id", "point_id", "date", "date_start", "date_end",
    "province", "crop_type", "qc_valid", "kcact",
    "etc_8d_mm", "et0_pm_8d_mm", "qc_mod16",
    ".geo", "system:index", "valid_obs",
}

# MODIS NDVI columns to merge (exclude metadata already in training table)
MODIS_MERGE_COLS = ["modis_ndvi", "modis_evi", "modis_qa"]


def load_feature_cols(df):
    return [c for c in df.select_dtypes(include=["number", "bool"]).columns
            if c not in EXCLUDE_COLS and c != "year"]


def make_catboost():
    return CatBoostRegressor(**make_catboost_params(extra={
        "iterations": 500, "depth": 6, "learning_rate": 0.05,
        "random_seed": 42, "verbose": 0}))


def load_modis_csvs():
    """Load and concatenate all MODIS NDVI CSV files."""
    files = sorted(glob.glob(str(MODIS_CSV_DIR / "hebei_kcact_modis_ndvi_*.csv")))
    if not files:
        print(f"No MODIS NDVI CSV files found in {MODIS_CSV_DIR}")
        return None

    dfs = []
    for f in files:
        print(f"  Reading {Path(f).name}...")
        df = pd.read_csv(f)
        # Normalize: point_id is the same as patch_id (pt_ prefixed)
        if "point_id" in df.columns and "patch_id" not in df.columns:
            df["patch_id"] = df["point_id"].astype(str)
        elif "patch_id" in df.columns:
            df["patch_id"] = df["patch_id"].astype(str)
        # Keep only merge keys + MODIS feature columns
        merge_cols = ["patch_id", "date_start", "date_end", "date"]
        available_merge = [c for c in merge_cols if c in df.columns]
        available_feat = [c for c in MODIS_MERGE_COLS if c in df.columns]
        keep = available_merge + available_feat
        dfs.append(df[keep].copy())

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Total MODIS rows: {len(result)}")
    return result


def merge_modis_into_training(table_df, modis_df):
    """Merge MODIS NDVI columns into training table on (patch_id, date_start, date_end)."""
    merge_keys = ["patch_id", "date_start", "date_end"]

    # Ensure types match
    for k in merge_keys:
        if k in table_df.columns and k in modis_df.columns:
            table_df[k] = table_df[k].astype(str)
            modis_df[k] = modis_df[k].astype(str)

    # Keep only merge keys + feature columns in modis
    feat_cols = [c for c in MODIS_MERGE_COLS if c in modis_df.columns]
    modis_subset = modis_df[merge_keys + feat_cols].drop_duplicates(subset=merge_keys)

    merged = table_df.merge(modis_subset, on=merge_keys, how="left")
    for c in feat_cols:
        merged[c] = merged[c].fillna(0.0)

    return merged


def run_loyo_cv(df, feature_cols, target, label, years):
    per_year = []
    y_true, y_pred = [], []
    for test_year in years:
        train = df[df["year"] != test_year]
        test = df[df["year"] == test_year]
        X_tr = train[feature_cols].fillna(0.0)
        y_tr = train[target].values
        X_te = test[feature_cols].fillna(0.0)
        y_te = test[target].values
        m = make_catboost()
        m.fit(X_tr, y_tr)
        yp = m.predict(X_te)
        per_year.append({"test_year": int(test_year), "r2": round(float(r2_score(y_te, yp)), 4),
                         "rmse": round(float(np.sqrt(mean_squared_error(y_te, yp))), 4),
                         "mae": round(float(mean_absolute_error(y_te, yp)), 4),
                         "n": len(y_te)})
        y_true.extend(y_te)
        y_pred.extend(yp)
    yt = np.array(y_true)
    yp_arr = np.array(y_pred)
    pooled = {"test_year": "pooled", "r2": round(float(r2_score(yt, yp_arr)), 4),
              "rmse": round(float(np.sqrt(mean_squared_error(yt, yp_arr))), 4),
              "mae": round(float(mean_absolute_error(yt, yp_arr)), 4),
              "n": len(yt), "features": len(feature_cols)}
    return per_year, pooled


def build_markdown(baseline_per, baseline_pooled, modis_per, modis_pooled,
                   n_merged, n_total, modis_feature_importance=None):
    lines = [
        "# MODIS NDVI Integration Results",
        "",
        f"**Date**: 2026-04-25 | **Model**: CatBoost (500 iters, depth=6, lr=0.05)",
        "",
        "## 1. Merge Statistics",
        "",
        f"- Total training samples: {n_total:,}",
        f"- Successfully merged with MODIS NDVI: {n_merged:,} ({n_merged/n_total*100:.1f}%)",
        f"- New feature columns: `modis_ndvi`, `modis_evi`, `modis_qa`",
        "",
        "##  2. Overall Comparison (Pooled LOYO)",
        "",
        "| Experiment | Features | R² | RMSE | MAE | Δ R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    delta = modis_pooled["r2"] - baseline_pooled["r2"]
    sgn = "+" if delta >= 0 else ""
    lines.append(f"| Baseline (46 S2 features) | {baseline_pooled['features']} | {baseline_pooled['r2']:.4f} | {baseline_pooled['rmse']:.4f} | {baseline_pooled['mae']:.4f} | — |")
    lines.append(f"| + MODIS NDVI | {modis_pooled['features']} | {modis_pooled['r2']:.4f} | {modis_pooled['rmse']:.4f} | {modis_pooled['mae']:.4f} | {sgn}{delta:.4f} |")

    lines.append("")
    lines.append("## 3. Per-Year Breakdown")
    lines.append("")
    lines.append("| Year | Baseline R² | +MODIS NDVI R² | Δ | Best |")
    lines.append("|---:|---:|---:|---:|---:|")

    by_yr = {}
    for r in baseline_per:
        by_yr[r["test_year"]] = {"base": r["r2"]}
    for r in modis_per:
        by_yr.setdefault(r["test_year"], {})["modis"] = r["r2"]

    for yr in sorted(by_yr.keys()):
        b = by_yr[yr]["base"]
        m_val = by_yr[yr].get("modis", float("nan"))
        d = m_val - b
        sgn2 = "+" if d >= 0 else ""
        best = "Baseline" if b >= m_val else "MODIS NDVI"
        lines.append(f"| {yr} | {b:.4f} | {m_val:.4f} | {sgn2}{d:.4f} | **{best}** |")

    lines.append("")
    lines.append("## 4. Feature Importance (Top-10, +MODIS NDVI model)")
    lines.append("")
    if modis_feature_importance:
        lines.append("| Rank | Feature | Importance |")
        lines.append("|---:|---|---:|")
        for i, (name, imp) in enumerate(modis_feature_importance[:10], 1):
            lines.append(f"| {i} | {name} | {imp:.4f} |")

    lines.append("")
    lines.append("## 5. Go/No-Go Decision")
    lines.append("")
    if delta > 0.005:
        lines.append(f"**GO**: MODIS NDVI improves R² by +{delta:.4f}. Proceed to Phase 2 (multi-year MODIS 2000–2018).")
    elif delta > 0:
        lines.append(f"**BORDERLINE**: MODIS NDVI improves R² by +{delta:.4f} (significant but small). Consider Phase 2 if operational value of cleaner predictions justifies cost.")
    else:
        lines.append(f"**NO-GO**: MODIS NDVI does not improve R² ({delta:+.4f}). Skip Phase 2 multi-year MODIS export.")

    return "\n".join(lines)


def main():
    cfg = get_gpu_config()
    print(cfg.summary())
    # ---- Load training table ----
    table = pd.read_parquet(INPUT_TABLE)
    table = table[table["qc_valid"]].copy()
    for c in ["patch_id", "date_start", "date_end"]:
        table[c] = table[c].astype(str)
    n_total = len(table)
    years = sorted(table["year"].dropna().unique().astype(int))

    # ---- Load and merge MODIS NDVI ----
    modis_df = load_modis_csvs()
    if modis_df is None:
        print("No MODIS NDVI files found. Skipping.")
        return

    merged = merge_modis_into_training(table, modis_df)
    n_merged = int(merged["modis_ndvi"].notna().sum())
    print(f"Merged: {n_merged}/{n_total} rows have MODIS NDVI data")

    # ---- Baseline: S2-only features ----
    base_feat = load_feature_cols(table)
    safe_base_feat = [c for c in base_feat if c in table.columns]
    print(f"\nRunning baseline ({len(safe_base_feat)} features)...")
    base_per, base_pooled = run_loyo_cv(table, safe_base_feat, "kcact", "baseline", years)
    print(f"  Pooled R²={base_pooled['r2']:.4f} RMSE={base_pooled['rmse']:.4f}")

    # ---- MODIS: S2 + MODIS NDVI features ----
    modis_feat = load_feature_cols(merged)
    safe_modis_feat = [c for c in modis_feat if c in merged.columns]
    new_modis_cols = [c for c in ["modis_ndvi", "modis_evi", "modis_qa"] if c in safe_modis_feat]
    print(f"\nRunning +MODIS NDVI ({len(safe_modis_feat)} features, +{new_modis_cols})...")
    modis_per, modis_pooled = run_loyo_cv(merged, safe_modis_feat, "kcact", "modis_ndvi", years)
    print(f"  Pooled R²={modis_pooled['r2']:.4f} RMSE={modis_pooled['rmse']:.4f}")

    # ---- Feature importance ----
    print("\nComputing feature importance...")
    m = make_catboost()
    X_all = merged[safe_modis_feat].fillna(0.0)
    m.fit(X_all, merged["kcact"])
    importance = sorted(zip(safe_modis_feat, m.feature_importances_), key=lambda x: x[1], reverse=True)

    # ---- Save ----
    merged.to_parquet(OUTPUT_TABLE, index=False)
    print(f"Merged table saved to {OUTPUT_TABLE}")

    # ---- Report ----
    md = build_markdown(base_per, base_pooled, modis_per, modis_pooled,
                        n_merged, n_total, importance)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(md)
    print(f"Report written to {REPORT_MD}")

    # ---- Summary ----
    delta = modis_pooled["r2"] - base_pooled["r2"]
    print(f"\n=== PHASE 1 RESULT ===")
    print(f"  Baseline R²: {base_pooled['r2']:.4f}")
    print(f"  +MODIS NDVI R²: {modis_pooled['r2']:.4f}")
    print(f"  Δ: {delta:+.4f}")
    if delta > 0.005:
        print("  → GO for Phase 2 (multi-year MODIS)")
    elif delta > 0:
        print("  → BORDERLINE — review before Phase 2")
    else:
        print("  → NO-GO — skip Phase 2")


if __name__ == "__main__":
    main()
