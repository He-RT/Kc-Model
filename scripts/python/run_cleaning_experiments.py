"""Run data-cleaning experiments and write results to a markdown document.

Experiments:
  1. Baseline: CatBoost LOYO, all qc_valid samples (R²≈0.702)
  2. Strict MOD16 QC: only qc_mod16 == 0 samples
  3. Kcact smoothing: train on 3-window rolling-median smoothed Kcact

All comparisons done on raw Kcact — no evaluation on smoothed data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

ROOT = Path("/Users/hert/Projects/dcsdxx")
INPUT_TABLE = ROOT / "data/processed/train/hebei_winter_wheat_kcact_train_ready.parquet"
OUTPUT_MD = ROOT / "outputs/reports/kcact_data_cleaning_experiments.md"

EXCLUDE_COLS = {
    "patch_id", "point_id", "date", "date_start", "date_end",
    "province", "crop_type", "qc_valid", "kcact",
    "etc_8d_mm", "et0_pm_8d_mm", "qc_mod16",
    ".geo", "system:index", "valid_obs",
}


def load_feature_cols(df):
    return [c for c in df.select_dtypes(include=["number", "bool"]).columns
            if c not in EXCLUDE_COLS and c != "year"]


def make_catboost():
    return CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        random_seed=42, verbose=0, thread_count=-1)


def ndvi_stage(ndvi):
    if ndvi < 0.35:
        return "early"
    elif ndvi < 0.60:
        return "mid"
    else:
        return "late"


def run_loyo_cv(df, feature_cols, target_col, label, years):
    """Run LOYO CV and return (per_year_df, pooled_dict)."""
    results = []
    y_all_true, y_all_pred = [], []
    y_all_ndvi = []

    for test_year in years:
        train = df[df["year"] != test_year]
        test = df[df["year"] == test_year]

        X_train = train[feature_cols].fillna(0.0)
        y_train = train[target_col].values
        X_test = test[feature_cols].fillna(0.0)

        y_test_raw = test["kcact"].values  # always evaluate on raw Kcact
        ndvi_test = test["ndvi"].values

        m = make_catboost()
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        r2 = r2_score(y_test_raw, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
        mae = mean_absolute_error(y_test_raw, y_pred)

        results.append({
            "experiment": label, "test_year": int(test_year),
            "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
            "n_samples": len(y_test_raw),
        })

        y_all_true.extend(y_test_raw)
        y_all_pred.extend(y_pred)
        y_all_ndvi.extend(ndvi_test)

    yt = np.array(y_all_true)
    yp = np.array(y_all_pred)
    ndvi_arr = np.array(y_all_ndvi)
    pooled = {
        "experiment": label, "test_year": "pooled",
        "r2": round(float(r2_score(yt, yp)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "mae": round(float(mean_absolute_error(yt, yp)), 4),
        "n_samples": len(yt),
        "n_features": len(feature_cols),
    }

    # Per-stage
    stages = {}
    for s_name, lo, hi in [("early", 0, 0.35), ("mid", 0.35, 0.60), ("late", 0.60, 1.0)]:
        m = (ndvi_arr >= lo) & (ndvi_arr < hi)
        if m.sum() > 0:
            stages[s_name] = {
                "n": int(m.sum()),
                "r2": round(float(r2_score(yt[m], yp[m])), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(yt[m], yp[m]))), 4),
                "mae": round(float(mean_absolute_error(yt[m], yp[m])), 4),
                "mean_kc": round(float(yt[m].mean()), 3),
            }

    return results, pooled, stages


def build_markdown(baseline_res, strict_qc_res, smooth_res, df_full, df_strict):
    """Assemble the markdown report."""
    b_per, b_pool, b_stages = baseline_res
    q_per, q_pool, q_stages = strict_qc_res
    s_per, s_pool, s_stages = smooth_res

    lines = []
    lines.append("# Kcact Data-Cleaning Experiments")
    lines.append("")
    lines.append(f"**Date**: 2026-04-25  |  **Model**: CatBoost (500 iters, depth=6, lr=0.05)")
    lines.append("")
    lines.append(f"**Baseline**: {b_pool['n_samples']:,} samples, {b_pool['n_features']} features, pooled R² = **{b_pool['r2']:.4f}**")
    lines.append("")

    # ---- Summary Table ----
    lines.append("## 1. Overall Comparison (Pooled LOYO)")
    lines.append("")
    lines.append("| Experiment | Samples | R² | RMSE | MAE | Δ R² vs Baseline |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for lab, p in [("Baseline (all QC)", b_pool), ("Strict MOD16 QC", q_pool), ("Smoothed Kcact", s_pool)]:
        delta = p["r2"] - b_pool["r2"]
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {lab} | {p['n_samples']:,} | {p['r2']:.4f} | {p['rmse']:.4f} | {p['mae']:.4f} | {sign}{delta:.4f} |")
    lines.append("")

    # ---- Per-Year Detail ----
    lines.append("## 2. Per-Year Breakdown")
    lines.append("")
    years = sorted(set(r["test_year"] for r in b_per if r["test_year"] != "pooled"))
    lines.append("| Year | Baseline R² | Strict QC R² | Smooth R² | Best |")
    lines.append("|---:|---:|---:|---:|---:|")

    by_year = {}
    for r in b_per:
        by_year[r["test_year"]] = {"base": r["r2"]}
    for r in q_per:
        by_year[r["test_year"]]["qc"] = r["r2"]
    for r in s_per:
        by_year[r["test_year"]]["smooth"] = r["r2"]

    for yr in years:
        b = by_year[yr]["base"]
        q = by_year[yr].get("qc", float("nan"))
        s = by_year[yr].get("smooth", float("nan"))
        best = max(b, q, s)
        best_label = ""
        if abs(best - b) < 0.001:
            best_label = "Baseline"
        elif abs(best - q) < 0.001:
            best_label = "Strict QC"
        elif abs(best - s) < 0.001:
            best_label = "Smooth"
        lines.append(f"| {yr} | {b:.4f} | {q:.4f} | {s:.4f} | **{best_label}** |")
    lines.append("")

    # ---- Per-Stage ----
    lines.append("## 3. Per-NDVI-Stage Comparison")
    lines.append("")
    lines.append("| Stage | Baseline R² | Strict QC R² | Smooth R² | Baseline n | Strict QC n |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for stage in ["early", "mid", "late"]:
        br2 = b_stages.get(stage, {}).get("r2", float("nan"))
        qr2 = q_stages.get(stage, {}).get("r2", float("nan"))
        sr2 = s_stages.get(stage, {}).get("r2", float("nan"))
        bn = b_stages.get(stage, {}).get("n", 0)
        qn = q_stages.get(stage, {}).get("n", 0)
        best = max(br2, qr2, sr2)
        bf = f"**{br2:.4f}**" if abs(best - br2) < 0.001 else f"{br2:.4f}"
        qf = f"**{qr2:.4f}**" if abs(best - qr2) < 0.001 else f"{qr2:.4f}"
        sf = f"**{sr2:.4f}**" if abs(best - sr2) < 0.001 else f"{sr2:.4f}"
        lines.append(f"| {stage} | {bf} | {qf} | {sf} | {bn:,} | {qn:,} |")
    lines.append("")

    # ---- Sample Count Details ----
    lines.append("## 4. Sample Counts")
    lines.append("")
    lines.append(f"- Baseline: {b_pool['n_samples']:,} samples (all qc_valid)")
    lines.append(f"- Strict QC: {q_pool['n_samples']:,} samples (qc_mod16 == 0 only)")
    lines.append(f"  - Lost: {b_pool['n_samples'] - q_pool['n_samples']:,} samples ({(b_pool['n_samples'] - q_pool['n_samples']) / b_pool['n_samples'] * 100:.1f}%)")
    lines.append(f"- Smoothing: same sample count as baseline ({b_pool['n_samples']:,})")
    lines.append("")

    # ---- Analysis ----
    lines.append("## 5. Analysis & Recommendations")
    lines.append("")

    delta_qc = q_pool["r2"] - b_pool["r2"]
    delta_smooth = s_pool["r2"] - b_pool["r2"]

    lines.append("### Strict MOD16 QC")
    if delta_qc > 0.002:
        lines.append(f"**IMPROVEMENT**: +{delta_qc:.4f} R². Stricter QC helps. Consider using qc_mod16==0 as the default filter.")
    elif delta_qc > -0.002:
        lines.append(f"**NEUTRAL**: {delta_qc:+.4f} R². The gain from cleaner data is offset by sample loss. May still be worthwhile for operational use (cleaner predictions).")
    else:
        lines.append(f"**DEGRADATION**: {delta_qc:+.4f} R². The lost samples contained useful signal. Keep current QC approach.")
    lines.append("")

    lines.append("### Kcact Time-Series Smoothing")
    if delta_smooth > 0.002:
        lines.append(f"**IMPROVEMENT**: +{delta_smooth:.4f} R². Smoothing removes real noise. Recommend adding to pipeline.")
    elif delta_smooth > -0.002:
        lines.append(f"**NEUTRAL**: {delta_smooth:+.4f} R². Smoothing doesn't help. The noise in Kcact appears to be genuine crop variability, not measurement artifact.")
    else:
        lines.append(f"**DEGRADATION**: {delta_smooth:+.4f} R². Smoothing removes real physiological signal. Do NOT use smoothing.")
    lines.append("")

    lines.append("### Recommendation")
    best_r2 = max(b_pool["r2"], q_pool["r2"], s_pool["r2"])
    best_name = "Baseline" if abs(best_r2 - b_pool["r2"]) < 0.001 else \
                ("Strict MOD16 QC" if abs(best_r2 - q_pool["r2"]) < 0.001 else "Smoothed Kcact")
    lines.append(f"The best-performing approach is **{best_name}** at R² = **{best_r2:.4f}**.")
    lines.append("")

    return "\n".join(lines)


def main():
    # ---- Load data ----
    df_all = pd.read_parquet(INPUT_TABLE)
    df_all = df_all[df_all["qc_valid"]].copy()
    feature_cols = load_feature_cols(df_all)
    years = sorted(df_all["year"].dropna().unique().astype(int))

    # ---- Dataset 2: Strict QC ----
    df_strict = df_all[df_all["qc_mod16"] == 0].copy()

    # ---- Compute smoothed Kcact ----
    df_smooth = df_all.copy()
    df_smooth = df_smooth.sort_values(["patch_id", "date"])
    df_smooth["kcact_smooth"] = (
        df_smooth.groupby("patch_id")["kcact"]
        .transform(lambda s: s.rolling(3, center=True, min_periods=1).median())
    )

    # ---- Run experiments ----
    print("Running baseline...")
    base_res = run_loyo_cv(df_all, feature_cols, "kcact", "baseline", years)

    print("Running strict QC...")
    qc_res = run_loyo_cv(df_strict, feature_cols, "kcact", "strict_qc", years)

    print("Running smoothed Kcact...")
    smooth_res = run_loyo_cv(df_smooth, feature_cols, "kcact_smooth", "smooth_kcact", years)

    # ---- Build & write markdown ----
    md = build_markdown(base_res, qc_res, smooth_res, df_all, df_strict)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(md)
    print(f"\nReport written to {OUTPUT_MD}")
    print(f"Baseline: R²={base_res[1]['r2']:.4f}  n={base_res[1]['n_samples']:,}")
    print(f"Strict QC: R²={qc_res[1]['r2']:.4f}  n={qc_res[1]['n_samples']:,}")
    print(f"Smoothed:  R²={smooth_res[1]['r2']:.4f}  n={smooth_res[1]['n_samples']:,}")


if __name__ == "__main__":
    main()
