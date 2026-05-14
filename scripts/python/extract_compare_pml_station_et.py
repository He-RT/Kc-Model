#!/usr/bin/env python3
"""Extract PML-V2 station ETa and plot linear comparisons with tower/MOD16.

PML-V2 bands are daily rates (mm/d) for 8-day composites. For cropland ETa we use
Ec + Es + Ei (vegetation transpiration + soil evaporation + interception), not
ET_water which is for water/snow/ice surfaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COORDS = ROOT / "data/processed/station_coordinates.csv"
OBS = ROOT / "data/processed/station_etc_with_et0.csv"
MOD16_MATCHED = ROOT / "outputs/tables/kc_mod16_vs_tower.csv"
PML_OUT = ROOT / "data/processed/pml_station_et.csv"
MATCH_OUT = ROOT / "outputs/tables/pml_vs_tower_mod16.csv"
STATS_OUT = ROOT / "outputs/tables/pml_vs_tower_mod16_linear_stats.csv"
STATS_JSON = ROOT / "outputs/tables/pml_vs_tower_mod16_linear_stats.json"
FIG_TOWER = ROOT / "outputs/figures/station_pml_vs_tower_eta_linear.png"
FIG_MOD16 = ROOT / "outputs/figures/station_pml_vs_mod16_eta_linear.png"
DEFAULT_PML_COLLECTION = "projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a"
LEGACY_PML_COLLECTION = "CAS/IGSNRR/PML/V2_v018"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-id", default="chuang-yaogan")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--asset", default=DEFAULT_PML_COLLECTION)
    p.add_argument("--tag", default=None, help="Output suffix, default inferred from asset")
    return p.parse_args()


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def output_paths(tag: str) -> dict[str, Path]:
    return {
        "pml": ROOT / f"data/processed/pml_station_et_{tag}.csv",
        "match": ROOT / f"outputs/tables/pml_{tag}_vs_tower_mod16.csv",
        "stats": ROOT / f"outputs/tables/pml_{tag}_vs_tower_mod16_linear_stats.csv",
        "stats_json": ROOT / f"outputs/tables/pml_{tag}_vs_tower_mod16_linear_stats.json",
        "fig_tower": ROOT / f"outputs/figures/station_pml_{tag}_vs_tower_eta_linear.png",
        "fig_mod16": ROOT / f"outputs/figures/station_pml_{tag}_vs_mod16_eta_linear.png",
    }


def extract_pml(project_id: str, asset: str, paths: dict[str, Path], force: bool = False) -> pd.DataFrame:
    pml_out = paths["pml"]
    if pml_out.exists() and not force:
        df = pd.read_csv(pml_out)
        df["date"] = pd.to_datetime(df["date"])
        df["date_end"] = pd.to_datetime(df["date_end"])
        return df

    init_ee(project_id)
    coords = pd.read_csv(COORDS)
    feats = []
    for row in coords.itertuples(index=False):
        feats.append(ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {"station": row.station}))
    pts = ee.FeatureCollection(feats)

    obs = pd.read_csv(OBS)
    start = pd.to_datetime(obs["date_prev"]).min() - pd.Timedelta(days=16)
    end = pd.to_datetime(obs["date"]).max() + pd.Timedelta(days=16)

    col = ee.ImageCollection(asset).filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    def per_img(img):
        img = ee.Image(img)
        date = ee.Date(img.get("system:time_start"))
        eta = img.select("Ec").add(img.select("Es")).add(img.select("Ei")).rename("pml_eta_mm_d")
        names = img.bandNames()
        ew = ee.Image(ee.Algorithms.If(names.contains("Ew"), img.select("Ew"), img.select("ET_water"))).rename("Ew")
        et = ee.Image(ee.Algorithms.If(names.contains("ET"), img.select("ET"), eta)).rename("pml_et_mm_d")
        bands = ee.Image.cat([img.select("Ec"), img.select("Es"), img.select("Ei"), ew, et, eta])
        fc = bands.reduceRegions(collection=pts, reducer=ee.Reducer.mean(), scale=500, tileScale=4)
        return fc.map(lambda f: f.set({"date": date.format("YYYY-MM-dd"), "date_end": date.advance(8, "day").format("YYYY-MM-dd"), "pml_asset": asset}))

    fc = ee.FeatureCollection(col.map(per_img).flatten())
    info = fc.getInfo()
    rows = []
    for feat in info["features"]:
        props = feat["properties"]
        rows.append({
            "station": props.get("station"),
            "date": props.get("date"),
            "date_end": props.get("date_end"),
            "Ec": props.get("Ec"),
            "Es": props.get("Es"),
            "Ei": props.get("Ei"),
            "Ew": props.get("Ew"),
            "pml_et_mm_d": props.get("pml_et_mm_d"),
            "pml_eta_mm_d": props.get("pml_eta_mm_d"),
            "pml_asset": props.get("pml_asset"),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["date_end"] = pd.to_datetime(df["date_end"])
    for c in ["Ec", "Es", "Ei", "Ew", "pml_et_mm_d", "pml_eta_mm_d"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # PML V22a stores flux bands with a 0.01 scale factor in this EE asset;
    # legacy v018 is already in mm/d. Guard by asset name and value magnitude.
    if "V22a" in asset or df["pml_eta_mm_d"].median(skipna=True) > 50:
        for c in ["Ec", "Es", "Ei", "Ew", "pml_et_mm_d", "pml_eta_mm_d"]:
            df[c] = df[c] * 0.01
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    pml_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(pml_out, index=False)
    return df


def weighted_overlap_mean(series: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, value_col: str) -> tuple[float, int, float]:
    overlapping = series[(series["date"] < t1) & (series["date_end"] > t0)].copy()
    if overlapping.empty:
        return np.nan, 0, 0.0
    overlap_start = overlapping["date"].where(overlapping["date"] > t0, t0)
    overlap_end = overlapping["date_end"].where(overlapping["date_end"] < t1, t1)
    days = (overlap_end - overlap_start).dt.total_seconds() / 86400.0
    vals = pd.to_numeric(overlapping[value_col], errors="coerce")
    ok = vals.notna() & np.isfinite(vals) & (days > 0)
    if not ok.any():
        return np.nan, int(len(overlapping)), float(days.sum())
    return float(np.average(vals[ok], weights=days[ok])), int(ok.sum()), float(days[ok].sum())


def match_windows(pml: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    obs = pd.read_csv(OBS)
    obs["date"] = pd.to_datetime(obs["date"])
    obs["date_prev"] = pd.to_datetime(obs["date_prev"])
    mod = pd.read_csv(MOD16_MATCHED)
    mod["date"] = pd.to_datetime(mod["date"])
    mod["date_prev"] = pd.to_datetime(mod["date_prev"])
    keep_mod = mod[["station", "date", "date_prev", "etc_mod16_mm_d", "n_mod16_overlap"]]

    rows = []
    for row in obs.itertuples(index=False):
        st = row.station
        t0, t1 = pd.Timestamp(row.date_prev), pd.Timestamp(row.date)
        s = pml[pml["station"] == st]
        pml_eta, n_overlap, overlap_days = weighted_overlap_mean(s, t0, t1, "pml_eta_mm_d")
        rows.append({
            "station": st,
            "date": t1,
            "date_prev": t0,
            "etc_tower_mm_d": row.etc_obs_mm_d,
            "et0_pm_mm_d": row.et0_pm_mean_mm_d,
            "n_days_window": row.n_days_window,
            "pml_eta_mm_d": pml_eta,
            "n_pml_overlap": n_overlap,
            "pml_overlap_days": overlap_days,
        })
    out = pd.DataFrame(rows)
    out = out.merge(keep_mod, on=["station", "date", "date_prev"], how="left")
    out["kcact_pml"] = out["pml_eta_mm_d"] / out["et0_pm_mm_d"]
    paths["match"].parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths["match"], index=False)
    return out


def setup_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    for f in ["PingFang SC", "Arial Unicode MS", "Heiti SC", "Songti SC", "Noto Sans CJK SC"]:
        if any(f in font.name for font in font_manager.fontManager.ttflist):
            rcParams["font.sans-serif"] = [f]
            break
    rcParams["axes.unicode_minus"] = False


def calc_stats(df: pd.DataFrame, xcol: str, ycol: str, label: str) -> pd.DataFrame:
    rows = []
    for station, g in [("ALL", df), *list(df.groupby("station"))]:
        d = g[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[(d[xcol] >= 0) & (d[ycol] >= 0)]
        if len(d) >= 2 and d[xcol].std() > 0 and d[ycol].std() > 0:
            r = float(d[xcol].corr(d[ycol]))
            slope, intercept = np.polyfit(d[xcol].to_numpy(), d[ycol].to_numpy(), 1)
            r2 = r * r
        else:
            r = r2 = slope = intercept = np.nan
        diff = d[ycol] - d[xcol]
        rows.append({
            "comparison": label,
            "station": station,
            "n": int(len(d)),
            "pearson_r": r,
            "linear_r2": r2,
            "slope": float(slope) if np.isfinite(slope) else np.nan,
            "intercept": float(intercept) if np.isfinite(intercept) else np.nan,
            "rmse_y_minus_x": float(np.sqrt(np.mean(diff**2))) if len(d) else np.nan,
            "mae_y_minus_x": float(np.mean(np.abs(diff))) if len(d) else np.nan,
            "bias_y_minus_x": float(np.mean(diff)) if len(d) else np.nan,
            "relative_bias_pct": float((d[ycol].sum() - d[xcol].sum()) / d[xcol].sum() * 100) if len(d) and d[xcol].sum() else np.nan,
            "x_mean": float(d[xcol].mean()) if len(d) else np.nan,
            "y_mean": float(d[ycol].mean()) if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_linear(df: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, title: str, out: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    setup_font()
    d = df[["station", xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d[xcol] >= 0) & (d[ycol] >= 0)].copy()
    stats = calc_stats(d, xcol, ycol, title)
    overall = stats[stats["station"] == "ALL"].iloc[0]
    slope = overall["slope"]
    intercept = overall["intercept"]
    colors = {"位山":"#4C78A8", "禹城":"#F58518", "馆陶":"#54A24B", "栾城":"#B279A2"}
    fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=180)
    for station, g in d.groupby("station"):
        ax.scatter(g[xcol], g[ycol], s=24, alpha=0.68, label=f"{station} (n={len(g)})", color=colors.get(station), edgecolor="white", linewidth=0.25)
    lim = max(float(d[xcol].max()), float(d[ycol].max())) * 1.05
    lim = max(lim, 7)
    xx = np.linspace(0, lim, 200)
    ax.plot(xx, xx, color="black", lw=1.2, ls="--", label="1:1 line")
    if np.isfinite(slope):
        ax.plot(xx, slope * xx + intercept, color="#D62728", lw=2.2, label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
    txt = (f"n = {int(overall['n'])}\n"
           f"Pearson r = {overall['pearson_r']:.3f}\n"
           f"Linear R² = {overall['linear_r2']:.3f}\n"
           f"RMSE = {overall['rmse_y_minus_x']:.2f} mm/day\n"
           f"Bias = {overall['bias_y_minus_x']:.2f} mm/day ({overall['relative_bias_pct']:.1f}%)")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#BBBBBB", alpha=0.92))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return stats


def main() -> None:
    args = parse_args()
    tag = args.tag or ("v22a" if "V22a" in args.asset else "v018")
    paths = output_paths(tag)
    pml = extract_pml(args.project_id, args.asset, paths, force=args.force_extract)
    print(f"PML station ET rows: {len(pml):,}; saved {paths['pml']}")
    print(f"coverage: {pml['date'].min().date()} to {pml['date'].max().date()}, stations={sorted(pml['station'].dropna().unique().tolist())}")
    matched = match_windows(pml, paths)
    print(f"Matched rows: {len(matched):,}; saved {paths['match']}")
    s1 = plot_linear(
        matched,
        "etc_tower_mm_d",
        "pml_eta_mm_d",
        "Tower observed ETa (mm/day)",
        f"PML {tag} ETa = Ec+Es+Ei (mm/day)",
        f"Station ETa: PML {tag} vs Tower Observations",
        paths["fig_tower"],
    )
    s2 = plot_linear(
        matched,
        "etc_mod16_mm_d",
        "pml_eta_mm_d",
        "MOD16 ETa (mm/day)",
        f"PML {tag} ETa = Ec+Es+Ei (mm/day)",
        f"Station ETa: PML {tag} vs MOD16",
        paths["fig_mod16"],
    )
    stats = pd.concat([s1, s2], ignore_index=True)
    stats.to_csv(paths["stats"], index=False)
    paths["stats_json"].write_text(json.dumps(stats.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Figures:\n  {paths['fig_tower']}\n  {paths['fig_mod16']}")
    print(f"Stats: {paths['stats']}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
