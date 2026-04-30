"""Merge GEE-exported MODIS indicators into maize training parquet and retrain."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
CSV_DIR = ROOT / "data/raw/gee/modis_indicators"
OUT_PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_with_modis.parquet"
OUT_DIR = ROOT / "outputs/tables"

def load_modis_csvs():
    frames = {}
    products = {"fpar": "Fpar_500m", "lst": ["LST_Day_1km","LST_Night_1km"],
                "albedo": "Albedo_WSA_shortwave", "sm": "volumetric_soil_water_layer_1"}
    for prefix, bands in products.items():
        files = sorted(CSV_DIR.glob(f"maize_{prefix}_*.csv"))
        if not files:
            print(f"  WARNING: No {prefix} files found in {CSV_DIR}")
            continue
        dfs = []
        for f in files:
            d = pd.read_csv(f)
            d["date"] = pd.to_datetime(d["date"])
            d["point_id"] = d["point_id"].astype(str)
            if isinstance(bands, list):
                keep = ["point_id","date"] + bands
            else:
                keep = ["point_id","date",bands]
            d = d[[c for c in keep if c in d.columns]]
            if isinstance(bands, str) and bands in d.columns:
                d = d.rename(columns={bands: prefix})
            dfs.append(d)
        frames[prefix] = pd.concat(dfs, ignore_index=True)
        print(f"  {prefix}: {len(frames[prefix])} rows from {len(files)} files")
    return frames

def merge_with_parquet(df, modis):
    df = df.copy()
    df["point_id"] = df["point_id"].astype(str)

    # Add fpar
    if "fpar" in modis:
        fpar = modis["fpar"].copy()
        fpar["fpar"] = fpar["fpar"].astype(float) * 0.01
        df = df.merge(fpar[["point_id","date","fpar"]], on=["point_id","date"], how="left")

    # Add LST + delta_lst
    if "lst" in modis:
        lst = modis["lst"].copy()
        lst["LST_Day_1km"] = lst["LST_Day_1km"].astype(float) * 0.02
        lst["LST_Night_1km"] = lst["LST_Night_1km"].astype(float) * 0.02
        lst["lst_day"] = lst["LST_Day_1km"]
        lst["lst_night"] = lst["LST_Night_1km"]
        lst["delta_lst"] = lst["lst_day"] - lst["lst_night"]
        df = df.merge(lst[["point_id","date","lst_day","lst_night","delta_lst"]],
                      on=["point_id","date"], how="left")

    # Add albedo
    if "albedo" in modis:
        alb = modis["albedo"].copy()
        alb["albedo_sw"] = alb["albedo"].astype(float)
        df = df.merge(alb[["point_id","date","albedo_sw"]], on=["point_id","date"], how="left")

    # Add SM
    if "sm" in modis:
        sm = modis["sm"].copy()
        sm["sm_surface"] = sm["sm"].astype(float)
        df = df.merge(sm[["point_id","date","sm_surface"]], on=["point_id","date"], how="left")

    return df

def main():
    print("Loading MODIS CSVs...")
    modis = load_modis_csvs()
    if not modis:
        print("No MODIS CSVs found. Download from Drive first:")
        print("  Folder: kcact_maize_modis_indicators")
        print("  Save to:", CSV_DIR)
        return

    print("\nLoading parquet...")
    df = pd.read_parquet(PARQUET)
    df = df[df['qc_valid']].copy()
    print(f"  {len(df)} valid samples")

    print("\nMerging MODIS indicators...")
    df = merge_with_parquet(df, modis)

    # Coverage check
    for c in ["fpar","lst_day","delta_lst","albedo_sw","sm_surface"]:
        n = df[c].notna().sum() if c in df.columns else 0
        print(f"  {c}: {n}/{len(df)} ({n/len(df)*100:.1f}%)")

    # Save enhanced parquet
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved: {OUT_PARQUET}")

    # ---- Train with 7 best indicators ----
    print("\n=== Training: 7 indicators on large maize dataset ===")
    feats_7 = ["ndvi","lswi","doy","fpar","delta_lst","sm_surface","albedo_sw"]
    feats_7 = [f for f in feats_7 if f in df.columns]
    sub = df.dropna(subset=feats_7 + ['kcact'])
    print(f"  7-indicator samples: {len(sub)}")

    years = sorted(sub['year'].unique())
    all_p, all_a = [], []
    for yr in years:
        tr = sub[sub['year']!=yr]; te = sub[sub['year']==yr]
        if len(te) < 10: continue
        m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6,
                              l2_leaf_reg=3, loss_function='RMSE', random_seed=42, verbose=False)
        m.fit(tr[feats_7].values, tr['kcact'].values)
        all_p.extend(m.predict(te[feats_7].values))
        all_a.extend(te['kcact'].values)
    r2_7 = r2_score(all_a, all_p)
    rmse_7 = np.sqrt(mean_squared_error(all_a, all_p))
    print(f"  7 indicators: LOYO R² = {r2_7:.5f}, RMSE = {rmse_7:.5f}")

    # ---- Baseline: full features ----
    exclude = {"patch_id","point_id","date","date_start","date_end","province","crop_type",
               "qc_valid","kcact","etc_8d_mm","et0_pm_8d_mm","qc_mod16",".geo","system:index",
               "valid_obs","obs_count_s2","summer_maize_candidate","season_year",
               "centroid_lat_weather","centroid_lon_weather","centroid_lat_s2","centroid_lon_s2",
               "greenup_doy","greenup_ndvi","days_since_greenup","gdd_since_greenup",
               "gdd_frac_greenup","vpd_days_greenup","vpd_gdd_frac_greenup"}
    feats_full = [c for c in df.columns if c not in exclude]
    sub_f = df.dropna(subset=feats_full + ['kcact'])
    print(f"\n  Full-feature samples: {len(sub_f)}")

    all_p, all_a = [], []
    for yr in years:
        tr = sub_f[sub_f['year']!=yr]; te = sub_f[sub_f['year']==yr]
        m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6,
                              l2_leaf_reg=3, loss_function='RMSE', random_seed=42, verbose=False)
        m.fit(tr[feats_full].values, tr['kcact'].values)
        all_p.extend(m.predict(te[feats_full].values))
        all_a.extend(te['kcact'].values)
    r2_full = r2_score(all_a, all_p)
    rmse_full = np.sqrt(mean_squared_error(all_a, all_p))
    print(f"  Full ({len(feats_full)} features): LOYO R² = {r2_full:.5f}, RMSE = {rmse_full:.5f}")

    # Save comparison
    result = pd.DataFrame([
        {"combo": "7_indicators", "n_feat": len(feats_7), "LOYO_R2": round(r2_7,5), "LOYO_RMSE": round(rmse_7,5)},
        {"combo": "full_features", "n_feat": len(feats_full), "LOYO_R2": round(r2_full,5), "LOYO_RMSE": round(rmse_full,5)},
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_DIR / "maize_7indicator_vs_full.csv", index=False)
    print(f"\n  ΔR² = {r2_7 - r2_full:+.4f} (7 vs {len(feats_full)} features)")
    print(f"  Results: {OUT_DIR / 'maize_7indicator_vs_full.csv'}")

if __name__ == "__main__":
    main()
