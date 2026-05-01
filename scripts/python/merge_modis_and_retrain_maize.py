"""Merge GEE-exported MODIS indicators into maize training parquet and retrain."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
CSV_DIR = ROOT / "data" / "raw" / "gee" / "kcact_maize_modis_indicators"
OUT_PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_with_modis.parquet"
OUT_DIR = ROOT / "outputs/tables"

def load_modis_csvs():
    frames = {}
    # product_key -> (file_pattern, band_name_in_csv, output_name)
    products = {
        "fpar":   ("maize_fpar",   "mean", "fpar"),
        "lst":    ("maize_lst",    ["LST_Day_1km","LST_Night_1km"], "lst"),
        # "sm": ("maize_era5_sm","mean", "sm_surface"),  # 29M rows OOM, skip
        "m09vi":  ("maize_m09vi",  ["ndvi_m09","sur_refl_b07"], "m09vi"),
    }
    for key, (pattern, bands, output_name) in products.items():
        files = sorted(CSV_DIR.glob(f"{pattern}_*.csv"))
        if not files:
            print(f"  WARNING: No {key} files found in {CSV_DIR}")
            continue
        dfs = []
        for f in files:
            d = pd.read_csv(f)
            d["date"] = pd.to_datetime(d["date"])
            d["point_id"] = d["point_id"].astype(str)
            # Handle "mean" → actual band name for single-band products
            if isinstance(bands, str) and bands == "mean" and "mean" in d.columns:
                d = d.rename(columns={"mean": output_name})
            keep = ["point_id","date"]
            if isinstance(bands, list):
                keep += bands
            else:
                col = output_name if (bands == "mean") else bands
                if col in d.columns:
                    keep.append(col)
            d = d[[c for c in keep if c in d.columns]]
            dfs.append(d)
        if dfs:
            frames[key] = pd.concat(dfs, ignore_index=True)
            print(f"  {key}: {len(frames[key])} rows from {len(files)} files")
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

    # Add MOD09A1 ndvi_m09 + b07 (handle per-province duplication)
    if "m09vi" in modis:
        m09 = modis["m09vi"].copy()
        m09["ndvi_m09"] = m09["ndvi_m09"].astype(float)
        m09["b07"] = m09["sur_refl_b07"].astype(float) * 0.0001
        # Drop duplicates (per-province exports may overlap)
        m09 = m09.drop_duplicates(subset=["point_id","date"], keep="first")
        df = df.merge(m09[["point_id","date","ndvi_m09","b07"]], on=["point_id","date"], how="left")

    # Add SRTM DEM
    srtm_files = sorted(CSV_DIR.glob("maize_srtm_dem*.csv"))
    if srtm_files:
        dem = pd.read_csv(srtm_files[0])
        dem["point_id"] = dem["point_id"].astype(str)
        dem["elevation"] = dem["elevation"].astype(float)
        df = df.merge(dem[["point_id","elevation"]], on=["point_id"], how="left")
        print(f"  srtm: {len(dem)} points")

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

    print("\nMerging MODIS indicators (step by step)...")
    import gc
    df = df.copy()
    df["point_id"] = df["point_id"].astype(str)

    # Step 1: fpar
    if "fpar" in modis:
        m = modis.pop("fpar"); m["fpar"] = m["fpar"].astype(float) * 0.01
        df = df.merge(m[["point_id","date","fpar"]], on=["point_id","date"], how="left")
        del m; gc.collect(); print("  fpar merged")

    # Step 2: LST
    if "lst" in modis:
        m = modis.pop("lst")
        for c in ["LST_Day_1km","LST_Night_1km"]:
            if c in m.columns: m[c] = m[c].astype(float) * 0.02
        m["lst_day"] = m["LST_Day_1km"]; m["lst_night"] = m["LST_Night_1km"]
        m["delta_lst"] = m["lst_day"] - m["lst_night"]
        df = df.merge(m[["point_id","date","lst_day","lst_night","delta_lst"]],
                      on=["point_id","date"], how="left")
        del m; gc.collect(); print("  lst merged")

    # Step 3: SM — skip for now (29M rows OOM). Use precip_30d as proxy later.
    print("  sm skipped (29M rows, use proxy)")

    # Step 4: MOD09A1
    if "m09vi" in modis:
        m = modis.pop("m09vi")
        m["ndvi_m09"] = m["ndvi_m09"].astype(float)
        m["b07"] = m["sur_refl_b07"].astype(float) * 0.0001
        m = m.drop_duplicates(subset=["point_id","date"], keep="first")
        df = df.merge(m[["point_id","date","ndvi_m09","b07"]], on=["point_id","date"], how="left")
        del m; gc.collect(); print("  m09vi merged")

    # Step 5: SRTM
    srtm_files = sorted(CSV_DIR.glob("maize_srtm_dem*.csv"))
    if srtm_files:
        m = pd.read_csv(srtm_files[0]); m["point_id"] = m["point_id"].astype(str)
        m["elevation"] = m["elevation"].astype(float)
        df = df.merge(m[["point_id","elevation"]], on=["point_id"], how="left")
        del m; gc.collect(); print("  srtm merged")

    # Coverage check
    for c in ["fpar","lst_day","delta_lst","albedo_sw","sm_surface","ndvi_m09","b07","elevation"]:
        n = df[c].notna().sum() if c in df.columns else 0
        print(f"  {c}: {n}/{len(df)} ({n/len(df)*100:.1f}%)")

    # Save enhanced parquet
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved: {OUT_PARQUET}")

    # ---- Train: multiple combos ----
    print("\n=== Training on large maize dataset ===")
    years = sorted(df['year'].unique())

    combos = {
        "ndvi+b07+doy": ["ndvi_m09","b07","doy"],
        "ndvi+lswi+doy": ["ndvi","lswi","doy"],
        "best7_station": ["ndvi_m09","lswi","doy","fpar","delta_lst","sm_surface","elevation"],
        "best7_plus_b07": ["ndvi_m09","b07","doy","fpar","delta_lst","sm_surface","elevation"],
    }

    results = []
    for name, feats in combos.items():
        feats = [f for f in feats if f in df.columns]
        sub = df.dropna(subset=feats + ['kcact'])
        all_p, all_a = [], []
        for yr in years:
            tr = sub[sub['year']!=yr]; te = sub[sub['year']==yr]
            if len(te) < 10: continue
            m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6,
                                  l2_leaf_reg=3, loss_function='RMSE', random_seed=42, verbose=False)
            m.fit(tr[feats].values, tr['kcact'].values)
            all_p.extend(m.predict(te[feats].values))
            all_a.extend(te['kcact'].values)
        r2 = r2_score(all_a, all_p)
        rmse = np.sqrt(mean_squared_error(all_a, all_p))
        print(f"  {name:20s}  n={len(feats):2d}  R²={r2:.5f}  RMSE={rmse:.5f}  samples={len(sub)}")
        results.append({"combo": name, "n_feat": len(feats), "LOYO_R2": round(r2,5), "LOYO_RMSE": round(rmse,5)})

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
        if len(te) < 10: continue
        m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6,
                              l2_leaf_reg=3, loss_function='RMSE', random_seed=42, verbose=False)
        m.fit(tr[feats_full].values, tr['kcact'].values)
        all_p.extend(m.predict(te[feats_full].values))
        all_a.extend(te['kcact'].values)
    r2_full = r2_score(all_a, all_p)
    rmse_full = np.sqrt(mean_squared_error(all_a, all_p))
    print(f"  Full ({len(feats_full)} feats): R²={r2_full:.5f}  RMSE={rmse_full:.5f}")

    results.append({"combo": "full_features", "n_feat": len(feats_full),
                    "LOYO_R2": round(r2_full,5), "LOYO_RMSE": round(rmse_full,5)})

    out = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "maize_real_modis_results.csv", index=False)
    print(f"\nResults: {OUT_DIR / 'maize_real_modis_results.csv'}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
