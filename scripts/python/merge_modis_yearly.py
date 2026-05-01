"""Merge MODIS indicators into maize parquet — year-by-year to avoid OOM."""

from pathlib import Path
import gc
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_train_ready.parquet"
CSV_DIR = ROOT / "data" / "raw" / "gee" / "kcact_maize_modis_indicators"
OUT_PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_with_modis.parquet"
OUT_DIR = ROOT / "outputs/tables"

def load_csvs_for_year(yr, pattern):
    files = sorted(CSV_DIR.glob(f"{pattern}_*{yr}*.csv"))
    if not files: return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    d = pd.concat(dfs, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d["point_id"] = d["point_id"].astype(str)
    return d

def main():
    df_all = pd.read_parquet(PARQUET)
    df_all = df_all[df_all['qc_valid']].copy()
    print(f"Total samples: {len(df_all)}, years: {sorted(df_all['year'].unique())}")

    years = sorted(df_all['year'].unique())
    enhanced_parts = []

    for yr in years:
        print(f"\n--- {yr} ---")
        yr_df = df_all[df_all['year'] == yr].copy()
        yr_df["date"] = pd.to_datetime(yr_df["date"])
        yr_df["point_id"] = yr_df["point_id"].astype(str)
        print(f"  {len(yr_df)} rows")

        # fpar
        fpar = load_csvs_for_year(yr, "maize_fpar")
        if len(fpar) > 0:
            fpar["fpar"] = fpar["mean"].astype(float) * 0.01
            fpar = fpar.groupby(["point_id","date"], as_index=False)["fpar"].mean()
            yr_df = yr_df.merge(fpar[["point_id","date","fpar"]], on=["point_id","date"], how="left")
            del fpar; gc.collect()
            print(f"  fpar: {yr_df['fpar'].notna().sum()}/{len(yr_df)}")

        # LST
        lst = load_csvs_for_year(yr, "maize_lst")
        if len(lst) > 0:
            for c in ["LST_Day_1km","LST_Night_1km"]:
                if c in lst.columns: lst[c] = lst[c].astype(float) * 0.02
            lst["lst_day"] = lst["LST_Day_1km"]
            lst["lst_night"] = lst["LST_Night_1km"]
            lst["delta_lst"] = lst["lst_day"] - lst["lst_night"]
            lst = lst.groupby(["point_id","date"], as_index=False)[["lst_day","lst_night","delta_lst"]].mean()
            yr_df = yr_df.merge(lst[["point_id","date","lst_day","lst_night","delta_lst"]],
                                on=["point_id","date"], how="left")
            del lst; gc.collect()
            print(f"  lst: {yr_df['delta_lst'].notna().sum()}/{len(yr_df)}")

        # MOD09A1 ndvi + b07
        m09 = load_csvs_for_year(yr, "maize_m09vi")
        if len(m09) > 0:
            if "ndvi_m09" in m09.columns:
                m09["ndvi_m09"] = m09["ndvi_m09"].astype(float)
            if "sur_refl_b07" in m09.columns:
                m09["b07"] = m09["sur_refl_b07"].astype(float) * 0.0001
            keep = ["point_id","date"]
            for c in ["ndvi_m09","b07"]:
                if c in m09.columns: keep.append(c)
            m09 = m09.groupby(["point_id","date"], as_index=False)[[c for c in keep if c!="point_id" and c!="date"]].mean()
            yr_df = yr_df.merge(m09[keep], on=["point_id","date"], how="left")
            del m09; gc.collect()
            for c in ["ndvi_m09","b07"]:
                if c in yr_df.columns:
                    print(f"  {c}: {yr_df[c].notna().sum()}/{len(yr_df)}")

        enhanced_parts.append(yr_df)

    df = pd.concat(enhanced_parts, ignore_index=True)
    del df_all, enhanced_parts; gc.collect()

    # SRTM
    srtm_f = sorted(CSV_DIR.glob("maize_srtm_dem*.csv"))
    if srtm_f:
        dem = pd.read_csv(srtm_f[0])
        dem["point_id"] = dem["point_id"].astype(str)
        dem["elevation"] = dem["elevation"].astype(float)
        df = df.merge(dem[["point_id","elevation"]], on=["point_id"], how="left")
        print(f"\nsrtm: {df['elevation'].notna().sum()}/{len(df)}")

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved: {OUT_PARQUET}")

    # ---- Train ----
    new_cols = ["fpar","lst_day","delta_lst","ndvi_m09","b07","elevation"]
    for c in new_cols:
        if c in df.columns:
            print(f"  {c}: {df[c].notna().sum()}/{len(df)} ({(df[c].notna().sum()/len(df)*100):.0f}%)")

    combs = {
        "ndvi_m09+b07+doy": ["ndvi_m09","b07","doy"],
        "best7_real": ["ndvi","lswi","doy","fpar","delta_lst","ndvi_m09","b07"],
        "best6_noSM": ["ndvi","lswi","doy","fpar","delta_lst","elevation"],
    }

    years_list = sorted(df['year'].unique())
    exclude = {"patch_id","point_id","date","date_start","date_end","province","crop_type",
               "qc_valid","kcact","etc_8d_mm","et0_pm_8d_mm","qc_mod16",".geo","system:index",
               "valid_obs","obs_count_s2","summer_maize_candidate","season_year",
               "centroid_lat_weather","centroid_lon_weather","centroid_lat_s2","centroid_lon_s2",
               "greenup_doy","greenup_ndvi","days_since_greenup","gdd_since_greenup",
               "gdd_frac_greenup","vpd_days_greenup","vpd_gdd_frac_greenup"}
    feats_full = [c for c in df.columns if c not in exclude]

    results = []
    for name, feats in [("full_features", feats_full)] + [(k,v) for k,v in combs.items()]:
        feats = [f for f in feats if f in df.columns]
        sub = df.dropna(subset=feats + ['kcact'])
        all_p, all_a = [], []
        for yr in years_list:
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
        results.append({"combo":name,"n_feat":len(feats),"LOYO_R2":round(r2,5),"LOYO_RMSE":round(rmse,5)})

    out = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "maize_real_modis_results.csv", index=False)
    print(f"\n{out.to_string(index=False)}")
    print(f"Saved: {OUT_DIR / 'maize_real_modis_results.csv'}")

if __name__ == "__main__":
    main()
