"""Desktop: merge MODIS CSVs into maize parquet + train top N station combos."""

import argparse, gc, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "processed" / "train" / "ncp_summer_maize_kcact_train_ready.parquet"
CSV_DIR = ROOT / "data" / "raw" / "gee" / "kcact_maize_modis_indicators"
OUT_PARQUET = ROOT / "data" / "processed" / "train" / "ncp_summer_maize_kcact_with_modis.parquet"

EXCLUDE = {"patch_id","point_id","date","date_start","date_end","province","crop_type",
           "qc_valid","kcact","etc_8d_mm","et0_pm_8d_mm","qc_mod16",".geo","system:index",
           "valid_obs","obs_count_s2","summer_maize_candidate","season_year",
           "centroid_lat_weather","centroid_lon_weather","centroid_lat_s2","centroid_lon_s2",
           "greenup_doy","greenup_ndvi","days_since_greenup","gdd_since_greenup",
           "gdd_frac_greenup","vpd_days_greenup","vpd_gdd_frac_greenup"}

FEAT_MAP = {
    "ndvi_m09":"ndvi_m09","evi_m09":"evi","gndvi_m09":"gndvi","savi_m09":"savi",
    "lswi_m09":"lswi","lswi":"lswi","fpar":"fpar","delta_lst":"delta_lst",
    "lst_day":"lst_day","lst_night":"lst_night","albedo_sw":"albedo_proxy",
    "sm_surface":"sm_proxy","doy":"doy","year":"year","doy_sin":"doy_sin",
    "doy_cos":"doy_cos","ndvi":"ndvi","evi":"evi","gndvi":"gndvi","savi":"savi",
    "tmean_c":"tmean_c","vpd_kpa":"vpd_kpa_mean_8d",
    "solar_rad_mj_m2_d":"solar_rad_mj_m2_d_sum_8d",
    "precip_mm":"precip_mm_8d","wind_10m_m_s":"wind_2m_m_s_mean_8d",
    "precip_7d":"precip_7d","precip_15d":"precip_15d","precip_30d":"precip_30d",
    "ndvi_lag1":"ndvi_lag1","ndvi_diff":"ndvi_diff","ndvi_vpd":"ndvi_vpd",
    "lswi_vpd":"lswi_vpd","lat":"centroid_lat","lon":"centroid_lon",
}

def merge_modis():
    print("[1/3] Merging MODIS CSVs year by year...")
    df = pd.read_parquet(PARQUET)
    df = df[df["qc_valid"]].copy()
    df["point_id"] = df["point_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    new_cols = {}

    for yr in sorted(df["year"].unique()):
        mask = df["year"] == yr
        y = df[mask][["point_id","date"]].copy()

        fp = pd.read_csv(CSV_DIR / f"maize_fpar_{yr}.csv")
        fp["date"] = pd.to_datetime(fp["date"]); fp["point_id"] = fp["point_id"].astype(str)
        fp = fp.groupby(["point_id","date"], as_index=False)["mean"].mean()
        fp["fpar"] = fp["mean"].astype(float) * 0.01
        y = y.merge(fp[["point_id","date","fpar"]], on=["point_id","date"], how="left")

        ls = pd.read_csv(CSV_DIR / f"maize_lst_{yr}.csv")
        for c in ["LST_Day_1km","LST_Night_1km"]:
            if c in ls.columns: ls[c] = ls[c].astype(float) * 0.02
        ls["lst_day"] = ls["LST_Day_1km"]; ls["lst_night"] = ls["LST_Night_1km"]
        ls["delta_lst"] = ls["lst_day"] - ls["lst_night"]
        ls["date"] = pd.to_datetime(ls["date"]); ls["point_id"] = ls["point_id"].astype(str)
        ls = ls.groupby(["point_id","date"], as_index=False)[["lst_day","lst_night","delta_lst"]].mean()
        y = y.merge(ls[["point_id","date","lst_day","lst_night","delta_lst"]], on=["point_id","date"], how="left")

        mfs = sorted(glob.glob(str(CSV_DIR / f"maize_m09vi*{yr}*.csv")))
        if mfs:
            m09 = pd.concat([pd.read_csv(f) for f in mfs], ignore_index=True)
            m09["date"] = pd.to_datetime(m09["date"]); m09["point_id"] = m09["point_id"].astype(str)
            if "ndvi_m09" in m09.columns:
                m09["ndvi_m09"] = m09["ndvi_m09"].astype(float)
                mc = m09.groupby(["point_id","date"], as_index=False)["ndvi_m09"].mean()
                y = y.merge(mc[["point_id","date","ndvi_m09"]], on=["point_id","date"], how="left")
            if "sur_refl_b07" in m09.columns:
                m09["b07"] = m09["sur_refl_b07"].astype(float) * 0.0001
                mc = m09.groupby(["point_id","date"], as_index=False)["b07"].mean()
                y = y.merge(mc[["point_id","date","b07"]], on=["point_id","date"], how="left")
            del m09

        for c in ["fpar","lst_day","lst_night","delta_lst","ndvi_m09","b07"]:
            if c in y.columns:
                new_cols.setdefault(c, pd.Series(np.nan, index=df.index))
                new_cols[c].loc[mask] = y[c].values
        del y, fp, ls; gc.collect()
        print(f"  {yr}")

    for c, vals in new_cols.items():
        df[c] = vals

    df["albedo_proxy"] = (0.15 + 0.05 * (1 - df["ndvi"])).clip(0.05, 0.35)
    df["sm_proxy"] = df["precip_30d"] / 100
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    df["elevation"] = 0.0

    df.to_parquet(OUT_PARQUET, index=False)
    for c in ["fpar","delta_lst","ndvi_m09","b07","elevation"]:
        print(f"  {c}: {df[c].notna().sum()}/{len(df)} ({(df[c].notna().sum()/len(df)*100):.0f}%)")
    print(f"Saved {OUT_PARQUET}: {len(df)} rows, {len(df.columns)} cols")
    return df


def train_combos(df, station_results_csv, top_n=50):
    print(f"\n[2/3] Loading station top {top_n} combos...")
    sr = pd.read_csv(station_results_csv)
    ok = sr[sr["status"] == "OK"].head(top_n)

    years = sorted(df["year"].unique())
    results = []

    for i, (_, r) in enumerate(ok.iterrows()):
        station_feats = r.feats.split(",")
        mapped = []
        for f in station_feats:
            f = f.strip()
            if f in FEAT_MAP:
                mf = FEAT_MAP[f]
                if mf in df.columns: mapped.append(mf)
            elif f.startswith("b0"):
                pass  # raw bands not available
            elif f in df.columns:
                mapped.append(f)
        mapped = list(dict.fromkeys(mapped))
        if len(mapped) < 2: continue

        sub = df.dropna(subset=mapped + ["kcact"])
        if len(sub) < 1000: continue

        all_p, all_a = [], []
        for yr in years:
            tr = sub[sub["year"] != yr]; te = sub[sub["year"] == yr]
            if len(te) < 10: continue
            m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                                  task_type="GPU", devices="0", loss_function="RMSE",
                                  random_seed=42, verbose=False)
            m.fit(tr[mapped].values, tr["kcact"].values)
            all_p.extend(m.predict(te[mapped].values))
            all_a.extend(te["kcact"].values)
        r2 = r2_score(all_a, all_p)
        rmse = np.sqrt(mean_squared_error(all_a, all_p)).item()
        rank = i + 1
        print(f"  Station#{rank:3d}  R²={r2:.5f}  n={len(mapped)}")
        results.append((rank, r2, rmse, len(mapped), len(sub), ",".join(mapped)))

    results.sort(key=lambda x: -x[1])
    print(f"\n[3/3] Top 20 (large-sample LOYO R²):")
    for i, (rank, r2, rmse, n, ns, feats) in enumerate(results[:20]):
        marker = " ***" if r2 > 0.70 else (" **" if r2 > 0.65 else "")
        print(f"  {i+1:2d}. Station#{rank:3d}  R²={r2:.5f}  n={n}{marker}")

    out_df = pd.DataFrame(results, columns=["station_rank","LOYO_R2","LOYO_RMSE","n_features","n_samples","feature_list"])
    out_path = ROOT / "outputs" / "tables" / "desktop_top50_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return results


def add_manual_combos(df):
    print("\n[BONUS] Manual key combos...")
    combos = [
        ("#2+weather (best)", ["ndvi_m09","b07","doy","vpd_kpa_mean_8d","precip_mm_8d",
                               "solar_rad_mj_m2_d_sum_8d","tmean_c"]),
        ("#2 bare", ["ndvi_m09","b07","doy"]),
        ("#2+weather+fpar", ["ndvi_m09","b07","doy","fpar","vpd_kpa_mean_8d",
                             "precip_mm_8d","solar_rad_mj_m2_d_sum_8d","tmean_c"]),
        ("#2+weather+dLST", ["ndvi_m09","b07","doy","delta_lst","vpd_kpa_mean_8d",
                             "precip_mm_8d","solar_rad_mj_m2_d_sum_8d","tmean_c"]),
        ("full", [c for c in df.columns if c not in EXCLUDE]),
    ]
    years = sorted(df["year"].unique())
    for name, feats in combos:
        feats = [f for f in feats if f in df.columns]
        sub = df.dropna(subset=feats + ["kcact"])
        all_p, all_a = [], []
        for yr in years:
            tr = sub[sub["year"] != yr]; te = sub[sub["year"] == yr]
            if len(te) < 10: continue
            m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                                  task_type="GPU", devices="0", loss_function="RMSE",
                                  random_seed=42, verbose=False)
            m.fit(tr[feats].values, tr["kcact"].values)
            all_p.extend(m.predict(te[feats].values))
            all_a.extend(te["kcact"].values)
        r2 = r2_score(all_a, all_p)
        rmse = np.sqrt(mean_squared_error(all_a, all_p)).item()
        print(f"  {name:30s}  R²={r2:.5f}  n={len(feats)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station-csv", default=str(ROOT / "outputs/tables/maize_500_combo_results.csv"))
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--skip-merge", action="store_true")
    args = p.parse_args()

    if args.skip_merge and OUT_PARQUET.exists():
        print("Loading cached merged parquet...")
        df = pd.read_parquet(OUT_PARQUET)
    else:
        df = merge_modis()

    train_combos(df, args.station_csv, args.top_n)
    add_manual_combos(df)


if __name__ == "__main__":
    main()
