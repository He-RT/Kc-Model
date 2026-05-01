"""Exhaustive combo sweep on large maize dataset."""

import itertools, random, gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/processed/train/ncp_summer_maize_kcact_with_modis.parquet"
OUT_DIR = ROOT / "outputs/tables"
TARGET = "kcact"

EXCLUDE = {"patch_id","point_id","date","date_start","date_end","province","crop_type",
           "qc_valid","kcact","etc_8d_mm","et0_pm_8d_mm","qc_mod16",".geo","system:index",
           "valid_obs","obs_count_s2","summer_maize_candidate","season_year",
           "centroid_lat_weather","centroid_lon_weather","centroid_lat_s2","centroid_lon_s2",
           "greenup_doy","greenup_ndvi","days_since_greenup","gdd_since_greenup",
           "gdd_frac_greenup","vpd_days_greenup","vpd_gdd_frac_greenup"}

def build_combos():
    combos = []

    # Pool definitions
    CORE = ["fpar","delta_lst","sm_proxy","ndvi_m09","b07","lswi","albedo_proxy","doy"]
    WEATHER = ["vpd_kpa_mean_8d","precip_mm_8d","solar_rad_mj_m2_d_sum_8d","tmean_c"]
    EXTRA_VI = ["evi","gndvi","savi","ndvi"]
    TEMP = ["doy_sin","doy_cos","year"]
    DERIV = ["ndvi_lag1","ndvi_diff","ndvi_vpd","lswi_vpd","precip_7d","precip_15d","precip_30d"]
    FULL_POOL = CORE + WEATHER + EXTRA_VI + TEMP + DERIV

    # 1. All CORE subsets (2^8 - 1 = 255)
    for k in range(1, len(CORE)+1):
        for combo in itertools.combinations(CORE, k):
            feats = list(combo)
            name = "core_" + "_".join(f[:5] for f in feats)
            combos.append((name, feats))
            # +doy variants for combos without doy
            if "doy" not in feats and len(feats) <= 6:
                combos.append((name + "+doy", feats + ["doy"]))
            # +weather
            if len(feats) <= 5:
                combos.append((name + "+W", feats + WEATHER))

    # 2. CORE subsets + each weather var individually
    for w in WEATHER:
        for k in range(1, 8):
            for combo in random.sample(list(itertools.combinations(CORE, k)),
                                       min(10, len(list(itertools.combinations(CORE, k))))):
                feats = list(combo) + [w]
                combos.append(("w_"+w[:6]+"_"+"_".join(f[:4] for f in combo), feats))

    # 3. NDVI variants head-to-head (with same supporting cast)
    ndvi_variants = {"ndvi_m09":"m09","ndvi":"s2","evi":"s2e","gndvi":"s2g","savi":"s2s"}
    base_support = ["delta_lst","doy","vpd_kpa_mean_8d","precip_mm_8d","solar_rad_mj_m2_d_sum_8d"]
    for ndvi_col, tag in ndvi_variants.items():
        feats = [ndvi_col] + base_support
        combos.append((f"ndvi_{tag}_base", feats))
        for extra in ["fpar","b07","lswi","sm_proxy","albedo_proxy"]:
            if extra in CORE:
                combos.append((f"ndvi_{tag}_+{extra}", feats + [extra]))

    # 4. Feature count sweep: best features added one by one
    for seed_order in [
        ["ndvi_m09","b07","doy","vpd_kpa_mean_8d","precip_mm_8d","solar_rad_mj_m2_d_sum_8d","tmean_c"],
        ["fpar","delta_lst","ndvi_m09","b07","doy","vpd_kpa_mean_8d","sm_proxy"],
        ["ndvi_m09","doy","vpd_kpa_mean_8d","fpar","lswi","delta_lst","b07"],
    ]:
        for i in range(1, len(seed_order)+1):
            combos.append((f"sweep{len(combos)}", seed_order[:i]))

    # 5. Random exploration (100 combos from full pool)
    for _ in range(100):
        k = random.randint(1, 10)
        feats = random.sample(FULL_POOL, min(k, len(FULL_POOL)))
        combos.append(("rand_"+"_".join(f[:4] for f in feats), list(dict.fromkeys(feats))))

    # 6. Full feature baseline
    combos.append(("full_all", list(FULL_POOL)))
    combos.append(("#2_best_7", ["ndvi_m09","b07","doy","vpd_kpa_mean_8d","precip_mm_8d","solar_rad_mj_m2_d_sum_8d","tmean_c"]))

    # Deduplicate by sorted feature set
    seen, unique = set(), []
    for name, feats in combos:
        key = "|".join(sorted(feats))
        if key not in seen:
            seen.add(key)
            unique.append((name[:50], feats))
    return unique


def main():
    print("Loading merged parquet...")
    df = pd.read_parquet(PARQUET)
    if "qc_valid" in df.columns:
        df = df[df["qc_valid"]]
    print(f"  {len(df)} rows, {len(df.columns)} cols")

    combos = build_combos()
    print(f"Generated {len(combos)} unique combos\n")

    years = sorted(df["year"].unique())
    results = []

    for idx, (name, feats) in enumerate(combos):
        avail = [f for f in feats if f in df.columns]
        if len(avail) < 2: continue
        sub = df.dropna(subset=avail + [TARGET])
        if len(sub) < 1000: continue

        all_p, all_a = [], []
        for yr in years:
            tr = sub[sub["year"] != yr]; te = sub[sub["year"] == yr]
            if len(te) < 10: continue
            m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                                  task_type="GPU", devices="0", loss_function="RMSE",
                                  random_seed=42, verbose=False)
            m.fit(tr[avail].values, tr[TARGET].values)
            all_p.extend(m.predict(te[avail].values))
            all_a.extend(te[TARGET].values)
        r2 = r2_score(all_a, all_p)
        rmse = np.sqrt(mean_squared_error(all_a, all_p)).item()
        results.append((name, r2, rmse, len(avail), len(sub)))
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(combos)} done, best so far: R²={max(r[1] for r in results):.5f}")

    results.sort(key=lambda x: -x[1])
    out = pd.DataFrame(results, columns=["combo","LOYO_R2","LOYO_RMSE","n_features","n_samples"])
    path = OUT_DIR / "exhaustive_combo_results.csv"
    out.to_csv(path, index=False)

    print(f"\nDone! {len(results)} combos → {path}")
    print(f"\nTop 20:")
    for i, (name, r2, rmse, n, ns) in enumerate(results[:20]):
        print(f"  {i+1:2d}. {name:50s} n={n:2d} R²={r2:.5f}")


if __name__ == "__main__":
    main()
