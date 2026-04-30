from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

PROJ = Path(__file__).resolve().parents[2]
DATA = PROJ / "data" / "processed" / "station_ml_features.csv"
RAW = PROJ / "data" / "processed" / "modis_cache" / "mod09a1_raw_bands.csv"
ERA5 = PROJ / "data" / "processed" / "modis_cache" / "era5_weather_all.csv"
STN = PROJ / "data" / "processed" / "station_coordinates.csv"
NDVI_DAILY = PROJ / "data" / "processed" / "modis_cache" / "mod09ga_ndvi_daily.csv"
NDVI_250M = PROJ / "data" / "processed" / "modis_cache" / "mod09q1_ndvi_250m.csv"
NDVI_NBAR = PROJ / "data" / "processed" / "modis_cache" / "mcd43a4_ndvi_nbar.csv"
OUT = PROJ / "outputs" / "tables"

MAIZE_DOY = (150, 300)
TARGET = "kcact"
NM = {"yucheng":"禹城","weishan":"位山","guantao":"馆陶","luancheng":"栾城"}
RAW_IN = ["sur_refl_b01","sur_refl_b02","sur_refl_b03","sur_refl_b04","sur_refl_b05","sur_refl_b06","sur_refl_b07"]
RAW_RN = ["b01","b02","b03","b04","b05","b06","b07"]
ERA5_V = ["tmean_c","tmin_c","tmax_c","dewpoint_c","solar_rad_mj_m2_d","precip_mm","pressure_kpa","wind_10m_m_s"]

def load_base():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df["date_prev"] = pd.to_datetime(df["date_prev"])
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    df = df[(df["doy"] >= MAIZE_DOY[0]) & (df["doy"] <= MAIZE_DOY[1])]
    df = df.dropna(subset=[TARGET])
    df["doy_sin"] = np.sin(2*np.pi*df["doy"]/365)
    df["doy_cos"] = np.cos(2*np.pi*df["doy"]/365)
    return df

def merge_time_series(df, path, col_map, date_col="date", scale=1.0):
    """Merge an external time series CSV into df by station + date window."""
    try:
        ts = pd.read_csv(path)
    except FileNotFoundError:
        return df
    if "station" in ts.columns and ts["station"].iloc[0] in NM:
        ts["station"] = ts["station"].map(NM).fillna(ts["station"])
    ts[date_col] = pd.to_datetime(ts[date_col])
    for src, dst in col_map.items():
        df[dst] = np.nan
        if src not in ts.columns:
            continue
        for _, row in df.iterrows():
            stn, tp, tc = row["station"], row["date_prev"], row["date"]
            w = ts[(ts["station"]==stn) & (ts[date_col]>tp) & (ts[date_col]<=tc)]
            if len(w) > 0:
                df.at[row.name, dst] = w[src].mean() * scale
        df[dst] = df.groupby("station")[dst].transform(lambda x: x.fillna(x.median()))
    return df

def load_all():
    df = load_base()
    # Raw bands + MOD09A1 VIs
    raw = pd.read_csv(RAW)
    raw["station"] = raw["station"].map(NM).fillna(raw["station"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.rename(dict(zip(RAW_IN, RAW_RN)), axis=1)
    for c in RAW_RN: df[c] = np.nan
    for _, row in df.iterrows():
        stn, tp, tc = row["station"], row["date_prev"], row["date"]
        w = raw[(raw["station"]==stn) & (raw["date"]>tp) & (raw["date"]<=tc)]
        if len(w) > 0:
            for c in RAW_RN: df.at[row.name, c] = w[c].mean()
    for c in RAW_RN:
        df[c] = df.groupby("station")[c].transform(lambda x: x.fillna(x.median()))

    eps = 1e-6
    df["ndvi_m09"] = (df["b02"] - df["b01"]) / (df["b02"] + df["b01"] + eps)
    df["evi_m09"] = 2.5*(df["b02"]-df["b01"])/(df["b02"]+6*df["b01"]-7.5*df["b03"]+1)
    df["lswi_m09"] = (df["b02"] - df["b06"]) / (df["b02"] + df["b06"] + eps)
    df["gndvi_m09"] = (df["b02"] - df["b04"]) / (df["b02"] + df["b04"] + eps)
    df["savi_m09"] = 1.5*(df["b02"]-df["b01"])/(df["b02"]+df["b01"]+0.5)

    # New NDVI sources
    df = merge_time_series(df, NDVI_DAILY, {"ndvi_daily":"ndvi_daily"}, scale=1.0)
    df = merge_time_series(df, NDVI_250M, {"ndvi_250m":"ndvi_250m"}, scale=1.0)
    df = merge_time_series(df, NDVI_NBAR, {"ndvi_nbar":"ndvi_nbar"}, scale=1.0)

    # ERA5 weather + derived
    try:
        w = pd.read_csv(ERA5); w["date"] = pd.to_datetime(w["date"])
        for c in ERA5_V: df[c] = np.nan
        for _, row in df.iterrows():
            stn, tp, tc = row["station"], row["date_prev"], row["date"]
            win = w[(w["station"]==stn) & (w["date"]>tp) & (w["date"]<=tc)]
            if len(win) > 0:
                for c in ERA5_V: df.at[row.name, c] = win[c].mean()
        for c in ERA5_V:
            df[c] = df.groupby("station")[c].transform(lambda x: x.fillna(x.median()))
        df["vpd_kpa"] = (0.6108*np.exp(17.27*df["tmean_c"]/(df["tmean_c"]+237.3))
                         - 0.6108*np.exp(17.27*df["dewpoint_c"]/(df["dewpoint_c"]+237.3))).clip(lower=0)
        df["precip_7d"] = df.groupby("station")["precip_mm"].transform(lambda s: s.rolling(1,min_periods=1).sum())
        df["precip_15d"] = df.groupby("station")["precip_mm"].transform(lambda s: s.rolling(2,min_periods=1).sum())
        df["precip_30d"] = df.groupby("station")["precip_mm"].transform(lambda s: s.rolling(4,min_periods=1).sum())
        df["ndvi_lag1"] = df.groupby("station")["ndvi_m09"].shift(1).fillna(df["ndvi_m09"])
        df["ndvi_diff"] = df["ndvi_m09"] - df["ndvi_lag1"]
        df["ndvi_vpd"] = df["ndvi_m09"] * df["vpd_kpa"]
        df["lswi_vpd"] = df["lswi_m09"] * df["vpd_kpa"]
        sl = pd.read_csv(STN).set_index("station")
        df["lat"] = df["station"].map(sl["lat"].to_dict())
        df["lon"] = df["station"].map(sl["lon"].to_dict())
    except FileNotFoundError: pass

    return df

def fill_nans(df, cols):
    for c in cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df.groupby("station")[c].transform(lambda x: x.fillna(x.median()))
    return df.dropna(subset=[c for c in cols if c in df.columns])

def run_loso(df, features):
    stations = df["station"].values
    X, y = df[features].values, df[TARGET].values
    preds = np.full(len(y), np.nan)
    for stn in np.unique(stations):
        tm = stations != stn; em = stations == stn
        if tm.sum() < 10 or em.sum() == 0: continue
        m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=4,
                              loss_function="RMSE", random_seed=42, verbose=False)
        m.fit(X[tm], y[tm])
        preds[em] = m.predict(X[em])
    v = ~np.isnan(preds)
    if v.sum() < 10: return np.nan, np.nan
    return r2_score(y[v], preds[v]), float(np.sqrt(mean_squared_error(y[v], preds[v])))

def gen_combos():
    c = []

    # Feature pools
    NDVI_POOL = ["ndvi","evi","ndvi_m09","evi_m09","ndvi_daily","ndvi_250m","ndvi_nbar","ndvi_m09","lswi_m09","gndvi_m09","savi_m09"]
    THERM_POOL = ["lst_day","delta_lst"]
    ENERGY_POOL = ["albedo_sw","fpar"]
    MOIST_POOL = ["lswi","sm_surface"]
    TEMP_POOL = ["doy","doy_sin","doy_cos","year"]
    WEATHER_POOL = ["tmean_c","vpd_kpa","solar_rad_mj_m2_d","precip_mm","wind_10m_m_s"]
    RAW_POOL = RAW_RN
    INTERACT_POOL = []  # added dynamically

    ALL_POOLS = {
        "ndvi": NDVI_POOL, "therm": THERM_POOL, "energy": ENERGY_POOL,
        "moist": MOIST_POOL, "temp": TEMP_POOL, "weather": WEATHER_POOL, "raw": RAW_POOL,
    }

    # 1. All single features (50+)
    all_features = NDVI_POOL + THERM_POOL + ENERGY_POOL + MOIST_POOL + TEMP_POOL + WEATHER_POOL + RAW_POOL
    all_features = list(dict.fromkeys(all_features))  # dedupe ordered
    for f in all_features:
        c.append((f"single_{f}", [f]))
        c.append((f"single_{f}_doy", [f, "doy"]))

    # 2. NDVI bake-off: each NDVI + fpar + lst_day + doy
    ndvi_candidates = ["ndvi_m09","ndvi_daily","ndvi_250m","ndvi_nbar","ndvi","evi_m09","lswi_m09"]
    for ndvi in ndvi_candidates:
        c.append((f"bakeoff_{ndvi}", [ndvi, "fpar", "lst_day", "doy"]))
        c.append((f"bakeoff_{ndvi}_full", [ndvi, "fpar", "lst_day", "delta_lst", "sm_surface", "doy"]))

    # 3. All subsets of core TOP6: {fpar, lst_day, delta_lst, sm_surface, ndvi_m09, lswi_m09}
    CORE = ["fpar","lst_day","delta_lst","sm_surface","ndvi_m09","lswi_m09"]
    for k in range(1, len(CORE)+1):
        for combo in itertools.combinations(CORE, k):
            feats = list(combo)
            c.append(("core_"+"_".join(f[:4] for f in feats), feats))
            if len(feats) <= 5:
                c.append(("core_"+"_".join(f[:4] for f in feats)+"_doy", feats+["doy"]))
                c.append(("core_"+"_".join(f[:4] for f in feats)+"_cyc", feats+["doy_sin","doy_cos"]))

    # 4. All subsets of TOP8: CORE + evi_m09 + albedo_sw
    TOP8 = CORE + ["evi_m09","albedo_sw"]
    for k in range(1, min(7, len(TOP8)+1)):
        for combo in random.sample(list(itertools.combinations(TOP8, k)), min(30, len(list(itertools.combinations(TOP8, k))))):
            feats = list(combo)
            c.append(("top8r_"+"_".join(f[:4] for f in feats), feats))
            c.append(("top8r_"+"_".join(f[:4] for f in feats)+"_doy", feats+["doy"]))

    # 5. Domain cross-product: 1 from each domain
    domains = {
        "vi": ["ndvi_m09","evi_m09","lswi_m09","gndvi_m09"],
        "therm": ["lst_day","delta_lst"],
        "energy": ["fpar","albedo_sw"],
        "moist": ["sm_surface","lswi"],
    }
    for vi in domains["vi"]:
        for th in domains["therm"]:
            c.append((f"cross_{vi[:6]}_{th[:6]}", [vi, th]))
            c.append((f"cross_{vi[:6]}_{th[:6]}_doy", [vi, th, "doy"]))
            for en in domains["energy"]:
                c.append((f"cross_{vi[:6]}_{th[:6]}_{en[:6]}", [vi, th, en]))
                c.append((f"cross_{vi[:6]}_{th[:6]}_{en[:6]}_doy", [vi, th, en, "doy"]))
                for mo in domains["moist"]:
                    c.append((f"cross_{vi[:6]}_{th[:6]}_{en[:6]}_{mo[:6]}", [vi, th, en, mo]))
                    c.append((f"cross_{vi[:6]}_{th[:6]}_{en[:6]}_{mo[:6]}_doy", [vi, th, en, mo, "doy"]))

    # 6. Weather-only combos
    for k in range(1, len(WEATHER_POOL)+1):
        for combo in itertools.combinations(WEATHER_POOL, k):
            feats = list(combo)
            c.append(("weather_"+"_".join(f[:4] for f in feats), feats))
            c.append(("weather_"+"_".join(f[:4] for f in feats)+"_doy", feats+["doy"]))

    # 7. Raw band subsets (random sample to avoid explosion)
    for k in range(1, min(8, len(RAW_POOL)+1)):
        combos = list(itertools.combinations(RAW_POOL, k))
        sample_n = min(15, len(combos))
        for combo in random.sample(combos, sample_n):
            feats = list(combo)
            c.append(("raw_"+"_".join(feats), feats))
            c.append(("raw_"+"_".join(feats)+"_doy", feats+["doy"]))

    # 8. Raw bands + VIs hybrids
    for vi in ["ndvi_m09","evi_m09","lswi_m09"]:
        for k in range(1, 5):
            for rcombo in random.sample(list(itertools.combinations(RAW_POOL, k)), min(8, len(list(itertools.combinations(RAW_POOL, k))))):
                feats = [vi] + list(rcombo)
                c.append(("hybrid_"+"_".join(f[:4] for f in feats), feats))
                c.append(("hybrid_"+"_".join(f[:4] for f in feats)+"_doy", feats+["doy"]))

    # 9. Interaction terms with top features
    for base in ["ndvi_m09","fpar","lswi_m09"]:
        for other in ["fpar","lst_day","sm_surface","delta_lst"]:
            if base == other: continue
            c.append((f"inter_{base[:6]}_x_{other[:6]}", [base, other]))
            c.append((f"inter_{base[:6]}_x_{other[:6]}_doy", [base, other, "doy"]))

    # 10. Existing model analogs
    c.append(("existing_m09", ["ndvi_m09","evi_m09","lswi_m09","tmean_c","vpd_kpa","solar_rad_mj_m2_d",
                                "precip_mm","wind_10m_m_s","precip_7d","doy","year"]))
    c.append(("existing_m09_full", ["ndvi_m09","evi_m09","lswi_m09","gndvi_m09",
                                     "tmean_c","vpd_kpa","solar_rad_mj_m2_d","precip_mm","wind_10m_m_s",
                                     "precip_7d","precip_15d","ndvi_lag1","ndvi_diff",
                                     "ndvi_vpd","lswi_vpd","doy","year","lat","lon"]))
    c.append(("existing_m09_daily", ["ndvi_daily","evi_m09","lswi_m09","tmean_c","vpd_kpa",
                                      "solar_rad_mj_m2_d","precip_mm","precip_7d","doy","year"]))

    # 11. Forward feature selection emulation: add best features one by one
    ffs_base = []
    for f in ["fpar","lst_day","delta_lst","doy","ndvi_m09","sm_surface","lswi_m09","evi_m09","albedo_sw"]:
        ffs_base.append(f)
        c.append(("ffs_"+"_".join(f[:4] for f in ffs_base), ffs_base.copy()))

    # 12. Random exploration (50 random combos from pool)
    pool = all_features + ["vpd_kpa","precip_7d","precip_15d","ndvi_lag1","ndvi_diff"]
    pool = list(dict.fromkeys(pool))
    for _ in range(50):
        k = random.randint(1, 8)
        feats = random.sample(pool, min(k, len(pool)))
        c.append(("random_"+"_".join(f[:4] for f in feats), feats))

    # Deduplicate
    seen, unique = set(), []
    for name, feats in c:
        key = "|".join(sorted(feats))
        if key not in seen:
            seen.add(key)
            unique.append((name[:50], feats))
    return unique

def main():
    print("Loading maize-only data...")
    df = load_all()
    df = fill_nans(df, [c for c in df.columns if c not in ("station","date","date_prev","etc_obs_mm_d","kcact")])
    print(f"Maize samples (DOY {MAIZE_DOY[0]}-{MAIZE_DOY[1]}): {len(df)}")

    combos = gen_combos()
    print(f"Generated {len(combos)} unique combos\n")

    results = []
    for i, (name, feats) in enumerate(combos):
        avail = [f for f in feats if f in df.columns]
        if len(avail) < 1:
            results.append(dict(combo=name, n=0, feats=",".join(feats),
                                R2=np.nan, RMSE=np.nan, ns=0, status="NO_COLS"))
            continue
        dfc = fill_nans(df, avail)
        if len(dfc) < 15:
            results.append(dict(combo=name, n=len(avail), feats=",".join(avail),
                                R2=np.nan, RMSE=np.nan, ns=len(dfc), status="FEW"))
            continue
        r2, rmse = run_loso(dfc, avail)
        results.append(dict(combo=name, n=len(avail), feats=",".join(avail),
                            R2=round(r2,5) if not np.isnan(r2) else r2,
                            RMSE=round(rmse,5) if not np.isnan(rmse) else rmse,
                            ns=len(dfc), status="OK"))
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    out_df = pd.DataFrame(results).sort_values("R2", ascending=False, na_position="last")
    OUT.mkdir(parents=True, exist_ok=True)
    op = OUT / "maize_500_combo_results.csv"
    out_df.to_csv(op, index=False)

    print(f"\n{'Rank':4s} {'Combo':45s} {'n':>3s} {'R²':>8s} {'RMSE':>8s}  Status")
    print("-"*80)
    for i, (_, r) in enumerate(out_df.iterrows()):
        if i >= 40: break
        r2s = f"{r.R2:.5f}" if pd.notna(r.R2) else "     N/A"
        rms = f"{r.RMSE:.5f}" if pd.notna(r.RMSE) else "     N/A"
        print(f"{i+1:4d} {r.combo:45s} {r.n:3d} {r2s:>8s} {rms:>8s}  {r.status}")
    print(f"\n{len(out_df)} total → {op}")


if __name__ == "__main__":
    main()
