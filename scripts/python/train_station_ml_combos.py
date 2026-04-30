from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "station_ml_features.csv"
RAW_PATH = PROJECT_ROOT / "data" / "processed" / "modis_cache" / "mod09a1_raw_bands.csv"
ERA5_PATH = PROJECT_ROOT / "data" / "processed" / "modis_cache" / "era5_weather_all.csv"
STN_PATH = PROJECT_ROOT / "data" / "processed" / "station_coordinates.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"
TARGET = "kcact"

RAW_BAND_NAMES = ["sur_refl_b01","sur_refl_b02","sur_refl_b03","sur_refl_b04","sur_refl_b05","sur_refl_b06","sur_refl_b07"]
RAW_RENAMED = ["b01","b02","b03","b04","b05","b06","b07"]
NAME_MAP = {"yucheng":"禹城","weishan":"位山","guantao":"馆陶","luancheng":"栾城"}
ERA5_VARS = ["tmean_c","tmin_c","tmax_c","dewpoint_c","solar_rad_mj_m2_d","precip_mm","pressure_kpa","wind_10m_m_s"]

ALL_RS = ["ndvi","evi","lswi","lst_day","albedo_sw","fpar","delta_lst","sm_surface"]

# ---- Derived VI from MOD09A1 raw bands ----
def compute_mod09_vis(df):
    """NDVI=(b02-b01)/(b02+b01), EVI09=2.5*(b02-b01)/(b02+6*b01-7.5*b03+1)"""
    r = df.copy()
    eps = 1e-6
    r["ndvi_m09"] = (r["b02"] - r["b01"]) / (r["b02"] + r["b01"] + eps)
    r["evi_m09"] = 2.5 * (r["b02"] - r["b01"]) / (r["b02"] + 6 * r["b01"] - 7.5 * r["b03"] + 1)
    r["lswi_m09"] = (r["b02"] - r["b06"]) / (r["b02"] + r["b06"] + eps)
    r["gndvi_m09"] = (r["b02"] - r["b04"]) / (r["b02"] + r["b04"] + eps)
    r["savi_m09"] = 1.5 * (r["b02"] - r["b01"]) / (r["b02"] + r["b01"] + 0.5)
    return r

# ---- Helpers ----
def prepare_base_df():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["date_prev"] = pd.to_datetime(df["date_prev"])
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    return df.dropna(subset=[TARGET])

def merge_raw_bands(df):
    raw = pd.read_csv(RAW_PATH)
    raw["station"] = raw["station"].map(NAME_MAP).fillna(raw["station"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.rename(dict(zip(RAW_BAND_NAMES, RAW_RENAMED)), axis=1)
    result = df.copy()
    for col in RAW_RENAMED: result[col] = np.nan
    for _, row in result.iterrows():
        stn, tp, tc = row["station"], row["date_prev"], row["date"]
        w = raw[(raw["station"]==stn) & (raw["date"]>tp) & (raw["date"]<=tc)]
        if len(w) > 0:
            for col in RAW_RENAMED: result.at[row.name, col] = w[col].mean()
    for col in RAW_RENAMED:
        result[col] = result.groupby("station")[col].transform(lambda x: x.fillna(x.median()))
    return result.dropna(subset=RAW_RENAMED)

def merge_era5_weather(df):
    w = pd.read_csv(ERA5_PATH)
    w["date"] = pd.to_datetime(w["date"])
    result = df.copy()
    for col in ERA5_VARS: result[col] = np.nan
    for _, row in result.iterrows():
        stn, tp, tc = row["station"], row["date_prev"], row["date"]
        win = w[(w["station"]==stn) & (w["date"]>tp) & (w["date"]<=tc)]
        if len(win) > 0:
            for col in ERA5_VARS: result.at[row.name, col] = win[col].mean()
    for col in ERA5_VARS:
        result[col] = result.groupby("station")[col].transform(lambda x: x.fillna(x.median()))
    return result

def add_derived_features(df):
    r = df.copy()
    r["vpd_kpa"] = (
        0.6108 * np.exp(17.27 * r["tmean_c"] / (r["tmean_c"] + 237.3))
        - 0.6108 * np.exp(17.27 * r["dewpoint_c"] / (r["dewpoint_c"] + 237.3))
    ).clip(lower=0)
    r["precip_7d"] = r.groupby("station")["precip_mm"].transform(lambda s: s.rolling(1, min_periods=1).sum())
    r["precip_15d"] = r.groupby("station")["precip_mm"].transform(lambda s: s.rolling(2, min_periods=1).sum())
    r["precip_30d"] = r.groupby("station")["precip_mm"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    r["ndvi_lag1"] = r.groupby("station")["ndvi"].shift(1).fillna(r["ndvi"])
    r["ndvi_diff"] = r["ndvi"] - r["ndvi_lag1"]
    r["ndvi_vpd"] = r["ndvi"] * r["vpd_kpa"]
    r["lswi_vpd"] = r["lswi"] * r["vpd_kpa"]
    stn_lookup = pd.read_csv(STN_PATH).set_index("station")
    r["lat"] = r["station"].map(stn_lookup["lat"].to_dict())
    r["lon"] = r["station"].map(stn_lookup["lon"].to_dict())
    return r

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
        train_m = stations != stn; test_m = stations == stn
        if train_m.sum() < 10 or test_m.sum() == 0: continue
        m = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4,
                              loss_function="RMSE", random_seed=42, verbose=False)
        m.fit(X[train_m], y[train_m])
        preds[test_m] = m.predict(X[test_m])
    valid = ~np.isnan(preds)
    return r2_score(y[valid], preds[valid]), float(np.sqrt(mean_squared_error(y[valid], preds[valid])))

# ---- Feature combo generator ----
def all_subsets(items):
    """All non-empty subsets."""
    import itertools
    result = []
    for k in range(1, len(items)+1):
        for combo in itertools.combinations(items, k):
            result.append(list(combo))
    return result

def build_combos():
    combos = []

    # A - Single features
    for f in ["ndvi","evi","lswi","lst_day","albedo_sw","fpar","delta_lst","sm_surface","doy"]:
        combos.append((f"single_{f}", [f]))

    # B - MOD09A1-derived VIs (100% coverage vs MOD13Q1 50%)
    M09_VI = ["ndvi_m09","evi_m09","lswi_m09","gndvi_m09","savi_m09"]
    for f in M09_VI: combos.append((f"single_{f}", [f]))
    combos.append(("VI_m09", M09_VI))
    combos.append(("VI_m09_doy", M09_VI + ["doy"]))
    combos.append(("VI_m09_top3", ["ndvi_m09","evi_m09","lswi_m09"]))
    combos.append(("VI_m09_top4_doy", ["ndvi_m09","evi_m09","lswi_m09","lswi","doy"]))

    # C - Systematic ablation of top {fpar, lst_day, lswi, delta_lst, sm_surface} subsets
    TOP5 = ["fpar","lst_day","lswi","delta_lst","sm_surface"]
    for subset in all_subsets(TOP5):
        name = "top5_" + "_".join(s[:4] for s in subset)
        combos.append((name, subset))
        # with doy
        if len(subset) <= 4:
            combos.append((name + "_doy", subset + ["doy"]))

    # D - MOD09A1 VI + thermal/energy combos
    combos.append(("m09vi_lst_fpar", ["ndvi_m09","lswi_m09","lst_day","fpar"]))
    combos.append(("m09vi_lst_fpar_doy", ["ndvi_m09","lswi_m09","lst_day","fpar","doy"]))
    combos.append(("m09vi_thermal", ["ndvi_m09","evi_m09","lswi_m09","lst_day","delta_lst"]))
    combos.append(("m09vi_thermal_doy", ["ndvi_m09","evi_m09","lswi_m09","lst_day","delta_lst","doy"]))
    combos.append(("m09vi_all_doy", M09_VI + ["lst_day","fpar","delta_lst","sm_surface","doy"]))

    # E - Interaction terms
    combos.append(("interact_fpar_lst", ["fpar","lst_day","fpar_x_lst"]))
    combos.append(("interact_fpar_lst_doy", ["fpar","lst_day","fpar_x_lst","doy"]))
    combos.append(("interact_lswi_dlt", ["lswi","delta_lst","lswi_x_dlt"]))
    combos.append(("interact_lswi_dlt_doy", ["lswi","delta_lst","lswi_x_dlt","doy"]))
    combos.append(("interact_full", ["fpar","lst_day","lswi","delta_lst","fpar_x_lst","lswi_x_dlt","doy"]))

    # F - Seasonal encoding
    combos.append(("top4_cyclic", ["fpar","lst_day","lswi","delta_lst","doy_sin","doy_cos"]))
    combos.append(("top4_cyclic_year", ["fpar","lst_day","lswi","delta_lst","doy_sin","doy_cos","year"]))
    combos.append(("all_8_cyclic", ALL_RS + ["doy_sin","doy_cos"]))
    combos.append(("all_8_cyclic_year", ALL_RS + ["doy_sin","doy_cos","year"]))
    combos.append(("m09vi_cyclic", M09_VI + ["doy_sin","doy_cos"]))

    # G - Original 39 from previous run (condensed, all reproduced)
    orig = [
        ("doy_only", ["doy"]),
        ("ndvi_only", ["ndvi"]), ("fpar_only", ["fpar"]), ("lst_day_only", ["lst_day"]),
        ("VI_only", ["ndvi","evi","lswi"]), ("VI_only_doy", ["ndvi","evi","lswi","doy"]),
        ("thermal_only", ["lst_day","delta_lst"]), ("thermal_only_doy", ["lst_day","delta_lst","doy"]),
        ("moisture_only", ["lswi","sm_surface"]), ("energy_only", ["albedo_sw","fpar"]),
        ("fpar_lswi", ["fpar","lswi"]), ("fpar_lswi_doy", ["fpar","lswi","doy"]),
        ("ndvi_fpar", ["ndvi","fpar"]), ("ndvi_fpar_doy", ["ndvi","fpar","doy"]),
        ("lst_albedo", ["lst_day","albedo_sw"]),
        ("lst_lswi_sm", ["lst_day","lswi","sm_surface"]),
        ("delta_lswi_sm", ["delta_lst","lswi","sm_surface"]),
        ("vi_thermal", ["ndvi","evi","lswi","lst_day","delta_lst"]),
        ("vi_energy", ["ndvi","evi","lswi","albedo_sw","fpar"]),
        ("thermal_energy", ["lst_day","delta_lst","albedo_sw","fpar"]),
        ("thermal_moisture", ["lst_day","delta_lst","lswi","sm_surface"]),
        ("all_7_nosavi", ["ndvi","evi","lswi","lst_day","albedo_sw","fpar","delta_lst"]),
        ("all_7_nosavi_doy", ["ndvi","evi","lswi","lst_day","albedo_sw","fpar","delta_lst","doy"]),
        ("all_8", ALL_RS), ("all_8_doy", ALL_RS + ["doy"]),
        ("all_8_doy_year", ALL_RS + ["doy","year"]),
        ("top2", ["fpar","lst_day"]), ("top2_doy", ["fpar","lst_day","doy"]),
        ("top3", ["fpar","lst_day","lswi"]), ("top3_doy", ["fpar","lst_day","lswi","doy"]),
        ("top4", ["fpar","lst_day","lswi","delta_lst"]),
        ("top4_doy", ["fpar","lst_day","lswi","delta_lst","doy"]),
        ("top5", ["fpar","lst_day","lswi","delta_lst","sm_surface"]),
        ("top6", ["fpar","lst_day","lswi","delta_lst","sm_surface","ndvi"]),
        ("top7", ["fpar","lst_day","lswi","delta_lst","sm_surface","ndvi","evi"]),
        ("raw_bands", RAW_RENAMED), ("raw_bands_doy", RAW_RENAMED + ["doy"]),
        ("existing_analog", ["ndvi","evi","lswi","tmean_c","vpd_kpa","solar_rad_mj_m2_d",
                             "precip_mm","wind_10m_m_s","precip_7d","doy","year","lat","lon"]),
        ("existing_analog_full", ["ndvi","evi","lswi","tmean_c","tmin_c","tmax_c","vpd_kpa",
                                  "solar_rad_mj_m2_d","precip_mm","pressure_kpa","wind_10m_m_s",
                                  "precip_7d","precip_15d","precip_30d","ndvi_lag1","ndvi_diff",
                                  "ndvi_vpd","lswi_vpd","doy","year","lat","lon"]),
    ]
    combos.extend(orig)

    # H - MOD09A1 VI replaces MOD13Q1 in key combos
    combos.append(("m09ndvi_fpar_doy", ["ndvi_m09","fpar","doy"]))
    combos.append(("m09ndvi_lswi_fpar_doy", ["ndvi_m09","lswi_m09","fpar","doy"]))
    combos.append(("m09vi_fpar_lst_doy", ["ndvi_m09","evi_m09","lswi_m09","fpar","lst_day","doy"]))

    # I - raw bands + VI hybrids
    combos.append(("raw_m09ndvi_doy", RAW_RENAMED + ["ndvi_m09","doy"]))
    combos.append(("raw_m09evi_lswi_doy", RAW_RENAMED + ["evi_m09","lswi_m09","doy"]))

    # J - Double-check NDVI from MOD09A1 vs MOD13Q1
    combos.append(("ndvi_m09_only", ["ndvi_m09"]))
    combos.append(("ndvi_m09_doy", ["ndvi_m09","doy"]))
    combos.append(("ndvi_m09_fpar_only", ["ndvi_m09","fpar"]))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for name, feats in combos:
        key = name
        if key not in seen:
            seen.add(key)
            unique.append((name, feats))
    return unique

# ---- Main ----
def main():
    base = prepare_base_df()
    base = fill_nans(base, ALL_RS + ["doy","year","doy_sin","doy_cos"])
    raw_df, existing_df = None, None

    combos = build_combos()
    print(f"Testing {len(combos)} feature combinations...\n")

    results = []
    for combo_name, features in combos:
        status = "OK"

        if combo_name.startswith("raw_") or any("m09" in f for f in features):
            if raw_df is None:
                try:
                    raw_df = merge_raw_bands(base)
                    raw_df = compute_mod09_vis(raw_df)
                    raw_df = fill_nans(raw_df, [c for c in ["doy","year","doy_sin","doy_cos"] if c in raw_df.columns])
                except FileNotFoundError:
                    results.append(dict(combo_name=combo_name, n_features=len(features),
                                        feature_list=", ".join(features),
                                        LOSO_R2=np.nan, LOSO_RMSE=np.nan,
                                        n_samples=0, status="SKIP_NOFILE"))
                    continue
            df_combo = raw_df.copy()
            features = [f + "_x_lst" if f == "fpar_x_lst" else f for f in features]
            if "fpar_x_lst" in features:
                df_combo["fpar_x_lst"] = df_combo["fpar"] * df_combo["lst_day"]
            if "lswi_x_dlt" in features:
                df_combo["lswi_x_dlt"] = df_combo["lswi"] * df_combo["delta_lst"]
        elif combo_name.startswith("existing_analog"):
            if existing_df is None:
                try:
                    existing_df = merge_era5_weather(base)
                    existing_df = add_derived_features(existing_df)
                except FileNotFoundError:
                    results.append(dict(combo_name=combo_name, n_features=len(features),
                                        feature_list=", ".join(features),
                                        LOSO_R2=np.nan, LOSO_RMSE=np.nan,
                                        n_samples=0, status="SKIP_NOFILE"))
                    continue
            df_combo = existing_df.copy()
        elif "interact" in combo_name:
            if raw_df is None:
                try:
                    raw_df = merge_raw_bands(base)
                    raw_df = compute_mod09_vis(raw_df)
                    raw_df = fill_nans(raw_df, [c for c in ["doy","year","doy_sin","doy_cos"] if c in raw_df.columns])
                except: pass
            df_combo = base.copy()
            if "fpar_x_lst" in features: df_combo["fpar_x_lst"] = df_combo["fpar"] * df_combo["lst_day"]
            if "lswi_x_dlt" in features: df_combo["lswi_x_dlt"] = df_combo["lswi"] * df_combo["delta_lst"]
        else:
            df_combo = base.copy()
            if "fpar_x_lst" in features: df_combo["fpar_x_lst"] = df_combo["fpar"] * df_combo["lst_day"]
            if "lswi_x_dlt" in features: df_combo["lswi_x_dlt"] = df_combo["lswi"] * df_combo["delta_lst"]

        df_combo = fill_nans(df_combo, features)
        if len(df_combo) < 20 or len(df_combo["station"].unique()) < 2:
            status = "FEW_SAMPLES"; r2, rmse = np.nan, np.nan
        else:
            r2, rmse = run_loso(df_combo, features)

        results.append(dict(combo_name=combo_name, n_features=len(features),
                            feature_list=", ".join(features),
                            LOSO_R2=round(r2,5) if not np.isnan(r2) else r2,
                            LOSO_RMSE=round(rmse,5) if not np.isnan(rmse) else rmse,
                            n_samples=len(df_combo), status=status))

    out = pd.DataFrame(results).sort_values("LOSO_R2", ascending=False, na_position="last")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    op = OUT_DIR / "station_ml_combo_results.csv"
    out.to_csv(op, index=False)

    # Print top 25
    print(f"{'Rank':4s} {'Combo':32s} {'n':3s} {'R²':>8s} {'RMSE':>8s}  Status")
    print("-"*68)
    for i, (_, r) in enumerate(out.iterrows()):
        if i >= 30: break
        r2s = f"{r.LOSO_R2:.5f}" if pd.notna(r.LOSO_R2) else "     N/A"
        rms = f"{r.LOSO_RMSE:.5f}" if pd.notna(r.LOSO_RMSE) else "     N/A"
        print(f"{i+1:4d} {r.combo_name:32s} {r.n_features:3d} {r2s:>8s} {rms:>8s}  {r.status}")

    print(f"\n... ({len(out)} total) → {op}")


if __name__ == "__main__":
    main()
