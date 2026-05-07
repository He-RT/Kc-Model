"""Merge MOD09A1 raw bands (b01-b06) + compute RDVI + run full permutation."""

import itertools, gc, glob
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path.home() / 'dcsdxx'
PARQUET = ROOT / 'data/processed/train/ncp_summer_maize_kcact_with_modis.parquet'
CSV_DIR = ROOT / 'data/raw/gee/kcact_maize_modis_indicators'

print('[1/3] Merging raw bands...')
df = pd.read_parquet(PARQUET)
df['sm_proxy'] = df['precip_30d'] / 100
df['date'] = pd.to_datetime(df['date'])
df['point_id'] = df['point_id'].astype(str)

new_cols = {}
for yr in sorted(df['year'].unique()):
    mask = df['year'] == yr
    y = df.loc[mask, ['point_id','date']].copy()

    # --- rb: sur_refl_b01,b02 + ndvi_m09 ---
    rbf = sorted(glob.glob(str(CSV_DIR / f'maize_rb_*{yr}*.csv')))
    if rbf:
        rb = pd.concat([pd.read_csv(f) for f in rbf], ignore_index=True)
        rb['date'] = pd.to_datetime(rb['date'])
        rb['point_id'] = rb['point_id'].astype(str)
        if 'sur_refl_b01' in rb.columns:
            rb['b01'] = rb['sur_refl_b01'].astype(float) * 0.0001
        if 'sur_refl_b02' in rb.columns:
            rb['b02'] = rb['sur_refl_b02'].astype(float) * 0.0001
        if 'ndvi_m09' in rb.columns:
            rb['ndvi_m09_rb'] = rb['ndvi_m09'].astype(float)
        for c in ['b01','b02','ndvi_m09_rb']:
            if c in rb.columns:
                mc = rb.groupby(['point_id','date'], as_index=False)[c].mean()
                y = y.merge(mc, on=['point_id','date'], how='left')

    # --- rb2: b03,b04,b05,b06 (already renamed in GEE) ---
    r2f = sorted(glob.glob(str(CSV_DIR / f'maize_rb2_*{yr}*.csv')))
    if r2f:
        rb2 = pd.concat([pd.read_csv(f) for f in r2f], ignore_index=True)
        rb2['date'] = pd.to_datetime(rb2['date'])
        rb2['point_id'] = rb2['point_id'].astype(str)
        for c in ['b03','b04','b05','b06']:
            if c in rb2.columns:
                rb2[c] = rb2[c].astype(float) * 0.0001
                mc = rb2.groupby(['point_id','date'], as_index=False)[c].mean()
                y = y.merge(mc, on=['point_id','date'], how='left')

    for c in y.columns:
        if c in ['point_id','date']: continue
        new_cols.setdefault(c, pd.Series(np.nan, index=df.index))
        new_cols[c].loc[mask] = y[c].values
    print(f'  {yr}: merged')

for c, vals in new_cols.items():
    df[c] = vals
df = df.drop(columns=['point_id'], errors='ignore')

# Compute RDVI: (NIR-Red)/sqrt(NIR+Red)
eps = 1e-6
df['rdvi'] = (df['b02'] - df['b01']) / (np.sqrt(df['b02'] + df['b01']) + eps)

print(f'\n[2/3] Coverage:')
for c in ['b01','b02','b03','b04','b05','b06','rdvi']:
    n = df[c].notna().sum() if c in df.columns else 0
    pct = n / len(df) * 100 if n > 0 else 0
    print(f'  {c}: {n}/{len(df)} ({pct:.0f}%)')

df.to_parquet(str(ROOT / 'data/processed/train/ncp_summer_maize_kcact_with_rb.parquet'), index=False)

# ---- Train ----
print(f'\n[3/3] Running combos...')
POOL = ['ndvi_m09','evi','gndvi','savi','rdvi','sm_proxy','doy','wind_2m_m_s_mean_8d']
avail = [f for f in POOL if f in df.columns]
print(f'Pool ({len(avail)}): {avail}')

years = sorted(df['year'].unique())
combos = []
for k in range(1, len(avail)+1):
    for combo in itertools.combinations(avail, k):
        combos.append(list(combo))
print(f'{len(combos)} combos')

results = []
for idx, feats in enumerate(combos):
    sub = df.dropna(subset=feats + ['kcact'])
    if len(sub) < 1000: continue
    ap, aa = [], []
    for yr in years:
        tr = sub[sub['year']!=yr]; te = sub[sub['year']==yr]
        if len(te) < 10: continue
        m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                              task_type='GPU', devices='0', loss_function='RMSE',
                              random_seed=42, verbose=False)
        m.fit(tr[feats].values, tr['kcact'].values)
        ap.extend(m.predict(te[feats])); aa.extend(te['kcact'].values)
    r2 = r2_score(aa, ap)
    rmse = np.sqrt(mean_squared_error(aa, ap)).item()
    results.append(('+'.join(feats), len(feats), r2, rmse, len(sub)))
    if (idx+1) % 50 == 0:
        best = max(r[2] for r in results)
        print(f'  {idx+1}/{len(combos)} done, best R²={best:.5f}')

results.sort(key=lambda x: -x[2])
print(f'\n=== Top 10 ===')
for i,(name,n,r2,rmse,ns) in enumerate(results[:10]):
    print(f'  {i+1:2d}. {name:85s} n={n} R²={r2:.5f}')

out = pd.DataFrame(results, columns=['combo','n_features','LOYO_R2','LOYO_RMSE','n_samples'])
out.to_csv(str(ROOT / 'outputs/tables/rdvi_combos.csv'), index=False)
print(f'Saved {len(results)} combos')

# Feature avg R2
fc = {}
for name,_,r2,_,_ in results:
    for f in name.split('+'):
        fc.setdefault(f,{'n':0,'r2_sum':0})
        fc[f]['n']+=1; fc[f]['r2_sum']+=r2
print('\nFeature avg R2:')
for f in avail:
    d=fc[f]
    print(f'  {f:30s}: {d["n"]:3d} combos, avg R²={d["r2_sum"]/d["n"]:.4f}')
