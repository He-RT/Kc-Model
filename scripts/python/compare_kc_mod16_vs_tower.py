"""Compare Kc from MOD16A2GF ET (500m) vs tower ET at 4 flux stations.

Both use the same ERA5-derived ET0 denominator, so this isolates the ETa source difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/Users/hert/Projects/dcsdxx/outputs/tables/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
mod16 = pd.read_csv(Path("/Users/hert/Projects/dcsdxx/data/processed/mod16_station_et.csv"))
mod16['date'] = pd.to_datetime(mod16['date'])
mod16 = mod16.rename(columns={'et_500m': 'etc_mod16_mm_d'})

etc_et0 = pd.read_csv(Path("/Users/hert/Projects/dcsdxx/data/processed/station_etc_with_et0.csv"))
etc_et0['date'] = pd.to_datetime(etc_et0['date'])
etc_et0['date_prev'] = pd.to_datetime(etc_et0['date_prev'])

# ── Convert MOD16 to daily mean ──────────────────────────────────────────────
# MOD16A2GF ET band: 8-day total ET (mm), scale factor 0.1
# Convert to average daily ET: total / 8
mod16['etc_mod16_mm_d'] = mod16['etc_mod16_mm_d'] / 8

# ── Match: for each station obs window, find overlapping MOD16 composites ──
print("Matching MOD16 windows to station observation windows...")
rows = []
skipped = 0
for _, obs in etc_et0.iterrows():
    stn = obs['station']
    t0, t1 = obs['date_prev'], obs['date']  # station observation window

    # MOD16 8-day composites whose [date, date+8d) overlaps with [t0, t1]
    stn_mod = mod16[mod16['station'] == stn].copy()
    stn_mod['date_end'] = stn_mod['date'] + pd.Timedelta(days=8)
    overlapping = stn_mod[
        (stn_mod['date'] < t1) & (stn_mod['date_end'] > t0)
    ]

    if overlapping.empty:
        # Fallback: nearest MOD16 window
        stn_mod['dist'] = abs((stn_mod['date'] - t1).dt.days)
        nearest = stn_mod.nsmallest(1, 'dist')
        etc_mod16 = nearest['etc_mod16_mm_d'].mean()
        n_mod16_windows = 0
        skipped += 1
    else:
        etc_mod16 = overlapping['etc_mod16_mm_d'].mean()
        n_mod16_windows = len(overlapping)

    rows.append({
        'station': stn,
        'date': obs['date'],
        'date_prev': obs['date_prev'],
        'etc_tower_mm_d': obs['etc_obs_mm_d'],
        'etc_mod16_mm_d': etc_mod16,
        'et0_pm_mm_d': obs['et0_pm_mean_mm_d'],
        'n_days_window': obs['n_days_window'],
        'kcact_tower': obs['kcact'],
        'kcact_mod16': etc_mod16 / obs['et0_pm_mean_mm_d'] if obs['et0_pm_mean_mm_d'] > 0 else np.nan,
        'n_mod16_overlap': n_mod16_windows,
    })

df = pd.DataFrame(rows)
print(f"Matched {len(df)} obs, {skipped} fell back to nearest MOD16 window")

# ── Quality flags ─────────────────────────────────────────────────────────────
df['kcact_mod16'] = df['kcact_mod16'].clip(0, 2)
df['delta_kc'] = df['kcact_mod16'] - df['kcact_tower']
df['bias_pct'] = 100 * df['delta_kc'] / df['kcact_tower']

# ── Overall statistics ───────────────────────────────────────────────────────
print("\n=== Overall ===")
print(f"Samples: {len(df)}")
print(f"Kcact_tower  mean={df['kcact_tower'].mean():.4f}  std={df['kcact_tower'].std():.4f}")
print(f"Kcact_mod16  mean={df['kcact_mod16'].mean():.4f}  std={df['kcact_mod16'].std():.4f}")
r = df['kcact_tower'].corr(df['kcact_mod16'])
rmse = np.sqrt(((df['kcact_mod16'] - df['kcact_tower'])**2).mean())
mae = (df['kcact_mod16'] - df['kcact_tower']).abs().mean()
print(f"Correlation r={r:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
print(f"Mean bias (MOD16 - tower): {df['delta_kc'].mean():.4f} ({df['bias_pct'].mean():.1f}%)")

# ── Per-station ──────────────────────────────────────────────────────────────
print("\n=== Per-station ===")
for stn in sorted(df['station'].unique()):
    s = df[df['station'] == stn]
    r_stn = s['kcact_tower'].corr(s['kcact_mod16'])
    print(f"  {stn}: n={len(s)}, r={r_stn:.4f}, "
          f"tower={s['kcact_tower'].mean():.3f}, mod16={s['kcact_mod16'].mean():.3f}, "
          f"bias={s['delta_kc'].mean():+.4f}")

# ── By window length ─────────────────────────────────────────────────────────
print("\n=== By window days ===")
for nd in sorted(df['n_days_window'].unique()):
    s = df[df['n_days_window'] == nd]
    if len(s) >= 5:
        print(f"  {int(nd)}d: n={len(s)}, r={s['kcact_tower'].corr(s['kcact_mod16']):.4f}, "
              f"bias={s['delta_kc'].mean():+.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
out = OUT_DIR / "kc_mod16_vs_tower.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")

# ── Quick scatter summary ────────────────────────────────────────────────────
import json
stats = {
    "n": len(df),
    "r": round(float(r), 4),
    "rmse": round(float(rmse), 4),
    "mae": round(float(mae), 4),
    "mean_bias": round(float(df['delta_kc'].mean()), 4),
    "kc_tower_mean": round(float(df['kcact_tower'].mean()), 4),
    "kc_mod16_mean": round(float(df['kcact_mod16'].mean()), 4),
}
(OUT_DIR / "kc_mod16_vs_tower_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
