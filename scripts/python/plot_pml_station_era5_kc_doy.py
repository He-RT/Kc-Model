from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.font_manager import FontProperties

ROOT = Path('/Users/hert/Projects/dcsdxx')
IN = ROOT / 'outputs/tables/pml_era5_kcact_vs_station_era5_kcact.csv'
OUT_FIG = ROOT / 'outputs/figures/pml_station_era5_kcact_doy_summer_maize.png'
OUT_PDF = ROOT / 'outputs/figures/pml_station_era5_kcact_doy_summer_maize.pdf'
OUT_CSV = ROOT / 'outputs/tables/pml_station_era5_kcact_doy_summer_maize.csv'
OUT_SCRIPT = ROOT / 'scripts/python/plot_pml_station_era5_kc_doy.py'
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_SCRIPT.parent.mkdir(parents=True, exist_ok=True)

font_paths = [
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/System/Library/Fonts/Supplemental/Songti.ttc',
]
zh_prop = None
for fp in font_paths:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        zh_prop = FontProperties(fname=fp)
        rcParams['font.family'] = zh_prop.get_name()
        break
rcParams['axes.unicode_minus'] = False
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Load paired station/PML table. Both Kcact definitions use ERA5 ET0 as denominator.
df = pd.read_csv(IN, parse_dates=['date', 'date_prev'])
if 'kcact_station_eta_era5' not in df.columns:
    df['kcact_station_eta_era5'] = df['station_eta_mm_d'] / df['era5_et0_mm_d']
if 'kcact_pml_era5' not in df.columns:
    df['kcact_pml_era5'] = df['pml_eta_mm_d'] / df['era5_et0_mm_d']

df['doy'] = df['date'].dt.dayofyear
# 夏玉米主要生长周期，和前面 PML 大样本图保持一致。
df = df[(df['doy'] >= 160) & (df['doy'] <= 305)].copy()
valid = (
    df[['kcact_station_eta_era5', 'kcact_pml_era5', 'era5_et0_mm_d', 'station_eta_mm_d', 'pml_eta_mm_d']]
      .replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    & (df['window_days'] == 8)
    & df['era5_et0_mm_d'].between(0.1, 12.0)
    & df['kcact_station_eta_era5'].between(0, 3.0)
    & df['kcact_pml_era5'].between(0, 3.0)
)
df = df.loc[valid].copy()
# 统一到 8 日 DOY 组，降低不同站点日期偏移造成的抖动。
df['doy_bin'] = 161 + ((df['doy'] - 161) // 8) * 8
df['doy_bin'] = df['doy_bin'].clip(161, 305).astype(int)

# 先做 station × doy_bin 均值，再对站点均值求平均，避免禹城样本数过多主导曲线。
station_bin = df.groupby(['station', 'doy_bin'], as_index=False).agg(
    station_kc=('kcact_station_eta_era5', 'mean'),
    pml_kc=('kcact_pml_era5', 'mean'),
    station_eta_mm_d=('station_eta_mm_d', 'mean'),
    pml_eta_mm_d=('pml_eta_mm_d', 'mean'),
    era5_et0_mm_d=('era5_et0_mm_d', 'mean'),
    n=('date', 'size'),
)

def se(x):
    x = pd.Series(x).dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan

summary = station_bin.groupby('doy_bin', as_index=False).agg(
    station_kc_mean=('station_kc', 'mean'),
    pml_kc_mean=('pml_kc', 'mean'),
    station_kc_se=('station_kc', se),
    pml_kc_se=('pml_kc', se),
    station_eta_mean=('station_eta_mm_d', 'mean'),
    pml_eta_mean=('pml_eta_mm_d', 'mean'),
    era5_et0_mean=('era5_et0_mm_d', 'mean'),
    n_station_bins=('station', 'size'),
    n_obs=('n', 'sum'),
)
summary['date_label'] = (pd.Timestamp('2020-01-01') + pd.to_timedelta(summary['doy_bin'] - 1, unit='D')).dt.strftime('%m-%d')
summary['station_ci_low'] = summary['station_kc_mean'] - 1.96 * summary['station_kc_se']
summary['station_ci_high'] = summary['station_kc_mean'] + 1.96 * summary['station_kc_se']
summary['pml_ci_low'] = summary['pml_kc_mean'] - 1.96 * summary['pml_kc_se']
summary['pml_ci_high'] = summary['pml_kc_mean'] + 1.96 * summary['pml_kc_se']
summary.to_csv(OUT_CSV, index=False)

plt.style.use('default')
rcParams.update({
    'figure.dpi': 180,
    'savefig.dpi': 320,
    'axes.linewidth': 0.9,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'legend.frameon': False,
})
fig, ax = plt.subplots(figsize=(7.7, 4.9))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Growth stage bands.
stages = [
    (160, 177, '苗期'),
    (178, 209, '拔节期'),
    (210, 241, '抽雄吐丝'),
    (242, 281, '灌浆期'),
    (282, 305, '成熟期'),
]
stage_colors = ['#F7F7F7', '#EEF5EE', '#FFF5E6', '#EEF4FB', '#F8EFEF']
for (start, end, label), color in zip(stages, stage_colors):
    ax.axvspan(start, end, color=color, zorder=0)
    ax.text((start + end) / 2, 1.72, label, ha='center', va='center', fontsize=8.4, color='#555555', fontproperties=zh_prop)

x = summary['doy_bin'].to_numpy(float)
st = summary['station_kc_mean'].to_numpy(float)
pm = summary['pml_kc_mean'].to_numpy(float)
# CI only where enough station bins exist.
for prefix, color in [('station', '#C2410C'), ('pml', '#1F4E79')]:
    lo = summary[f'{prefix}_ci_low'].to_numpy(float)
    hi = summary[f'{prefix}_ci_high'].to_numpy(float)
    y = summary[f'{prefix}_kc_mean'].to_numpy(float)
    good = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(x[good], lo[good], hi[good], color=color, alpha=0.10, lw=0, zorder=1)

ax.plot(x, st, color='#C2410C', lw=2.25, marker='o', markersize=4.5, markeredgecolor='white', markeredgewidth=0.4, label='站点 ETa / ERA5 ET0', zorder=3)
ax.plot(x, pm, color='#1F4E79', lw=2.25, marker='o', markersize=4.5, markeredgecolor='white', markeredgewidth=0.4, label='PML ETa / ERA5 ET0', zorder=4)

ax.set_xlim(158, 307)
ax.set_ylim(0.0, 1.8)
xticks = [161, 177, 193, 209, 225, 241, 257, 273, 289, 305]
ax.set_xticks(xticks)
ax.set_xticklabels([(pd.Timestamp('2020-01-01') + pd.Timedelta(days=d-1)).strftime('%-m/%-d') for d in xticks], fontproperties=zh_prop)
ax.set_xlabel('夏玉米生长季时间', fontsize=10.5, fontproperties=zh_prop)
ax.set_ylabel('Kcact = ETa / ERA5 ET0', fontsize=10.5, fontproperties=zh_prop)
ax.set_title('PML 与站点实测 Kcact 的生长季变化\n统一使用 ERA5 ET0 作为分母', fontsize=12, pad=10, fontproperties=zh_prop)
ax.grid(axis='y', color='#D9D9D9', lw=0.55, alpha=0.7)
ax.grid(axis='x', visible=False)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
leg = ax.legend(loc='upper left', fontsize=8.8, handlelength=2.4)
for text in leg.get_texts():
    text.set_fontproperties(zh_prop)

# Annotate largest gap.
summary['gap'] = summary['station_kc_mean'] - summary['pml_kc_mean']
if len(summary):
    row = summary.loc[summary['gap'].abs().idxmax()]
    ax.annotate(f"差值 {row['gap']:+.2f}\nDOY {int(row['doy_bin'])}",
                xy=(row['doy_bin'], max(row['station_kc_mean'], row['pml_kc_mean'])),
                xytext=(row['doy_bin'] - 22, min(1.55, max(row['station_kc_mean'], row['pml_kc_mean']) + 0.26)),
                arrowprops=dict(arrowstyle='-', color='#333333', lw=0.8),
                fontsize=8.4, color='#222222', fontproperties=zh_prop)

caption = f"数据：4个站点与PML V2.2a同窗口对齐；筛选DOY 160–305，n={int(summary['n_obs'].sum())} 条8日记录。曲线先按站点-DOY求均值，再跨站点平均。"
fig.text(0.02, 0.012, caption, fontsize=7.4, color='#555555', fontproperties=zh_prop)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT_FIG, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
print(OUT_FIG)
print(OUT_PDF)
print(OUT_CSV)
print(summary[['doy_bin','date_label','station_kc_mean','pml_kc_mean','gap','n_station_bins','n_obs']].to_string(index=False))
