from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.font_manager import FontProperties

ROOT = Path('/Users/hert/Projects/dcsdxx')
IN = ROOT / 'outputs/tables/pml_era5_kcact_vs_station_weather_kcact.csv'
OUT_FIG = ROOT / 'outputs/figures/pml_era5_vs_station_met_kcact_doy_summer_maize_paper.png'
OUT_PDF = ROOT / 'outputs/figures/pml_era5_vs_station_met_kcact_doy_summer_maize_paper.pdf'
OUT_SVG = ROOT / 'outputs/figures/pml_era5_vs_station_met_kcact_doy_summer_maize_paper.svg'
OUT_CSV = ROOT / 'outputs/tables/pml_era5_vs_station_met_kcact_doy_summer_maize_paper.csv'
OUT_SCRIPT = ROOT / 'scripts/python/plot_pml_era5_vs_station_met_kc_doy.py'
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_SCRIPT.parent.mkdir(parents=True, exist_ok=True)

# Chinese-capable font; keep vector text embeddable in PDF/SVG.
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
rcParams['svg.fonttype'] = 'none'

# Data: paired station observations and station-location PML extraction.
df = pd.read_csv(IN, parse_dates=['date', 'date_prev'])
# Use month-day based DOY on a non-leap reference year. This avoids leap-year
# offsets when multiple years are folded into one summer-maize growth curve.
def md_to_doy(month: int, day: int) -> int:
    return int(pd.Timestamp(year=2001, month=month, day=day).dayofyear)

df['doy_common'] = df['date'].apply(lambda d: md_to_doy(int(d.month), int(d.day)))
# Requested ratios.
df['kc_pml_era5'] = df['pml_eta_mm_d'] / df['era5_et0_mm_d']
df['kc_station_met'] = df['station_eta_mm_d'] / df['met_station_et0_mm_d']

# Summer maize stages from the report table:
# Initial stage 06-15–07-06; developing 07-07–08-08;
# mid stage 08-09–09-12; end stage 09-13–10-11.
#
# The source data are 8-day windows. For stage-wise plotting, keep the
# overlapping window before the first stage and the overlapping window after
# the last stage: 06-09–10-15.
STAGE_START = md_to_doy(6, 15)
STAGE_END = md_to_doy(10, 11)
PLOT_START = md_to_doy(6, 9)
PLOT_END = md_to_doy(10, 15)

# 8-day bins anchored to 06-09, matching the existing station/PML windows.
BIN_ANCHOR = PLOT_START
df['doy_bin'] = BIN_ANCHOR + ((df['doy_common'] - BIN_ANCHOR) // 8) * 8

valid = (
    df['doy_bin'].between(PLOT_START, PLOT_END)
    & (df['window_days'] == 8)
    & df[['pml_eta_mm_d', 'station_eta_mm_d', 'era5_et0_mm_d', 'met_station_et0_mm_d', 'kc_pml_era5', 'kc_station_met']]
        .replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    & df['era5_et0_mm_d'].between(0.1, 12.0)
    & df['met_station_et0_mm_d'].between(0.1, 12.0)
    & df['kc_pml_era5'].between(0, 3.0)
    & df['kc_station_met'].between(0, 3.0)
)
df = df.loc[valid].copy()
df['doy_bin'] = df['doy_bin'].clip(PLOT_START, PLOT_END).astype(int)

# First average within station × DOY bin, then average stations, so no station dominates by sample count.
station_bin = df.groupby(['station', 'doy_bin'], as_index=False).agg(
    kc_pml_era5=('kc_pml_era5', 'mean'),
    kc_station_met=('kc_station_met', 'mean'),
    pml_eta_mean=('pml_eta_mm_d', 'mean'),
    station_eta_mean=('station_eta_mm_d', 'mean'),
    era5_et0_mean=('era5_et0_mm_d', 'mean'),
    met_et0_mean=('met_station_et0_mm_d', 'mean'),
    n_obs=('date', 'size'),
)

def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))

summary = station_bin.groupby('doy_bin', as_index=False).agg(
    pml_kc_mean=('kc_pml_era5', 'mean'),
    station_kc_mean=('kc_station_met', 'mean'),
    pml_kc_se=('kc_pml_era5', sem),
    station_kc_se=('kc_station_met', sem),
    pml_eta_mean=('pml_eta_mean', 'mean'),
    station_eta_mean=('station_eta_mean', 'mean'),
    era5_et0_mean=('era5_et0_mean', 'mean'),
    met_et0_mean=('met_et0_mean', 'mean'),
    n_station_bins=('station', 'size'),
    n_obs=('n_obs', 'sum'),
)
summary['date_label'] = (pd.Timestamp('2001-01-01') + pd.to_timedelta(summary['doy_bin'] - 1, unit='D')).dt.strftime('%m-%d')
summary['gap_pml_minus_station'] = summary['pml_kc_mean'] - summary['station_kc_mean']
summary['pml_ci_low'] = summary['pml_kc_mean'] - 1.96 * summary['pml_kc_se']
summary['pml_ci_high'] = summary['pml_kc_mean'] + 1.96 * summary['pml_kc_se']
summary['station_ci_low'] = summary['station_kc_mean'] - 1.96 * summary['station_kc_se']
summary['station_ci_high'] = summary['station_kc_mean'] + 1.96 * summary['station_kc_se']

# Agreement between the two mean seasonal curves. For a DOY-series comparison,
# report R² from a linear fit across the 8-day mean points.
fit_mask = summary[['station_kc_mean', 'pml_kc_mean']].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
fit_x = summary.loc[fit_mask, 'station_kc_mean'].to_numpy(float)
fit_y = summary.loc[fit_mask, 'pml_kc_mean'].to_numpy(float)
if len(fit_x) >= 2 and np.nanstd(fit_y) > 0:
    fit_coef = np.polyfit(fit_x, fit_y, 1)
    fit_pred = np.polyval(fit_coef, fit_x)
    mean_curve_r2 = 1.0 - np.sum((fit_y - fit_pred) ** 2) / np.sum((fit_y - np.mean(fit_y)) ** 2)
else:
    mean_curve_r2 = np.nan

summary.to_csv(OUT_CSV, index=False)

# Paper-style figure: restrained colors, clear axis, 300+ dpi output.
plt.style.use('default')
rcParams.update({
    'figure.dpi': 180,
    'savefig.dpi': 600,
    'axes.linewidth': 0.75,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'legend.frameon': False,
})
# 170 mm × 95 mm, suitable for report insertion.
fig, ax = plt.subplots(figsize=(6.7, 3.75))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Subtle growth-stage background, using the stage dates in the report table.
stages = [
    (md_to_doy(6, 15), md_to_doy(7, 6), '生长初期'),
    (md_to_doy(7, 7), md_to_doy(8, 8), '快速生长期'),
    (md_to_doy(8, 9), md_to_doy(9, 12), '生长中期'),
    (md_to_doy(9, 13), md_to_doy(10, 11), '生长后期'),
]
colors_bg = ['#F7F7F7', '#EDF5EF', '#FFF5E6', '#EEF4FB']
for (start, end, label), color in zip(stages, colors_bg):
    ax.axvspan(start, end, color=color, zorder=0)
    # Stage names are described in the table/caption rather than annotated
    # inside the plotting area, keeping the figure suitable for manuscript use.

pml_color = '#1F4E79'      # blue
station_color = '#C2410C'  # orange-red
# Per-station thin lines to make clear this is station-location PML, not large-sample mean.
for st, g in station_bin.groupby('station'):
    g = g.sort_values('doy_bin')
    ax.plot(g['doy_bin'], g['kc_pml_era5'], color=pml_color, lw=0.7, alpha=0.18, zorder=1)
    ax.plot(g['doy_bin'], g['kc_station_met'], color=station_color, lw=0.7, alpha=0.18, zorder=1)

x = summary['doy_bin'].to_numpy(float)
# Confidence ribbons across stations.
for prefix, color in [('pml', pml_color), ('station', station_color)]:
    lo = summary[f'{prefix}_ci_low'].to_numpy(float)
    hi = summary[f'{prefix}_ci_high'].to_numpy(float)
    good = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(x[good], lo[good], hi[good], color=color, alpha=0.10, lw=0, zorder=2)

ax.plot(x, summary['pml_kc_mean'], color=pml_color, lw=1.9, marker='o', markersize=3.7,
        markeredgecolor='white', markeredgewidth=0.35, label='PML/ERA5', zorder=4)
ax.plot(x, summary['station_kc_mean'], color=station_color, lw=1.9, marker='o', markersize=3.7,
        markeredgecolor='white', markeredgewidth=0.35, label='站点/气象站', zorder=5)

if np.isfinite(mean_curve_r2):
    ax.text(
        0.025, 0.945, f'$R^2$ = {mean_curve_r2:.3f}',
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=8.0,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#BDBDBD', linewidth=0.45, alpha=0.86),
        zorder=6,
    )

# No point annotation in the paper version; keep the plot clean for insertion.
ax.set_xlim(PLOT_START - 1, PLOT_END + 1)
ax.set_ylim(0.25, 1.22)
# Keep the leading/trailing 8-day windows in the plotted range, but avoid
# crowding the axis with very close labels such as 06/09–06/15 and
# 10/11–10/15. Stage limits remain visible through the shaded bands.
xticks = [
    PLOT_START,
    md_to_doy(7, 6),
    md_to_doy(8, 8),
    md_to_doy(9, 12),
    PLOT_END,
]
ax.set_xticks(xticks)
ax.set_xticks([md_to_doy(6, 15), md_to_doy(7, 7), md_to_doy(8, 9), md_to_doy(9, 13), md_to_doy(10, 11)], minor=True)
ax.set_xticklabels(
    [(pd.Timestamp('2001-01-01') + pd.Timedelta(days=d-1)).strftime('%-m/%-d') for d in xticks],
    fontsize=7.8,
    rotation=0,
    ha='center',
    fontproperties=zh_prop,
)
ax.tick_params(axis='x', which='minor', length=2.2, width=0.45, color='#777777')
ax.tick_params(axis='y', labelsize=8.0)
ax.set_xlabel('夏玉米生长季时间', fontsize=9.0, fontproperties=zh_prop)
ax.set_ylabel('Kcact', fontsize=9.0, fontproperties=zh_prop)
ax.grid(axis='y', color='#D9D9D9', lw=0.45, alpha=0.75)
ax.grid(axis='x', visible=False)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2,
                fontsize=6.0, handlelength=1.7, columnspacing=1.2)
for t in leg.get_texts():
    t.set_fontproperties(zh_prop)

# 按论文图件规范：图题和说明文字放在正文图注中，不放在图内。
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(OUT_FIG, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
fig.savefig(OUT_SVG, bbox_inches='tight')
print(OUT_FIG)
print(OUT_PDF)
print(OUT_SVG)
print(OUT_CSV)
print(summary[['doy_bin','date_label','pml_kc_mean','station_kc_mean','gap_pml_minus_station','n_station_bins','n_obs']].to_string(index=False))
