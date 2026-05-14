from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.font_manager import FontProperties

ROOT = Path('/Users/hert/Projects/dcsdxx')
infile = ROOT/'data/processed/train/ncp_summer_maize_selected_indicators_pml_era5grid.parquet'
out_png = ROOT/'outputs/figures/maize_pml_kcact_growth_cycle_nature_cn.png'
out_pdf = ROOT/'outputs/figures/maize_pml_kcact_growth_cycle_nature_cn.pdf'
out_csv = ROOT/'outputs/tables/maize_pml_kcact_growth_cycle_summary.csv'
out_png.parent.mkdir(parents=True, exist_ok=True)
out_csv.parent.mkdir(parents=True, exist_ok=True)

font_paths = [
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/System/Library/Fonts/Supplemental/Songti.ttc',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/Library/Fonts/Arial Unicode.ttf',
]
zh_prop = None
for fp in font_paths:
    p = Path(fp)
    if p.exists():
        font_manager.fontManager.addfont(str(p))
        zh_prop = FontProperties(fname=str(p))
        rcParams['font.family'] = zh_prop.get_name()
        break
rcParams['axes.unicode_minus'] = False
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Nature-like restrained style
plt.style.use('default')
rcParams.update({
    'figure.dpi': 180,
    'savefig.dpi': 320,
    'axes.linewidth': 0.9,
    'axes.edgecolor': '#222222',
    'xtick.color': '#222222',
    'ytick.color': '#222222',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'legend.frameon': False,
})

df = pd.read_parquet(infile)
# 只保留华北平原夏玉米主要生长季；313 以后/极晚熟窗口样本少且 ET0 很低，容易放大 Kcact 尾端噪声。
df = df[(df['doy'] >= 160) & (df['doy'] <= 305)].copy()
df = df[(df['kcact_pml'] >= 0.02) & (df['kcact_pml'] <= 1.60)]
# 保险：聚合表中每行是一个 0.1°格网×8日窗口，n_points 表示该格内原始点数量。
df['w'] = df.get('n_points', pd.Series(1, index=df.index)).fillna(1).clip(lower=1)
if 'cell_id' not in df.columns:
    df['cell_id'] = df.get('era5_cell', df.get('grid_patch_id', pd.Series(range(len(df)), index=df.index)))
if 'season_year' not in df.columns:
    df['season_year'] = df.get('year', pd.to_datetime(df['date']).dt.year)

def wmean(g, col):
    return np.average(g[col], weights=g['w'])

def wstd(g, col):
    x = g[col].to_numpy(float)
    w = g['w'].to_numpy(float)
    mu = np.average(x, weights=w)
    return np.sqrt(np.average((x - mu) ** 2, weights=w))

rows = []
for doy, g in df.groupby('doy'):
    n_cells = len(g)
    rows.append({
        'doy': int(doy),
        'date_label': (pd.Timestamp('2020-01-01') + pd.Timedelta(days=int(doy)-1)).strftime('%m-%d'),
        'kcact_mean': wmean(g, 'kcact_pml'),
        'kcact_std': wstd(g, 'kcact_pml'),
        'kcact_q25': g['kcact_pml'].quantile(0.25),
        'kcact_q75': g['kcact_pml'].quantile(0.75),
        'n_cells': n_cells,
        'n_points_sum': int(g['w'].sum()),
        'pml_eta_mean_mm_d': wmean(g, 'pml_eta_crop_mm_d') if 'pml_eta_crop_mm_d' in g else np.nan,
        'et0_mean_mm_d': wmean(g, 'et0_pm_daily_mm') if 'et0_pm_daily_mm' in g else np.nan,
    })
sumdf = pd.DataFrame(rows).sort_values('doy')
sumdf['kcact_se'] = sumdf['kcact_std'] / np.sqrt(sumdf['n_cells'].clip(lower=1))
sumdf['ci_low'] = sumdf['kcact_mean'] - 1.96 * sumdf['kcact_se']
sumdf['ci_high'] = sumdf['kcact_mean'] + 1.96 * sumdf['kcact_se']
sumdf.to_csv(out_csv, index=False)

yearly = []
for (year, doy), g in df.groupby(['season_year', 'doy']):
    yearly.append({'season_year': int(year), 'doy': int(doy), 'kcact_mean': wmean(g, 'kcact_pml'), 'n_cells': len(g)})
ydf = pd.DataFrame(yearly)

fig, ax = plt.subplots(figsize=(7.8, 4.7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

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
    ax.text((start+end)/2, 1.48, label, ha='center', va='center', fontsize=8.5, color='#555555', fontproperties=zh_prop)

# 各年份浅灰线，主趋势深蓝线
for year, g in ydf.groupby('season_year'):
    g = g.sort_values('doy')
    ax.plot(g['doy'], g['kcact_mean'], color='#B9B9B9', lw=0.85, alpha=0.45, zorder=1)

x = sumdf['doy'].to_numpy(dtype=float)
y = sumdf['kcact_mean'].to_numpy(dtype=float)
lo = sumdf['ci_low'].to_numpy(dtype=float)
hi = sumdf['ci_high'].to_numpy(dtype=float)
ax.fill_between(x, lo, hi, color='#3B6EA8', alpha=0.16, lw=0, zorder=2, label='95% 置信区间')
ax.plot(x, y, color='#1F4E79', lw=2.25, zorder=3, label='多年加权均值')
ax.scatter(x, y, s=16, color='#1F4E79', edgecolor='white', linewidth=0.35, zorder=4)

peak = sumdf.loc[sumdf['kcact_mean'].idxmax()]
ax.annotate(f"峰值 {peak['kcact_mean']:.2f}\nDOY {int(peak['doy'])}",
            xy=(peak['doy'], peak['kcact_mean']), xytext=(peak['doy']-34, peak['kcact_mean']+0.13),
            arrowprops=dict(arrowstyle='-', color='#333333', lw=0.8),
            fontsize=8.5, color='#222222', ha='left', fontproperties=zh_prop)

ax.set_xlim(158, 307)
ax.set_ylim(0.0, 1.55)
ax.set_xticks([161, 177, 193, 209, 225, 241, 257, 273, 289, 305])
ax.set_xticklabels(['6/9','6/25','7/11','7/27','8/12','8/28','9/13','9/29','10/15','10/31'], fontproperties=zh_prop)
ax.set_ylabel('PML 反演 Kcact（ETa / ET0）', fontsize=10.5, fontproperties=zh_prop)
ax.set_xlabel('夏玉米生长季时间', fontsize=10.5, fontproperties=zh_prop)
ax.set_title('华北平原夏玉米 Kcact 生长季变化\nPML-V2.2a ETa 替代 MOD16 ETa，2019–2024',
             fontsize=12, pad=10, fontproperties=zh_prop)
ax.grid(axis='y', color='#D9D9D9', lw=0.55, alpha=0.7)
ax.grid(axis='x', visible=False)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.985), fontsize=8.5, handlelength=2.2)
for text in leg.get_texts():
    text.set_fontproperties(zh_prop)

caption = f"数据：0.1°统一尺度格网×8日窗口；严格阈值过滤后 n={len(df):,}，格网数={df['cell_id'].nunique():,}。灰线为单年份，蓝线为按格内样本数加权的多年均值。"
fig.text(0.02, 0.012, caption, fontsize=7.6, color='#555555', fontproperties=zh_prop)
fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.savefig(out_png, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(out_png)
print(out_pdf)
print(out_csv)
print(sumdf[['doy','date_label','kcact_mean','n_cells','n_points_sum']].head().to_string(index=False))
print(sumdf[['doy','date_label','kcact_mean','n_cells','n_points_sum']].tail().to_string(index=False))
