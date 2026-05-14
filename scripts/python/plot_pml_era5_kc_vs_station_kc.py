from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.font_manager import FontProperties
from scipy import stats

ROOT = Path('/Users/hert/Projects/dcsdxx')
XLS = Path('/Users/hert/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_uwvqbkxg35qp22_e6f5/temp/RWTemp/2026-05/7b290b31f4c2dab912ccf3497d9146fe/KsKc+ET0(1).xlsx')
PML_MATCH = ROOT/'outputs/tables/pml_v22a_vs_tower_mod16.csv'
OUT_FIG = ROOT/'outputs/figures/pml_era5_kcact_vs_station_weather_kcact.png'
OUT_PDF = ROOT/'outputs/figures/pml_era5_kcact_vs_station_weather_kcact.pdf'
OUT_CSV = ROOT/'outputs/tables/pml_era5_kcact_vs_station_weather_kcact.csv'
OUT_STATS = ROOT/'outputs/tables/pml_era5_kcact_vs_station_weather_kcact_stats.csv'
for p in [OUT_FIG.parent, OUT_CSV.parent]: p.mkdir(parents=True, exist_ok=True)

# Chinese font
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

# Parse station workbook. Header row is row 2 (0-based header=1), one sheet per station.
frames = []
for sheet in pd.ExcelFile(XLS).sheet_names:
    df = pd.read_excel(XLS, sheet_name=sheet, header=1)
    era_col = 'ERA5ET0' if 'ERA5ET0' in df.columns else ('ER5ET0' if 'ER5ET0' in df.columns else None)
    if era_col is None:
        continue
    sub = pd.DataFrame({
        'station': sheet,
        'date': pd.to_datetime(df['日期\nDate'], errors='coerce'),
        'station_eta_mm_d': pd.to_numeric(df['实际蒸散发\nETc (mm/d)'], errors='coerce'),
        'era5_et0_mm_d': pd.to_numeric(df[era_col], errors='coerce'),
        'met_station_et0_mm_d': pd.to_numeric(df['气象站ET0'], errors='coerce'),
        'window_days': pd.to_numeric(df['窗口天数'], errors='coerce'),
    })
    frames.append(sub)
station = pd.concat(frames, ignore_index=True)
station['date'] = station['date'].dt.strftime('%Y-%m-%d')

pml = pd.read_csv(PML_MATCH, parse_dates=['date', 'date_prev'])
pml['date'] = pml['date'].dt.strftime('%Y-%m-%d')
keep = pml[['station','date','date_prev','pml_eta_mm_d','n_pml_overlap','pml_overlap_days']].copy()

df = station.merge(keep, on=['station','date'], how='inner')
# Requested two Kcact definitions:
df['kcact_station_met'] = df['station_eta_mm_d'] / df['met_station_et0_mm_d']
df['kcact_pml_era5'] = df['pml_eta_mm_d'] / df['era5_et0_mm_d']
# Valid 8-day observations and physically usable ET0. Keep a wide Kc range for station comparison but drop pathological tiny ET0/outliers.
valid = (
    df[['station_eta_mm_d','met_station_et0_mm_d','era5_et0_mm_d','pml_eta_mm_d','kcact_station_met','kcact_pml_era5']]
      .replace([np.inf,-np.inf], np.nan).notna().all(axis=1)
    & (df['window_days'] == 8)
    & (df['met_station_et0_mm_d'] > 0.1)
    & (df['era5_et0_mm_d'] > 0.1)
    & (df['station_eta_mm_d'] >= 0)
    & (df['pml_eta_mm_d'] >= 0)
    & (df['kcact_station_met'].between(0, 3.0))
    & (df['kcact_pml_era5'].between(0, 3.0))
)
df = df.loc[valid].copy()
df.to_csv(OUT_CSV, index=False)

def calc_stats(d):
    x=d['kcact_station_met'].to_numpy(float); y=d['kcact_pml_era5'].to_numpy(float)
    n=len(d)
    if n<3:
        return dict(n=n, pearson_r=np.nan, r2=np.nan, slope=np.nan, intercept=np.nan, rmse=np.nan, mae=np.nan, bias=np.nan)
    lr=stats.linregress(x,y)
    pred=lr.slope*x+lr.intercept
    return dict(n=n, pearson_r=lr.rvalue, r2=lr.rvalue**2, slope=lr.slope, intercept=lr.intercept,
                rmse=float(np.sqrt(np.mean((y-x)**2))), mae=float(np.mean(np.abs(y-x))), bias=float(np.mean(y-x)),
                fit_rmse=float(np.sqrt(np.mean((y-pred)**2))))
rows=[{'station':'全部', **calc_stats(df)}]
for st,g in df.groupby('station'):
    rows.append({'station':st, **calc_stats(g)})
stats_df=pd.DataFrame(rows)
stats_df.to_csv(OUT_STATS,index=False)
all_stats=stats_df.iloc[0]

# Nature-like scatter
plt.style.use('default')
rcParams.update({
    'figure.dpi': 180, 'savefig.dpi': 320, 'axes.linewidth': 0.9,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'legend.frameon': False,
})
colors={'禹城':'#1f77b4','位山':'#d62728','馆陶':'#2ca02c','栾城':'#9467bd'}
fig, ax=plt.subplots(figsize=(6.3,5.4))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
for st,g in df.groupby('station'):
    ax.scatter(g['kcact_station_met'], g['kcact_pml_era5'], s=24, alpha=0.72,
               color=colors.get(st), label=f'{st} (n={len(g)})', edgecolor='white', linewidth=0.35)

maxv=float(np.nanpercentile(pd.concat([df['kcact_station_met'],df['kcact_pml_era5']]),99))
lim=max(1.6, min(2.4, maxv*1.08))
xx=np.linspace(0,lim,100)
ax.plot(xx, xx, color='#333333', lw=1.0, ls='--', alpha=0.75, label='1:1线')
# regression line
lr=stats.linregress(df['kcact_station_met'], df['kcact_pml_era5'])
ax.plot(xx, lr.slope*xx+lr.intercept, color='#1F4E79', lw=2.0, label='线性拟合')
ax.set_xlim(0,lim); ax.set_ylim(0,lim)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, color='#D9D9D9', lw=0.55, alpha=0.7)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
ax.set_xlabel('站点 Kcact = 实测 ETa / 气象站 ET0', fontsize=10.5, fontproperties=zh_prop)
ax.set_ylabel('PML Kcact = PML ETa / ERA5 ET0', fontsize=10.5, fontproperties=zh_prop)
ax.set_title('PML/ERA5 Kcact 与站点实测 Kcact 的相关性', fontsize=12, pad=10, fontproperties=zh_prop)
text=(f"n = {int(all_stats.n)}\n"
      f"R² = {all_stats.r2:.3f}\n"
      f"r = {all_stats.pearson_r:.3f}\n"
      f"y = {all_stats.slope:.2f}x + {all_stats.intercept:.2f}\n"
      f"Bias = {all_stats.bias:+.3f}")
ax.text(0.035,0.965,text,transform=ax.transAxes,ha='left',va='top',fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, boxstyle='round,pad=0.32'), fontproperties=zh_prop)
leg=ax.legend(loc='lower right', fontsize=8.5, handletextpad=0.4)
for t in leg.get_texts(): t.set_fontproperties(zh_prop)
caption='注：PML 使用 V2.2a 站点像元 8日 ETa；ERA5 ET0 与气象站 ET0 取自补充表，同一站点同一观测窗口对齐。'
fig.text(0.02,0.012,caption,fontsize=7.4,color='#555555',fontproperties=zh_prop)
fig.tight_layout(rect=[0,0.04,1,1])
fig.savefig(OUT_FIG,bbox_inches='tight')
fig.savefig(OUT_PDF,bbox_inches='tight')
print(OUT_FIG)
print(OUT_PDF)
print(OUT_CSV)
print(OUT_STATS)
print(stats_df.to_string(index=False))
