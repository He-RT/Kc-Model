from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.font_manager import FontProperties
from scipy import stats

ROOT = Path('/Users/hert/Projects/dcsdxx')
# 优先用微信补充表；若临时路径失效，则用已整理过的对齐表。
XLS_CANDIDATES = [
    Path('/Users/hert/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_uwvqbkxg35qp22_e6f5/temp/RWTemp/2026-05/7b290b31f4c2dab912ccf3497d9146fe/KsKc+ET0(1).xlsx'),
    Path('/Users/hert/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_uwvqbkxg35qp22_e6f5/temp/drag/KsKc+ET0(1).xlsx'),
]
FALLBACK_CSV = ROOT / 'outputs/tables/pml_era5_kcact_vs_station_weather_kcact.csv'
OUT_FIG = ROOT / 'outputs/figures/station_era5_vs_met_et0_linear.png'
OUT_PDF = ROOT / 'outputs/figures/station_era5_vs_met_et0_linear.pdf'
OUT_CSV = ROOT / 'outputs/tables/station_era5_vs_met_et0_linear.csv'
OUT_STATS = ROOT / 'outputs/tables/station_era5_vs_met_et0_linear_stats.csv'
OUT_SCRIPT = ROOT / 'scripts/python/plot_station_era5_vs_met_et0.py'
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_SCRIPT.parent.mkdir(parents=True, exist_ok=True)

font_paths = ['/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/STHeiti Light.ttc','/System/Library/Fonts/Supplemental/Songti.ttc']
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

def load_from_xlsx(path: Path) -> pd.DataFrame:
    frames = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=1)
        era_col = 'ERA5ET0' if 'ERA5ET0' in df.columns else ('ER5ET0' if 'ER5ET0' in df.columns else None)
        if era_col is None or '气象站ET0' not in df.columns:
            continue
        sub = pd.DataFrame({
            'station': sheet,
            'date': pd.to_datetime(df['日期\nDate'], errors='coerce'),
            'window_days': pd.to_numeric(df['窗口天数'], errors='coerce'),
            'era5_et0_mm_d': pd.to_numeric(df[era_col], errors='coerce'),
            'met_station_et0_mm_d': pd.to_numeric(df['气象站ET0'], errors='coerce'),
        })
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)

xlsx = next((p for p in XLS_CANDIDATES if p.exists()), None)
if xlsx is not None:
    df = load_from_xlsx(xlsx)
else:
    df = pd.read_csv(FALLBACK_CSV, parse_dates=['date'])
    df = df[['station','date','window_days','era5_et0_mm_d','met_station_et0_mm_d']].copy()

valid = (
    df[['era5_et0_mm_d','met_station_et0_mm_d']].replace([np.inf,-np.inf], np.nan).notna().all(axis=1)
    & (df['window_days'] == 8)
    & df['era5_et0_mm_d'].between(0.1, 12.0)
    & df['met_station_et0_mm_d'].between(0.1, 12.0)
)
df = df.loc[valid].copy().sort_values(['station','date'])
df.to_csv(OUT_CSV, index=False)

def calc(d):
    x = d['met_station_et0_mm_d'].to_numpy(float)
    y = d['era5_et0_mm_d'].to_numpy(float)
    n = len(d)
    if n < 3:
        return dict(n=n, pearson_r=np.nan, r2=np.nan, slope=np.nan, intercept=np.nan, rmse=np.nan, mae=np.nan, bias=np.nan, ratio=np.nan, fit_rmse=np.nan)
    lr = stats.linregress(x, y)
    pred = lr.slope*x + lr.intercept
    return dict(
        n=n,
        pearson_r=float(lr.rvalue),
        r2=float(lr.rvalue**2),
        slope=float(lr.slope),
        intercept=float(lr.intercept),
        rmse=float(np.sqrt(np.mean((y-x)**2))),
        mae=float(np.mean(np.abs(y-x))),
        bias=float(np.mean(y-x)),
        ratio=float(np.mean(y)/np.mean(x)),
        fit_rmse=float(np.sqrt(np.mean((y-pred)**2))),
    )

rows = [{'station':'全部', **calc(df)}]
for st,g in df.groupby('station'):
    rows.append({'station':st, **calc(g)})
stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUT_STATS, index=False)
all_s = stats_df.iloc[0]

plt.style.use('default')
rcParams.update({'figure.dpi':180,'savefig.dpi':320,'axes.linewidth':0.9,'legend.frameon':False})
colors = {'禹城':'#1f77b4','位山':'#d62728','馆陶':'#2ca02c','栾城':'#9467bd'}
fig, ax = plt.subplots(figsize=(6.35,5.35))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
for st,g in df.groupby('station'):
    ax.scatter(g['met_station_et0_mm_d'], g['era5_et0_mm_d'], s=24, alpha=0.70,
               color=colors.get(st), label=f'{st} (n={len(g)})', edgecolor='white', linewidth=0.35)

lim = max(7.5, min(10.5, float(np.nanpercentile(pd.concat([df['met_station_et0_mm_d'], df['era5_et0_mm_d']]), 99))*1.08))
xx = np.linspace(0, lim, 100)
ax.plot(xx, xx, color='#333333', lw=1.0, ls='--', alpha=0.75, label='1:1线')
lr = stats.linregress(df['met_station_et0_mm_d'], df['era5_et0_mm_d'])
ax.plot(xx, lr.slope*xx + lr.intercept, color='#1F4E79', lw=2.0, label='线性拟合')
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal', adjustable='box')
ax.grid(True, color='#D9D9D9', lw=0.55, alpha=0.7)
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
ax.set_xlabel('气象站 ET0 (mm/d)', fontsize=10.5, fontproperties=zh_prop)
ax.set_ylabel('ERA5 ET0 (mm/d)', fontsize=10.5, fontproperties=zh_prop)
ax.set_title('站点 ERA5 ET0 与气象站 ET0 的相关性', fontsize=12, pad=10, fontproperties=zh_prop)
text = (f"n = {int(all_s.n)}\n"
        f"R² = {all_s.r2:.3f}\n"
        f"r = {all_s.pearson_r:.3f}\n"
        f"y = {all_s.slope:.2f}x + {all_s.intercept:.2f}\n"
        f"Bias = {all_s.bias:+.3f} mm/d\n"
        f"ERA5/站点均值 = {all_s.ratio:.2f}")
ax.text(0.035,0.965,text,transform=ax.transAxes,ha='left',va='top',fontsize=8.8,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.84, boxstyle='round,pad=0.32'), fontproperties=zh_prop)
leg = ax.legend(loc='lower right', fontsize=8.5, handletextpad=0.4)
for t in leg.get_texts():
    t.set_fontproperties(zh_prop)
caption = '注：仅保留8日窗口且ET0在0.1–12 mm/d范围内的站点记录。'
fig.text(0.02,0.012,caption,fontsize=7.4,color='#555555',fontproperties=zh_prop)
fig.tight_layout(rect=[0,0.04,1,1])
fig.savefig(OUT_FIG, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
print(OUT_FIG)
print(OUT_PDF)
print(OUT_CSV)
print(OUT_STATS)
print(stats_df.to_string(index=False))
