#!/usr/bin/env python3
"""
Station Kcact line charts — Nature-style with Songti SC (宋体) typography.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 宋体 + Nature rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Songti SC", "STSong", "SimSun", "Times New Roman"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "lines.linewidth": 0.8,
    "axes.grid": False,
    "axes.unicode_minus": False,
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "station_etc_with_et0.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"禹城": "#E64B35", "位山": "#4DBBD5", "馆陶": "#00A087", "栾城": "#3C5488"}

# Load
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["doy"] = df["date"].dt.dayofyear
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def doy_smooth(sdf: pd.DataFrame, col: str = "kcact", step: int = 4,
               half_win: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DOY-binned mean ± std for a single station's column."""
    bins = np.arange(1, 366, step)
    means, stds = [], []
    for d in bins:
        win = sdf[(sdf["doy"] >= d) & (sdf["doy"] < d + 2 * half_win)][col].dropna()
        means.append(win.mean() if len(win) >= 3 else np.nan)
        stds.append(win.std() if len(win) >= 3 else np.nan)
    x = bins + half_win
    return x, np.array(means), np.array(stds)


def add_crop_zones(ax):
    """Light background shading for NCP crop seasons."""
    ax.axvspan(1, 60, alpha=0.05, color="blue", lw=0)
    ax.axvspan(60, 155, alpha=0.05, color="green", lw=0)
    ax.axvspan(155, 175, alpha=0.05, color="orange", lw=0)
    ax.axvspan(175, 275, alpha=0.05, color="green", lw=0)
    ax.axvspan(275, 365, alpha=0.05, color="blue", lw=0)

    annotations = [
        (30, "越冬期"), (107, "返青-成熟"), (165, "收获"), (225, "夏玉米"), (320, "出苗-越冬"),
    ]
    for doy, txt in annotations:
        ax.text(doy, 1.65, txt, fontsize=5.5, ha="center", va="top",
                color="#888888", fontfamily="Songti SC")


# ===========================================================================
# Figure 1 — 四站季节Kcact折线图 (主图, 4 panels)
# ===========================================================================
def fig1_seasonal_four_panel():
    stations = ["禹城", "位山", "馆陶", "栾城"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    ((ax0, ax1), (ax2, ax3)) = axes
    ax_map = dict(zip(stations, [ax0, ax1, ax2, ax3]))
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for stn, ax, lbl in zip(stations, [ax0, ax1, ax2, ax3], labels):
        sdf = df[df["station"] == stn]

        # Individual year traces (faint)
        for yr, grp in sdf.groupby("year"):
            grp_sorted = grp.sort_values("doy")
            ax.plot(grp_sorted["doy"], grp_sorted["kcact"],
                    color=COLORS[stn], alpha=0.12, lw=0.4, zorder=0)

        # Mean ± std
        x, mean, std = doy_smooth(sdf)
        ax.plot(x, mean, color=COLORS[stn], lw=1.4, zorder=2)
        ax.fill_between(x, np.maximum(mean - std, 0), mean + std,
                        color=COLORS[stn], alpha=0.15, lw=0, zorder=1)

        add_crop_zones(ax)

        ax.set_title(stn, fontsize=10, fontweight="bold", color=COLORS[stn],
                     fontfamily="Songti SC")
        ax.set_xlim(0, 365)
        ax.set_ylim(0, 1.8)
        ax.set_ylabel("作物系数 Kcact", fontfamily="Songti SC")
        ax.set_xlabel("日序 DOY", fontfamily="Songti SC")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(60))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(15))
        ax.text(0.02, 0.95, lbl, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")
        ax.text(0.98, 0.95, f"n={len(sdf)}", transform=ax.transAxes,
                fontsize=6, ha="right", va="top", color="grey")

    fig.tight_layout(pad=0.8, w_pad=1.2, h_pad=1.2)
    out = FIG_DIR / "fig1_kcact_seasonal_four_panel.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 2 — 四站平均季节Kcact叠加 + 月均值折线
# ===========================================================================
def fig2_seasonal_comparison():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) DOY smoothed curves overlaid
    for stn in ["禹城", "位山", "馆陶", "栾城"]:
        x, mean, _ = doy_smooth(df[df["station"] == stn])
        ax_a.plot(x, mean, color=COLORS[stn], label=stn, lw=1.2)
    add_crop_zones(ax_a)
    ax_a.set_xlim(0, 365)
    ax_a.set_ylim(0, 1.4)
    ax_a.set_xlabel("日序 DOY")
    ax_a.set_ylabel("作物系数 Kcact")
    ax_a.legend(frameon=False, loc="upper right", ncol=2,
                handlelength=1.0, prop={"family": "Songti SC", "size": 7})
    ax_a.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax_a.xaxis.set_minor_locator(mticker.MultipleLocator(15))
    ax_a.text(0.02, 0.95, "(a)", transform=ax_a.transAxes, fontsize=10,
              fontweight="bold", va="top")

    # (b) Monthly mean Kcact line — each station one line
    months = np.arange(1, 13)
    month_label_pos = months
    for stn in ["禹城", "位山", "馆陶", "栾城"]:
        sdf = df[df["station"] == stn]
        mon_means = [sdf[sdf["month"] == m]["kcact"].mean() for m in months]
        ax_b.plot(month_label_pos, mon_means, "o-", color=COLORS[stn],
                  label=stn, lw=1.0, ms=3, markerfacecolor="white",
                  markeredgewidth=0.8)
    ax_b.set_xlabel("月份")
    ax_b.set_ylabel("作物系数 Kcact")
    ax_b.set_xticks(months)
    ax_b.set_xticklabels(["1月","2月","3月","4月","5月","6月",
                          "7月","8月","9月","10月","11月","12月"],
                         fontfamily="Songti SC", fontsize=6)
    ax_b.set_xlim(0.5, 12.5)
    ax_b.set_ylim(0, 1.4)
    ax_b.legend(frameon=False, loc="upper right", ncol=2,
                handlelength=1.0, prop={"family": "Songti SC", "size": 7})
    ax_b.text(0.02, 0.95, "(b)", transform=ax_b.transAxes, fontsize=10,
              fontweight="bold", va="top")

    fig.tight_layout(pad=0.8, w_pad=1.2)
    out = FIG_DIR / "fig2_kcact_seasonal_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 3 — 时序折线: 每站一条全时段Kcact
# ===========================================================================
def fig3_timeseries():
    stations = ["禹城", "位山", "馆陶", "栾城"]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.5), sharex=True)

    for si, (stn, ax) in enumerate(zip(stations, axes)):
        sdf = df[df["station"] == stn].sort_values("date")

        # Kcact line
        ax.plot(sdf["date"], sdf["kcact"], color=COLORS[stn], lw=0.5, alpha=0.85)

        # Lowess-like smooth via rolling median
        roll = sdf.set_index("date")["kcact"].rolling(5, center=True, min_periods=3).median()
        ax.plot(roll.index, roll.values, color="#222222", lw=0.9, alpha=0.7)

        ax.set_ylabel("Kcact", fontsize=7)
        ax.set_ylim(0, 2.2)
        ax.text(0.01, 0.92, stn, transform=ax.transAxes, fontsize=9,
                fontweight="bold", color=COLORS[stn], fontfamily="Songti SC")
        ax.text(0.99, 0.92, f"n={len(sdf)}", transform=ax.transAxes,
                fontsize=6, ha="right", va="top", color="grey")
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))

    axes[-1].set_xlabel("日期 Date")
    fig.tight_layout(pad=0.5, h_pad=0.3)
    out = FIG_DIR / "fig3_kcact_timeseries.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 4 — ET0 + ETc 双变量季节折线 (代表性站点)
# ===========================================================================
def fig4_et0_etc_seasonal():
    """ET0 / ETc seasonal mean curves — clean side-by-side comparison, no twin axes."""
    stations = ["禹城", "位山", "馆陶", "栾城"]
    fig, (ax_et0, ax_etc) = plt.subplots(2, 1, figsize=(7.2, 5.5))

    for stn in stations:
        sdf = df[df["station"] == stn]

        # ---- ET0 ----
        x, et0_mean, et0_std = doy_smooth(sdf, "et0_pm_mean_mm_d")
        ax_et0.plot(x, et0_mean, color=COLORS[stn], lw=1.2, label=stn)
        ax_et0.fill_between(x,
                            np.maximum(et0_mean - et0_std, 0),
                            et0_mean + et0_std,
                            color=COLORS[stn], alpha=0.10, lw=0)

        # ---- ETc ----
        x, etc_mean, etc_std = doy_smooth(sdf, "etc_obs_mm_d")
        ax_etc.plot(x, etc_mean, color=COLORS[stn], lw=1.2, label=stn)
        ax_etc.fill_between(x,
                            np.maximum(etc_mean - etc_std, 0),
                            etc_mean + etc_std,
                            color=COLORS[stn], alpha=0.10, lw=0)

    # ---- Styling: ET0 panel ----
    add_crop_zones(ax_et0)
    ax_et0.set_xlim(0, 365)
    ax_et0.set_ylim(bottom=0)
    ax_et0.set_ylabel("参考蒸散发 ET0 (mm/d)", fontfamily="Songti SC")
    ax_et0.legend(frameon=False, loc="upper right", ncol=4,
                  handlelength=1.0, prop={"size": 7.5})
    ax_et0.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax_et0.xaxis.set_minor_locator(mticker.MultipleLocator(15))
    ax_et0.text(0.01, 0.95, "(a)", transform=ax_et0.transAxes,
                fontsize=10, fontweight="bold", va="top")

    # ---- Styling: ETc panel ----
    add_crop_zones(ax_etc)
    ax_etc.set_xlim(0, 365)
    ax_etc.set_ylim(bottom=0)
    ax_etc.set_xlabel("日序 DOY")
    ax_etc.set_ylabel("实际蒸散发 ETc (mm/d)", fontfamily="Songti SC")
    ax_etc.legend(frameon=False, loc="upper right", ncol=4,
                  handlelength=1.0, prop={"size": 7.5})
    ax_etc.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax_etc.xaxis.set_minor_locator(mticker.MultipleLocator(15))
    ax_etc.text(0.01, 0.95, "(b)", transform=ax_etc.transAxes,
                fontsize=10, fontweight="bold", va="top")

    fig.tight_layout(pad=0.8, h_pad=0.6)
    out = FIG_DIR / "fig4_et0_etc_seasonal.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 5 — 禹城单站详图: DOY × Year 热力图 + 季节曲线
# ===========================================================================
def fig5_yucheng_detailed():
    """Detailed view for Yucheng (longest record)."""
    sdf = df[df["station"] == "禹城"]
    fig = plt.figure(figsize=(7.2, 4.0))

    # (a) DOY × Year heatmap of Kcact
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])

    years = sorted(sdf["year"].unique())
    doy_bins = np.arange(1, 366, 8)

    # Build matrix
    matrix = np.full((len(years), len(doy_bins)), np.nan)
    for yi, yr in enumerate(years):
        yr_df = sdf[sdf["year"] == yr]
        for di, d in enumerate(doy_bins):
            row = yr_df[(yr_df["doy"] >= d) & (yr_df["doy"] < d + 16)]
            if len(row) >= 1:
                matrix[yi, di] = row["kcact"].mean()

    im = ax.pcolormesh(doy_bins, np.arange(len(years)), matrix,
                       cmap="YlOrRd", shading="auto", vmin=0, vmax=1.5,
                       rasterized=True)
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=6)
    ax.set_xlim(0, 365)
    ax.set_xlabel("日序 DOY")
    ax.set_ylabel("年份")
    ax.set_title("禹城 Kcact 逐日-逐年分布", fontsize=9, fontweight="bold",
                 fontfamily="Songti SC", color=COLORS["禹城"])
    cbar = fig.colorbar(im, ax=ax, pad=0.01, aspect=25, shrink=0.9)
    cbar.set_label("Kcact", fontsize=7)

    # (b) Multi-year DOY lines (thin colored by year)
    ax2 = fig.add_subplot(gs[0, 1])
    yr_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(years)))
    for yi, yr in enumerate(years):
        yr_df = sdf[sdf["year"] == yr].sort_values("doy")
        ax2.plot(yr_df["doy"], yr_df["kcact"], color=yr_colors[yi],
                 lw=0.5, alpha=0.7, label=str(yr))
    # Bold mean
    x, mean, _ = doy_smooth(sdf)
    ax2.plot(x, mean, color="black", lw=1.5)
    ax2.set_xlim(0, 365)
    ax2.set_ylim(0, 1.8)
    ax2.set_xlabel("日序 DOY")
    ax2.set_ylabel("Kcact")
    ax2.set_title("逐年折线 + 均值", fontsize=9, fontfamily="Songti SC")
    ax2.legend(frameon=False, fontsize=5, ncol=2, handlelength=0.8,
               loc="upper right")

    fig.tight_layout(pad=0.8)
    out = FIG_DIR / "fig5_yucheng_detail.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 6 — 夏玉米 Kcact 季节折线 (四站并排, DOY 150–300)
# ===========================================================================
def fig6_maize_seasonal():
    stations = ["禹城", "位山", "馆陶", "栾城"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for stn, ax, lbl in zip(stations, axes.flat, labels):
        sdf = df[df["station"] == stn]
        maize = sdf[(sdf["doy"] >= 150) & (sdf["doy"] <= 300)]

        # Individual year traces
        for yr, grp in maize.groupby("year"):
            grp_sorted = grp.sort_values("doy")
            ax.plot(grp_sorted["doy"], grp_sorted["kcact"],
                    color=COLORS[stn], alpha=0.15, lw=0.4, zorder=0)

        # Mean ± std
        x, mean, std = doy_smooth(maize, "kcact", step=2, half_win=8)
        ax.plot(x, mean, color=COLORS[stn], lw=1.4, zorder=2)
        ax.fill_between(x, np.maximum(mean - std, 0), mean + std,
                        color=COLORS[stn], alpha=0.15, lw=0, zorder=1)

        # Maize phenology zones
        ax.axvspan(150, 170, alpha=0.06, color="#8B7355", lw=0)   # emergence
        ax.axvspan(170, 210, alpha=0.06, color="#66CC66", lw=0)   # vegetative
        ax.axvspan(210, 240, alpha=0.06, color="#FF9933", lw=0)   # peak/reproductive
        ax.axvspan(240, 270, alpha=0.06, color="#FFDD55", lw=0)   # grain fill
        ax.axvspan(270, 300, alpha=0.06, color="#8B7355", lw=0)   # maturity

        # Phenology labels
        pheno = [(160, "出苗"), (190, "营养生长"), (225, "抽雄吐丝"), (255, "灌浆"), (285, "成熟")]
        for doy, txt in pheno:
            ax.text(doy, 1.35, txt, fontsize=5, ha="center", color="#888888",
                    fontfamily="Songti SC")

        ax.set_title(stn, fontsize=10, fontweight="bold", color=COLORS[stn],
                     fontfamily="Songti SC")
        ax.set_xlim(150, 300)
        ax.set_ylim(0, 1.5)
        ax.set_ylabel("作物系数 Kcact", fontfamily="Songti SC")
        ax.set_xlabel("日序 DOY")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.text(0.02, 0.95, lbl, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")
        n_maize = len(maize)
        ax.text(0.98, 0.95, f"n={n_maize}", transform=ax.transAxes,
                fontsize=6, ha="right", va="top", color="grey")

    fig.tight_layout(pad=0.8, w_pad=1.2, h_pad=1.2)
    out = FIG_DIR / "fig6_maize_kcact_seasonal.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 7 — 夏玉米四站叠加对比 + ET0/ETc/Kcact 三线图
# ===========================================================================
def fig7_maize_comparison():
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # (a) Four-station mean Kcact overlay, DOY 150-300
    ax_a = fig.add_subplot(gs[0, 0])
    for stn in ["禹城", "位山", "馆陶", "栾城"]:
        maize = df[(df["station"] == stn) & (df["doy"] >= 150) & (df["doy"] <= 300)]
        x, mean, std = doy_smooth(maize, "kcact", step=2, half_win=8)
        ax_a.plot(x, mean, color=COLORS[stn], lw=1.2, label=stn)
        ax_a.fill_between(x, np.maximum(mean - std, 0), mean + std,
                          color=COLORS[stn], alpha=0.08, lw=0)

    # Phenology zones
    ax_a.axvspan(150, 170, alpha=0.05, color="#8B7355", lw=0)
    ax_a.axvspan(170, 210, alpha=0.05, color="#66CC66", lw=0)
    ax_a.axvspan(210, 240, alpha=0.05, color="#FF9933", lw=0)
    ax_a.axvspan(240, 270, alpha=0.05, color="#FFDD55", lw=0)
    ax_a.axvspan(270, 300, alpha=0.05, color="#8B7355", lw=0)

    ax_a.set_xlim(150, 300)
    ax_a.set_ylim(0, 1.2)
    ax_a.set_xlabel("日序 DOY")
    ax_a.set_ylabel("作物系数 Kcact", fontfamily="Songti SC")
    ax_a.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.0,
                prop={"size": 7.5})
    ax_a.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax_a.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax_a.text(0.02, 0.95, "(a) 四站 Kcact 季节曲线", transform=ax_a.transAxes,
              fontsize=8, fontweight="bold", va="top", fontfamily="Songti SC")

    # (b) Monthly boxplot, maize months (6-10月)
    ax_b = fig.add_subplot(gs[0, 1])
    maize_months = [6, 7, 8, 9, 10]
    positions = list(range(len(maize_months)))
    bp_data = [df[(df["month"] == m)]["kcact"].dropna().values for m in maize_months]
    bp = ax_b.boxplot(bp_data, positions=positions, widths=0.55,
                      patch_artist=True,
                      boxprops={"linewidth": 0.5, "facecolor": "#EEEEEE"},
                      whiskerprops={"linewidth": 0.5},
                      capprops={"linewidth": 0.5},
                      medianprops={"linewidth": 0.8, "color": "#333333"},
                      flierprops={"markersize": 2, "markerfacecolor": "#AAAAAA",
                                  "linewidth": 0.3, "alpha": 0.5})

    # Overlay per-station mean lines
    for stn in ["禹城", "位山", "馆陶", "栾城"]:
        sdf = df[df["station"] == stn]
        mon_means = [sdf[sdf["month"] == m]["kcact"].mean() for m in maize_months]
        ax_b.plot(positions, mon_means, "o-", color=COLORS[stn], lw=1.0, ms=4,
                  markerfacecolor="white", markeredgewidth=0.8, label=stn)

    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(["6月", "7月", "8月", "9月", "10月"],
                         fontfamily="Songti SC", fontsize=7)
    ax_b.set_ylabel("作物系数 Kcact", fontfamily="Songti SC")
    ax_b.set_ylim(0, 1.5)
    ax_b.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.0,
                prop={"size": 7})
    ax_b.text(0.02, 0.95, "(b) 夏玉米季月均 Kcact", transform=ax_b.transAxes,
              fontsize=8, fontweight="bold", va="top", fontfamily="Songti SC")
    # n= above each box
    for i, vals in enumerate(bp_data):
        ax_b.text(i, 1.42, f"n={len(vals)}", fontsize=5.5, ha="center", color="grey")

    out = FIG_DIR / "fig7_maize_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 8 — 夏玉米 ET0 + ETc 双变量季节线 (禹城 & 位山)
# ===========================================================================
def fig8_maize_et0_etc():
    """Clean 2-panel: Yucheng + Weishan ET0/ETc during maize season."""
    stns = ["禹城", "位山"]
    fig, (ax_yc, ax_ws) = plt.subplots(2, 1, figsize=(7.2, 5.0))

    for stn, ax in zip(stns, [ax_yc, ax_ws]):
        sdf = df[(df["station"] == stn) & (df["doy"] >= 150) & (df["doy"] <= 300)]

        # ET0
        x, et0_m, et0_s = doy_smooth(sdf, "et0_pm_mean_mm_d", step=2, half_win=8)
        ax.plot(x, et0_m, color="#777777", lw=1.2, label="ET0")
        ax.fill_between(x, np.maximum(et0_m - et0_s, 0), et0_m + et0_s,
                        color="#777777", alpha=0.10, lw=0)

        # ETc
        x, etc_m, etc_s = doy_smooth(sdf, "etc_obs_mm_d", step=2, half_win=8)
        ax.plot(x, etc_m, color=COLORS[stn], lw=1.4, label="ETc")
        ax.fill_between(x, np.maximum(etc_m - etc_s, 0), etc_m + etc_s,
                        color=COLORS[stn], alpha=0.10, lw=0)

        # Phenology zones
        for start, end, c in [(150, 170, "#8B7355"), (170, 210, "#66CC66"),
                                (210, 240, "#FF9933"), (240, 270, "#FFDD55"),
                                (270, 300, "#8B7355")]:
            ax.axvspan(start, end, alpha=0.05, color=c, lw=0)

        ax.set_xlim(150, 300)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("蒸散发 (mm/d)", fontfamily="Songti SC")
        ax.legend(frameon=False, loc="upper right", handlelength=1.0,
                  prop={"size": 8})
        ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.set_title(stn, fontsize=10, fontweight="bold", color=COLORS[stn],
                     fontfamily="Songti SC", loc="left")
        ax.text(0.98, 0.95, f"n={len(sdf)}", transform=ax.transAxes,
                fontsize=7, ha="right", va="top", color="grey")

    ax_ws.set_xlabel("日序 DOY")
    ax_yc.text(0.01, 0.95, "(a)", transform=ax_yc.transAxes, fontsize=10,
               fontweight="bold", va="top")
    ax_ws.text(0.01, 0.95, "(b)", transform=ax_ws.transAxes, fontsize=10,
               fontweight="bold", va="top")

    fig.tight_layout(pad=0.8, h_pad=0.8)
    out = FIG_DIR / "fig8_maize_et0_etc.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating Songti-SC Kcact line charts ...")
    fig1_seasonal_four_panel()
    fig2_seasonal_comparison()
    fig3_timeseries()
    fig4_et0_etc_seasonal()
    fig5_yucheng_detailed()
    fig6_maize_seasonal()
    fig7_maize_comparison()
    fig8_maize_et0_etc()
    print(f"\nDone → {FIG_DIR}")


if __name__ == "__main__":
    main()
