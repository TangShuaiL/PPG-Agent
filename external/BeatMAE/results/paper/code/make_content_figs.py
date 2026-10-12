#!/usr/bin/env python3
"""论文三张内容图的绘图代码（定稿版）。

出三张：
    fig_distribution.{png,pdf}      数据分布（SBP/DBP 同轴 + 年龄人级）
    fig_scatter_vital_row.{png,pdf} 预测散点（vital 一行：SBP/DBP 段级、Age 人级）
    fig_history_gain.{png,pdf}      历史累积 ΔMAE

2026-09-21 从作者交付的 1244 行版本**按调用闭包裁出**：原版还含若干方案对照卡
（`_scatter_options` 等）、误差脊线 `fig_history_error`、数据统计表 `write_stats_table`
与 6 张独立散点 `fig_scatter`，都不在论文里，已裁掉。裁法用 AST 算闭包（含常量读取），
并做过健全性检查（无保留项引用被删名字）。

**图 1（框架图）不在这里** —— 那张是作者手绘的，见 `../figs/fig1_framework.pdf`。

数据源只有 `../figure_data/*.csv.gz` 与 `../{history/history.csv,analysis/dataset_stats.csv}`；
样式见 `_paper_style.py`（配色经 `_cvd.py` 校验：三色两两 ΔE 最差 17.8，门槛 15）。

    python make_content_figs.py              # 三张全出
    python make_content_figs.py --only dist  # 只出分布
"""
import argparse
import csv
import gzip
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import _paper_style as S                                    # noqa: E402


HERE = Path(__file__).resolve().parent


ROOT = HERE.parent                       # …/results/paper


FD = ROOT / 'figure_data'


TB_HIST = ROOT / 'history' / 'history.csv'


OUT = ROOT / 'figs'          # 原为 figs/mainfigs_<日期>/；收归单点后去掉日期层


TARGETS = ['sbp', 'dbp', 'age']


CORPORA_MAIN = ['mimic', 'vital', 'aurora']


UNIT = {'sbp': 'mmHg', 'dbp': 'mmHg', 'age': 'years'}


INK2 = '#5B6469'          # 参考线/次要文字


EDGE = '#AEB6BB'


_M1 = np.array([[0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]])


_M2 = np.linalg.inv(_M1)


_W = np.array([0.95047, 1.0, 1.08883])


def _lab_f(t):
    return np.where(t > (6 / 29) ** 3, np.cbrt(t),
                    t / (3 * (6 / 29) ** 2) + 4 / 29)


def _lab_fi(t):
    return np.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))


def hex2lab(h):
    rgb = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    x, y, z = _lab_f(_M1 @ lin / _W)
    return np.array([116 * y - 16, 500 * (x - y), 200 * (y - z)])


def lab2hex(lab):
    L, a, b = lab
    fy = (L + 16) / 116
    xyz = _lab_fi(np.array([fy + a / 500, fy, fy - b / 200])) * _W
    lin = np.clip(_M2 @ xyz, 0, 1)
    rgb = np.where(lin <= 0.0031308, lin * 12.92,
                   1.055 * lin ** (1 / 2.4) - 0.055)
    return '#%02X%02X%02X' % tuple(int(round(255 * v)) for v in np.clip(rgb, 0, 1))


def hue_ramp(base, n=6, L_light=93.5, fC=0.45, gc=0.55):
    """目标色的色相保持不变，只把 L* 从 L_light 拉到原色、C* 从 fC·C0 拉到 C0。

    fC/gc 控制彩度衰减：C = C0·(fC + (1-fC)·t^gc)。gc<1 ⇒ 中间调仍带彩度，不发灰。
    """
    L0, a0, b0 = hex2lab(base)
    C0, h0 = np.hypot(a0, b0), np.arctan2(b0, a0)
    out = []
    for k in range(n):
        t = k / (n - 1)
        L = L_light + (L0 - L_light) * t
        C = C0 * (fC + (1 - fC) * t ** gc)
        out.append(lab2hex((L, C * np.cos(h0), C * np.sin(h0))))
    return out


SEQ = {t: hue_ramp(S.TARGET[t]) for t in TARGETS}


def _read(path, opener, cols=None):
    with opener(path, 'rt', encoding='utf-8', newline='') as f:
        r = csv.reader(f)
        hdr = next(r)
        cols = cols or hdr
        idx = {c: hdr.index(c) for c in cols}
        out = {c: [] for c in cols}
        for row in r:
            for c in out:
                out[c].append(row[idx[c]])
    return out


def load_gz(p, cols=None):
    return _read(p, gzip.open, cols)


def bp_hist():
    d = load_gz(FD / 'dist_bp.csv.gz')
    bl = np.asarray(d['bin_left'], float)
    return bl, np.asarray(d['sbp_count'], np.int64), np.asarray(d['dbp_count'], np.int64)


def ages():
    d = load_gz(FD / 'dist_age.csv.gz')
    return (np.asarray(d['age'], float), np.asarray(d['source']),
            np.asarray(d['subject_id']))


def seg_frames():
    """逐段 y_true/y_pred + src/sid（下游评测池 = 报告域测试折 stream 池）。"""
    out = {}
    for t in TARGETS:
        d = load_gz(FD / f'{t}.csv.gz')
        out[t] = dict(src=np.asarray(d['src']), sid=np.asarray(d['sid']),
                      x=np.asarray(d['y_true'], float),
                      y=np.asarray(d['y_pred'], float))
    return out


def history_rows():
    return list(csv.DictReader(open(TB_HIST, encoding='utf-8')))


def kde(v, grid, bw):
    """高斯 KDE（不依赖 scipy）。"""
    d = (grid[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(axis=1) / (v.size * bw * np.sqrt(2 * np.pi))


def gsmooth(y, sigma):
    """一维高斯平滑（仅用于**显示**：计数型直方图有泊松散粒噪声）。"""
    if sigma <= 0:
        return y
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, r, mode='edge'), k, mode='same')[r:-r]


def seq_cmap(target):
    return LinearSegmentedColormap.from_list(f'{target}_seq', SEQ[target])


def tidy(ax, keep=('left', 'bottom'), color=EDGE):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color(color)
            ax.spines[s].set_linewidth(0.7)


def halo(w=2.6):
    return [pe.withStroke(linewidth=w, foreground='white')]


def panels(bp=True, age=True, figsize=None, wspace=0.34):
    n = int(bp) + int(age)
    if figsize is None:
        figsize = (7.30, 2.80) if n == 2 else (3.62, 2.72)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = list(axes[0])
    if n == 2:
        fig.subplots_adjust(left=0.064, right=0.982, bottom=0.212, top=0.900,
                            wspace=wspace)
    else:
        fig.subplots_adjust(left=0.122, right=0.978, bottom=0.200, top=0.925)
    return fig, axes


def _draw_bp(ax, frames, tag='(a)'):
    """血压分布：两条曲线**各自归一到自身总量**，纵轴读占比 ⇒ 分布形状可直接对比。

    为什么不用堆叠：这张图要回答的是"评测池的分布与全库是否一致"。归一到各自总量后，
    两条曲线重合即说明一致（评测池实际只占全库 4.1%，堆叠时那一层只有 4% 高，
    形状根本看不出来）。绝对数量交给 `table_data_stats`。
    """
    bl, cs, cd = bp_hist()
    cent = bl + 0.5
    pool = {'sbp': frames['sbp']['x'], 'dbp': frames['dbp']['x']}
    edges = np.arange(30, 258, 1)

    curves, peak = {}, 0.0
    for tgt, cnt in (('sbp', cs), ('dbp', cd)):
        corp = gsmooth(cnt, 1.3)
        corp = corp / corp.sum()                   # 归一化：曲线下面积 = 1
        h, _ = np.histogram(pool[tgt], bins=edges)
        pl = gsmooth(h, 1.3)
        pl = pl / pl.sum()
        curves[tgt] = (corp, pl)
        peak = max(peak, float(corp.max()), float(pl.max()))

    for tgt in ('dbp', 'sbp'):
        hue = S.TARGET[tgt]
        corp, pl = curves[tgt]
        ax.fill_between(cent, 0, corp * 100, color=hue, alpha=0.26, lw=0, zorder=3)
        ax.plot(cent, corp * 100, color=hue, lw=1.40, zorder=6)
        ax.plot(cent, pl * 100, color=hue, lw=1.10, ls=(0, (4.5, 2.4)), zorder=7)

    yhi = peak * 100 * 1.30
    ax.set_xlim(30, 206)
    ax.set_ylim(0, yhi)
    ax.set_xlabel('Blood pressure (mmHg)')
    ax.set_ylabel('Share of segments per 1 mmHg bin  (%)')
    ax.set_xticks([40, 60, 80, 100, 120, 140, 160, 180, 200])
    ax.set_yticks(np.arange(0, np.floor(yhi) + 0.5, 1.0))

    handles = [Patch(fc=S.TARGET['sbp'], alpha=0.30, ec=S.TARGET['sbp'], lw=0.9,
                     label='SBP  ·  corpus'),
               Line2D([], [], color=S.TARGET['sbp'], lw=1.10, ls=(0, (4.5, 2.4)),
                      label='SBP  ·  evaluation pool'),
               Patch(fc=S.TARGET['dbp'], alpha=0.30, ec=S.TARGET['dbp'], lw=0.9,
                     label='DBP  ·  corpus'),
               Line2D([], [], color=S.TARGET['dbp'], lw=1.10, ls=(0, (4.5, 2.4)),
                      label='DBP  ·  evaluation pool')]
    ax.legend(handles=handles, loc='upper right', ncol=1, fontsize=6.1,
              handlelength=1.55, borderaxespad=0.26,
              labelspacing=0.30, handletextpad=0.5)
    # 均值±SD（启用口径 = 训练池+测试池，剔除 mimic.test）：
    # 出自 paper/tables/pretrain_pools.csv（2026-09-17，make_data_tables.py 现算）
    ax.text(0.030, 0.955, 'SBP 121.1$\\pm$22.0 mmHg\nDBP 62.1$\\pm$13.0 mmHg',
            transform=ax.transAxes, ha='left', va='top', fontsize=6.2,
            color=S.INK, linespacing=1.45, path_effects=halo(2.4))
    S.panel_tag(ax, tag, dy=1.045)
    tidy(ax)
    S.grid_y(ax, alpha=0.5)


def _draw_age(ax, ages_all, age_src, age_sid, frames, tag='(b)'):
    """年龄分布（人级）：同样两条曲线各自归一化，纵轴读占比。"""
    hue = S.TARGET['age']
    keep = np.isin(age_src, CORPORA_MAIN)         # ⚠ 去 iwshi_bp 域
    corp = ages_all[keep]
    # 评测池受试者：由散点 sid 反查（vital 555 + aurora 238 = 793）
    pool_mask = np.zeros(ages_all.size, bool)
    for dom in ('vital', 'aurora'):
        ids = np.unique(frames['age']['sid'][frames['age']['src'] == dom])
        pool_mask |= (age_src == dom) & np.isin(age_sid, ids)
    pool = ages_all[pool_mask]

    grid = np.linspace(18.0, 92.0, 500)
    kc = kde(corp, grid, 2.5) * 100                # KDE 已归一化 ⇒ 直接是 %/年
    kp = kde(pool, grid, 2.5) * 100
    peak = max(float(kc.max()), float(kp.max()))

    ax.fill_between(grid, 0, kc, color=hue, alpha=0.26, lw=0, zorder=3)
    ax.plot(grid, kc, color=hue, lw=1.40, zorder=6)
    ax.plot(grid, kp, color=hue, lw=1.10, ls=(0, (4.5, 2.4)), zorder=7)

    yhi = peak * 1.30
    ax.set_xlim(16.2, 93.8)
    ax.set_ylim(0, yhi)
    ax.set_yticks(np.arange(0, np.floor(yhi) + 0.5, 1.0))
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Share of subjects per 1-year bin  (%)')
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80, 90])
    handles = [Patch(fc=hue, alpha=0.30, ec=hue, lw=0.9, label='corpus'),
               Line2D([], [], color=hue, lw=1.10, ls=(0, (4.5, 2.4)),
                      label='evaluation pool')]
    ax.legend(handles=handles, loc='upper right', fontsize=6.1,
              handlelength=1.55, borderaxespad=0.26, labelspacing=0.30,
              handletextpad=0.5)
    # 均值±SD（启用口径 = 训练池+测试池，剔除 mimic.test）：
    # 出自 paper/tables/pretrain_pools.csv（2026-09-17，make_data_tables.py 现算）
    ax.text(0.030, 0.955, '57.4$\\pm$15.7 yr',
            transform=ax.transAxes, ha='left', va='top', fontsize=6.2,
            color=S.INK, path_effects=halo(2.4))
    S.panel_tag(ax, tag, dy=1.045)
    tidy(ax)
    S.grid_y(ax, alpha=0.5)


def fig_distribution(frames):
    ages_all, age_src, age_sid = ages()
    S.apply(7.5)

    fig, ax = panels(bp=True, age=False)
    _draw_bp(ax[0], frames)
    S.save(fig, OUT, 'fig_dist_bp')

    fig, ax = panels(bp=False, age=True)
    _draw_age(ax[0], ages_all, age_src, age_sid, frames)
    S.save(fig, OUT, 'fig_dist_age')

    fig, ax = panels(bp=True, age=True)
    _draw_bp(ax[0], frames, '(a)')
    _draw_age(ax[1], ages_all, age_src, age_sid, frames, '(b)')
    S.save(fig, OUT, 'fig_distribution')


def _smooth2d(H, sigma=1.0):
    """二维高斯平滑：消除"密度查表"在核心区留下的方格感。"""
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        H = np.apply_along_axis(
            lambda m: np.convolve(np.pad(m, r, mode='edge'), k, 'same')[r:-r],
            axis, H)
    return H


def _cloud(ax, x, y, tgt, lo, hi, nbin=None, ms=None):
    """密度着色点云（稀疏点浅、核心深）。"""
    n = x.size
    nb = nbin or (200 if n > 50000 else (120 if n > 3000 else 60))
    H, xe, ye = np.histogram2d(x, y, bins=nb, range=[[lo, hi], [lo, hi]])
    H = _smooth2d(H, 1.0)
    ix = np.clip(np.digitize(x, xe) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, nb - 1)
    dens = H[ix, iy]
    order = np.argsort(dens, kind='stable')
    vmin = max(float(np.percentile(dens, 1.0)), 1e-12)
    vmax = max(float(np.percentile(dens, 99.4)), vmin * 10.0)
    ms = ms or (1.7 if n > 50000 else (6.0 if n > 3000 else 16.0))
    ax.scatter(x[order], y[order], c=dens[order], cmap=seq_cmap(tgt),
               norm=LogNorm(vmin=vmin, vmax=vmax), s=ms, linewidths=0,
               alpha=0.88, rasterized=True, zorder=2)


REF_COL = '#7E898F'      # 参考分布：中性灰（不与目标色争夺注意力）


MARG_BW = 1.45           # 显示带宽倍率：Silverman 偏细，乘一点更顺眼


_MARG_N = 360            # KDE 求值网格点数


def _silverman(v):
    """Silverman 正态参考带宽：0.9·min(σ, IQR/1.34)·n^(-1/5)。"""
    v = np.asarray(v, float)
    n = v.size
    if n < 2:
        return 1.0
    sd = float(v.std(ddof=1))
    q1, q3 = np.percentile(v, [25, 75])
    iqr = float(q3 - q1)
    a = min(sd, iqr / 1.34) if iqr > 0 else sd
    if not np.isfinite(a) or a <= 0:
        a = sd if sd > 0 else 1.0
    return float(0.9 * a * n ** (-0.2))


def _kde_curve(v, grid, bw, chunk=12000):
    """定带宽高斯 KDE。分块累加 —— 一次建 (360 × 188620) 会吃掉约 0.5 GB。"""
    v = np.asarray(v, float)
    out = np.zeros(grid.size, float)
    for i in range(0, v.size, chunk):
        d = (grid[:, None] - v[i:i + chunk][None, :]) / bw
        out += np.exp(-0.5 * d * d).sum(axis=1)
    return out / (v.size * bw * np.sqrt(2.0 * np.pi))


def _marg_curve(v, lo, hi, grid):
    """返回 (density, bandwidth)。密度**不归一化** —— 参考/估计共用同一密度
    尺度，峰高差（窄 ⇒ 高）才是能读出来的信息。"""
    bw = _silverman(v) * MARG_BW
    return _kde_curve(v, grid, bw), bw


def _marg_plot(ax, grid, dens, color, kind, fill=0.0, ls='-', lw=1.05, z=3):
    """把一条密度曲线画到边缘面板上（kind='top' 沿 x / 'right' 沿 y）。"""
    if fill > 0:
        if kind == 'top':
            ax.fill_between(grid, 0, dens, color=color, alpha=fill, lw=0, zorder=z)
        else:
            ax.fill_betweenx(grid, 0, dens, color=color, alpha=fill, lw=0, zorder=z)
    if kind == 'top':
        ax.plot(grid, dens, color=color, ls=ls, lw=lw, zorder=z + 3)
    else:
        ax.plot(dens, grid, color=color, ls=ls, lw=lw, zorder=z + 3)
    return float(dens.max())


def _marg_strip(ax):
    """边缘面板去轴：只留形状。"""
    ax.tick_params(bottom=False, left=False, labelbottom=False,
                   labelleft=False, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_scatter_vital_row(frames):
    """vital 三联横排版（单栏图）：SBP / DBP / Age 并排，真实与估计分布都放上方。

    2026-09-17 用户定：散点只放 vital（aurora 撤下省空间）；边缘分布用 overlay
    方案（参考灰虚线 + 估计色填充，同轴叠在上方），不要右侧条。
    """
    S.apply(7.5)
    fig = plt.figure(figsize=(3.55, 1.98))
    outer = fig.add_gridspec(2, 3, height_ratios=[0.21, 1],
                             left=0.088, right=0.988, bottom=0.215, top=0.808,
                             wspace=0.30, hspace=0.07)
    for ci, tgt in enumerate(TARGETS):
        fr = frames[tgt]
        m = fr['src'] == 'vital'
        if tgt == 'age':                      # 人级：逐段预测先按人取均值
            ids = np.unique(fr['sid'][m])
            keep = np.isin(fr['sid'], ids)
            x = np.array([fr['x'][keep & (fr['sid'] == s)].mean() for s in ids])
            y = np.array([fr['y'][keep & (fr['sid'] == s)].mean() for s in ids])
        else:
            x, y = fr['x'][m], fr['y'][m]

        lo = min(np.percentile(x, 0.2), np.percentile(y, 0.2))
        hi = max(np.percentile(x, 99.8), np.percentile(y, 99.8))
        pad = (hi - lo) * 0.10
        lo, hi = lo - pad, hi + pad
        grid = np.linspace(lo, hi, _MARG_N)
        dx, _ = _marg_curve(x, lo, hi, grid)
        dy, _ = _marg_curve(y, lo, hi, grid)
        gmax = max(dx.max(), dy.max()) * 1.18

        ax = fig.add_subplot(outer[1, ci])
        axt = fig.add_subplot(outer[0, ci], sharex=ax)
        hue = S.TARGET[tgt]

        _cloud(ax, x, y, tgt, lo, hi, ms=0.9 if x.size > 50000 else 4.5)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls=(0, (5, 3)), zorder=6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        u = UNIT[tgt]
        ax.set_xlabel(f'Ref. {S.TARGET_LABEL[tgt]} ({u})', fontsize=6.0,
                      labelpad=1.5)
        if ci == 0:
            ax.set_ylabel(f'Est. ({u})', fontsize=6.0, labelpad=1.5)
        ax.tick_params(labelsize=5.2, length=1.6, pad=1.0)
        mae = float(np.abs(x - y).mean())
        r = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.955, 0.055, f'MAE {mae:.2f}\n$r$ {r:.2f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=5.4,
                color='black', linespacing=1.4, path_effects=halo(2.2))

        _marg_plot(axt, grid, dx, REF_COL, 'top', fill=0.18, ls=(0, (3.2, 2.0)),
                   lw=0.9)
        _marg_plot(axt, grid, dy, hue, 'top', fill=0.28, lw=0.9)
        axt.set_ylim(0, gmax)
        _marg_strip(axt)
        tidy(ax)

    handles = [Line2D([], [], color=REF_COL, ls=(0, (3.2, 2.0)), lw=1.0,
                      label='reference'),
               Patch(fc=S.TARGET['sbp'], alpha=0.35, ec=S.TARGET['sbp'], lw=0.8,
                     label='est. SBP'),
               Patch(fc=S.TARGET['dbp'], alpha=0.35, ec=S.TARGET['dbp'], lw=0.8,
                     label='est. DBP'),
               Patch(fc=S.TARGET['age'], alpha=0.35, ec=S.TARGET['age'], lw=0.8,
                     label='est. Age'),
               Line2D([], [], color=INK2, ls=(0, (5, 3)), lw=0.9, label='$y=x$')]
    # 横排图例压底部，留出与 x 轴标签的呼吸间距
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=5.4,
               frameon=False, handlelength=1.35, columnspacing=0.75,
               labelspacing=0.25, handletextpad=0.42,
               bbox_to_anchor=(0.5, 0.010))
    # 点云已在 _cloud 里 rasterized=True：PDF 文字/线条保持矢量（放大不糊），
    # 只有点云按 600 dpi 栅格化；PNG 整张位图给到 1000 dpi。
    png = OUT / 'fig_scatter_vital_row.png'
    pdf = OUT / 'fig_scatter_vital_row.pdf'
    fig.savefig(png, dpi=1000)
    fig.savefig(pdf, dpi=600)
    print(f'  → {png}')
    print(f'  → {pdf}')
    print('     (vital row: SBP/DBP segment-level, Age subject-level)')


def k_pos(k_lo, k_hi):
    """桶在 **log2 k 轴**上的位置 = 1-based k 区间的几何中点。

    k=1 → 1；k2–4 → √(2·4)=2.83；k5–16 → √(5·16)=8.94 …… 每个桶的宽度本来
    就是"翻倍"，等距摆放才是对的（原来按序号等距会把 log 尺度画成线性）。
    开区间桶（k65+）按一个八度封顶，只影响摆放位置，不影响数字。
    """
    lo = max(int(k_lo) + 1, 1)
    hi = int(k_hi)
    if hi > lo * 128:
        hi = lo * 2
    return float(np.sqrt(lo * max(hi, lo)))


def fig_history_gain():
    """ΔMAE = seg-MLP − causal-attn 随历史长度 k 的走势（x 轴为 log2 k）。

    09-15 15:30「**保持原样，只提美感**」精修 —— **结构、口径、数字一个没动**
    （仍是绝对 ΔMAE，不做相对增益；跨目标排序与自己正文的表述保持一致）：
      · x 轴刻度原来是 `k=1\\nn = 551` 两行**同样黑度**，笨重。改成桶名走刻度、
        下面的 n 用**独立灰色小字**（`get_xaxis_transform()` 手写，不占刻度标签）。
      · 收紧右侧空白（1.85 → 1.62）：原来末端数字后面留了近半格空地。
      · y 轴改整数量级刻度，上端多留一点（Age 末值 3.98 原来几乎贴着顶框）。
      · 网格更淡、线宽与标记微调、图例间距舒展。
    """
    S.apply(7.5)
    rows = history_rows()
    doms = ['vital', 'aurora']
    ylo = min(float(r['gain_MAE']) for r in rows)
    yhi = max(float(r['gain_MAE']) for r in rows)
    span = yhi - ylo

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.62))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.305, top=0.870,
                        wspace=0.145)
    for ax, dom in zip(axes, doms):
        sub = [r for r in rows if r['domain'] == dom]
        buckets = list(dict.fromkeys(r['bucket'] for r in sub))
        first = {r['bucket']: r for r in sub}          # 每桶任取一行拿 k_lo/k_hi
        xs = np.array([k_pos(int(first[b]['k_lo']), int(first[b]['k_hi']))
                       for b in buckets])
        vals, ns = {}, {}
        for tgt in TARGETS:
            rs = {r['bucket']: r for r in sub if r['target'] == tgt}
            vals[tgt] = np.array([float(rs[b]['gain_MAE']) for b in buckets])
            ns[tgt] = [int(rs[b]['n']) for b in buckets]

        ax.axhline(0, color='#B2B9BD', lw=0.75, ls=(0, (4, 2.6)), zorder=2)
        for tgt in TARGETS:
            ax.plot(xs, vals[tgt], color=S.TARGET[tgt], lw=1.55, zorder=4,
                    ms=4.0, mec='white', mew=0.8, **S.TARGET_STYLE[tgt])
            ax.annotate(f'{vals[tgt][-1]:.2f}', (xs[-1], vals[tgt][-1]),
                        xytext=(5.5, 0), textcoords='offset points',
                        va='center', ha='left', fontsize=6.5,
                        color=S.TARGET[tgt], path_effects=halo(2.6))

        ax.set_xscale('log', base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(buckets, fontsize=6.3)
        ax.minorticks_off()
        ax.set_xlim(xs[0] * 0.74, xs[-1] * 1.62)
        ax.set_ylim(ylo - span * 0.075, yhi + span * 0.135)
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # 桶名之下的评测段数：灰色小字，与黑色桶名分层（原来一样黑、一样大）
        for x, b, i in zip(xs, buckets, range(len(buckets))):
            ax.text(x, -0.115, f'$n$ = {ns["sbp"][i]:,}',
                    transform=ax.get_xaxis_transform(), ha='center', va='top',
                    fontsize=5.2, color=INK2)

        ax.set_xlabel('History length $k$ (segment rank within subject)')
        ax.xaxis.set_label_coords(0.5, -0.240)
        ax.set_ylabel(r'$\Delta$MAE (mmHg / yr)')
        S.panel_tag(ax, f'({"ab"[doms.index(dom)]}) {dom} domain', dy=1.05)
        tidy(ax)
        S.grid_y(ax, alpha=0.42)

    handles = [Line2D([], [], color=S.TARGET[t], lw=1.55,
                      label=S.TARGET_LABEL[t], **S.TARGET_STYLE[t])
               for t in TARGETS]
    axes[0].legend(handles=handles, loc='upper left', fontsize=6.6,
                   handlelength=2.4, borderaxespad=0.30, labelspacing=0.42)
    return S.save(fig, OUT, 'fig_history_gain')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='all', choices=['all', 'dist', 'scatter', 'history'])
    a = ap.parse_args()
    if a.only in ('all', 'dist', 'scatter'):
        frames = seg_frames()
    if a.only in ('all', 'dist'):
        fig_distribution(frames)
    if a.only in ('all', 'scatter'):
        fig_scatter_vital_row(frames)
    if a.only in ('all', 'history'):
        fig_history_gain()


if __name__ == '__main__':
    main()
