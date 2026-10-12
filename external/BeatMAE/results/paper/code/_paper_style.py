#!/usr/bin/env python3
"""论文数据图统一风格。

配色不是"挑好看的"，而是两条约束同时满足：
  ① **与主结构图同源** —— 色相全部取自框架图的调色板家族
     （drawio 绿/蓝/紫 + 自定义深绿），全篇一套调子；
  ② **可辨性经量化校验** —— 见 `_cvd.py`：四色觉条件（normal/protan/deutan/tritan）
     下三色两两 ΔE 最差 **17.8**（门槛 15），各色对白底对比度最低 **3.02:1**（门槛 3:1）。
     随时可复验：`python results/paper/code/_cvd.py`（退出码 0 = 全部 PASS）。

## 颜色通道

**目标**（SBP / DBP / Age）是唯一的系列配色通道：

| 目标 | 色 | 线型 + 标记（冗余编码，色觉障碍下仍可追踪，WCAG 1.4.1） |
|---|---|---|
| SBP | `#197040` 深绿 | 实线 + 圆点 |
| DBP | `#529BCF` 蓝 | 虚线 + 方块 |
| Age | `#6B3A93` 紫 | 点线 + 三角 |

正/负增益这类单色语义由各图自行处理，不在本模块固化。
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 目标三色（经 _cvd.py 校验：min ΔE 17.8 / 对比度 3.02:1）────────────────
TARGET = {'sbp': '#197040', 'dbp': '#529BCF', 'age': '#6B3A93'}
TARGET_LABEL = {'sbp': 'SBP', 'dbp': 'DBP', 'age': 'Age'}
# 冗余编码：颜色之外再给线型与标记，色觉障碍下仍可追踪（WCAG 1.4.1）
TARGET_STYLE = {'sbp': dict(ls='-',  marker='o'),
                'dbp': dict(ls='--', marker='s'),
                'age': dict(ls=':',  marker='^')}

# ── 墨色与网格（供 apply() 与各图直接取用）────────────────────────────────
INK = '#1A1A1A'
GRID = '#D6DBDF'

FONT = 'Times New Roman'


def apply(fs=7.5):
    """全局 rcParams。fs = 正文基准字号（pt），论文双栏图 7–8 合适。"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [FONT, 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': fs,
        'axes.titlesize': fs + 1.0,
        'axes.labelsize': fs,
        'xtick.labelsize': fs - 0.5,
        'ytick.labelsize': fs - 0.5,
        'legend.fontsize': fs - 0.5,
        'axes.edgecolor': '#8C9499',
        'axes.linewidth': 0.6,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
        'grid.color': GRID, 'grid.linewidth': 0.5,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.bbox': 'tight',
        'legend.frameon': False,
        'figure.dpi': 110,
        'lines.linewidth': 1.5,
        'lines.markersize': 3.4,
        'lines.markeredgewidth': 0.7,
        'pdf.fonttype': 42,      # Type 42 = 可编辑嵌入（IEEE Xplore 要求）
        'ps.fonttype': 42,
    })


def grid_y(ax, alpha=0.55):
    ax.grid(axis='y', alpha=alpha, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_tag(ax, text, dx=0.012, dy=0.985):
    """左上角 (a)(b)(c) 面板标号 —— 论文里引用子图用。"""
    ax.text(dx, dy, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=plt.rcParams['font.size'] + 0.5, fontweight='bold', color=INK)


def save(fig, out_dir, name, also_pdf=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f'{name}.png'
    fig.savefig(png, dpi=300)
    outs = [png]
    if also_pdf:
        pdf = out_dir / f'{name}.pdf'
        fig.savefig(pdf)
        outs.append(pdf)
    plt.close(fig)
    for o in outs:
        print(f'  → {o}')
    return outs
