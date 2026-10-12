#!/usr/bin/env python3
"""把绘图所需的**逐段预测**导出成精简文件 → results/paper/figure_data/。

为什么需要
    `results/paper/` 只有汇总（MAE/r），**画不出散点**——散点要逐段的 y_true/y_pred。
    而原始 preds 目录 223 MB（含 5 seed × 3 域 × 3 变体），其中散点只用 11.5%。
    故导出**只含所需部分**的精简版，让 `paper/` 能独立重画图。

导出内容（口径与论文表一致）
    seed = **s4**（best seed）
    域   = **报告域** vital + aurora（不含 mimic）
    列   = src, sid, y_true, y_pred
    文件 = results/paper/figure_data/<target>.csv.gz（三个目标各一份）

    另出**分布图**的两份（`--skip-dist` 可跳过）：
      dist_bp.csv.gz    SBP/DBP 的**1 mmHg 细粒度计数**（四库汇总，含 mimic）
      dist_age.csv.gz   Age 的**人级中位数**（逐 source×subject，6,081 行）

    BP 为什么存计数而不是逐段原值：有标签段 480 万，存原值 gz 后 **75.9 MB**
    （散点全套才 4.6 MB），而分布图本身就只需要直方图。故存**1 mmHg** 的细粒度
    计数——图要用 3 mmHg 时把整数计数相加即可，**完全精确**，且只有 ~2 KB。
    Age 逐段原值不存则是因为**图只用得到人级**（每人一个值），存人级即无损。

溯源
    **源出处的唯一权威仍是 `results/probe/<exp>/full/stream/preds/`**（散点部分）
    与语料 `corpus_v2c/{manifest,labels}.parquet`（分布部分）；
    本目录是**派生副本**，只为让 paper/ 可独立重画。数值不一致时以源为准，
    并重跑本脚本。

用法：python tools/export_figure_data.py [--out results/paper/figure_data]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.paths import CORPUS_V2C, resolve_corpus

P = ROOT / 'results' / 'probe'
EXP = 'beatmae_v25c3/v2c_3src_mva'
SEED = 4
DOMS = ['vital', 'aurora']
TGT = ['sbp', 'dbp', 'age']

# 分布图导出用：细粒度 bin。**网格从画图脚本导入**——导出域必须与图的分箱域
# 逐位一致，否则重采样对不齐（这是必须成立的不变量，不能靠两处常量各自维护）。
from tools._calibers import BINS_BP, FINE_STEP      # noqa: E402  口径独立成模块

DIST_BIN = FINE_STEP
DIST_LO, DIST_HI = float(BINS_BP[0]), float(BINS_BP[-1])   # 半开区间 [lo, hi)


def export_dist(out):
    """分布图数据 → dist_bp.csv.gz（细粒度计数）+ dist_age.csv.gz（人级中位数）。

    ✅ 只存计数即可**精确**复现 `hist(..., density=True)`：
      实测 numpy 的 density 是按**界内计数**归一化的（`n/db/n.sum()`），
      不是按输入总数——越界极值被丢弃后**不进分母**。
      故重采样时 density = count / (count.sum() * bin_width)，与绘图逐位一致。
      （上面的"越界"只作信息打印，不参与归一化。）
    """
    R = Path(resolve_corpus(CORPUS_V2C, tag='corpus_v2c'))
    lab = pd.read_parquet(R / 'labels.parquet', columns=['row', 'sbp', 'dbp', 'age'])

    # ── BP：逐目标的细粒度计数 ──
    edges = np.arange(DIST_LO, DIST_HI, DIST_BIN)      # 与图的 BINS_BP 同域、半开
    off = (np.asarray(BINS_BP, float) - edges[0]) / DIST_BIN
    assert np.allclose(off, np.round(off), atol=1e-9), \
        f'细粒度({DIST_BIN:g})与图的分箱({np.diff(BINS_BP)[0]:g})边界未对齐'
    cols = {'bin_left': edges[:-1].astype(np.float32)}
    meta, stats = [], []
    for t in ('sbp', 'dbp'):
        v = lab[t].dropna().to_numpy(np.float64)
        c, _ = np.histogram(v, bins=edges)
        cols[f'{t}_count'] = c.astype(np.int64)
        oob = int(((v < DIST_LO) | (v >= DIST_HI)).sum())
        # ⚠ 恰好等于末边界的值会触发 numpy 的「归入最后一桶」规则，而计数型导出
        #   无法复现（同一格里混着边界内外的值）→ 必须当场炸，不许静默差几个计数。
        at_top = int((v == DIST_HI).sum())
        assert at_top == 0, f'{t}: {at_top} 个值恰好 == 末边界 {DIST_HI}，两条路径会不一致'
        meta.append((t, v.size, oob))
        # 标注统计：图上要写 N/中位/范围，而**计数恢复不出中位数**，故单存。
        # （age 不存 —— 那边逐人的原值就在文件里，统计可直接算出，避免双份真相。）
        stats.append(dict(target=t, N=v.size, median=float(np.median(v)),
                          lo=float(v.min()), hi=float(v.max())))
    bp = pd.DataFrame(cols)
    fp = out / 'dist_bp.csv.gz'
    bp.to_csv(fp, index=False, compression='gzip')
    st = pd.DataFrame(stats)
    st.to_csv(out / 'dist_bp_stats.csv', index=False)
    for t, n, oob in meta:
        print(f'  dist_bp: {t}  N_true={n:,}  越界={oob}  计数和={int(bp[f"{t}_count"].sum()):,}'
              f'  bin={DIST_BIN:g}mmHg×{len(bp)}')

    # ── Age：人级中位数（逐 source×subject，**不跨库合并同名**）──
    man = pd.read_parquet(R / 'manifest.parquet',
                          columns=['source', 'subject_id']).reset_index()
    man = man.rename(columns={'index': 'row'})
    d = man.merge(lab, on='row', how='inner')
    per = (d.groupby(['source', 'subject_id'], as_index=False).age.median()
             .dropna(subset=['age']))
    fp = out / 'dist_age.csv.gz'
    per.to_csv(fp, index=False, compression='gzip')
    print(f'  dist_age: {len(per):,} 人  中位 {per.age.median():.1f}  '
          f'范围 [{per.age.min():.0f}, {per.age.max():.0f}]')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/paper/figure_data')
    ap.add_argument('--skip-dist', action='store_true',
                    help='只导散点数据，不导分布图数据（分布要读语料，慢）')
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    src_dir = P / EXP / 'full' / 'stream' / 'preds'

    print(f'源: {src_dir.relative_to(ROOT)}')
    print(f'口径: seed=s{SEED} · 域={DOMS} · 列=src,sid,y_true,y_pred')
    total = 0
    for t in TGT:
        d = pd.read_csv(src_dir / f'{t}__causal_attn.csv.gz',
                        usecols=['seed', 'src', 'sid', 'y_true', 'y_pred'])
        n0 = len(d)
        d = d[(d.seed == SEED) & (d.src.isin(DOMS))][['src', 'sid', 'y_true', 'y_pred']]
        fp = out / f'{t}.csv.gz'
        d.to_csv(fp, index=False, compression='gzip')
        mb = fp.stat().st_size / 1048576
        total += fp.stat().st_size
        print(f'  {t}: {n0:,} → {len(d):,} 行  {mb:.1f} MB  → {fp.relative_to(ROOT)}')
    print(f'  合计 {total/1048576:.1f} MB（原始 preds 目录约 223 MB）')

    if not a.skip_dist:
        print('\n分布图数据（读语料 labels…）', flush=True)
        export_dist(out)


if __name__ == '__main__':
    main()
