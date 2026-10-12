#!/usr/bin/env python3
"""从日志现算论文的五张表 → results/paper/*.csv。

**不抄数字。** 每张表都从 `results/` 的 preds 明细现算，数字只有日志一个出处。

口径（2026-09-13 用户定，与主表/迁移表既有惯例一致）：
  · **一律 best-seed**：每个臂取 5 个 seed 中最优的那个，**不用 5-seed 平均**。
  · 选优判据 = **该臂在"本表所报格子"上的 MAE 均值**（跨域+目标），选中后
    **同一 seed 应用到该臂全部格子**（不做逐格挑 seed —— 那是逐格挑数据）。
  · 本脚本把每个臂选中的 seed 一并打印，供人工核对。

五张表：
  1. main       主实验：★2.5c3(causal_attn) vs 4 个对比模型(seg_mlp)
  2. ablation   消融：完整模型 / w/o 历史通路 / w/o c_long / w/o c_recent / w/o 预训练
  3. transfer   causal_attn 迁移进对比模型
  4. history    历史数据累积（per-bucket ΔMAE）
  5. iwshi_bp   iwshi_bp 穿戴域迁移（六臂，含污染臂标记）

用法：python tools/make_paper_tables.py [--out results/paper]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools._calibers import BUCKETS, load_k          # 口径独立成模块（原在绘图脚本里）

P = ROOT / 'results' / 'probe'
S = ROOT / 'results' / 'cache' / 'stream'
TGT = ('sbp', 'dbp', 'age')
DOMS = ('vital', 'aurora')
SLOT = 'v2c_3src_mva'
SEEDS = (0, 1, 2, 3, 4)
KEYCOL = ('sid', 'seg_idx', 'src')

MAIN = 'beatmae_v25c3'
RAND = 'beatmae_v25c3_randinit'


def preds(model, t, var, slot=SLOT, scope='full', family='stream'):
    return P / model / slot / scope / family / 'preds' / f'{t}__{var}.csv.gz'


def read(model, t, var, common=None, **kw):
    f = preds(model, t, var, **kw)
    if not f.exists():
        return None
    d = pd.read_csv(f, usecols=['seed', 'sid', 'seg_idx', 'src', 'y_true', 'y_pred'])
    if common is not None:
        m = np.fromiter((k in common for k in zip(d.sid, d.seg_idx, d.src)),
                        dtype=bool, count=len(d))
        d = d[m]
    return d


def cell(d, dom, seed):
    """(MAE, r)，先按域切、再按 seed 切。"""
    g = d[(d.src == dom) & (d.seed == seed)]
    if not len(g):
        return None
    yt, yp = g.y_true.to_numpy(np.float64), g.y_pred.to_numpy(np.float64)
    return float(np.mean(np.abs(yt - yp))), float(np.corrcoef(yt, yp)[0, 1])


def pick_seed(frames, rule=lambda vals: np.mean([v[0] for v in vals])):
    """per-arm best seed：按 rule（默认 = 所报格子 MAE 均值）挑，返回 (seed, 逐 seed 值)。"""
    per = {}
    for s in SEEDS:
        vals = [v for v in (frames.get((d, t, s)) for d in DOMS for t in TGT) if v]
        if len(vals) == len(DOMS) * len(TGT):
            per[s] = rule(vals)
    if not per:
        return None, {}
    return min(per, key=per.get), per


def main_seed(m, var):
    """主实验口径的选优：按 `var`（该臂在主表里用的头）的**六格 MAE 均值**挑 seed。

    ★ 这是**全篇选种的唯一出处**：`table_main` 与 `table_transfer` 都必须走这里。
    2026-09-18 之前 transfer 自己按 causal_attn 另挑一遍，导致 MMR / PaPaGei-S
    两个臂在两表里选中不同 seed，同一编码器的 seg_mlp 给出两组数（MMR vital/SBP
    主表 12.58 vs 增益表 12.54，六格全不同）——审稿人一眼可见的矛盾。
    另：transfer 自己的判据（挑 causal 最好的 seed）**本身有选择偏向**，
    会系统性抬高 gain（MMR 实测选中 gain 最大的 s3 = 0.811，五 seed 均值仅 0.565）。
    """
    fr = {}
    for d in DOMS:
        for t in TGT:
            dd = read(m, t, var)
            if dd is None:
                continue
            for s in SEEDS:
                v = cell(dd, d, s)
                if v:
                    fr[(d, t, s)] = v
    return pick_seed(fr)


def table_main():
    """主实验：每个模型取自己最好的 seed（判据 = 六格 MAE 均值）。"""
    models = [('beatmae_v25c3', 'causal_attn', '★ 2.5c3 3src（本方）'),
              ('xmae', 'seg_mlp', 'xMAE'), ('himae', 'seg_mlp', 'HiMAE'),
              ('mmr', 'seg_mlp', 'MMR'),
              ('papagei_s', 'seg_mlp', 'PaPaGei-S'), ('papagei_p', 'seg_mlp', 'PaPaGei-P')]
    rows = []
    for m, var, lab in models:
        bs, per = main_seed(m, var)
        if bs is None:
            print(f'  [缺] {m} 无完整 seed'); continue
        row = {'模型': lab, 'seed': f's{bs}'}
        for d in DOMS:
            for t in TGT:
                dd = read(m, t, var)
                v = cell(dd, d, bs) if dd is not None else None
                row[f'{d}_{t}_MAE'] = round(v[0], 3) if v else None
                row[f'{d}_{t}_r'] = round(v[1], 3) if v else None
        rows.append(row)
        print(f'  {lab:<22} s{bs}  逐seed六格均值 ' +
              ' '.join(f's{s}={per[s]:.3f}' for s in sorted(per)))
    return pd.DataFrame(rows)


def table_ablation():
    """消融：五个臂各取自己最好的 seed（判据 = 六格 MAE 均值）。

    与主表的差异：主表是不同模型比，消融是同一模型改部件——两表的 seed 各自独立选。
    ⚠ w/o 预训练 臂的段集与其余四臂不同（见下），故只在**交集**上算，口径才可比。
    """
    arms = [(MAIN, 'causal_attn', '完整模型', False),
            (MAIN, 'seg_mlp', 'w/o 历史通路', True),
            (MAIN, 'causal_droplong', 'w/o c_long', True),
            (MAIN, 'causal_droprecent', 'w/o c_recent', True),
            (RAND, 'causal_attn', 'w/o 预训练', True)]
    # 交集段集：主力流与 randinit 流的共有段（两臂段集互不为子集，见 ablation 分析）
    ka, kb = stream_keys(f'beatmae_v25c3__{SLOT}__full'), stream_keys(f'{RAND}__{SLOT}__full')
    common = ka & kb
    print(f'  段集配对：主力 {len(ka):,} / randinit {len(kb):,} / 交集 {len(common):,}')
    rows = []
    base = {}
    for m, var, lab, is_abl in arms:
        fr = {}
        for d in DOMS:
            for t in TGT:
                dd = read(m, t, var, common=common)
                if dd is None:
                    continue
                for s in SEEDS:
                    v = cell(dd, d, s)
                    if v:
                        fr[(d, t, s)] = v
        bs, per = pick_seed(fr)
        if bs is None:
            print(f'  [缺] {lab}: 无完整 seed'); continue
        row = {'消融项': lab, 'seed': f's{bs}'}
        for d in DOMS:
            for t in TGT:
                v = fr.get((d, t, bs))
                row[f'{d}_{t}_MAE'] = round(v[0], 3) if v else None
                row[f'{d}_{t}_r'] = round(v[1], 3) if v else None
                if not is_abl:
                    base[(d, t)] = v[0] if v else None
        rows.append(row)
        print(f'  {lab:<18} s{bs}  逐seed六格均值 ' +
              ' '.join(f's{s}={per[s]:.3f}' for s in sorted(per)))
    # Δ 列（正 = 拿掉后变差）
    for row in rows[1:]:
        for d in DOMS:
            for t in TGT:
                k = f'{d}_{t}_MAE'
                if row.get(k) is not None and base.get((d, t)) is not None:
                    row[f'{d}_{t}_d'] = round(row[k] - base[(d, t)], 3)
    return pd.DataFrame(rows)


def stream_keys(exp):
    z = np.load(S / exp / 'stream_meta.npz', allow_pickle=True)
    nm, si, src, bnd = (z['sub_names'].astype(str), z['seg_idx'],
                        z['src'].astype(str), z['bnd'])
    sub = np.empty(len(si), dtype=object)
    for i in range(len(bnd) - 1):
        sub[bnd[i]:bnd[i + 1]] = nm[i]
    return set(zip(sub.tolist(), si.tolist(), src.tolist()))


def table_transfer():
    """迁移：**复用主表的 seed**（同一 seed 贯穿两表），在该 seed 上比两种头。

    ⚠ seed 来自 `main_seed()`，不再自己按 causal_attn 另挑（2026-09-18 用户定）。
    这样同一编码器的 seg_mlp 在 main.csv 与 transfer.csv 里**必然是同一个数**；
    两表可以并排放进论文而不打架。代价是 gain 不再被"挑 causal 最好的 seed"抬高
    （MMR 0.811→0.433），但这正是我们要去掉的偏向。
    ★ 用 causal_attn 判据挑（它主表里用的就是 causal_attn），其余臂用 seg_mlp。
    """
    models = [('beatmae_v25c3', 'causal_attn', '★ 2.5c3 3src（本方）'),
              ('xmae', 'seg_mlp', 'xMAE'), ('himae', 'seg_mlp', 'HiMAE'),
              ('mmr', 'seg_mlp', 'MMR'),
              ('papagei_s', 'seg_mlp', 'PaPaGei-S'), ('papagei_p', 'seg_mlp', 'PaPaGei-P')]
    rows = []
    for m, main_var, lab in models:
        bs, _ = main_seed(m, main_var)
        if bs is None:
            print(f'  [缺] {lab}'); continue
        row = {'模型': lab, 'seed': f's{bs}'}
        for d in DOMS:
            for t in TGT:
                for var, tag in (('seg_mlp', 'seg'), ('causal_attn', 'causal')):
                    dd = read(m, t, var)
                    v = cell(dd, d, bs) if dd is not None else None
                    row[f'{d}_{t}_{tag}_MAE'] = round(v[0], 3) if v else None
                    row[f'{d}_{t}_{tag}_r'] = round(v[1], 3) if v else None
                a, b = row.get(f'{d}_{t}_seg_MAE'), row.get(f'{d}_{t}_causal_MAE')
                if a is not None and b is not None:
                    row[f'{d}_{t}_gain'] = round(a - b, 3)   # 正 = 因果头更好
        rows.append(row)
        print(f'  {lab:<22} s{bs}（沿用主表）')
    return pd.DataFrame(rows)


def bucket_mae_one_seed(preds_dir, t, var, dom, key2k, seed):
    """单 seed 的逐桶 MAE。口径与 plot_history_gain.bucket_mae 一致，只是不跨 seed 平均。"""
    d = pd.read_csv(preds_dir / f'{t}__{var}.csv.gz',
                    usecols=['seed', 'sid', 'seg_idx', 'src', 'y_true', 'y_pred'])
    d = d[(d.src == dom) & (d.seed == seed)]
    if not len(d):
        return None
    kk = np.array([key2k[(n, int(g))] for n, g in zip(d.sid, d.seg_idx)])
    err = np.abs(d.y_pred.to_numpy() - d.y_true.to_numpy())
    yt = d.y_true.to_numpy(np.float64)
    yp = d.y_pred.to_numpy(np.float64)
    rows = []
    for lo, hi, name in BUCKETS[dom]:
        m = (kk >= lo) & (kk < hi)
        if not m.any():
            continue
        rows.append(dict(bucket=name, k_lo=lo, k_hi=hi, n=int(m.sum()),
                         mae=float(err[m].mean()),
                         r=float(np.corrcoef(yt[m], yp[m])[0, 1]) if m.sum() > 2 else np.nan))
    return pd.DataFrame(rows) if rows else None


def table_history():
    """历史累积：主力模型 best-seed（判据同主表 = 六格 MAE 均值）。"""
    key2k, _ = load_k(S / f'{MAIN}__{SLOT}__full' / 'stream_meta.npz')
    fr = {}
    for d in DOMS:
        for t in TGT:
            dd = read(MAIN, t, 'causal_attn')
            for s in SEEDS:
                v = cell(dd, d, s)
                if v:
                    fr[(d, t, s)] = v
    bs, per = pick_seed(fr)
    print(f'  用 seed s{bs}（判据同主表）  逐seed六格均值 ' +
          ' '.join(f's{s}={per[s]:.3f}' for s in sorted(per)))
    rows = []
    for dom in DOMS:
        for t in TGT:
            pd_dir = P / MAIN / SLOT / 'full/stream/preds'
            a = bucket_mae_one_seed(pd_dir, t, 'seg_mlp', dom, key2k, bs)
            b = bucket_mae_one_seed(pd_dir, t, 'causal_attn', dom, key2k, bs)
            if a is None or b is None:
                continue
            m = a.rename(columns={'mae': 'seg_mlp_MAE', 'r': 'seg_mlp_r'})
            m['causal_attn_MAE'] = b['mae'].to_numpy()
            m['causal_attn_r'] = b['r'].to_numpy()
            m['gain_MAE'] = (m.seg_mlp_MAE - m.causal_attn_MAE).round(3)
            for c in ('seg_mlp_MAE', 'causal_attn_MAE', 'seg_mlp_r', 'causal_attn_r'):
                m[c] = m[c].round(3)
            m.insert(0, 'target', t)
            m.insert(0, 'domain', dom)
            rows.append(m)
    return pd.concat(rows, ignore_index=True)


def table_iwshi():
    """iwshi_bp 穿戴域迁移：stream5f 域内五折；每个臂取自己最好的 seed。

    **从 preds 现算**（不再读 results.json 的 `seg` 列表），理由有二：
      1. `results.json` 的 `seg_r` 是**跨 seed 平均后的标量**，拿它配 best-seed 的 MAE
         不自洽；preds 有 `fold` + `seed` 两列，可算出所选 seed 自己的 r。
      2. 与其余四表同源（都从 preds 现算），溯源一致。
    选优判据 = 所选 seed 在三个目标上的「逐折均值」的平均 → 再取最优的那一个 seed。
    ⚠ 表内混了两代模型（v25c 五臂 + v25c3 ★主力），可比性说明见 claims.md §5.4。
    """
    arms = [('beatmae_v25c', 'v2c_aurora', '① aurora', False),
            ('beatmae_v25c', 'v2c_2src_av', '③ aurora+vital', False),
            ('beatmae_v25c3', SLOT, '★ 2.5c3 (mva)', False),
            ('beatmae_v25c', 'v2c_aurora_iwb', '② aurora+iwb', True),
            ('beatmae_v25c', 'v2c_3src_avi', '④ aurora+vital+iwb', True),
            ('beatmae_v25c', 'v2c_4src_mvaiwb', '⑤ mvak', True)]
    rows = []
    for m, slot, lab, contaminated in arms:
        d0 = P / m / slot / 'iwshi_bp' / 'stream5f' / 'preds'
        if not (d0 / 'sbp__causal_attn.csv.gz').exists():
            print(f'  [缺] {lab}'); continue
        # 逐 (target, seed, fold) 的 MAE 与 r
        per = {}
        for t in TGT:
            d = pd.read_csv(d0 / f'{t}__causal_attn.csv.gz',
                            usecols=['fold', 'seed', 'y_true', 'y_pred'])
            for (s, f_), g in d.groupby(['seed', 'fold']):
                yt, yp = g.y_true.to_numpy(np.float64), g.y_pred.to_numpy(np.float64)
                per[(t, int(s), int(f_))] = (
                    float(np.mean(np.abs(yt - yp))),
                    float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 else np.nan)
        folds = sorted({f_ for (_, _, f_) in per})
        per_seed = {}
        for s in SEEDS:
            vals = [float(np.mean([per[(t, s, f_)][0] for f_ in folds
                                   if (t, s, f_) in per])) for t in TGT]
            if all(not np.isnan(v) for v in vals):
                per_seed[s] = float(np.mean(vals))
        if not per_seed:
            print(f'  [缺] {lab}: 无 causal_attn 结果'); continue
        bs = min(per_seed, key=per_seed.get)
        row = {'臂': lab, 'seed': f's{bs}', '含 iwshi_bp': '是' if contaminated else '否'}
        for t in TGT:
            mae = float(np.mean([per[(t, bs, f_)][0] for f_ in folds if (t, bs, f_) in per]))
            r = float(np.nanmean([per[(t, bs, f_)][1] for f_ in folds if (t, bs, f_) in per]))
            row[f'{t}_MAE'] = round(mae, 3)
            row[f'{t}_r'] = round(r, 3)
        rows.append(row)
        print(f'  {lab:<22} s{bs}  逐seed三目标均值 ' +
              ' '.join(f's{s}={per_seed[s]:.3f}' for s in sorted(per_seed)))
    return pd.DataFrame(rows)


# 每张表归到 results/paper/ 下的哪个实验子目录。**加表或改归属只改这里。**
TABLE_DIR = {'main': 'main', 'ablation': 'ablation', 'transfer': 'transfer',
             'history': 'history', 'iwshi_bp': 'wearable'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/paper',
                    help='产物根（各实验进其子目录，见 TABLE_DIR）')
    a = ap.parse_args()
    out = ROOT / a.out
    for name, fn in (('main', table_main), ('ablation', table_ablation),
                     ('transfer', table_transfer), ('history', table_history),
                     ('iwshi_bp', table_iwshi)):
        print(f'\n══ {name}')
        df = fn()
        d = out / TABLE_DIR[name]
        d.mkdir(parents=True, exist_ok=True)
        fp = d / f'{name}.csv'
        df.to_csv(fp, index=False)
        print(f'  → {fp.relative_to(ROOT)}  ({len(df)} 行)')


if __name__ == '__main__':
    main()
