#!/usr/bin/env python3
"""数据说明与个人级结果两张表 → results/paper/*.csv。

与 `tools/make_paper_tables.py`（五张实验表）分开，因为这两张是为**数据说明**与
**人级口径**服务，来源也不同（语料 labels/manifest vs probe preds），跑得快。

产物
  1. dataset_stats.csv    逐库 × **split（train/test/all）** 数据统计（供 §4.1 数据说明）
       人数、段数、BP 有标签段数、age 有标签段数、女/男、人级年龄中位与范围
       ***四个库全收（含 mimic）*** —— 用户 2026-09-15 指出数据说明该含它
       **2026-09-17 加 split 列**：此前只有「全库总数 + 测试折数」，女/男、有标签段、
       年龄这些汇总量**拆不开训练/测试**，统计时无法区分。现在每库出 train/test/all 三行。
  2. pretrain_pools.csv   预训练**实际使用的池**（训练池 / 测试池 / 排除池）及其构成
       本设计：训练池 = mimic/vital/aurora 的**训练折**；
               测试池 = vital/aurora 的**测试折**；
               **mimic 的测试折整折不用**（既不训练也不测试）→ 单列成排除池。
  3. subject_level.csv    个人级（person-pooled）下游结果（供人级散点/人级口径）
       每个域×目标：人级 MAE / r / 人数；**均值池化与中位数池化两种都出**
       注：段预测在个体内分布右偏且有离群段，故人级血压的常见做法是**中位数**池化；
       均值池化并列给出以便对比。正文引用哪一个由作者决定，两列都在。

用法：python tools/make_data_tables.py [--out results/paper/analysis]
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
SEED = 4                       # best seed，与其余各表一致
TGT = ('sbp', 'dbp', 'age')
DOMS = ('vital', 'aurora')

# 全局测试折 = 折 1、2（三库同号折合并）。**来源：主模型 run_meta.json 的 test_folds**，
# 不是这里拍的——改这里必须同步改 run_meta，否则表与实验对不上。
TEST_FOLDS = (1, 2)


def _r1(x):
    """安全的 1 位小数舍入；空集返回 None（不写 NaN）。"""
    return None if x is None or not np.isfinite(x) else round(float(x), 1)


def _stats(g):
    """一个子集上的汇总量。人数/年龄均**按人**聚合，不按段；血压**按段**（有BP标签段）。

    ⚠ 必须按 (source, subject_id) 分组，**不能只按 subject_id**：本仓存在跨源同名
    受试者（mimic/vital 有重名），单键分组会把两个不同的人并成一个，人数少算。
    实测训练池会因此少 18 人（4,832 → 4,814）。单库内两种分组等价。

    ⚠ 两个聚合层级**不同**，不可混用（2026-09-17 用户要求补均值/SD）：
      · 年龄：先按人取段中位数 → 再在**人**上算 mean/SD（人级）
      · 血压：直接在**段**上算 mean/SD（段级），分母 = 有BP标签段数
    这样图注的 mean±SD 可直接取用，**不必从直方图反推**（直方图有分箱精度损失，
    且两条路径的分母口径不同）。
    """
    KEY = ['source', 'subject_id']
    per_gender = g.groupby(KEY).gender.agg(
        lambda x: x.dropna().mode().iloc[0] if x.notna().any() else None)
    per_age = g.groupby(KEY).age.median().dropna()
    vc = per_gender.value_counts().to_dict()
    sbp, dbp = g.sbp.dropna(), g.dbp.dropna()
    n_age = len(per_age)
    return dict(
        人数=len(per_gender), 段数=len(g),      # len(per_gender) = 唯一 (source, subject_id) 数
        有BP标签段=int(g.sbp.notna().sum()), 有age标签段=int(g.age.notna().sum()),
        女=int(vc.get('F', 0)), 男=int(vc.get('M', 0)),
        性别缺失=int(per_gender.isna().sum()),
        人级年龄中位=_r1(per_age.median()) if n_age else None,
        人级年龄均值=_r1(per_age.mean()) if n_age else None,
        人级年龄SD=_r1(per_age.std(ddof=1)) if n_age > 1 else None,
        人级年龄最小=int(per_age.min()) if n_age else None,
        人级年龄最大=int(per_age.max()) if n_age else None,
        段级SBP均值=_r1(sbp.mean()) if len(sbp) else None,
        段级SBP_SD=_r1(sbp.std(ddof=1)) if len(sbp) > 1 else None,
        段级DBP均值=_r1(dbp.mean()) if len(dbp) else None,
        段级DBP_SD=_r1(dbp.std(ddof=1)) if len(dbp) > 1 else None,
    )


def load_corpus():
    R = Path(resolve_corpus(CORPUS_V2C, tag='corpus_v2c'))
    man = pd.read_parquet(R / 'manifest.parquet',
                          columns=['source', 'subject_id', 'fold', 'sig_ok'])
    man = man.reset_index().rename(columns={'index': 'row'})
    lab = pd.read_parquet(R / 'labels.parquet',
                          columns=['row', 'sbp', 'dbp', 'age', 'gender'])
    return man.merge(lab, on='row', how='left')


def dataset_stats(out, d):
    """逐库 × split ∈ {train, test, all}。

    ⚠ 折号必须是**受试者级**的，否则同一个人会被拆到 train 和 test 两侧。
    下面断言这一点（0 个跨折受试者才算通过）。
    """
    n_multi = int((d.groupby(['source', 'subject_id']).fold.nunique() > 1).sum())
    assert n_multi == 0, f'{n_multi} 个受试者跨多折，折号不是受试者级，不能按折切分'

    rows = []
    for src, g in d.groupby('source'):
        is_te = g.fold.isin(TEST_FOLDS)
        for split, sub in (('train', g[~is_te]), ('test', g[is_te]), ('all', g)):
            rows.append(dict(库=src, split=split, **_stats(sub)))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'dataset_stats.csv', index=False)
    print(f'  → dataset_stats.csv  ({len(df)} 行 = {df.库.nunique()} 库 × 3 split)')
    return df


# 预训练实际使用的池。改这里 = 改实验设计，必须与 run_meta.json 的 sources/test_folds 一致。
POOL_SPEC = [
    ('训练池', ['mimic', 'vital', 'aurora'], 'train',
     'mimic/vital/aurora 的训练折；主模型 pretrain 的 sources 即此三库'),
    ('测试池', ['vital', 'aurora'], 'test',
     'vital/aurora 的测试折；报告域在这两个库上评估'),
    ('排除池', ['mimic'], 'test',
     'mimic 的测试折整折不用（既不训练也不测试）——故数据说明中它只出现在这里'),
]


def pretrain_pools(out, d):
    """训练池 / 测试池 / 排除池 的构成与合计（直接回答"统计时怎么区分开"）。"""
    rows = []
    for name, srcs, split, note in POOL_SPEC:
        sub = d[d.source.isin(srcs)]
        sub = sub[sub.fold.isin(TEST_FOLDS)] if split == 'test' \
            else sub[~sub.fold.isin(TEST_FOLDS)]
        rows.append(dict(池=name, 构成=' + '.join(f'{s}.{split}' for s in srcs),
                         库数=len(srcs), 说明=note, **_stats(sub)))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'pretrain_pools.csv', index=False)
    print(f'  → pretrain_pools.csv  ({len(df)} 行)')
    return df


def subject_level(out):
    rows = []
    for dom in DOMS:
        f = P / EXP / 'full' / 'stream' / 'preds' / 'age__causal_attn.csv.gz'
        del f
        for t in TGT:
            fp = P / EXP / 'full' / 'stream' / 'preds' / f'{t}__causal_attn.csv.gz'
            d = pd.read_csv(fp, usecols=['seed', 'src', 'sid', 'y_true', 'y_pred'])
            d = d[(d.seed == SEED) & (d.src == dom)]
            g = d.groupby('sid')[['y_true', 'y_pred']]
            for how in ('mean', 'median'):
                agg = getattr(g, how)()
                yt = agg.y_true.to_numpy(np.float64)
                yp = agg.y_pred.to_numpy(np.float64)
                rows.append(dict(
                    domain=dom, target=t, pooling=how, n_subject=len(agg),
                    MAE=round(float(np.mean(np.abs(yt - yp))), 3),
                    r=round(float(np.corrcoef(yt, yp)[0, 1]), 3),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'subject_level.csv', index=False)
    print(f'  → subject_level.csv  ({len(df)} 行)')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/paper/analysis')
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    print('══ 读语料 manifest + labels…', flush=True)
    d = load_corpus()

    print('\n══ dataset_stats（逐库 × train/test/all）')
    ds = dataset_stats(out, d)
    print(ds.to_string(index=False))
    print('\n══ pretrain_pools（训练池/测试池/排除池）')
    pp = pretrain_pools(out, d)
    print(pp.to_string(index=False))
    print('\n══ subject_level（best seed s%d）' % SEED)
    sl = subject_level(out)
    print(sl.to_string(index=False))


if __name__ == '__main__':
    main()
