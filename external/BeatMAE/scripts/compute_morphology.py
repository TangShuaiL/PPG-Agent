#!/usr/bin/env python
"""compute_morphology.py — PaPaGei 形态学伪标签预计算（段级 sVRI / skewness / IPA）。

PaPaGei 的"伪标签"不是现成的：sVRI、skewness、IPA 三个形态学量都要从波形现算
（官方 `morphology.py`，本脚本逐行搬运其 `extract_svri` / `skewness_sqi` /
`compute_ipa` 三个函数，© 2024 Nokia, BSD-3-Clause-Clear；**不需要关键点、
不需要任何官方权重**）。P 只用 sVRI 分箱当对比学习的正样本定义；S 另外把
IPA 与 skewness 当 MoE 头的回归目标。所以这一张表是 P/S 两族共用的输入。

**计算口径已与训练输入逐项对齐**——全部在 `zscore(段)` 上算，也就是
`CorpusDataset(input_mode='raw_waveform')` 喂给模型的那张量：
- sVRI：内部先 min-max 归一化再取两段均值之比 → 对 z-score 变换不变；
- skewness：`scipy.stats.skew` 对平移与缩放都不变；
- IPA：`trapz` 的比值只对缩放不变、**对平移会变**，所以必须同口径 z-score。
  官方 `compute_ipa_for_dataset` 里也是先 `Normalize('z-score')` 再算，一致。
（唯一残差：训练侧 z-score 走 float32、本脚本走 float64，量级 ~1e-7，
对三个形态学量无影响。）

---
**与官方的三处偏差（都是"保住段池同构"换来的，必须知情）**

1. **不丢段。** 官方 `harmonize_datasets` 算完形态学后按
   `svri∈(0,2) & skewness∈(-3,3) & ipa∈(-10,10)` **过滤掉约 3.4% 的段**再进
   Dataset。本库另三个对比族（xMAE / HiMAE / MMR）用的是同一张段池，PaPaGei
   单独丢段主表就不是同池对比了。故本表**保留全部段**，过滤与否留给下游按需做。

2. **失败哨兵另立布尔列，不污染数值列。** 官方把 IPA 计算失败（`IndexError`）
   记成 `ipa = 0`，但 0 在真实 IPA 分布里几乎不出现（本库实测中位数 ~1），
   所以 0 是哨兵不是测量值。本表数值列仍按官方语义写 0（下游"跟官方一样回归
   0"直接读 `ipa`），同时用 `ipa_fail` 标出来（下游"屏蔽假标签"读它）。
   **本库实测 ipa_fail 占 55%**——官方 `clean_ipa_only` 默认为 False，
   意味着官方也是拿这些 0 当目标在训；这个比例对 S 的 IPA 头意味着什么，
   跑之前先看一眼脚本末尾打出的统计。sVRI 同理有 `svri_fail`（实测 ~0.3%，
   成因是段内最大值落在首/末样本时均值取空段；官方此时得 NaN、会被上面的
   过滤顺手丢掉）。

3. **不做分箱。** 官方在 Dataset 里用 `bin_data()` 对**传进来的 df** 按 min/max
   等宽分 8 箱。传进去的是训练切分，所以箱边界属于训练超参、必须由训练脚本按
   自己的池子现算（本库落在 `baselines/papagei/labels.py`），不能在这里预先固化
   ——固化会让验证/测试折的 sVRI 泄漏进箱边界。本表只出原始测量值。

输出（`--out`，默认 `results/cache/morphology/morphology_v2c.parquet`，原子落盘）：
- parquet  `row`(int64) / `svri`(f32) / `skewness`(f32) / `ipa`(f32) /
           `svri_fail`(bool) / `ipa_fail`(bool)
           按 `row` 与 manifest 对齐，可任意子集 join；行序 = manifest 行序。
- meta.json 语料根、manifest sha1、行数、各列分位、失败率、git/host 溯源。

用法：
    python scripts/compute_morphology.py                       # 全量（~5 分钟 / 32 进程）
    python scripts/compute_morphology.py --limit=20000         # 抽样试跑（均匀抽，看分布用）
    python scripts/compute_morphology.py --procs=16 --out=/tmp/m.parquet
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import warnings
# 官方写 `from scipy.integrate import trapz`：trapz 在 scipy 1.14 已删除，别名换成
# 自 1.6 起就有的 trapezoid（同一函数）。下面的调用点仍写 trapz，保持与官方对照。
from scipy.integrate import trapezoid as trapz
from scipy.signal import argrelmin      # 官方 compute_ipa 里的 argrelmax 未参与计算，略去
from scipy.stats import skew

from data.beat_ops import FS, SEG_LEN, zscore
from data.paths import CORPUS_V2C, resolve_corpus
from train.run_meta import collect_meta

FS_INT = int(FS)             # 125
ORDER = FS_INT // 5          # 25 —— 官方 argrelmax/argrelmin 的 prominence 窗（0.2 s）
WIN = 5 * FS_INT             # 625 —— 官方 SQI 的 5 秒切片长（1250 点恰好切两片）

DEFAULT_OUT = 'results/cache/morphology/morphology_v2c.parquet'

# scipy 的 argrelmin/trapz 在退化段上会出 RuntimeWarning（空切片、除零）。
# 那正是我们要用 *_fail 记录的情形，不该刷满日志。
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ────────────────────── 官方三个函数（逐行搬运）──────────────────────
def extract_svri(w: np.ndarray) -> float:
    """sVRI：段内最大值之后与之前的 min-max 归一化均值之比（官方 `extract_svri`）。

    对 z-score 不变（min-max 归一化吃掉平移缩放）。首/末样本即最大值时
    切片为空 → 均值 NaN，调用方记 `svri_fail`。
    """
    def __scale(data):
        data_max, data_min = max(data), min(data)
        return [(x - data_min) / (data_max - data_min) for x in data]

    max_index = int(np.argmax(w))
    scaled = np.asarray(__scale(w), dtype=np.float64)
    with np.errstate(invalid='ignore'):
        return float(np.mean(scaled[max_index:]) / np.mean(scaled[:max_index]))


def skewness_sqi(w: np.ndarray) -> float:
    """sQI：全段按 5 秒切片后逐片求偏度再平均（官方 `skewness_sqi` + 切片循环）。"""
    ppg = np.vstack([w[p:p + WIN] for p in range(0, len(w), WIN)])
    return float(np.mean(skew(ppg, axis=1, bias=True)))


def compute_ipa(sig: np.ndarray) -> float:
    """IPA：首个舒张期切迹前后的两段面积之比（官方 `compute_ipa`）。

    官方失败哨兵是 0（只 catch IndexError），本脚本照写 0 并另标 `ipa_fail`——
    见模块 docstring 偏差 2。注意官方只 catch IndexError，除零这类会漏出去变成
    inf/nan，这里一并收进 fail 标记，数值列按官方语义退化成 0。
    """
    try:
        minima_index = argrelmin(sig, order=ORDER)[0]
        single_beat = sig[minima_index[0]:minima_index[1]]
        minima_beats = argrelmin(single_beat)[0]
        minima_beat = minima_beats[0]

        sys_values = single_beat[:minima_beat]
        dias_values = single_beat[minima_beat:]
        sys_x = np.linspace(0, len(sys_values) - 1, len(sys_values))
        dias_x = np.linspace(0, len(dias_values) - 1, len(dias_values))
        ipa = trapz(y=sys_values, x=sys_x) / trapz(y=dias_values, x=dias_x)
    except IndexError:
        return 0.0, True
    if not np.isfinite(ipa):
        return 0.0, True
    return float(ipa), False


# ────────────────────────── 多进程外壳 ──────────────────────────
# fork 后子进程经 COW 共享父进程的只读数组（shard / row_in_shard），
# 每次任务只传一个下标切片。
_W: dict = {}


def _worker_init(root: str, shard: np.ndarray, ris: np.ndarray):
    # 逐进程单线程：scipy/numpy 的 BLAS 线程与进程池叠加会互相抢核
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = '1'
    _W.update(root=root, shard=shard, ris=ris)


def _worker(chunk: np.ndarray):
    """处理一批 manifest 行下标，返回 (chunk, svri, skewness, ipa, svri_fail, ipa_fail)。"""
    root, shard, ris = _W['root'], _W['shard'], _W['ris']
    n = len(chunk)
    svri = np.empty(n, np.float32)
    skew_ = np.empty(n, np.float32)
    ipa = np.empty(n, np.float32)
    svri_fail = np.zeros(n, bool)
    ipa_fail = np.zeros(n, bool)

    sig = None
    cur_shard = -1
    for k, p in enumerate(chunk):
        s = int(shard[p])
        if s != cur_shard:
            sig = np.load(Path(root) / 'signals' / f'ppg_shard_{s:04d}.npy',
                          mmap_mode='r')
            cur_shard = s
        # eps 用 beat_ops 默认值：与训练侧 `CorpusDataset` 完全同一口径
        w = zscore(np.asarray(sig[int(ris[p]), 0], dtype=np.float64))

        v = extract_svri(w)
        ipa_v, ipa_bad = compute_ipa(w)
        svri[k] = v
        skew_[k] = skewness_sqi(w)
        ipa[k] = ipa_v
        svri_fail[k] = not np.isfinite(v)
        ipa_fail[k] = ipa_bad

    return chunk, svri, skew_, ipa, svri_fail, ipa_fail


# ────────────────────────── 主流程 ──────────────────────────
def _sha1_head(path: Path, nbytes: int = 1 << 20) -> str:
    """manifest 前 1MB 的内容 sha1 —— 语料身份指纹（全量哈希太慢，前 1MB 足够区分）。"""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:12]


def compute(root: str, procs: int, limit: int | None) -> pd.DataFrame:
    man = pd.read_parquet(Path(root) / 'manifest.parquet',
                          columns=['row', 'shard', 'row_in_shard', 'sig_ok'])
    # 与 CorpusDataset 同一入口条件：sig_ok==1。其余过滤（折 / 源 / 最少拍数）
    # 都是池子级选择，由下游按 row 自己挑；这里全量算，表是池无关的。
    man = man[man['sig_ok'] == 1].reset_index(drop=True)
    if limit is not None and limit < len(man):
        # 均匀抽样（head 只会取到单一源：manifest 按源分块）
        idx = np.linspace(0, len(man) - 1, limit).astype(int)
        man = man.iloc[idx].reset_index(drop=True)
    N = len(man)

    row = man['row'].to_numpy(np.int64)
    shard = man['shard'].to_numpy(np.int64)
    ris = man['row_in_shard'].to_numpy(np.int64)

    # 按 shard 聚成连续段再均分给进程：每个任务只碰 1~2 个分片，且负载均衡
    order = np.argsort(shard, kind='stable')
    chunks = [c for c in np.array_split(order, procs) if len(c)]

    print(f'[morph] 语料 {root}', flush=True)
    print(f'[morph] 段数 {N:,} | 分片 {len(np.unique(shard))} | 进程 {procs} '
          f'| chunk {[len(c) for c in chunks]}', flush=True)

    out = {k: np.empty(N, d) for k, d in
           (('svri', np.float32), ('skewness', np.float32), ('ipa', np.float32),
            ('svri_fail', bool), ('ipa_fail', bool))}

    t0 = time.time()
    done = 0
    with Pool(procs, initializer=_worker_init, initargs=(root, shard, ris)) as pool:
        # 分片大小差异很小（同一源内均分），imap_unordered 已足够均衡；
        # 大分片先发可减少尾部空等。
        for i, res in enumerate(pool.imap(_worker, chunks)):
            chunk, a, b, c, f1, f2 = res
            out['svri'][chunk] = a
            out['skewness'][chunk] = b
            out['ipa'][chunk] = c
            out['svri_fail'][chunk] = f1
            out['ipa_fail'][chunk] = f2
            done += len(chunk)
            el = time.time() - t0
            print(f'[morph] {done:,}/{N:,} ({done / N:5.1%}) 已用 {el:6.1f}s '
                  f'预计总 {el * N / done:6.1f}s', flush=True)

    df = pd.DataFrame({'row': row, **out})
    stats = {
        'n_seg': int(N),
        'svri': _col_stats(df['svri']),
        'skewness': _col_stats(df['skewness']),
        'ipa': _col_stats(df['ipa']),
        'svri_fail': int(df['svri_fail'].sum()),
        'ipa_fail': int(df['ipa_fail'].sum()),
        'ipa_fail_frac': round(float(df['ipa_fail'].mean()), 4),
        'svri_in_0_2_frac': round(float(((df['svri'] > 0) & (df['svri'] < 2)).mean()), 4),
        'skewness_in_m3_3_frac': round(float(((df['skewness'] > -3) & (df['skewness'] < 3)).mean()), 4),
        'ipa_in_m10_10_frac': round(float(((df['ipa'] > -10) & (df['ipa'] < 10)).mean()), 4),
        'all_in_range_frac': round(float((((df['svri'] > 0) & (df['svri'] < 2))
                                          & ((df['skewness'] > -3) & (df['skewness'] < 3))
                                          & ((df['ipa'] > -10) & (df['ipa'] < 10))).mean()), 4),
    }
    return df, stats


def _col_stats(s: pd.Series) -> dict:
    """数值列分位（有限值上算；失败位单独统计，不混进来）。"""
    a = s.to_numpy(dtype=np.float64)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return {'n_finite': 0}
    ps = np.percentile(fin, [0.1, 1, 25, 50, 75, 99, 99.9])
    return {'n_finite': int(fin.size),
            'min': round(float(fin.min()), 4), 'max': round(float(fin.max()), 4),
            'p0.1': round(float(ps[0]), 4), 'p1': round(float(ps[1]), 4),
            'p25': round(float(ps[2]), 4), 'p50': round(float(ps[3]), 4),
            'p75': round(float(ps[4]), 4), 'p99': round(float(ps[5]), 4),
            'p99.9': round(float(ps[6]), 4),
            'mean': round(float(fin.mean()), 4)}


def main():
    ap = argparse.ArgumentParser(
        description='PaPaGei 形态学伪标签预计算（svri / skewness / IPA）',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', default=None, help='语料根（缺省走 data/paths 探测）')
    ap.add_argument('--out', default=DEFAULT_OUT, help=f'输出 parquet（默认 {DEFAULT_OUT}）')
    ap.add_argument('--procs', type=int, default=min(32, os.cpu_count() or 8),
                    help='进程数（默认 min(32, CPU 数)）')
    ap.add_argument('--limit', type=int, default=None,
                    help='调试用：均匀抽 N 段跑（默认全量）')
    args = ap.parse_args()

    root = resolve_corpus(CORPUS_V2C, explicit=args.corpus, tag='corpus_v2c')
    df, stats = compute(root, args.procs, args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(out)                                   # 原子落盘，崩了不留半张表

    meta = collect_meta({'script': 'scripts/compute_morphology.py',
                         'corpus_root': root,
                         'manifest_sha1_head': _sha1_head(Path(root) / 'manifest.parquet'),
                         'procs': args.procs, 'limit': args.limit,
                         'fs': FS_INT, 'seg_len': SEG_LEN,
                         'order': ORDER, 'sqi_window': WIN,
                         'stats': stats})
    (out.with_name(out.stem + '_meta.json')).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str))

    print(f'\n[morph] 写出 {out}  {out.stat().st_size / 1e6:.1f} MB', flush=True)
    print(f'[morph] 写出 {out.with_name(out.stem + "_meta.json")}', flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    if stats['ipa_fail_frac'] > 0.2:
        print(f"\n[morph] ⚠ IPA 计算失败占 {stats['ipa_fail_frac']:.1%}（官方哨兵值 0）。"
              f"\n        官方 clean_ipa_only 默认为 False，即官方也拿这些 0 当 S 的"
              f"\n        IPA 回归目标。要不要屏蔽成 --ipa-valid-only 见 pretrain 配置。",
              flush=True)


if __name__ == '__main__':
    main()
