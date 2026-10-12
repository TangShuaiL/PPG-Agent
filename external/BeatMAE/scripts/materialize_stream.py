#!/usr/bin/env python3
"""29 号：流式数据物化——把"排序+分组+cap采样"一次性做进磁盘文件。

背景：27/28/29 号下游脚本每次启动都要重复付"pandas merge + 字符串排序 + cap 采样"
三件套（~10 分钟）。本脚本把这套一次性操作的结果固化成两个文件，之后所有下游脚本
启动只花一次加载时间。

产出（results/cache/stream/）:
  stream_emb.npy   段嵌入 float32，已按 (subject, seg_idx) 保序排列 + cap 采样
  stream_meta.npz  ei      int64  每段在原始 npz 里的行号（溯源用）
                   bnd     int64  受试者边界数组（len = n_sub+1），人 p 段区间 [bnd[p], bnd[p+1])
                   sub     int32  受试者编码（仅分组用，无语义）；一组 = 一个 (源, 人名) 对
                   sub_names      受试者原始 id 字符串（逐组；跨源重名时同名字符串会出现两次）
                   sub_src        逐组来源库字符串（与 sub_names 一一对应，切域用）
                   fold    int8   受试者折（取自该人首段，受试者级隔离）
                   seg_idx int32  段序（人内时间序，因果历史资格认定依据；跳号=时间空隙）
                   y       float32 (N,3) raw 段级标签，列序见 targets；NaN 段保留（可做历史不做 query）
                   targets str(3) y 的列名快照（列序权威在 train/stream_probe.py:TARGETS，用于读侧校验）
                   src     bytes  数据源字符串

约定（与 2026-09-04 定稿的段级流式设计一致）:
  - 人内 seg_idx 升序 = 时间序；物化后由哨兵校验（人内 diff >= 0）
  - cap 512：超长受试者保序均匀采样（不破坏时间结构）
  - 训练域 mimic+vital+aurora；iwshi_bp 纯测试不训，不进流式库
用法: python materialize_stream.py
"""
import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 仓库根入 path（直接跑 scripts/*.py 时必须）

from data.paths import CORPUS_V2C, STREAM_CACHE, resolve_corpus
# y 的列序权威在读侧（train/stream_probe.py:TARGETS），本脚本是写侧，从那里 import——
# 2026-09-11 之前两边各写一份字面量，任一处改序即静默错位，现只此一份。
from train.stream_probe import TARGETS

CORPUS = resolve_corpus(CORPUS_V2C, tag='corpus_v2c')
NPZ = 'results/cache/embeddings/beatmae_v2__v2c_4src_mvaiwb__full_lab.npz'
OUT = STREAM_CACHE
CAP_M = 512
# 注：此处原有 SRCS = ('mimic','vital','aurora')，已于 2026-09-10 删除——
# 全仓零引用，且与下面 --srcs 的默认值重复（后者还能被 CLI 覆盖）。


def main():
    # 参数化（2026-09-06）：--exp 决定物化子目录 STREAM_CACHE/<exp>/，--npz 指定源缓存；
    # 无参时保持旧行为（beatmae 4src 主实验）
    global NPZ, OUT
    ap = argparse.ArgumentParser(description='流式物化：排序+分组+cap采样一次固化')
    ap.add_argument('--exp', default='beatmae_v2__v2c_4src_mvaiwb__full')
    ap.add_argument('--npz', default=None,
                    help='源嵌入缓存（缺省 results/cache/embeddings/<exp>_lab.npz）')
    ap.add_argument('--srcs', default='mimic,vital,aurora',
                    help='来源过滤（逗号分隔；"all"=不过滤，用于 iwshi_bp 跨域评估物化）')
    a = ap.parse_args()
    NPZ = a.npz or f'results/cache/embeddings/{a.exp}_lab.npz'
    OUT = STREAM_CACHE / a.exp
    t0 = time.time()
    print('① 读取 npz + parquet ...')
    z = np.load(NPZ)
    row = z['row']; emb = z['emb']; fold = z['fold']; sub_all = z['subject_id']
    man = pd.read_parquet(f'{CORPUS}/manifest.parquet').set_index('row')
    lab = pd.read_parquet(f'{CORPUS}/labels.parquet').set_index('row')

    print('② 行对齐（hash reindex，免排序）...')
    src = man.reindex(row)['source'].values
    seg = man.reindex(row)['seg_idx'].values.astype(np.int32)
    lab_df = lab.reindex(row)
    y = np.stack([lab_df[t].values for t in TARGETS], 1).astype(np.float32)

    print('③ 过滤训练域 ...')
    srcs = None if a.srcs == 'all' else tuple(a.srcs.split(','))
    keep = np.isin(src, srcs) if srcs is not None else np.ones(len(src), bool)
    row, emb, fold, seg, y = row[keep], emb[keep], fold[keep], seg[keep], y[keep]
    sub_s = sub_all[keep]
    print(f'   来源过滤[{a.srcs}] {keep.sum()} / {len(keep)} 段')

    print('④ 受试者 (源, 人名) 复合键 → int 编码（唯一一次字符串操作）...')
    # 2026-09-11 修复：原先只按 subject_id 单键分组。manifest 里有 28 个名字同时挂在
    # mimic 与 vital 下——是**两个不同的人**（28 对里 26 对的人在两侧折号不同）。单键会把
    # 这两人并成一组，下游取折是"首段优先"（train/stream_probe.py: fold_f[bnd[:-1]]），
    # 于是两人的折被强行绑定、cap 512 预算也并成一个。改为 (源, 人名) 复合键分组：
    # 源编小整数（≤8 个源），key = name_code * 8 + src_code，一次 np.unique 出全部组。
    src_s = src[keep]
    src_uniq, src_code = np.unique(src_s, return_inverse=True)
    assert len(src_uniq) <= 8, f'来源数 {len(src_uniq)} 超出编码位宽 8，请调大乘数'
    name_uniq, name_code = np.unique(sub_s, return_inverse=True)
    uniq, first, sub = np.unique(name_code.astype(np.int64) * 8 + src_code,
                                 return_index=True, return_inverse=True)
    sub = sub.astype(np.int32)
    sub_names, sub_src = sub_s[first], np.asarray(src_s)[first]   # 逐组的名字与源
    n_cross = int((np.bincount(uniq // 8) > 1).sum())             # 跨源重名（同名不同人）数
    print(f'   来源 {list(src_uniq)} | {len(name_uniq)} 个名字 → {len(uniq)} 组 (源, 人名)'
          + (f' | ⚠ 跨源同名 {n_cross} 个：同名不同源视为不同人，已各占一组' if n_cross else ''))

    print('⑤ int 双键 lexsort（分组 + 人内时间序）...')
    order = np.lexsort((seg, sub))
    ei = np.flatnonzero(keep)[order]           # 原始 npz 行号
    emb, fold, seg, y = emb[order], fold[order], seg[order], y[order]
    sub = sub[order]

    print('⑥ cap 512 保序采样 ...')
    n_sub = len(uniq)
    _, start = np.unique(sub, return_index=True)
    bounds = np.append(start, len(sub))
    keep2 = []
    for i in range(n_sub):
        b0, b1 = bounds[i], bounds[i + 1]
        m = b1 - b0
        if m > CAP_M:
            sel = np.unique(np.linspace(0, m - 1, CAP_M).astype(int))
            keep2.append(np.arange(b0, b0 + m)[sel])
        else:
            keep2.append(np.arange(b0, b1))
    keep2 = np.concatenate(keep2)
    ei, emb, fold, seg, y, sub = (a[keep2] for a in (ei, emb, fold, seg, y, sub))
    _, start2 = np.unique(sub, return_index=True)
    bnd = np.append(start2, len(sub))

    print('⑦ 哨兵：人内时间序校验 ...')
    diffs = np.concatenate([np.diff(seg[bnd[i]:bnd[i + 1]]) for i in range(n_sub)])
    assert (diffs >= 0).all(), '哨兵失败：人内 seg_idx 出现倒序！'
    print(f'   通过：{n_sub} 人内 seg_idx 全部非降序（跳号保留=时间空隙特征）')

    print('⑧ 落盘 ...')
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'stream_emb.npy', emb)
    np.savez(OUT / 'stream_meta.npz',
             ei=ei, bnd=bnd, sub=sub, sub_names=sub_names, sub_src=sub_src.astype('S'),
             fold=fold.astype(np.int8),
             seg_idx=seg, y=y, src=src[keep][order][keep2].astype('S'),
             targets=np.array(TARGETS))    # y 列序快照：读侧 load_all 据此校验

    nv = {t: int((~np.isnan(y[:, i])).sum()) for i, t in enumerate(TARGETS)}
    print(f'完成: {len(sub)} 段 / {n_sub} 人 | 有效标签 {nv} | 用时 {time.time()-t0:.0f}s')
    print(f'   stream_emb.npy {emb.nbytes/1e9:.1f}GB | stream_meta.npz')


if __name__ == '__main__':
    main()
