#!/usr/bin/env python
"""compute_feat_stats.py — 拍级 40 维特征的均值/标准差（辅助任务 z 参数化）。

从语料全库分层抽样若干段，仅用 feat_valid 拍统计，
产出 {names, mean(40), std(40)} 的 npz。

用法：compute_feat_stats.py [corpus_root] [out.npz]
  默认根是 corpus_v2（legacy 线）；**跑现役 corpus_v2c 必须显式传语料根**，
  否则统计量与 `config/base/beatmae.yaml` 的 `feat_stats:` 对不上。
  默认输出 `config/feat_stats_v2c.npz`（现役统计量，beatmae.yaml 直接引用）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # 仓库根（本文件在 preprocess/corpus_build/ 下）

import numpy as np

from data import memmap_npz_member

ROOT = '/data/corpus_v2'
# 默认输出**对齐现役统计量**（config/base/beatmae.yaml 的 feat_stats: 读的就是它）。
# 2026-09-21 前这里写的是 legacy 的 feat_stats.npz（corpus_v2 线），用默认值跑会生成
# 错的那一个，而 config/README.md 的手工命令显式传了路径，故一直没暴露。
OUT = Path(__file__).resolve().parents[2] / 'config' / 'feat_stats_v2c.npz'
N_SEGS = 2000          # 抽样段数（覆盖 >2 万拍，统计稳定）

# 可选命令行覆盖：compute_feat_stats.py [corpus_root] [out.npz]
if len(sys.argv) > 1:
    ROOT = sys.argv[1]
if len(sys.argv) > 2:
    OUT = Path(sys.argv[2])


def main():
    meta = json.load(open(f'{ROOT}/features_meta.json'))
    names = meta['beat_features']['columns']
    F = len(names)
    assert F == 40

    off = np.asarray(memmap_npz_member(f'{ROOT}/keypoints.npz', 'offsets'))
    feat = memmap_npz_member(f'{ROOT}/beat_features.npz', 'features')
    fv = np.asarray(memmap_npz_member(f'{ROOT}/beat_features.npz', 'feat_valid'))
    n_seg = off.shape[0] - 1

    rng = np.random.default_rng(0)
    segs = rng.choice(n_seg, size=N_SEGS, replace=False)
    xs, ys = [], []
    for s in segs:
        a, b = int(off[s]), int(off[s + 1])
        ok = fv[a:b].astype(bool)
        if ok.any():
            x = np.asarray(feat[a:b], dtype=np.float32)[ok]
            xs.append(x)
    x = np.concatenate(xs, axis=0).astype(np.float64)
    print(f'统计拍数: {x.shape[0]}')

    # 非有限值按列掩掉（±Inf 不参与统计；语料侧已由 22 号脚本 clip，此处兜底）
    finite = np.isfinite(x)
    x = np.where(finite, x, np.nan)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    print(f'非有限值掩掉的单元格: {(~finite).sum()} / {finite.size}')
    std[std < 1e-6] = 1.0
    np.savez(OUT, names=np.array(names), mean=mean.astype(np.float32),
             std=std.astype(np.float32))
    print(f'已保存 {OUT}')
    for i, n in enumerate(names):
        print(f'  {n:30s} mean={mean[i]:12.4f} std={std[i]:12.4f}')


if __name__ == '__main__':
    main()
