# -*- coding: utf-8 -*-
"""data/windows 的协议回归测试（2026-09-05）。

覆盖抽取自脚本的两个函数的协议铁律：
  build_bags: 无重叠、不越人、窗均值、尾窗合并、cap_win 保序采样
  recent_history: 与 28 号原内联 padded unfold 逐位一致、首段冷启动

（train_once 冒烟已于 2026-09-11 随 models/probe_heads/train_utils.py 删除：
 该模块只被本节使用，唯一调用方是它自己的冒烟测试。）
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.windows import build_bags, make_slices, recent_history


def _toy(n_sub=20, per_sub=10, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_sub * per_sub, d)).astype(np.float32)
    sids = np.repeat(np.arange(n_sub), per_sub)
    return X, sids


def test_build_bags_protocol():
    X, sids = _toy()
    slices = make_slices(sids)
    B, M, wby = build_bags(X, slices, W=6, cap_win=128)
    for si, chunks in enumerate(wby):
        b0, e0 = slices[si][1], slices[si][2]
        assert M[si, :len(chunks)].all() and not M[si, len(chunks):].any()
        for k, c in enumerate(chunks):
            assert np.allclose(B[si, k], X[c].mean(0), rtol=1e-5)   # 窗均值
            assert c[0] >= b0 and c[-1] < e0                        # 不越人
        for a, b in zip(chunks, chunks[1:]):
            assert b[0] == a[-1] + 1                                # 无重叠


def test_build_bags_cap_win_and_tail_merge():
    X, sids = _toy(per_sub=7)   # W=6 → 2 窗（6+1），尾窗 1 < max(2, 3) 并入 → 1 窗
    slices = make_slices(sids)
    _, _, wby = build_bags(X, slices, W=6, cap_win=128)
    assert all(len(c) == 1 for c in wby) and len(wby[0][0]) == 7   # 尾窗并入
    _, _, wby2 = build_bags(X, slices, W=6, cap_win=3)
    assert all(len(c) <= 3 for c in wby2)                          # cap 上限


def test_recent_history_bitwise_equal_to_inline():
    """与 28 号原内联实现逐位一致（抽取行为零变化的钉子）。"""
    torch.manual_seed(0)
    kk, m, d = 32, 50, 16
    E = torch.randn(m, d)
    pad = torch.zeros(kk, d)
    Ep = torch.cat([pad, E], 0)
    wins_old = Ep.unfold(0, kk, 1)[:m].transpose(1, 2)
    cnt = torch.clamp(torch.arange(m), max=kk)
    km_old = torch.arange(kk)[None, :] >= (kk - cnt)[:, None]
    wins_new, km_new = recent_history(E, kk)
    assert torch.equal(wins_old, wins_new)
    assert torch.equal(km_old, km_new)
    assert not km_new[0].any()      # 首段冷启动：无历史
    assert km_new[1].sum() == 1 and km_new[kk].all()   # min(i,kk) 历史数
