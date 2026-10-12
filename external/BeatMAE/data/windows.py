# -*- coding: utf-8 -*-
"""data/windows — 时序切窗与历史取窗（2026-09-05 从脚本抽取集中）。

协议铁律（docstring 即规范，改动须过消融）：
  切窗: 无重叠（步长=窗长）；尾部 < max(2, W//2) 段并入前窗；
        窗数 > cap_win 时 linspace 保序均匀采样；
        窗嵌入 = 窗内段嵌入均值（无注意力，注意力粒度在窗间）。
  历史: 因果——窗口 i 只含 E[max(i-kk,0):i]（严格过去，不含当前段）；
        人内时序必须真实升序，打乱即时间前视偏差（建样保序铁律，勿破）；
        唯一例外是显式点名的反时间对照消融 causal_shuf（28 号，非默认行为）。

| 函数 | 来源 | 用途 |
|------|------|------|
| make_slices    | scripts/21（原 _slices，21/26 逐字副本） | 段行 → 受试者切片 |
| build_bags     | scripts/21（21/25/26 三份逐字副本）      | 切窗 + 窗均值 → bag |
| recent_history | scripts/28 build_batch 内联段            | 近 kk 段因果历史窗 |
"""
import numpy as np
import torch


def make_slices(sids):
    """按受试者 id 序列分组 → [(sid, begin, end)]，要求同受试者行连续。"""
    bounds = np.flatnonzero(np.r_[True, sids[1:] != sids[:-1]])
    ends = np.r_[bounds[1:], len(sids)]
    return [(sids[b], b, e) for b, e in zip(bounds, ends)]


def build_bags(X, slices, W, cap_win):
    """受试者行序列切窗；窗数 > cap_win 时保序均匀采样。
    返回 (B (n, N_max, d) 窗均值嵌入, M (n, N_max) 有效掩码, win_by_sub 窗索引)。"""
    win_by_sub = []
    for _, b, e in slices:
        n = e - b
        chunks = [np.arange(b + s, min(b + s + W, e)) for s in range(0, max(n, 1), W)]
        if len(chunks) >= 2 and len(chunks[-1]) < max(2, W // 2):
            last = chunks.pop()
            chunks[-1] = np.r_[chunks[-1], last]
        if len(chunks) > cap_win:
            sel = np.unique(np.linspace(0, len(chunks) - 1, cap_win).astype(int))
            chunks = [chunks[i] for i in sel]
        win_by_sub.append(chunks)
    N = max(len(c) for c in win_by_sub)
    D = X.shape[1]
    n = len(slices)
    B = np.zeros((n, N, D), np.float32)
    M = np.zeros((n, N), bool)
    for si, chunks in enumerate(win_by_sub):
        for k, c in enumerate(chunks):
            B[si, k] = X[c].mean(0)
            M[si, k] = True
    return B, M, win_by_sub


def recent_history(E, kk):
    """单人段嵌入序列 E (m, d) → 近 kk 段因果历史窗。
    返回 (winsAll (m, kk, d), kmaskAll (m, kk) bool)：
      窗口 i = E[max(i-kk,0):i]（padded unfold，严格过去，不含当前段）；
      kmask: 窗口 i 内有效历史数 = min(i, kk)，首段 kmask 全 False（冷启动）。"""
    m, d = E.shape
    pad = torch.zeros(kk, d, device=E.device)
    Ep = torch.cat([pad, E], 0)
    winsAll = Ep.unfold(0, kk, 1)[:m].transpose(1, 2)
    pos_idx = torch.arange(m, device=E.device)
    cnt = torch.clamp(pos_idx, max=kk)
    kmaskAll = torch.arange(kk, device=E.device)[None, :] >= (kk - cnt)[:, None]
    return winsAll, kmaskAll
