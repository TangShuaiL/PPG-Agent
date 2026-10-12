# -*- coding: utf-8 -*-
"""段级流式因果探针头。来源：scripts/probe_full_stream.py，2026-09-05 抽取。

双参照设计（v4.0 定稿，2026-09-04）：
    第 k 段预测 = MLP([e_k ; c_recent ; c_long])
    c_recent = 加性注意力池化近 K=32 个可用历史段（唯一梯度组件）
    c_long   = 前缀均值 mean(e_{<k})（无参 cumsum，因果）
变体: CausalAttn=主形态（双参照拼接）；CausalDropLong / CausalDropRecent = 各砍掉
一条参照的消融（2026-09-13 增，见各自 docstring）；SegMLP=无历史对照（接口同型，
忽略历史入参）。
因果铁律: 输入只准用过去（历史在建样时保序固化），打乱人内时序 = 反时间对照消融。

（早期对照变体 CausalFuse / CausalRAtt 已在开发中移除，未采用。）


**d（嵌入维）必填，无默认值。** 它随模型族而变：beatmae/himae/mmr/xmae 是 256，
PaPaGei 是 512（官方 n_classes=512）。这里曾默认 d=256，PaPaGei 一接进来就崩在
`mat1 and mat2 shapes cannot be multiplied (…x512 and 256x128)`（2026-09-11 实测）；
去掉默认值是为了让同类错误在构造那一刻就报参数缺失，而不是训到前向才崩
（也可用 train/stream_probe.py::head_dim 从权重反推）。
"""
import torch
import torch.nn as nn


class CausalAttn(nn.Module):
    def __init__(self, d, hid=128):
        super().__init__()
        self.W = nn.Linear(d, hid)
        self.a = nn.Parameter(torch.randn(hid) * 0.01)
        self.mlp = nn.Sequential(nn.Linear(3 * d, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, ek, win, kmask, clong):
        # win (B,K,d) 历史段; kmask (B,K); k=0 的位置 kmask 全 False
        s = torch.tanh(self.W(win)) @ self.a
        s = s.masked_fill(~kmask, -1e9)
        cnt = kmask.sum(1)
        alpha = torch.where(cnt[:, None] > 0, torch.softmax(s, 1), torch.zeros_like(s))
        c_recent = torch.einsum('bk,bkd->bd', alpha, win)
        return self.mlp(torch.cat([ek, c_recent, clong], 1)).squeeze(-1)


class CausalDropLong(nn.Module):
    """消融：砍掉 c_long 参照 → MLP([e_k ; c_recent])（近端腿单独称重）。

    检验「双历史」的远端腿：前缀均值那条**无参**长程参照到底有没有贡献。
    去掉后历史只剩近 K 段注意力这一条通路。
    注意冷启动：k=0 时 kmask 全 False，c_recent 为 0 向量，退化成 MLP([e_k; 0])。
    forward 签名与 CausalAttn 一致，clong 收下但不用（调用方无需改）。
    """

    def __init__(self, d, hid=128):
        super().__init__()
        self.W = nn.Linear(d, hid)
        self.a = nn.Parameter(torch.randn(hid) * 0.01)
        self.mlp = nn.Sequential(nn.Linear(2 * d, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, ek, win, kmask, clong):
        s = torch.tanh(self.W(win)) @ self.a
        s = s.masked_fill(~kmask, -1e9)
        cnt = kmask.sum(1)
        alpha = torch.where(cnt[:, None] > 0, torch.softmax(s, 1), torch.zeros_like(s))
        c_recent = torch.einsum('bk,bkd->bd', alpha, win)
        return self.mlp(torch.cat([ek, c_recent], 1)).squeeze(-1)


class CausalDropRecent(nn.Module):
    """消融：砍掉 c_recent 参照 → MLP([e_k ; c_long])（远端腿单独称重）。

    检验「双历史」的近端腿：近 K 段加性注意力池化有没有贡献。去掉后**没有任何
    可学参数参与历史聚合**，历史只剩前缀均值这一条无参通路——即"完全不看局部
    近期模式、只看全程平均"。
    注意：本类没有 W（没有窗口注意力），故 `train/stream_probe.py::head_dim`
    对它必须走 mlp.0 反推，不能落到 W.weight 分支。
    """

    def __init__(self, d, hid=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, ek, win, kmask, clong):
        return self.mlp(torch.cat([ek, clong], 1)).squeeze(-1)


# CausalFuse（38 号）与 CausalRAtt（39 号）已于 2026-09-13 移除，理由与留痕见文件头
# 「已移除的变体」段。要复原代码见本次提交之前的 git history。


class SegMLP(nn.Module):
    def __init__(self, d, hid=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, ek, win=None, kmask=None, clong=None):
        return self.mlp(ek).squeeze(-1)
