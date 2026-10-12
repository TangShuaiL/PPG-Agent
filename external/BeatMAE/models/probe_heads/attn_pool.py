# -*- coding: utf-8 -*-
"""单查询门控注意力池化探针头（head=attn）。来源：train/probe.py，2026-09-05 抽取。

ABMIL（Ilse et al. 2018）：
    score_i = (tanh(W·e_i) ⊙ sigmoid(U·e_i))·q → softmax → 加权聚合段嵌入 → Linear。
参数量 ~2·d·hidden，与 MLP 头同量级。
"""
import torch


class AttnPoolProbe(torch.nn.Module):
    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.V = torch.nn.Linear(d_in, hidden)
        self.U = torch.nn.Linear(d_in, hidden)
        self.q = torch.nn.Parameter(torch.randn(hidden) * 0.01)
        self.head = torch.nn.Linear(d_in, 1)

    def forward(self, bag, mask):
        """bag (B, K, d)，mask (B, K) bool（True=有效段）。返回 (B,) 预测。"""
        score = (torch.tanh(self.V(bag)) * torch.sigmoid(self.U(bag))) @ self.q
        score = score.masked_fill(~mask, float('-inf'))
        a = torch.softmax(score, dim=1)                       # (B, K)
        z = torch.einsum('bk,bkd->bd', a, bag)                # (B, d)
        return self.head(z).squeeze(-1)
