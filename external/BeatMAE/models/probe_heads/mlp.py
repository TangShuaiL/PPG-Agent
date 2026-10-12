# -*- coding: utf-8 -*-
"""MLP 探针头（主口径 head=mlp）。来源：train/probe.py，2026-09-05 抽取。

结构: Linear(d,h1) → ReLU → Linear(h1,h2) → ReLU → Linear(h2,1)
协议: Adam(lr=1e-3, wd=1e-4)，早停 patience=20；SBP/DBP/Age 各自独立头。
超参配置: cfg probe.mlp 段（hidden1=256, hidden2=128）。
"""
import torch


class MLPProbe(torch.nn.Module):
    """2 层 MLP 探针头：Linear(d, h1) → ReLU → Linear(h1, h2) → ReLU → Linear(h2, 1)。"""
    def __init__(self, d_in, h1=256, h2=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, h1), torch.nn.ReLU(),
            torch.nn.Linear(h1, h2), torch.nn.ReLU(),
            torch.nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)
