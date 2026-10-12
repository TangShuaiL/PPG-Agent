"""PaPaGei 的两个数据增强（对比模型）。

复现自官方 `references/papagei-foundation-model/augmentations.py`
（© 2024 Nokia, BSD-3-Clause-Clear）。官方的 `get_transformations` 提供六个模块，
但 P/S 两份训练脚本的 `prob_dictionary` 只把其中两个的概率开成非 0：

    P (training_pospair.py): g_p=0.25, c_p=0.25, 其余 0
    S (training_mt.py):      g_p=0.00, c_p=0.25, 其余 0

所以本库只移植这两个（Negation/TimeWarp/TimeFlip/Scaling 概率全 0，移植过来是死代码）。

**逐样本 vs 整批**：官方 dataset 每次只喂一个样本，`RandomApply(p)` 与 `TSRandomCropPad`
都是逐样本独立掷骰。本库是整批前向，故把两处都向量化成**逐样本**独立采样，语义等价。

**量纲**：官方 dataset 先 `Normalize(z-score)` 再 transform，所以 sigma=0.05 是
z-score 尺度下的噪声幅度；本库上游 `input_mode='raw_waveform'` 同样是 z-score 后的段
（`data/corpus_dataset.py`），量纲一致，无需换算。
"""
import numpy as np
import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """加性高斯噪声，官方 sigma=0.05。"""

    def __init__(self, sigma=0.05):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return x + torch.randn_like(x) * self.sigma


class TSRandomCropPad(nn.Module):
    """随机裁一段、窗外补 0（官方 magnitude=0.5）。

    官方采样：`λ ~ Beta(0.5, 0.5)` 再 `λ = max(λ, 1-λ)`。Beta(0.5,0.5) 是 U 形的，
    取 max 后 λ ∈ [0.5, 1] 且**偏向 1**，所以多数样本只丢一小段。窗户起点
    `randint(0, L - win)` 与官方同（上界不开，窗不会被贴到最右端）。
    """

    def __init__(self, magnitude=0.5):
        super().__init__()
        self.magnitude = magnitude

    def forward(self, x):
        m = self.magnitude
        if not m or m <= 0:
            return x
        B, L = x.shape[0], x.shape[-1]
        lam = np.random.beta(m, m, size=B)          # 必须只抽一次再取 max(λ,1-λ)，
        lam = np.maximum(lam, 1 - lam)              # 抽两次是不同的分布
        win = np.round(L * lam).astype(int)
        out = torch.zeros_like(x)
        for i in range(B):
            if win[i] >= L:
                out[i] = x[i]
                continue
            s = np.random.randint(0, L - win[i])
            out[i, ..., s:s + win[i]] = x[i, ..., s:s + win[i]]
        return out


class RandomApply(nn.Module):
    """逐样本以概率 p 施加 tf，否则原样放行（官方借用 torchvision 同名模块）。"""

    def __init__(self, tf, p):
        super().__init__()
        self.tf = tf
        self.p = p

    def forward(self, x):
        if self.p <= 0:
            return x
        sel = (torch.rand(x.shape[0], device=x.device) < self.p)
        return torch.where(sel.view(-1, *([1] * (x.dim() - 1))), self.tf(x), x)


def build_transform(g_p=0.0, c_p=0.0) -> nn.Module:
    """按官方 Compose 顺序组装：噪声在前、裁剪在后（官方列表里 crop 排最后）。

    g_p / c_p 用官方 `prob_dictionary` 的取值：P=(0.25, 0.25)、S=(0.0, 0.25)。
    """
    return nn.Sequential(
        RandomApply(GaussianNoise(sigma=0.05), p=g_p),
        RandomApply(TSRandomCropPad(magnitude=0.5), p=c_p),
    )
