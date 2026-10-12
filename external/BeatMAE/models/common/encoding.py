"""共享组件：位置/时间编码。

- TimeEncoding:       连续时间（秒）→ 正弦编码，注意力借此感知 RR 间期
- PositionalEncoding: 标准正弦序列位置编码（patch 序列用）
"""
import math

import torch
import torch.nn as nn


class TimeEncoding(nn.Module):
    """连续时间（秒）→ 正弦编码。注意力借此感知 RR 间期。"""

    def __init__(self, d_model: int, max_period: float = 10.0):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        half = d_model // 2
        # 频率从 1/max_period Hz 到 ~8 Hz 对数分布（覆盖呼吸~心率频带）
        freqs = torch.exp(torch.linspace(
            math.log(1.0 / max_period), math.log(8.0), half))
        self.register_buffer('freqs', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (...) 秒 → (..., d_model)"""
        ang = t.unsqueeze(-1) * self.freqs * (2 * math.pi)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class PositionalEncoding(nn.Module):
    """标准可加正弦序列位置编码（patch 序列用）。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        return x + self.pe[:x.size(1)].unsqueeze(0)
