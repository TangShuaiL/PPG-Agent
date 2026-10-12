"""MMR 的 DWT 前处理（复现自 mmr.md 规格）。

Haar 小波 3 级分解，纯 torch 实现（batch 友好，无 PyWavelets 依赖）；
系数图构建：各子带 nearest 上采样回原长后按频率高→低堆叠。
论文 Appendix A.6.1：排除频率过高的前两个细节带（cD1、cD2），
只保留 [cD3, cA3]（n_subbands=2）。
"""
import math

import torch

SQRT2 = math.sqrt(2.0)


def haar_dwt(x: torch.Tensor, level: int = 3):
    """1D Haar 离散小波变换（正交归一）。

    Args:
        x: (B, T) 时域信号
        level: 分解级数
    Returns:
        cA: (B, T//2^level) 近似系数
        details: list，从粗到细 [cD_level, ..., cD_1]
    """
    details = []
    for _ in range(level):
        if x.shape[-1] % 2 == 1:                    # 奇数长丢弃末尾样本
            x = x[..., :-1]
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        cA = (x0 + x1) / SQRT2
        cD = (x0 - x1) / SQRT2
        details.append(cD)
        x = cA
    return x, details                               # details: [cD_L, ..., cD_1]


def _upsample_to(band: torch.Tensor, T: int) -> torch.Tensor:
    """nearest（zero-order）插值到原长，等价论文做法。"""
    n = band.shape[-1]
    rep = T // n
    out = band.repeat_interleave(rep, dim=-1)
    if out.shape[-1] < T:                           # 整除余数补最后一值
        out = torch.cat([out, out[..., -1:].expand(*out.shape[:-1],
                                                   T - out.shape[-1])], dim=-1)
    return out[..., :T]


def build_coeff_map(seg: torch.Tensor, level: int = 3,
                    drop_high: int = 2) -> torch.Tensor:
    """段波形 → DWT 系数图 (B, n_subbands, T)。

    子带按频率高→低堆叠；drop_high 个最高频细节带被排除。
    """
    B, T = seg.shape
    cA, details = haar_dwt(seg, level)              # details: 粗→细
    bands = details[:len(details) - drop_high] + [cA]   # 高频→低频
    return torch.stack([_upsample_to(b, T) for b in bands], dim=1)
