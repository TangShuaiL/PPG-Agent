"""HimaeAE：HiMAE 组装（对比模型）。

论文设定：输入 [-1,1]、patch P=5、遮 80%（随机/连续交替）、
MSE 只算被遮区域、输出 Tanh。
本库训练侧约定：输入逐段归一化、patch_mask (B, N) bool 传入，
损失只算被遮样本点（与主模型一致）。
"""
import torch
import torch.nn as nn

from .himae_decoder import HimaeDecoder
from .himae_encoder import CHANNELS, HimaeEncoder


class HimaeAE(nn.Module):
    def __init__(self, patch_len: int = 5, channels=CHANNELS):
        super().__init__()
        self.patch_len = patch_len
        self.encoder = HimaeEncoder(1, channels)
        self.decoder = HimaeDecoder(channels)
        self.bott_ch = channels[-1]

    def forward(self, seg, patch_mask=None):
        """seg: (B, L) 或 (B, 1, L)，L 需被 patch_len 整除。

        被遮 patch 的输入样本置 0（论文做法），输出全量重建波形；
        训练侧用 patch_mask 只对被遮样本点算损失。
        """
        if seg.dim() == 3:
            seg = seg.squeeze(1)
        B, L = seg.shape
        P = self.patch_len
        assert L % P == 0, f'段长 {L} 需被 patch_len {P} 整除'
        x = seg
        if patch_mask is not None:
            zero = (~patch_mask).to(seg.dtype)          # (B, N) 被遮=0
            x = x * zero.repeat_interleave(P, dim=1)

        skips, bott, lens = self.encoder(x.unsqueeze(1))
        recon = self.decoder(bott, skips, lens).squeeze(1)   # (B, L)
        return {'tokens': None, 'recon': recon, 'aux': None,
                'token_pad': None, 'latent': bott}

    def embed_segment(self, seg) -> torch.Tensor:
        """推理接口：瓶颈特征全局均值池化 → (B, 256)。"""
        if seg.dim() == 3:
            seg = seg.squeeze(1)
        _, bott, _ = self.encoder(seg.unsqueeze(1))
        return bott.mean(dim=-1)
