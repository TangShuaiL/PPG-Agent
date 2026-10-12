"""beat_mae 解码器：拍波形重建头。

约定：解码器只负责产出全量重建波形，掩码损失由训练侧统一计算。
"""
import torch.nn as nn

from ..common import TransformerStack


class BeatWaveformDecoder(nn.Module):
    """beat-token → 每拍 L 点波形。

    tokens: (B, K, d_enc)（被遮拍已是 encoder 处理过的 mask token）
    输出:   (B, K, beat_len)
    """

    def __init__(self, d_enc: int, beat_len: int = 128, d_dec: int = 128,
                 depth: int = 2, nhead: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(d_enc, d_dec)
        self.stack = TransformerStack(d_dec, depth, nhead)
        self.head = nn.Linear(d_dec, beat_len)

    def forward(self, tokens, token_pad_mask=None):
        h = self.in_proj(tokens)
        h = self.stack(h, key_padding_mask=token_pad_mask)
        return self.head(h)


class ConvBeatDecoder(nn.Module):
    """拍 token → 转置卷积上采样重建波形（HiMAE 风格轻量替代）。

    tokens: (B, K, d_enc)；输出: (B, K, beat_len)
    """

    def __init__(self, d_enc: int, beat_len: int = 128, base_ch: int = 64):
        super().__init__()
        # beat_len 需被 16 整除（4 次 2x 上采样）；128 → 初始长度 8
        assert beat_len % 16 == 0
        self.init_len = beat_len // 16
        self.base_ch = base_ch
        self.proj = nn.Linear(d_enc, base_ch * self.init_len)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(base_ch, base_ch // 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(base_ch // 2, base_ch // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(base_ch // 4, base_ch // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(base_ch // 4, 1, 4, stride=2, padding=1))

    def forward(self, tokens, token_pad_mask=None):
        B, K, _ = tokens.shape
        h = self.proj(tokens).reshape(B * K, self.base_ch, self.init_len)
        x = self.up(h)                                    # (B*K, 1, beat_len)
        return x.reshape(B, K, -1)
