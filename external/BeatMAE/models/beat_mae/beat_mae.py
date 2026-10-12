"""BeatMAE：BeatTokenEncoder + 拍波形解码器（v1 主架构）。

掩码约定：被遮拍整体替换为 mask token；损失可细化到被遮区域样本点
（sample_mask: (B, K, L) bool，True=计入损失）。
"""
import torch
import torch.nn as nn

from ..common import AuxFeatureHead, masked_mean_pool
from .beat_decoder import BeatWaveformDecoder, ConvBeatDecoder
from .beat_encoder import BeatTokenEncoder


def beatmae_arch_from_sd(sd) -> str:
    """从 state_dict 识别编码器架构版本：v1 stem 键形如 encoder.stem.0.*；
    v2 为 encoder.stem1.0.* 等。用于加载旧 checkpoint（如 mr065 轮）时自动兼容。"""
    return 'v1' if any(k.startswith('encoder.stem.') for k in sd) else 'v2'


class BeatMAE(nn.Module):
    def __init__(self, beat_len: int = 128, d_model: int = 256, depth: int = 6,
                 nhead: int = 8, dec_kind: str = 'transformer',
                 d_dec: int = 128, dec_depth: int = 2, n_aux_feats: int = 0,
                 n_ce_codes: int = 0, arch: str = 'v2',
                 dim_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.encoder = BeatTokenEncoder(beat_len, d_model, depth, nhead, arch=arch,
                                        dim_ff=dim_ff, dropout=dropout)
        if dec_kind == 'transformer':
            self.decoder = BeatWaveformDecoder(d_model, beat_len, d_dec, dec_depth)
        elif dec_kind == 'conv':
            self.decoder = ConvBeatDecoder(d_model, beat_len)
        else:
            raise ValueError(dec_kind)
        self.aux_head = AuxFeatureHead(d_model, n_aux_feats) if n_aux_feats else None
        # V3-B 拍形态离散辅助头：线性读 token → 码本簇 id（训练期存在、推理丢弃）
        self.ce_head = nn.Linear(d_model, n_ce_codes) if n_ce_codes else None

    def forward(self, beats, beat_pad_mask=None, onset_times=None, side=None,
                beat_mask=None, region_mask=None):
        tokens, pad = self.encoder(beats, beat_pad_mask, onset_times, side,
                                   beat_mask, region_mask)
        recon = self.decoder(tokens, pad)                   # (B, K, L)
        aux = self.aux_head(tokens) if self.aux_head else None
        ce = self.ce_head(tokens) if self.ce_head else None
        return {'tokens': tokens, 'recon': recon, 'aux': aux,
                'ce_logits': ce, 'token_pad': pad}

    def embed_segment(self, beats, beat_pad_mask=None, onset_times=None,
                      side=None) -> torch.Tensor:
        """推理接口：段嵌入（掩码池化去掉 padding 拍）。"""
        tokens, pad = self.encoder(beats, beat_pad_mask, onset_times, side)
        return masked_mean_pool(tokens, ~pad)
