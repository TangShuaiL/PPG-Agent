"""xMAE 的 PPG 单模态通路（复现自 xMAE/utils/model_arch/xmae.py 的 SinglePath）。

原作为 PPG→ECG 跨模态重建；此处只取 PPG encoder 侧作为对比模型，
重建任务退化为 PPG 自重建（官方 cfg 中 mwm_ppg 即此形态）。

官方超参：
fs=100, seq=1000, patch=40, d_model=256,
nhead=8, depth_ppg=2, stem_ch=32, latent_dim=256, dropout=0.1。
本库语料为 125Hz/1250 点：seq_len 默认 1250，patch_len 默认 50
（保持 0.4s 物理感受野不变）。
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU())


class InceptionBlock1D(nn.Module):
    """并行小卷积核 + 1x1 融合（原作结构）。"""

    def __init__(self, in_ch, out_ch, ks=(3, 5, 7)):
        super().__init__()
        mid = out_ch // len(ks)
        mids = [mid] * (len(ks) - 1) + [out_ch - mid * (len(ks) - 1)]
        self.branches = nn.ModuleList(
            [ConvBNAct(in_ch, m, k=k) for m, k in zip(mids, ks)])
        self.fuse = ConvBNAct(out_ch, out_ch, k=1, p=0)

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class UNetHalfInceptionStem1D(nn.Module):
    """Down: L→L/2→L/4（Inception 块）；Up-fuse: L/4→L/2→L（FPN 式）。

    输出长度 == 输入长度，含输入残差。
    """

    def __init__(self, in_ch=1, out_ch=32, widths=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = widths
        self.enc1 = nn.Sequential(InceptionBlock1D(in_ch, c1),
                                  InceptionBlock1D(c1, c1))
        self.down1 = ConvBNAct(c1, c2, k=3, s=2)      # L/2
        self.enc2 = nn.Sequential(InceptionBlock1D(c2, c2),
                                  InceptionBlock1D(c2, c2))
        self.down2 = ConvBNAct(c2, c3, k=3, s=2)      # L/4
        self.enc3 = nn.Sequential(InceptionBlock1D(c3, c3),
                                  InceptionBlock1D(c3, c3))
        self.lat2 = nn.Conv1d(c2, c2, 1, bias=False)
        self.lat1 = nn.Conv1d(c1, c1, 1, bias=False)
        self.red3to2 = nn.Conv1d(c3, c2, 1, bias=False)
        self.red2to1 = nn.Conv1d(c2, c1, 1, bias=False)
        self.smooth2 = ConvBNAct(c2, c2, k=3)
        self.smooth1 = ConvBNAct(c1, c1, k=3)
        self.head = ConvBNAct(c1, out_ch, k=3)
        self.skip = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.skip_bn = (nn.BatchNorm1d(out_ch)
                        if in_ch != out_ch else nn.Identity())

    @staticmethod
    def _upsample_to(x, L):
        return F.interpolate(x, size=L, mode='linear', align_corners=False)

    def forward(self, x):                            # (B, in_ch, L)
        e1 = self.enc1(x)                            # (B, c1, L)
        e2 = self.enc2(self.down1(e1))               # (B, c2, L/2)
        e3 = self.enc3(self.down2(e2))               # (B, c3, L/4)
        t2 = self.lat2(e2) + self._upsample_to(self.red3to2(e3), e2.shape[-1])
        t2 = self.smooth2(t2)                        # (B, c2, L/2)
        t1 = self.lat1(e1) + self._upsample_to(self.red2to1(t2), e1.shape[-1])
        t1 = self.smooth1(t1)                        # (B, c1, L)
        return self.head(t1) + self.skip_bn(self.skip(x))


class XmaePpgEncoder(nn.Module):
    """PPG 通路：UNet stem → patchify → Transformer（等价原作 SinglePath）。"""

    def __init__(self, seq_len: int = 1250, patch_len: int = 50,
                 d_model: int = 256, nhead: int = 8, depth: int = 2,
                 stem_ch: int = 32, proj_dim: int = 256, dropout: float = 0.1,
                 widths: Tuple[int, int, int] = (32, 64, 128)):
        super().__init__()
        assert seq_len % patch_len == 0
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.d_model = d_model

        self.stem = UNetHalfInceptionStem1D(1, stem_ch, widths)
        self.patcher = nn.Conv1d(stem_ch, d_model, patch_len, stride=patch_len)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList([
            _XmaeBlock(d_model, nhead, dropout) for _ in range(depth)])
        self.proj = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, proj_dim))
        self.dec_blk = _XmaeBlock(d_model, nhead, dropout, mlp_ratio=2.0)
        self.deconv = nn.ConvTranspose1d(d_model, 1, patch_len, stride=patch_len)
        self.mask_loc_table = nn.Embedding(self.n_patches, d_model)
        nn.init.xavier_uniform_(self.mask_loc_table.weight)

    # ------------------------------------------------ 编码（仅可见 patch）
    def encode_visible(self, x_B1L: torch.Tensor,
                       ids_keep: torch.Tensor) -> torch.Tensor:
        """聚集可见波形段 → stem → patch token → Transformer。

        ids_keep: (B, Nk) long，升序的可见 patch 序号。
        """
        B, _, L = x_B1L.shape
        P, N = self.patch_len, self.n_patches
        Nk = ids_keep.size(1)
        if Nk == 0:                                  # 全遮（实际不会发生）
            return self.pos[:, :0].expand(B, 0, self.d_model).contiguous()
        x_bp = x_B1L.view(B, 1, N, P)
        idx = ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, P)
        x_vis = x_bp.gather(2, idx).reshape(B, 1, Nk * P)
        t = self.patcher(self.stem(x_vis)).transpose(1, 2)   # (B, Nk, D)
        t = t + self.pos[:, :Nk]
        for blk in self.blocks:
            t = blk(t)
        return t

    # ------------------------------------------------ 解码（补 mask token）
    def decode_full(self, tokens_vis: torch.Tensor, ids_restore: torch.Tensor,
                    mask_token: torch.Tensor) -> torch.Tensor:
        """(B,Nk,D) → 补 mask token 还原 N → 1 层解码 → 波形 (B,1,L)。"""
        B, Nk, D = tokens_vis.shape
        N = ids_restore.size(1)
        if Nk != N:
            dec = torch.cat([tokens_vis,
                             mask_token.expand(B, N - Nk, D)], dim=1)
            dec = torch.gather(dec, 1,
                               ids_restore.unsqueeze(-1).expand(-1, -1, D))
            dec = dec + self.pos[:, :N]
        else:
            dec = tokens_vis
        dec = self.dec_blk(dec)
        return self.deconv(dec.transpose(1, 2))      # (B, 1, L)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """token 均值 → 投影，等价原作出嵌入方式。(B, N, D) → (B, proj_dim)"""
        return self.proj(tokens.mean(dim=1))


class _XmaeBlock(nn.Module):
    """Pre-LN Transformer 块（原作 TransformerEncoderBlock/DecoderLite 的统一版）。"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)), nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model), nn.Dropout(dropout))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))
