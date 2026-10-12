"""MMR ViT 编码器（复现自 mmr.md 规格）。

输入为 DWT 系数图 (B, n_subbands, T)；patch (1, 25)：频域不切分，
时域每 25 点一个 token（kernel 跨全部子带）。
位置编码：固定 2D sine-cosine（频域轴高=1，退化为时域轴）。
官方 MMR-Small：8 层 / hidden 256 / 4 头 / FFN 1024。
被遮 patch 不参与编码（非对称 MAE，由组装层 gather 可见 token）。
"""
import math

import torch
import torch.nn as nn

from models.common import TransformerStack   # 绝对导入：本族已移出 models/


def get_2d_sincos_pos_embed(d_model: int, grid_h: int, grid_w: int):
    """标准 2D sine-cosine 位置编码 → (grid_h*grid_w, d_model)。"""
    assert d_model % 4 == 0, '2D sincos 需要 d_model 被 4 整除'
    gh = torch.arange(grid_h, dtype=torch.float)
    gw = torch.arange(grid_w, dtype=torch.float)
    grid = torch.stack(torch.meshgrid(gh, gw, indexing='ij'), dim=-1)  # (H,W,2)
    grid = grid.reshape(-1, 2)

    def enc1d(pos, dim):
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float)
                        * (-math.log(10000.0) / dim))
        ang = pos.unsqueeze(-1) * div                 # (n, dim/2)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (n, dim)

    pe_h = enc1d(grid[:, 0], d_model // 2)
    pe_w = enc1d(grid[:, 1], d_model // 2)
    return torch.cat([pe_h, pe_w], dim=-1)


class MmrViTEncoder(nn.Module):
    def __init__(self, seg_len: int = 1250, patch_len: int = 25,
                 n_subbands: int = 2, d_model: int = 256, depth: int = 8,
                 nhead: int = 4, dim_ff: int = 1024, dropout: float = 0.0):
        super().__init__()
        assert seg_len % patch_len == 0
        self.patch_len = patch_len
        self.n_patches = seg_len // patch_len
        self.d_model = d_model
        # patch (1, 25)：频域不切分（通道维吃全部子带），时域 25 点一拍；
        # 等价于对 (B, S, N, P) 在时域维按 patch 做 Conv2d
        self.patch_embed = nn.Conv1d(n_subbands, d_model,
                                     kernel_size=patch_len, stride=patch_len)
        pe = get_2d_sincos_pos_embed(d_model, 1, self.n_patches)
        self.register_buffer('pos', pe.unsqueeze(0))          # (1, N, d)
        self.stack = TransformerStack(d_model, depth, nhead, dim_ff, dropout)

    def embed_visible(self, coeff_map: torch.Tensor,
                      ids_keep: torch.Tensor) -> torch.Tensor:
        """系数图 → 可见 patch token。

        coeff_map: (B, S, T)；ids_keep: (B, Nk) long，升序可见序号。
        """
        B, S, T = coeff_map.shape
        N, P = self.n_patches, self.patch_len
        Nk = ids_keep.size(1)
        if Nk == 0:
            return self.pos[:, :0].expand(B, 0, self.d_model).contiguous()
        x = coeff_map.view(B, S, N, P)
        idx = ids_keep.view(B, 1, Nk, 1).expand(-1, S, -1, P)
        x_vis = x.gather(2, idx).reshape(B, S, Nk * P)      # (B, S, Nk*P)
        t = self.patch_embed(x_vis).transpose(1, 2)         # (B, Nk, d)
        t = t + self.pos[:, :Nk]
        return self.stack(t)

    def embed_all(self, coeff_map: torch.Tensor) -> torch.Tensor:
        """全量编码（推理用）→ (B, N, d)。"""
        t = self.patch_embed(coeff_map).transpose(1, 2)     # (B, N, d)
        return self.stack(t + self.pos)
