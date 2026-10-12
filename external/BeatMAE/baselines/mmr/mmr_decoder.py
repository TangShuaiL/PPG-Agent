"""MMR 轻量解码器（复现自 mmr.md 规格）。

官方：2 层 / hidden 192 / 4 头。输入 = 可见 token（投影 256→192）
+ 被遮位补可学习 mask token，按 ids_restore 还原顺序；
输出头重建每 patch 的小波系数（n_subbands × patch_len）。
"""
import torch
import torch.nn as nn

from models.common import TransformerStack   # 绝对导入：本族已移出 models/


class MmrDecoder(nn.Module):
    def __init__(self, d_enc: int = 256, d_dec: int = 192, depth: int = 2,
                 nhead: int = 4, n_patches: int = 50,
                 n_subbands: int = 2, patch_len: int = 25):
        super().__init__()
        self.n_subbands = n_subbands
        self.patch_len = patch_len
        self.in_proj = nn.Linear(d_enc, d_dec)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_dec))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.stack = TransformerStack(d_dec, depth, nhead)
        self.head = nn.Linear(d_dec, n_subbands * patch_len)

    def forward(self, tokens_vis: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        """tokens_vis: (B, Nk, d_enc) → 系数重建 (B, N, n_subbands*patch_len)。"""
        B, Nk, _ = tokens_vis.shape
        N = ids_restore.size(1)
        h = self.in_proj(tokens_vis)
        D = h.shape[-1]
        if Nk != N:
            h = torch.cat([h, self.mask_token.expand(B, N - Nk, D)], dim=1)
            h = torch.gather(h, 1,
                             ids_restore.unsqueeze(-1).expand(-1, -1, D))
        h = h + self.pos[:, :N]
        h = self.stack(h)
        return self.head(h)
