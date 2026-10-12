"""共享组件：Pre-LN Transformer 块（V100/fp16 友好：无 bf16 依赖）。"""
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-LN Transformer 块。"""

    def __init__(self, d_model: int, nhead: int, dim_ff: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        dim_ff = dim_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerStack(nn.Module):
    def __init__(self, d_model: int, depth: int, nhead: int,
                 dim_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


def masked_mean_pool(x: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
    """token 序列 → 段嵌入。(B, N, D) → (B, D)；mask: (B, N) bool，True=保留。"""
    if mask is None:
        return x.mean(dim=1)
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
