"""XmaePPGMAE：xMAE PPG 通路 + PPG 自重建（对比模型）。

原作任务是用 PPG 重建被遮的 ECG；不含 ECG 时退化为 PPG 自身
掩码重建（官方 cfg 的 mwm_ppg 形态）。掩码接口与本库统一约定对齐：
patch_mask (B, N) bool，True=被遮。

注意：原作要求每样本被遮 patch 数一致（ids_keep 为规整张量），
训练侧按固定比例采样即可满足。
"""
import torch
import torch.nn as nn

from .xmae_ppg_encoder import XmaePpgEncoder


class XmaePPGMAE(nn.Module):
    def __init__(self, seq_len: int = 1250, patch_len: int = 50,
                 d_model: int = 256, nhead: int = 8, depth: int = 2,
                 stem_ch: int = 32, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = XmaePpgEncoder(seq_len, patch_len, d_model, nhead,
                                      depth, stem_ch, proj_dim, dropout)
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _ids_from_mask(self, patch_mask: torch.Tensor):
        """bool 掩码 → (ids_keep, ids_restore)，等价 MAE 的 shuffle 索引。"""
        B, N = patch_mask.shape
        noise = patch_mask.to(torch.float) + torch.rand(
            B, N, device=patch_mask.device) * 0.5      # 被遮位置噪声更大
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        n_mask = int(patch_mask[0].sum())
        assert int(patch_mask.sum(dim=1).min()) == n_mask and \
            int(patch_mask.sum(dim=1).max()) == n_mask, \
            'xMAE 要求 batch 内每样本被遮 patch 数一致'
        ids_keep = torch.sort(ids_shuffle[:, :N - n_mask], dim=1).values
        return ids_keep, ids_restore

    def forward(self, seg, patch_mask=None):
        """seg: (B, L) 或 (B, 1, L)；返回统一 dict。"""
        if seg.dim() == 3:
            seg = seg.squeeze(1)
        B = seg.shape[0]
        x = seg.unsqueeze(1)                            # (B, 1, L)
        if patch_mask is None:
            ids_keep = (torch.arange(self.n_patches, device=seg.device)
                        .unsqueeze(0).expand(B, -1))
            ids_restore = ids_keep
        else:
            ids_keep, ids_restore = self._ids_from_mask(patch_mask)
        tokens_vis = self.encoder.encode_visible(x, ids_keep)
        recon = self.encoder.decode_full(tokens_vis, ids_restore,
                                         self.mask_token)   # (B, 1, L)
        # 全量 token 位（被遮处为 mask token 占位）供下游/对齐用
        return {'tokens': tokens_vis, 'recon': recon.squeeze(1),
                'aux': None, 'token_pad': None, 'ids_restore': ids_restore}

    def embed_segment(self, seg) -> torch.Tensor:
        """推理接口：无掩码全量编码 → 段嵌入 (B, proj_dim)。"""
        if seg.dim() == 3:
            seg = seg.squeeze(1)
        B = seg.shape[0]
        ids = (torch.arange(self.n_patches, device=seg.device)
               .unsqueeze(0).expand(B, -1))
        tokens = self.encoder.encode_visible(seg.unsqueeze(1), ids)
        return self.encoder.embed(tokens)
