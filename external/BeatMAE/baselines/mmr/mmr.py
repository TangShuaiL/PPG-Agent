"""MMRMAE：小波系数图 + ViT 非对称 MAE（对比模型）。

重建目标是被遮 patch 的 DWT 系数（训练侧只对被遮位置算 MSE）。
接口与本库统一约定对齐：patch_mask (B, N) bool，True=被遮。
"""
import torch
import torch.nn as nn

from .mmr_decoder import MmrDecoder
from .mmr_dwt import build_coeff_map
from .mmr_encoder import MmrViTEncoder


class MMRMAE(nn.Module):
    def __init__(self, seg_len: int = 1250, patch_len: int = 25,
                 dwt_level: int = 3, drop_high: int = 2,
                 d_model: int = 256, depth: int = 8, nhead: int = 4,
                 dim_ff: int = 1024, d_dec: int = 192, dec_depth: int = 2):
        super().__init__()
        self.seg_len = seg_len
        self.patch_len = patch_len
        self.n_patches = seg_len // patch_len
        self.dwt_level = dwt_level
        self.drop_high = drop_high
        # 3 级 Haar 排除 2 个高频细节带 → 保留子带数 = level - drop_high + 1
        self.n_subbands = dwt_level - drop_high + 1
        self.encoder = MmrViTEncoder(seg_len, patch_len, self.n_subbands,
                                     d_model, depth, nhead, dim_ff)
        self.decoder = MmrDecoder(d_model, d_dec, dec_depth, 4,
                                  self.n_patches, self.n_subbands, patch_len)

    def _ids_from_mask(self, patch_mask: torch.Tensor):
        """bool 掩码 → (ids_keep, ids_restore)。要求每样本被遮数一致。"""
        B, N = patch_mask.shape
        noise = patch_mask.to(torch.float) + torch.rand(
            B, N, device=patch_mask.device) * 0.5
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        n_mask = int(patch_mask[0].sum())
        assert int(patch_mask.sum(dim=1).min()) == n_mask and \
            int(patch_mask.sum(dim=1).max()) == n_mask, \
            'MMR 要求 batch 内每样本被遮 patch 数一致'
        ids_keep = torch.sort(ids_shuffle[:, :N - n_mask], dim=1).values
        return ids_keep, ids_restore

    def forward(self, seg, patch_mask=None):
        """seg: (B, L) 或 (B, 1, L)。

        返回 dict 中：
            recon:       (B, N, n_subbands*patch_len) 系数预测
            coeff_target:(B, N, n_subbands*patch_len) 真值（训练算损失用）
        """
        if seg.dim() == 3:
            seg = seg.squeeze(1)
        B = seg.shape[0]
        coeff_map = build_coeff_map(seg, self.dwt_level, self.drop_high)
        if patch_mask is None:
            ids_keep = (torch.arange(self.n_patches, device=seg.device)
                        .unsqueeze(0).expand(B, -1))
            ids_restore = ids_keep
        else:
            ids_keep, ids_restore = self._ids_from_mask(patch_mask)
        tokens_vis = self.encoder.embed_visible(coeff_map, ids_keep)
        recon = self.decoder(tokens_vis, ids_restore)
        # patchify 真值系数：(B, S, T) → (B, N, S*P)
        S, P = self.n_subbands, self.patch_len
        target = (coeff_map.view(B, S, self.n_patches, P)
                  .permute(0, 2, 1, 3).reshape(B, self.n_patches, S * P))
        return {'tokens': tokens_vis, 'recon': recon, 'aux': None,
                'token_pad': None, 'coeff_target': target,
                'ids_restore': ids_restore}

    def embed_segment(self, seg) -> torch.Tensor:
        """推理接口：全量编码 → token 均值 → (B, d_model)。"""
        if seg.dim() == 3:
            seg = seg.squeeze(1)
        coeff_map = build_coeff_map(seg, self.dwt_level, self.drop_high)
        return self.encoder.embed_all(coeff_map).mean(dim=1)
