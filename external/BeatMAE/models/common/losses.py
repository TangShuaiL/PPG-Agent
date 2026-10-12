"""共享组件：辅助特征头与损失。

损失约定（v1_plan §4.3/§5.3，v3_plan V3-B）：
- L_recon = MSE(重建, 原波形)，只算被遮样本点
- L_aux   = ICC 加权 SmoothL1（拍级代表特征预测），训练期存在、推理丢弃
- L_ce    = 类别加权 CE（V3-B 拍形态离散码预测），只算可见拍（被遮拍 token
  是 mask_token，其形态码预测属 BEiT 式上下文预测，另行消融）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxFeatureHead(nn.Module):
    """拍/区域 token → 代表特征向量（~20 维，z 参数化）。"""

    def __init__(self, d_model: int, n_feats: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_feats))

    def forward(self, tokens):
        return self.mlp(tokens)


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """mask: 与 pred 同形，True=计入损失。无有效点时返回 0。"""
    m = mask.to(pred.dtype)
    n = m.sum()
    if n == 0:
        return pred.sum() * 0.0
    return ((pred - target) ** 2 * m).sum() / n


def aux_loss(pred: torch.Tensor, target: torch.Tensor,
             valid: torch.Tensor, icc_weights: torch.Tensor,
             beat_w_masked: float = 1.0, beat_w_visible: float = 0.3,
             beat_mask: torch.Tensor | None = None) -> torch.Tensor:
    """SmoothL1，ICC 加权，被遮拍/可见拍双权重。

    pred:     (B, N, F)      target: (B, N, F)（已截断 ±5）
    valid:    (B, N) bool，True=该 token 有特征目标（feat_valid）
    icc_weights: (F,)
    beat_mask: (B, N) bool，True=被遮（给 1.0 权重，其余 0.3）；None=全 1.0
    """
    diff = F.smooth_l1_loss(pred, target, reduction='none')   # (B, N, F)
    diff = (diff * icc_weights.to(diff.dtype)).mean(dim=-1)   # (B, N)
    if beat_mask is not None:
        wtok = torch.where(beat_mask, beat_w_masked, beat_w_visible)
        diff = diff * wtok.to(diff.dtype)
    m = valid.to(diff.dtype)
    n = m.sum()
    if n == 0:
        return diff.sum() * 0.0
    return (diff * m).sum() / n


def beat_codes(beats: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """V3-B 目标分配：逐拍 z-norm（去幅值尺度只留形态）后取最近质心。

    beats: (B, N, L) 原始拍波形（含 padding 拍，其码任意、由 ce_loss 的
    valid 掩掉）；centroids: (C, L)（码本脚本在同一 z-norm 口径下拟合）。
    返回 (B, N) long。
    """
    z = (beats - beats.mean(-1, keepdims=True)) \
        / beats.std(-1, keepdims=True).clamp_min(1e-6)
    z = z.reshape(-1, beats.shape[-1]).float()
    d = torch.cdist(z, centroids.float())
    return d.argmin(dim=1).reshape(beats.shape[:-1])


def ce_loss(logits: torch.Tensor, targets: torch.Tensor,
            valid: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """类别加权 CE，只算 valid token（缓解形态簇严重类不平衡）。

    logits: (B, N, C)   targets: (B, N) long   valid: (B, N) bool
    class_weights: (C,)（码本脚本给出的逆频率权重，均值归一为 1）
    """
    flat = logits.reshape(-1, logits.shape[-1]).float()
    tgt = targets.reshape(-1)
    loss = F.cross_entropy(flat, tgt, reduction='none')
    w = valid.reshape(-1).to(flat.dtype) * class_weights.to(flat.dtype)[tgt]
    n = w.sum()
    if n == 0:
        return flat.sum() * 0.0
    return (loss * w).sum() / n
