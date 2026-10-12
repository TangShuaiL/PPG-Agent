"""PaPaGei P / S 两个模型族（对比模型）。

论文与官方仓库把同一套骨干配上两种预训练配方，产出两个可比的模型：

    P  `ResNet1D`            `training_pospair.py`  NT-Xent(sVRI 箱)
    S  `ResNet1DMoE`(+3专家)  `training_mt.py`       0.6·NT-Xent + 0.2·L1(IPA)
                                                    + 0.2·L1(skewness)

官方 README 明确 `ResNet1DMoE` 是 PaPaGei-S 的模型。两者的输出头维数、
下游嵌入口径都相同，所以**下游只需一个 kind**：`embed_segment` 一律返回
`dense(out_emb)`，即官方 notebook 的 `outputs[0]`
（`use_projection=False`、`output_idx="0"`、512 维），与官方预训练权重对得上。

两者共用的架构不变量（与官方 config 逐字一致）：
`base_filters=32, kernel_size=3, stride=2, groups=1, n_block=18, n_classes=512`；
S 额外 `n_experts=3`。输入 1250 点（125 Hz × 10 s）。

**S 的辅助回归项不在本模块里**：`forward` 只负责把 MoE 头的原始输出交出去，
加权的三项损失组合在 `train/pretrain_papagei.py` 里写（与官方 train_step 逐行对应）。

P/S 的**构建与判别**集中在文件末尾的 `build_model` / `name_from_sd`：训练入口与
下游嵌入提取（`train/probe.py::load_frozen`）都走这两个函数，避免两处各写一份
判别逻辑而分叉。
"""
import torch
import torch.nn as nn

from .resnet1d import ResNet1D, ResNet1DMoE

# 官方两份训练脚本的 model_config，逐字一致
ARCH = dict(base_filters=32, kernel_size=3, stride=2, groups=1,
            n_block=18, n_classes=512)


def _as_bcl(seg: torch.Tensor) -> torch.Tensor:
    """(B, L) / (B, 1, L) → (B, 1, L)。"""
    if seg.dim() == 2:
        return seg.unsqueeze(1)
    if seg.dim() == 3:
        return seg
    raise ValueError(f'段张量应为 (B, L) 或 (B, 1, L)，实际 {tuple(seg.shape)}')


class PapageiP(nn.Module):
    """PaPaGei-P：ResNet1D + NT-Xent 对比头（无 MoE 专家）。"""

    def __init__(self, n_classes: int = 512):
        super().__init__()
        self.backbone = ResNet1D(in_channels=1, use_projection=False,
                                 **dict(ARCH, n_classes=n_classes))

    def forward(self, seg):
        """训练前向：seg (B, L) → (emb, out_emb)，emb = dense(out_emb) (B, 512)。

        对比损失吃 `emb`（官方 `embeddings, _ = model(signal)`）。
        """
        return self.backbone(_as_bcl(seg))

    def embed_segment(self, seg) -> torch.Tensor:
        """推理接口 → (B, 512)。"""
        return self.backbone(_as_bcl(seg))[0]


class PapageiS(nn.Module):
    """PaPaGei-S：ResNet1DMoE，对比头 + 两个 3 专家 MoE 回归头。"""

    def __init__(self, n_classes: int = 512, n_experts: int = 3):
        super().__init__()
        self.backbone = ResNet1DMoE(in_channels=1, n_experts=n_experts,
                                    use_projection=False,
                                    **dict(ARCH, n_classes=n_classes))

    def forward(self, seg):
        """训练前向 → (out_class, out_moe1, out_moe2, out)。

        out_class 吃对比损失；out_moe1 回归 IPA、out_moe2 回归 skewness
        （官方 `embeddings, ipa_pred, sqi_pred, _ = model(signal)`）。
        """
        return self.backbone(_as_bcl(seg))

    def embed_segment(self, seg) -> torch.Tensor:
        """推理接口 → (B, 512)。与 P 同口径（均为 dense(out_emb)）。"""
        return self.backbone(_as_bcl(seg))[0]


# ── P/S 判别与构建：训练入口与下游提取共用这一份，别再各写一个 if/else ──
def build_model(name: str) -> nn.Module:
    """`papagei_p` | `papagei_s` → 未初始化的模型。"""
    if name == 'papagei_p':
        return PapageiP()
    if name == 'papagei_s':
        return PapageiS()
    raise ValueError(f'未知模型: {name}（papagei_p | papagei_s）')


def name_from_sd(sd) -> str:
    """state_dict → 模型名（ckpt 没带 cfg 时的兜底判别）。

    只有 S 是 MoE（`backbone.expert_layers_1/2` + `gating_network_1/2`），P 的骨干
    后直接接 `dense`，所以看有没有专家层就够，不必逐键比对。
    """
    return 'papagei_s' if any(k.startswith('backbone.expert_layers_1') for k in sd) \
        else 'papagei_p'
