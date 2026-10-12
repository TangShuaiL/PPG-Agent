"""PPG-MAE 模型库 —— **本文模型侧**。

目录组织（一族一目录，文件名对应角色）：
    common/      共享组件：编码 / Transformer / 区域定义 / 损失
    beat_mae/    拍粒度 MAE（v1 主架构）：encoder/decoder/组装
    probe_heads/ 下游探针头（段级流式注意力头 + 岭回归/MLP 头）

**对比模型不在本目录**：HiMAE / MMR / xMAE / PaPaGei 已于 2026-09-20 移到
`baselines/`。它们是按论文与官方仓库的**独立重实现**（非作者官方代码），
与本文模型分开摆放，既让主干突出，也让「哪些是本文贡献」一目了然。
需要时改为 `from baselines import HimaeAE, ...`。

历史：region_mae/ 与 patch_mae/ 两族已于 2026-09-10 删除——两者只在早期
基线对比里跑过，未进任何主表与权重树（results/pretrain/ 无对应目录），
全仓除本文件与 tests 冒烟外零引用。注意**区域掩码机制是另一回事**：
models/common/regions.py + data/region_mask.py 仍然在用
（config/base/beatmae.yaml 的 mask.region_share=0.6），未受影响。
"""
from .beat_mae import (BeatMAE, BeatTokenEncoder, BeatWaveformDecoder,
                       ConvBeatDecoder)
from .common import (KP_A, KP_B, KP_D, KP_DN, KP_IP, KP_MD, KP_O, KP_S, KP_V,
                     N_REGIONS, R2_HALF_WIDTH_MS, REGION_BOUNDS,
                     AuxFeatureHead, PositionalEncoding, TimeEncoding,
                     TransformerBlock, TransformerStack, aux_loss, beat_codes,
                     ce_loss, masked_mean_pool, masked_mse, n_params,
                     region_slices)

__all__ = [
    # common
    'N_REGIONS', 'REGION_BOUNDS', 'R2_HALF_WIDTH_MS',
    'KP_O', 'KP_A', 'KP_MD', 'KP_B', 'KP_S', 'KP_DN', 'KP_IP', 'KP_D', 'KP_V',
    'TimeEncoding', 'PositionalEncoding',
    'TransformerBlock', 'TransformerStack', 'masked_mean_pool',
    'region_slices', 'AuxFeatureHead', 'masked_mse', 'aux_loss', 'n_params',
    # beat_mae
    'BeatTokenEncoder', 'BeatWaveformDecoder', 'ConvBeatDecoder', 'BeatMAE',
    # 对比模型见 baselines/（独立子包，不在此重复导出）
]
