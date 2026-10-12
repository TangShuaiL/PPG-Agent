"""common：跨模型族共享组件（编码、Transformer、区域、损失）。"""
from .encoding import PositionalEncoding, TimeEncoding
from .losses import (AuxFeatureHead, aux_loss, beat_codes, ce_loss,
                     masked_mse)
from .regions import (KP_A, KP_B, KP_D, KP_DN, KP_IP, KP_MD, KP_O, KP_S, KP_V,
                      N_REGIONS, R2_HALF_WIDTH_MS, REGION_BOUNDS, region_slices)
from .transformer import TransformerBlock, TransformerStack, masked_mean_pool
from .util import n_params

__all__ = [
    'PositionalEncoding', 'TimeEncoding',
    'AuxFeatureHead', 'aux_loss', 'masked_mse',
    'KP_O', 'KP_A', 'KP_MD', 'KP_B', 'KP_S', 'KP_DN', 'KP_IP', 'KP_D', 'KP_V',
    'N_REGIONS', 'R2_HALF_WIDTH_MS', 'REGION_BOUNDS', 'region_slices',
    'TransformerBlock', 'TransformerStack', 'masked_mean_pool',
    'n_params',
]
