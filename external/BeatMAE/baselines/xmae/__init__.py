"""xmae：xMAE 工作的 PPG encoder 复现（对比模型，只取 PPG 通路）。"""
from .xmae_mae import XmaePPGMAE
from .xmae_ppg_encoder import XmaePpgEncoder

__all__ = ['XmaePpgEncoder', 'XmaePPGMAE']
