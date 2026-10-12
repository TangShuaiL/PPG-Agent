"""40 维拍级形态学特征提取（原始 PPG 段 → 关键点 + 40 维）。

见 `scripts/extract_beat_features.py` 与 README 的说明。
"""
from .beat_features import FEATURE_GROUPS
from .segment import extract_segment_features, process_ppg_segment

__all__ = ['FEATURE_GROUPS', 'extract_segment_features', 'process_ppg_segment']
