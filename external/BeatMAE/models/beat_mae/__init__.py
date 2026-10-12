"""beat_mae：拍粒度 MAE（BeatMAE 三代共用一族代码，版本由 arch 参数区分）。

版本对照（与 results/pretrain/ 权重树 beatmae_v1|v2|v3b 对应，2026-09-05 注）：
    arch='v1'  → beatmae_v1   4.26M  v1 Flatten-Linear stem（仅加载旧 ckpt，如 mr065 轮）
    arch='v2'  → beatmae_v2   5.95M  现役主架构：整拍 CNN 多尺度 stem（d_model=256 对齐基线）
    v3b        → beatmae_v3b  同 v2  唯一变量 = ce_aux 离散辅助头（cfg ce_aux 字段开关，
                                      BeatMAE(n_ce_codes=64)；probe 按 sd 内 ce_head.* 自动识别）
加载时 beatmae_arch_from_sd() 依权重键前缀自动选架构，权重树目录名是版本真相源。
"""
from .beat_decoder import BeatWaveformDecoder, ConvBeatDecoder
from .beat_encoder import BeatTokenEncoder
from .beat_mae import BeatMAE

__all__ = ['BeatTokenEncoder', 'BeatWaveformDecoder', 'ConvBeatDecoder',
           'BeatMAE']
