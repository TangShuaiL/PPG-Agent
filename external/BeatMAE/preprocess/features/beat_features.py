"""40 维拍级形态学特征提取 —— 单拍 40 维装配（FEATURE_GROUPS 映射 + 舒张点覆盖）

本文件 2026-09-20 迁入本仓。代码为原项目自写，不派生自第三方；
第三方派生部分见 `third_party/vitalpy/`。
"""
# 从上游引擎裁取。40 维的键名映射表 FEATURE_GROUPS 是本模块的核心。
import numpy as np
import pandas as pd

from third_party.vitalpy.params import MINIMUM_PULSE_CYCLE
from third_party.vitalpy.features.utils import (
    extract_key_points, extract_key_temporal, extract_statistical,
    extract_spectral, branch_width_features)

from .keypoints import fix_diastolic_keypoints

DEFAULT_FS = 500

FEATURE_GROUPS = {
    "arterial_stiffness": {
        "label": "动脉硬度",
        "features": {
            "lasi":              "LASI",
            "slope_systolic":    "slp_s",
            "slope_systolic_n":  "slp_s_n",
            "slope_md":          "slp_md",
            "slope_md_n":        "slp_md_n",
            "pulse_width_50":    "bw50",
            "pulse_width_90":    "bw90",
            "bwr_50":            "bwr50",
        },
        "description": "大动脉弹性、血管硬度、收缩期动力学",
    },
    "wave_reflection": {
        "label": "波反射",
        "features": {
            "reflection_index":  "RI",
            "ni_dn":             "ni_dn",
            "ni_ip":             "ni_ip",
            "t_dn":              "t_dn",
            "nt_dn":             "nt_dn",
        },
        "description": "外周波反射、重搏切迹特征、血管树状态",
    },
    "pulse_morphology": {
        "label": "脉宽形态",
        "features": {
            "systolic_width_50":  "sbw50",
            "systolic_width_75":  "sbw75",
            "diastolic_width_50": "dbw50",
            "diastolic_width_75": "dbw75",
        },
        "description": "收缩期/舒张期各高度脉宽，与顺应性和外周阻力相关",
    },
    "perfusion": {
        "label": "灌注",
        "features": {
            "perfusion_index":       "p",
            "normalized_pulse_vol":  "mNPV",
            "ppg_characteristic":    "PPGK",
            "mean_intensity":        "Im",
            "pulse_amp":             "pulse_amp",  # AC component, computed below
        },
        "description": "外周血流灌注水平、脉容积",
    },
    "statistical_shape": {
        "label": "统计形态",
        "features": {
            "skewness":   "s",
            "kurtosis":   "k",
            "rms":        "rms",
            "crest_factor": "cf",
            "iqr":        "irq",
        },
        "description": "波形整体形态的统计描述",
    },
    "spectral": {
        "label": "频域特征",
        "features": {
            "freq_dominant_1":   "f1",
            "freq_dominant_2":   "f2",
            "freq_dominant_3":   "f3",
            "spectral_entropy":  "spectral_entropy",
            "wavelet_entropy":   "wavelet_entropy",
            "spectral_centroid": "spectral_centroid",
            "fsqi":              "fsqi",
        },
        "description": "频谱分布、信号复杂度、频域质量指标",
    },
}


_FEATURE_KEY_TO_GROUP = {}


def validate_template_lenient(template: np.ndarray, fs: float) -> bool:
    """Lenient validation: relaxed period check, skip trough-difference.

    Used for beats that received fragment merges (may have irregular
    morphology). Still enforces a reasonable upper bound to prevent
    over-merged beats.
    """
    period = len(template) / fs
    if period < MINIMUM_PULSE_CYCLE or period > 1.2:  # 50 bpm lower bound
        return False
    s_pos = float(np.argmax(template)) / float(len(template))
    if s_pos < 0.05 or s_pos >= 0.75:  # peak at edge or too late
        return False
    return True


def extract_beat_features_rich(
    beat: np.ndarray,
    fs: int = DEFAULT_FS,
    groups: list | str = "all",
) -> dict:
    """提取单个 beat 的全维度 Rich 特征集。

    从 VitalPy 的 ~385 维特征中精选 ~50 个最具生理意义
    的特征，按 7 个生理类别分组返回。

    Args:
        beat: 单个 beat 模板
        fs: 采样率
        groups: 要提取的特征组。
                "all" — 全部 7 组
                "vascular" — arterial_stiffness + wave_reflection + pulse_morphology
                "basic" — 基础组（perfusion + statistical_shape）
                或 list of group names

    Returns:
        dict: {
            "beat_info": {周期, 心率},
            "groups": {
                "arterial_stiffness": {"label": "...", "features": {...}},
                ...
            },
            "all_features_flat": {...}  # 扁平化版本，方便下游直接引用
        }
    """
    import pandas as pd

    template = pd.Series(beat)

    # 解析 groups 参数
    if groups == "all":
        active_groups = list(FEATURE_GROUPS.keys())
    elif groups == "vascular":
        active_groups = ["arterial_stiffness", "wave_reflection", "pulse_morphology"]
    elif groups == "basic":
        active_groups = ["perfusion", "statistical_shape"]
    elif isinstance(groups, list):
        active_groups = [g for g in groups if g in FEATURE_GROUPS]
    else:
        active_groups = list(FEATURE_GROUPS.keys())

    # 提取关键点
    valid, key_points = extract_key_points(template, fs)

    # Only call VitalPy functions whose output is actually used by FEATURE_GROUPS.
    # extract_temporal (KDE entropy, 74% of time) and extract_coeffs are NOT used.
    all_features = {}
    try:
        all_features.update(extract_key_temporal(template, fs, key_points, norm_method='z_score'))
    except Exception:
        pass
    try:
        all_features.update(branch_width_features(template, fs))
    except Exception:
        pass
    try:
        all_features.update(extract_statistical(template))
    except Exception:
        pass
    try:
        all_features.update(extract_spectral(template, fs))
    except Exception:
        pass

    # 心率
    period = len(beat) / fs
    heart_rate = round(60.0 / period, 1) if period > 0 else None

    # 手动补充：pulse_amp = AC component (systolic peak - beat minimum)
    s_idx = int(np.argmax(beat))
    pulse_amp_val = float(beat[s_idx] - np.min(beat))
    all_features['pulse_amp'] = pulse_amp_val

    result = {
        "beat_info": {
            "period_seconds": round(period, 3),
            "heart_rate_bpm": heart_rate,
            "sample_count": len(beat),
            "keypoints_valid": valid,
        },
        "groups": {},
        "all_features_flat": {},
    }

    for group_name in active_groups:
        group_info = FEATURE_GROUPS[group_name]
        group_features = {}

        for output_key, vitalpy_key in group_info["features"].items():
            value = all_features.get(vitalpy_key)
            if value is not None and isinstance(value, (int, float, np.floating)):
                value = float(value)
            group_features[output_key] = value
            result["all_features_flat"][output_key] = value

        result["groups"][group_name] = {
            "label": group_info["label"],
            "description": group_info["description"],
            "features": group_features,
        }

    return result


def extract_beat_features_fixed(beat: np.ndarray, fs: int = DEFAULT_FS,
                                 groups: str = "all",
                                 diastolic_search_ms: int = 200,
                                 notch_depth_threshold: float = 0.90) -> dict:
    """Extract rich features with CORRECTED diastolic keypoints.

    Wraps extract_beat_features_rich but fixes dn/ip/d before computing
    keypoint-dependent features (RI, LASI, ratio_sys_dia, etc.).
    """
    result = extract_beat_features_rich(beat, fs, groups=groups)
    feats = result.get("all_features_flat", {})
    if not feats:
        return result

    fixed = fix_diastolic_keypoints(beat, fs, diastolic_search_ms=diastolic_search_ms)
    if not fixed:
        return result

    s = int(np.argmax(beat))
    dn = fixed['dn']
    ip = fixed['ip']
    d = fixed['d']
    v = len(beat) - 1

    # Recompute keypoint-dependent features
    # Only compute these if the dicrotic notch amplitude is substantially below
    # systolic peak (indicating a real notch, not just a slight slope change)
    dn_depth = float(beat[dn] / beat[s]) if beat[s] != 0 else 1.0
    # For subtle/shoulder-type notches, amplitude 75-85% is normal
    # Only reject if notch is essentially at systolic level (no real decay)
    is_real_notch = (dn_depth < notch_depth_threshold)

    if is_real_notch:
        # Overwrite VitalPy values with corrected-keypoint versions
        feats['reflection_index'] = round(float(beat[ip] / beat[s]), 6)
        t_s = s / fs
        t_ip = ip / fs
        feats['lasi'] = round(1.0 / (t_ip - t_s), 4) if t_ip > t_s else 0.0
        feats['ni_dn'] = round(float(beat[dn] / beat[s]), 6)
        feats['ni_ip'] = round(float(beat[ip] / beat[s]), 6)
        feats['t_dn'] = round(float((dn - s) / fs), 6)
        feats['nt_dn'] = round(float((dn - s) / (v - s)), 6) if v > s else 0.0
    else:
        feats['reflection_index'] = None
        feats['lasi'] = None
        feats['ni_dn'] = None
        feats['ni_ip'] = None
        feats['t_dn'] = None
        feats['nt_dn'] = None

    # Intensity ratio s/d (keep as is, not in FEATURE_GROUPS but useful)
    feats['intensity_ratio_s_d'] = round(float(beat[s] / beat[d]), 6) if beat[d] != 0 else 0.0

    # Slope features need derivative, keep VitalPy's values for now
    # (they use normalized templates, so keypoint shift matters less)

    result["all_features_flat"] = feats
    return result


__all__ = ['FEATURE_GROUPS', 'extract_beat_features_rich', 'extract_beat_features_fixed']
