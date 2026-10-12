# ============================================================================
# 本文件派生自 VitalPy
#
#   VitalPy: A Vital Signal Analysis Package
#   Copyright (c) SCAI-Lab, ETH Zürich
#   作者: Ana Cisnal, Yanke Li, Bertram Fuchs, Mehdi Ejtehadi,
#         Robert Riener, Diego Paez-Granados
#   上游: https://github.com/SCAI-Lab/VitalPy
#   许可: GNU General Public License v3.0（见仓库根 LICENSE）
#   论文: A. Cisnal et al., "Robust Feature Selection for BP Estimation in
#         Multiple Populations: Towards Cuffless Ambulatory BP Monitoring,"
#         IEEE J. Biomed. Health Inform., doi: 10.1109/JBHI.2024.3411693
#
# 【修改声明 — GPL-3.0 §5】
#   本文件由 BeatMAE 项目转承自 VitalPy，并于 2026-09-20 迁入本仓
#   的 third_party/vitalpy/。相对于 VitalPy 原件的改动：25 行（import 路径；删除 TEMPORAL_FEATURES 列表）
#   完整改动清单与致谢见同目录 README.md。
# ============================================================================
from third_party.vitalpy.features.statistical_features import *
from third_party.vitalpy.features.spectral_features import *

"""
Set the features to be calculated.
"""

# List of x% of pulse height of PPG for calculating branch width (BW) and branch width ratio (BWR)
X_VALUES_BW_FEATURES = [10, 25, 33, 50, 66, 75, 90]


# TEMPORAL_FEATURES 已删除：它引用的 18 个函数来自 temporal_features.py，
# 而该模块不在本闭包内（extract_key_temporal 是自包含的，不用它们）。
# 保留此说明是为了让读者知道上游有此一组，见 third_party/vitalpy/README.md。
STATISTIC_FEATURES = [
    (skewness, 's', True, 'skewness',),
    (kurtosis, 'k', True, 'kurtosis',),
    (mav, 'mav', True, 'mean absolute value'),
    (median, 'median', True, 'median',),
    (mad, 'mad', True, 'mean absolute deviation',),
    (med_ad, 'med_ad', True, 'median absolute deviation',),
    (rms, 'rms', True, 'rms',),
    (sd, 'sd', True, 'Standard Deviation',),
    (shape_factor, 'sf', True, 'Shape factor',),
    (impulse_factor, 'if', True, 'Impulse factor',),
    (crest_factor, 'cf', True, 'Crest factor',),
    (variance, 'v', True, 'Variance',),
    (irq, 'irq', True, 'Interquartile range',),
    (perfussion, 'p', True, 'perfussion',)
]

SPECTRAL_FEATURES = [
    (fsqi, 'fsqi', True),
    (fsqi_first, 'fsqi1', True),
    (fsqi_second, 'fsqi2', True),
    (fsqi_third, 'fsqi3', True),
    (frequency_first, 'f1', True),
    (mag_first, 'mag_f1', True),
    (frequency_second, 'f2', True),
    (mag_second, 'mag_f2', True),
    (frequency_third, 'f3', True),
    (mag_third, 'mag_f3', True),
    (spectral_distance, 'spectral_distance', True),
    (fundamental_frequency, 'fundamental_frequency', True,),
    (max_power_spectrum, 'max_power_spectrum', True,),
    (max_frequency, 'max_frequency', True,),
    (median_frequency, 'median_frequency', True,),
    (spectral_centroid, 'spectral_centroid', True,),
    (spectral_decrease, 'spectral_decrease', True,),
    (spectral_kurtosis, 'spectral_kurtosis', True,),
    (spectral_skewness, 'spectral_skewness', True,),
    (spectral_spread, 'spectral_spread', True,),
    (spectral_slope, 'spectral_slope', True,),
    (spectral_variation, 'spectral_variation', True,),
    (spectral_maxpeaks, 'spectral_maxpeaks', True,),
    (spectral_roll_off, 'spectral_roll_off', True,),
    (spectral_roll_on, 'spectral_roll_on', True,),
    (human_range_energy, 'human_range_energy', True,),
    (power_bandwidth, 'power_bandwidth', True,),
    (fft_mean_coeff, 'fft_mean_coeff', False,),
    (spectral_entropy, 'spectral_entropy', True,),
    (wavelet_entropy, 'wavelet_entropy', True,),
    (wavelet_abs_mean, 'wavelet_labs_mean', False,),
    (wavelet_std, 'wavelet_std', False,),
    (wavelet_var, 'wavelet_var', False,),
]
