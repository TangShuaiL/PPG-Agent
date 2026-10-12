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
#   的 third_party/vitalpy/。相对于 VitalPy 原件的改动：2 行（仅 import 路径）
#   完整改动清单与致谢见同目录 README.md。
# ============================================================================
PPG_PEAK_DETECTION_THRESHOLD = 0.2
MINIMUM_PULSE_CYCLE = 0.3         # 200 bpm
MAXIMUM_PULSE_CYCLE = 1.2         # 50 bpm (matches segment_beats hard cap)

PPG_SAMPLE_RATE = 500
ECG_SAMPLE_RATE = 500


# number of coefficients for doing polynomial approximation of ppg waveform
POLY_COEFFS = 15       # whole template
POLY_COEFFS_SYS = 5    # systolic part
POLY_COEFFS_DIAS = 10  # diastolic part

POUND_TO_KILO = 0.453592
INCH_TO_MTS = 0.0254


# DEFAULT THRESHOLD FOR EUCLIDIAN DISTANCE CHECKING
MIN_MAX_EUC_DIST_THRESHOLD = 1.5
Z_SCORE_EUC_DIST_THRESHOLD = 3

# MINIMUM NUMBER OF VALID BEAT-TO-BEAT WAVEFORMS IN A MEASUREMENT (t=30s) TO BE CONSIDERED VALID
MIN_TEMPLATES = 5
