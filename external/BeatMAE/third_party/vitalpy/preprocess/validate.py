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
import numpy as np
from third_party.vitalpy.params import MINIMUM_PULSE_CYCLE, MAXIMUM_PULSE_CYCLE


def validate_template(template: np.ndarray, fs: float,
                      trough_diff: float = 0.2) -> bool:
    """
    Validate the quality of the beat-to-beat single waveforms by analyzing the template morphology.
    Specifically:
        1) Pulse width
        2) Location of the maximum value
        3) Location of the through
        4) Trough difference

    Parameters:
        template (np.ndarray): Beat-to-beat single PPG waveform.
        fs (float): Sample frequency of the signal (Hz).
        trough_diff (float): Max allowed |first-last| trough difference as a
            fraction of beat amplitude. 0.2 = historical strict default (clinical
            environments). Free-living wrist data with residual drift needs a
            larger value (e.g. 0.4) — pair with template_corr_min so the extra
            admitted beats are shape-consistent with the segment median.

    Returns:
        bool: True if the waveform meets the requirements, False otherwise.
    """

    # 1) Check that the pulse width is within the normal range
    period = len(template) / fs
    if period < MINIMUM_PULSE_CYCLE or period > MAXIMUM_PULSE_CYCLE:
        return False

    # 2) Check whether the systolic peak is in the half first part of the waveform
    max_index = np.argmax(template)
    if float(max_index) / float(len(template)) >= 0.75:
        return False

    # 3) Check where is the minimum point (through)
    # DISABLED: PulseDB normalized data often has valley slightly offset
    # min_index = np.argmin(template)
    # if not (min_index == 0 or min_index == len(template) - 1):
    #     return False

    # 4) Check if there is little difference between the init and final min point (through)
    if abs(template[0] - template[-1]) / (max(template) - min(template)) > trough_diff:
        return False

    return True


def validate_waveform(templates: list, fs: float):
    """
    Validate the quality of the PPG signal by analyzing the morphology.

    Parameters:
        templates (list): List of templates (pd.Series).
        fs (float): Sample frequency of the signal (Hz).

    Returns:
        list: List of good quality templates (pd.Series).
    """
    
    gq_templates = []
    for template in templates:
        if validate_template(template.to_numpy(), fs):
            gq_templates.append(template)

    return gq_templates
