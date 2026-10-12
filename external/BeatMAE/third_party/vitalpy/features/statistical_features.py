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
#   的 third_party/vitalpy/。相对于 VitalPy 原件的改动：0 行（逐字未改）
#   完整改动清单与致谢见同目录 README.md。
# ============================================================================
import scipy
import numpy as np


def skewness(signal: np.ndarray):
    """ Calculate the skewness (S) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Skewness
    """

    return scipy.stats.skew(signal)


def kurtosis(signal: np.ndarray):
    """ Calculate the kurtosis (K) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Kurtosis
    """
    return scipy.stats.kurtosis(signal)


def mav(signal: np.ndarray):
    """ Calculate the Mean Absolute Value (MAV) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Mean Absolute Value
    """

    return np.mean(abs(signal))


def median(signal: np.ndarray):
    """ Calculate the median of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Median
    """
    return np.median(signal)


def mad(signal: np.ndarray):
    """ Calculate the Mean Absolute Deviation (MAD) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Mean Absolute Deviation
    """
    return np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)


def med_ad(signal: np.ndarray):
    """ Calculate the Median absolute deviation (mAD) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Median Absolute Deviation
    """
    return scipy.stats.median_abs_deviation(signal, scale=1)


def rms(signal: np.ndarray):
    """ Calculate the Root-Mean-Square (RMS) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Root-Mean-Square
    """
    return np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))


def sd(signal: np.ndarray):
    """ Calculate the Standard Deviation (SD) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Standard Deviation
    """
    return np.std(signal)


def shape_factor(signal: np.ndarray):
    """ Calculate the Shape Factor (SF) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Shape Factor
    """
    rms = np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))
    mav = np.mean(abs(signal))
    return rms / mav


def impulse_factor(signal: np.ndarray):
    """ Calculate the Impulse Factor (IF) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Impulse Factor
    """
    return max(signal) / np.mean(signal)


def crest_factor(signal: np.ndarray):
    """ Calculate the Crest Factor (CF) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Crest Factor
    """
    rms = np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))
    return max(signal) / rms


def variance(signal: np.ndarray):
    """ Calculate the variance of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Variance
    """
    return np.var(signal)


def irq(signal: np.ndarray):
    """ Calculate the Interquartile Range (IRQ) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Interquartile Range
    """
    return scipy.stats.iqr(signal)


def perfussion(signal: np.ndarray):
    """ Calculate the perfussion of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        perfussion
    """
    return (max(signal) - min(signal)) / np.mean(signal)



