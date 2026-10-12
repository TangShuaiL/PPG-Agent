"""Shared feature lists and condition definitions for PulseDB statistics."""

from __future__ import annotations


CORE_REFERENCE_FEATURES = [
    "reflection_index_mean",
    "lasi_mean",
    "pulse_width_50_mean",
    "pulse_width_90_mean",
    "perfusion_index_mean",
    "skewness_mean",
    "kurtosis_mean",
    "spectral_entropy_mean",
    "wavelet_entropy_mean",
    "signal_quality_fsqi_mean",
    "ptt_mean_ms",
    "ptt_normalized",
    "hrv_sdnn",
    "hrv_rmssd",
    "abp_pp_mean",
]

CORRELATION_TARGETS = ["sbp_label", "dbp_label"]

QUALITY_COLUMNS = ["extraction_status", "quality_tier", "quality_ok"]

REFERENCE_QUANTILES = {
    "p05": 0.05,
    "p20": 0.20,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p85": 0.85,
    "p95": 0.95,
}

CONDITION_FEATURES = [
    "reflection_index_mean",
    "lasi_mean",
    "ptt_mean_ms",
    "quality_tier",
    "quality_score",
    "coverage",
    "sbp_label",
    "dbp_label",
]
