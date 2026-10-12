"""Signal access service — load raw waveforms from .mat files.

Mirrors the loading convention used by analysis scripts and signal_engine:
  - h5py Subj_Wins['PPG_F'][0, seg_idx] per segment window
  - 1250-sample windows are resampled to 5000 (fs normalized to 500 Hz)
Raw arrays are only ever sent to the frontend (downsampled); never to an LLM.
"""
from __future__ import annotations

import numpy as np
import h5py
from scipy.signal import resample

from .. import config


def load_segment_signal(waveform_path: str, seg_idx: int,
                        fs: int = config.TARGET_FS) -> np.ndarray:
    """Load one segment window from its .mat file as float64 @ 500 Hz."""
    with h5py.File(waveform_path, "r") as f:
        subj = f["Subj_Wins"]
        ppg_ref = subj["PPG_F"][0, seg_idx]
        if isinstance(ppg_ref, np.ndarray):
            ppg = ppg_ref.flatten().astype(np.float64)
        elif subj["PPG_F"].dtype == np.dtype("object"):
            ppg = f[ppg_ref][:].flatten().astype(np.float64)
        else:
            ppg = subj["PPG_F"][0, :].flatten().astype(np.float64)
    if len(ppg) == 1250:  # Aurora half-length windows → normalize length
        ppg = resample(ppg, 5000)
    return ppg


def downsample(signal: np.ndarray, n_points: int) -> list[float]:
    """Downsample to at most n_points via linear interpolation (display only)."""
    n = len(signal)
    if n <= n_points:
        return [round(float(x), 5) for x in signal]
    xs = np.linspace(0, n - 1, n_points)
    out = np.interp(xs, np.arange(n), signal)
    return [round(float(x), 5) for x in out]


def get_waveform_payload(waveform_path: str, seg_idx: int,
                         n_points: int = config.DEFAULT_DISPLAY_POINTS) -> dict:
    """Full payload for the waveform chart endpoint."""
    sig = load_segment_signal(waveform_path, seg_idx)
    n_points = min(max(int(n_points), 200), config.MAX_DISPLAY_POINTS)
    return {
        "fs": config.TARGET_FS,
        "n_samples": len(sig),
        "duration_sec": round(len(sig) / config.TARGET_FS, 3),
        "points": downsample(sig, n_points),
        "stats": {
            "min": round(float(np.min(sig)), 5),
            "max": round(float(np.max(sig)), 5),
            "mean": round(float(np.mean(sig)), 5),
            "std": round(float(np.std(sig)), 5),
        },
    }
