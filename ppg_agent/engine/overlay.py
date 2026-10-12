"""Shared waveform-overlay payload builders (pure numpy — no web imports).

Both the web analysis flow and the dataset-adaptation flow render the same
ECharts waveform on the frontend, so the overlay/waveform payloads must be
built by ONE helper set.  This module lives in ``ppg_agent/engine`` to keep
the import direction one-way (web.api → ppg_agent; ppg_agent never imports
web.api).

Payload contracts:
    overlay = {fs: 500, beats: [{beat_idx, start, end?, valid,
                                 keypoints?: {name: absolute_sample},
                                 feats?: {feature: {v, level, percentile}}}]}
    waveform = {fs, n_samples, duration_sec, points: [...], overlay}

``end`` mirrors the matplotlib convention (visualization.plot_signal_with_beats:
t_end = t_start + len(beat)/fs), with fallbacks when beat lengths are unknown.
"""
from __future__ import annotations

import math

import numpy as np


def downsample(signal: np.ndarray, n_points: int) -> list[float]:
    """Linear-interpolation downsampling for display only (same approach as
    web.api.services.waveform_reader.downsample)."""
    if len(signal) <= n_points:
        return [float(x) for x in signal]
    idx = np.linspace(0, len(signal) - 1, n_points)
    return [float(x) for x in np.interp(idx, np.arange(len(signal)), signal)]


def _clean(value):
    """Non-finite floats → None (FastAPI rejects NaN/inf in JSON)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def build_beat_overlay(beat_starts, valid_mask, keypoints,
                       beat_lengths: list[int] | None = None,
                       fs: int = 500) -> dict:
    """Base overlay: per-beat {beat_idx, start, end?, valid, keypoints?}.

    ``end`` resolution order (first hit wins):
      1. beat_lengths[i]  — start + len(beat)/... (start + len, mirrors the
                            matplotlib figure exactly, incl. the last beat)
      2. keypoints['v']   — absolute diastolic-end sample
      3. next beat's start — contiguous onset-to-onset beats
      4. absent            — no end info (callers may rely on absence; the
                            frontend clamps to the waveform duration)
    Invalid beats keep start/end/valid but carry no keypoints.
    """
    starts = [int(s) for s in (beat_starts or [])]
    valid = [bool(v) for v in (valid_mask or [])]
    kps = list(keypoints or [])
    n = len(starts)
    lengths = list(beat_lengths) if beat_lengths is not None else None

    def _end(i: int) -> int | None:
        if lengths is not None and i < len(lengths) and lengths[i] is not None:
            return starts[i] + int(lengths[i])
        kp = kps[i] if i < len(kps) else None
        v = (kp or {}).get("v")
        if v is not None:
            return starts[i] + int(v)
        if i + 1 < n:
            return starts[i + 1]
        return None

    beats = []
    for i in range(n):
        entry = {"beat_idx": i, "start": starts[i],
                 "end": _end(i), "valid": valid[i] if i < len(valid) else False}
        kp = kps[i] if i < len(kps) else None
        if valid[i] and kp:
            entry["keypoints"] = {k: int(starts[i]) + int(v)
                                  for k, v in kp.items() if v is not None}
        beats.append(entry)
    return {"fs": fs, "beats": beats}


def attach_beat_feats(overlay: dict, features_list: list[dict]) -> dict:
    """Attach raw per-beat feature dicts to valid beats (mutates + returns).

    Used by the adaptation flow for the per-beat feature-curve chart.
    ``features_list`` is the per-beat feature list from process_ppg_segment
    ({} for invalid beats); non-finite values are cleaned for JSON.
    """
    for i, b in enumerate(overlay["beats"]):
        if not b.get("valid"):
            continue
        feats = features_list[i] if i < len(features_list) else None
        if feats:
            cleaned = _clean(feats)
            if isinstance(cleaned, dict) and cleaned:
                b["feats"] = cleaned
    return overlay


def waveform_payload(ppg: np.ndarray, beat_starts, valid_mask, keypoints,
                     beats: list | None = None, fs: int = 500,
                     max_points: int = 2000,
                     features_list: list[dict] | None = None) -> dict:
    """Full adaptation-mode payload: waveform points + overlay (+ per-beat
    feature dicts for the feature-curve chart)."""
    lengths = [len(b) for b in beats] if beats else None
    overlay = build_beat_overlay(beat_starts, valid_mask, keypoints,
                                 lengths, fs)
    if features_list is not None:
        attach_beat_feats(overlay, features_list)
    sig = np.asarray(ppg)
    return {"fs": fs, "n_samples": int(len(sig)),
            "duration_sec": round(len(sig) / fs, 3),
            "points": downsample(sig, max_points),
            "overlay": overlay}
