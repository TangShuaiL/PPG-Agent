"""Analysis pipelines for the web layer (blocking, run in executor).

Wraps the deterministic ppg_agent engine.  All functions operate on the
request-scoped PPGContext installed per call by the chat router.
Raw signals never leave the process except as downsampled display points.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample as scipy_resample

from ppg_agent.context import get_context
from ppg_agent.data_loader import load_pulse_db_signal
from ppg_agent.engine.signal_engine import process_ppg_segment, extract_segment_features
from ppg_agent.schemas.signal import SignalRef

VITALPY_FS = 500

# Non-feature metadata keys inside segment feature dicts
_META_KEYS = {"subject_id", "seg_idx", "source", "sbp_label", "dbp_label",
              "age", "gender", "n_beats_ppg", "n_beats_valid"}


def _resolve_params(source: str) -> tuple[str, dict]:
    """Resolve processing params for a source via the preset chain.

    Same resolution as batch_extract_v2.py: SOURCE_PRESET_MAP → load_preset,
    missing preset falls back to 'default' (which holds the engine defaults).
    Returns (preset_name, params).
    """
    from ppg_agent.processing_params import preset_for_source, load_preset
    name = preset_for_source(source)
    try:
        params = load_preset(name)
    except FileNotFoundError:
        name, params = "default", load_preset("default")
    return name, params


def load_signal(source: str, subject_id: str, seg_idx: int) -> dict:
    """Load PPG (+ABP/R-peaks when present) into the request context."""
    ctx = get_context()
    signal = load_pulse_db_signal(source, subject_id, seg_idx)
    ctx.ppg_signal = np.asarray(signal["ppg"], dtype=np.float64)
    ctx.abp_signal = (np.asarray(signal["abp"], dtype=np.float64)
                      if signal.get("abp") is not None else None)
    ctx.rpeaks_raw = (np.asarray(signal["rpeaks"], dtype=int)
                      if signal.get("rpeaks") is not None else None)
    ctx.age = float(signal["age"]) if signal.get("age") is not None else None
    ctx.sbp_label = float(signal["sbp"]) if signal.get("sbp") is not None else None
    ctx.dbp_label = float(signal["dbp"]) if signal.get("dbp") is not None else None
    ctx.has_filtered = bool(signal.get("has_filtered", False))
    ctx.fs = int(signal.get("fs", VITALPY_FS))
    ctx.signal_ref = SignalRef(
        source=source, subject_id=subject_id, seg_idx=seg_idx,
        fs=ctx.fs, duration_sec=float(len(ctx.ppg_signal)) / ctx.fs,
    )
    return {"n_samples": len(ctx.ppg_signal), "fs": ctx.fs,
            "has_filtered": ctx.has_filtered, "has_abp": ctx.abp_signal is not None}


def normalize() -> dict:
    """Min-max normalize to [0, 1] (idempotent)."""
    ctx = get_context()
    pmin, pmax = float(np.min(ctx.ppg_signal)), float(np.max(ctx.ppg_signal))
    if pmax - pmin < 1e-8:
        raise ValueError("signal is flat (zero range)")
    if not (abs(pmin) < 0.01 and abs(pmax - 1.0) < 0.01):
        ctx.ppg_signal = (ctx.ppg_signal - pmin) / (pmax - pmin)
    return {"range": [round(pmin, 4), round(pmax, 4)]}


def _to_500hz() -> tuple[np.ndarray, np.ndarray]:
    ctx = get_context()
    sig = ctx.ppg_signal
    if ctx.fs != VITALPY_FS:
        target_len = int(len(sig) * VITALPY_FS / ctx.fs)
        ppg_500 = scipy_resample(sig, target_len)
        abp_500 = (scipy_resample(ctx.abp_signal, target_len)
                   if ctx.abp_signal is not None else np.zeros(target_len))
    else:
        ppg_500 = sig.copy()
        abp_500 = ctx.abp_signal.copy() if ctx.abp_signal is not None else np.zeros_like(sig)
    return ppg_500, abp_500


def segment_beats_step() -> dict:
    """Beat segmentation with onset correction (source preset params)."""
    ctx = get_context()
    ppg_500, _ = _to_500hz()
    preset_name, params = _resolve_params(ctx.signal_ref.source)
    r = process_ppg_segment(ppg_500, VITALPY_FS, merge_fragments=True,
                            trim_to_onset=True, params=params)
    ctx.beats = r["beats"]
    ctx.beat_starts = r["beat_starts"]
    ctx.beat_valid_mask = r["valid_mask"]
    ctx.beat_keypoints = r["keypoints"]
    ctx.num_beats = r["n_total"]
    ctx.n_valid_beats = r["n_valid"]
    ctx.ppg_quality = r["quality"]
    ctx.ppg_quality_tier = r["quality_tier"]
    return {"n_beats": r["n_total"], "n_valid": r["n_valid"],
            "quality": round(r["quality"], 3), "quality_tier": r["quality_tier"],
            "preset": preset_name, "n_rejected_mad": r["n_rejected_mad"]}


def beat_overlay() -> dict:
    """Keypoints in absolute sample coordinates — for waveform chart overlay.

    Uses the shared overlay builder (``end`` mirrors the matplotlib figure:
    start + beat template length).  Used by the demo result page, the chat
    analysis panel, and the Aurora keypoint QC.
    """
    from ppg_agent.engine.overlay import build_beat_overlay
    ctx = get_context()
    lengths = [len(b) for b in (ctx.beats or [])] or None
    return build_beat_overlay(ctx.beat_starts, ctx.beat_valid_mask,
                              ctx.beat_keypoints, lengths, VITALPY_FS)


def enrich_overlay_feats(overlay: dict, report: dict) -> dict:
    """Attach per-beat percentile/level annotations for abnormal features.

    Phenotype-mode driver: for every report feature whose segment-level is
    abnormal, each valid beat gets ``feats[f] = {v, level, percentile}``
    computed with the SAME stratified query (``query_stratified`` +
    ``baseline_source`` + age/gender) the report used — the per-beat
    annotation speaks the report's language, so the frontend can highlight
    exactly the beats that drove the abnormal segment statistic.  Non-finite
    or missing per-beat values are skipped.  Mutates and returns overlay.
    """
    from ppg_agent.engine.percentile_engine import PercentileEngine
    abnormal = {f: e for f, e in (report.get("stratified_percentiles") or {}).items()
                if e.get("level") not in ("normal", "unknown", None)}
    if not abnormal or not overlay.get("beats"):
        return overlay
    ctx = get_context()
    baseline = report.get("baseline_source", "vital")
    engine = PercentileEngine()
    age = ctx.age
    gender = (ctx.segment_features_raw or {}).get("gender")
    records = ctx.beat_records or []
    for b in overlay["beats"]:
        if not b.get("valid"):
            continue
        rec = records[b["beat_idx"]] if b["beat_idx"] < len(records) else {}
        feats = {}
        for fname in abnormal:
            v = rec.get(fname)
            if not isinstance(v, (int, float)) or not np.isfinite(float(v)):
                continue
            q = engine.query_stratified(fname, float(v), baseline, age, gender)
            if q.get("error"):
                continue
            feats[fname] = {"v": round(float(v), 4),
                            "level": q.get("level"),
                            "percentile": q.get("percentile")}
        if feats:
            b["feats"] = feats
    return overlay


def extract_features_step(source: str, subject_id: str, seg_idx: int) -> dict:
    """Full segment feature extraction (38 morphology features + PTT + HRV)."""
    ctx = get_context()
    ppg_500, abp_500 = _to_500hz()
    preset_name, params = _resolve_params(source)
    seg_result = extract_segment_features(
        ppg_500, abp_500, ctx.rpeaks_raw,
        subject_id, seg_idx, source,
        sbp_label=ctx.sbp_label, dbp_label=ctx.dbp_label,
        fs=VITALPY_FS, params=params,
    )
    ctx.segment_features_raw = seg_result["segment"]
    ctx.beat_records = seg_result["beats"]
    seg = seg_result["segment"]
    means = {k[:-5]: v for k, v in seg.items()
             if k.endswith("_mean") and k[:-5] not in _META_KEYS
             and isinstance(v, (int, float))}
    return {"n_features": len(means), "n_beat_records": len(seg_result["beats"]),
            "quality_score": seg.get("quality_score"),
            "quality_tier": seg.get("quality_tier"),
            "preset": preset_name,
            "feature_means": {k: round(float(v), 5) for k, v in means.items()}}


def run_phenotype(source: str, scenario: str | None = None) -> dict:
    """Agent 2 deterministic assessment: percentiles + VAI + cross-population.

    IMPORTANT: no silent cross-dataset fallback.  aurora resolves to its
    scenario-specific baseline (``aurora_a``/``aurora_o``, never pooled) via
    ``scenario`` ('a'/'o'); when the scenario is missing from the caller it is
    inferred from the context subject id's first char.  If the resolved source
    has no baseline, the report is still produced against 'vital' but flagged
    with an explicit baseline_warning (VITAL→Aurora transfer is known-bad).
    """
    from ppg_agent.engine.percentile_engine import PercentileEngine

    ctx = get_context()
    seg = ctx.segment_features_raw
    if not seg:
        raise ValueError("no features extracted yet")

    features = {k[:-5]: v for k, v in seg.items()
                if k.endswith("_mean") and k[:-5] not in _META_KEYS}

    # Beat-to-beat variability of this segment (CV = std/|mean|, ddof=1) —
    # same definition as knowledge/beat_cv_reference.json; drives the
    # beat_consistency / cv_level output of the report.
    beat_cv = {}
    for key, val in seg.items():
        if key.endswith("_mean"):
            fname = key[:-5]
            std_key = fname + "_std"
            m, s = val, seg.get(std_key)
            if isinstance(m, (int, float)) and isinstance(s, (int, float)) \
                    and abs(float(m)) > 1e-12:
                beat_cv[fname] = float(s) / abs(float(m))

    # aurora: scenario defaults to the subject id's first char ('a'/'o')
    if scenario is None and source in ("aurora", "aurorabp"):
        sid = getattr(ctx, "subject_id", None)
        if sid:
            scenario = str(sid)[0]
    baseline_source = PercentileEngine.resolve_source(source, scenario)

    engine = PercentileEngine()
    report: dict = {}
    if baseline_source not in engine.available_sources:
        report["baseline_warning"] = (
            f"No baseline for source '{source}' (scenario='{scenario}') — "
            f"assessed against 'vital'. Cross-dataset prior transfer is NOT "
            f"validated for this source; treat all percentiles/VAI as low "
            f"confidence."
        )
        baseline_source = "vital"

    report.update(engine.generate_report(
        features=features,
        actual_age=ctx.age,
        gender=seg.get("gender"),
        n_segments=seg.get("n_beats_valid", 1),
        source=baseline_source,
        beat_cv=beat_cv,
    ))
    report["baseline_source"] = baseline_source
    report["requested_source"] = source

    # Cross-population comparison for the most reliable features
    key_features = ["reflection_index", "lasi", "t_sys_ms",
                    "systolic_width_50", "pulse_width_50"]
    report["cross_population"] = {
        kf: engine.query_compare(kf, features[kf], ctx.age, seg.get("gender"))
        for kf in key_features if kf in features
    }
    return report
