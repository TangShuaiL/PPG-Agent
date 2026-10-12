"""Agent 1 — Signal Processor: load → segment → extract features.

Foundation (strict edges, no LLM):
    START → load → normalize → scan → segment → extract_features

Uses the new unified pipeline directly (process_ppg_segment + extract_segment_features).
Features extracted on-the-fly from raw signal — does NOT read pre-computed parquet.
Parquet is for Agent 2/3's offline analysis.

Entry: conditional edge classifies the user's intent —
    "adapt"   → dataset-adaptation mode: the AdaptationSession runs as a graph
                tool (adapt_pipeline); interactive rounds, L2 batch validation,
                L3 acceptance gate, preset saved on finalize. Full session
                functionality is preserved (see ppg_agent.agents.adaptation_session).
    "process" → the signal-processing pipeline below.

Decision (LLM tools):
    visualize_signal  — single-axis plot with onset-corrected beats + keypoints
    assess_quality    — segment-level quality report
    scan_report       — read signal_scan diagnostics
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from scipy.signal import resample as scipy_resample

from ppg_agent.context import get_context
from ppg_agent.llm_service import get_llm
from ppg_agent.engine.signal_engine import (
    process_ppg_segment,
    extract_segment_features,
)
from ppg_agent.schemas.signal import SignalRef
from ppg_agent.agents.adaptation_session import adapt_pipeline

VITALPY_FS = 500

# ---------------------------------------------------------------------------
# Entry intent router
# ---------------------------------------------------------------------------

# Deterministic keywords (fast path; LLM fallback handles other phrasings).
# _PROCESS_KEYWORDS are a STRONG signal — "适配好了，提取特征" must route to
# process; adaptation-session letters ("选B", "f") carry neither keyword set.
_ADAPT_KEYWORDS = ("适配", "调参", "调整参数", "参数调整", "优化参数", "参数优化",
                   "adapt", "tune", "tuning", "preset", "调整一下参数")
_PROCESS_KEYWORDS = ("提取", "处理", "分析", "特征", "管线", "可视化", "质量",
                     "process", "extract", "segment")


class _Intent(BaseModel):
    route: str = Field(description="One of: adapt | process")


def _classify_intent(text: str, llm: Any) -> str:
    """User intent → 'adapt' (dataset adaptation) or 'process' (pipeline)."""
    low = text.lower()
    if any(k in low for k in _ADAPT_KEYWORDS):
        return "adapt"
    if llm is None:
        return "process"
    try:
        clf = llm.with_structured_output(_Intent)
        r = clf.invoke(
            "Is the user asking to ADAPT a dataset (tune processing parameters, "
            "save a preset) or to PROCESS a signal (load/segment/extract "
            "features of a specific segment)?\n"
            f"User: {text[:2000]}\n"
            "Answer with exactly one of: adapt | process")
        route = getattr(r, "route", None)
        if route in ("adapt", "process"):
            return route
    except Exception:
        pass
    return "process"


def _route_entry(state: dict, llm: Any = None) -> str:
    """START conditional edge: classify the user's LATEST message.

    Priority: explicit process keywords (strongest — "适配好了，提取特征"
    goes to process) → adapt keywords → current conversation mode (so
    adaptation-session letters like "选B"/"f" stay in adapt) → LLM fallback.

    Classification lives in the edge itself (not a node) because
    StateGraph(dict) replaces state wholesale on each node write.
    """
    ctx = get_context()
    # Web orchestration may route authoritatively: the Supervisor already
    # classified the intent, so state["mode"] wins over keyword heuristics.
    # (Standalone graph runs pass no "mode" key → keyword priority unchanged.)
    mode = state.get("mode")
    if mode in ("adapt", "process"):
        ctx.session_mode = mode
        return mode
    msgs = state.get("messages", [])
    text = ""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            c = getattr(m, "content", None)
            if isinstance(c, str):
                text = c
            break
    low = text.lower()
    if any(k in low for k in _PROCESS_KEYWORDS):
        ctx.session_mode = "process"
        return "process"
    if any(k in low for k in _ADAPT_KEYWORDS):
        ctx.session_mode = "adapt"
        return "adapt"
    mode = getattr(ctx, "session_mode", None)
    if mode in ("adapt", "process"):
        return mode
    route = _classify_intent(text, llm)
    ctx.session_mode = route
    return route


# ---------------------------------------------------------------------------
# Foundation node: load
# ---------------------------------------------------------------------------

def _load_node(state: dict) -> dict:
    """Load PPG_F + ABP + ECG_RPeaks from PulseDB .mat file."""
    ctx = get_context()
    source = state.get("source", "vital")
    subject_id = state.get("subject_id", "p000001")
    seg_idx = state.get("seg_idx", 0)
    result: dict = {"messages": state.get("messages", [])}

    if ctx.is_loaded and ctx.signal_ref is not None:
        ref = ctx.signal_ref
        if ref.source == source and ref.subject_id == subject_id and ref.seg_idx == seg_idx:
            return {**result, "loaded": True, "fs": ctx.fs, "has_filtered": ctx.has_filtered}

    from ppg_agent.data_loader import load_pulse_db_signal

    try:
        signal = load_pulse_db_signal(source, subject_id, seg_idx)
    except FileNotFoundError as e:
        return {**result, "error": f"Signal file not found: {source}/{subject_id}/{seg_idx}. {e}"}
    except Exception as e:
        return {**result, "error": f"Failed to load: {e}"}

    ctx.ppg_signal = np.asarray(signal["ppg"], dtype=np.float64)
    ctx.abp_signal = np.asarray(signal.get("abp"), dtype=np.float64) if signal.get("abp") is not None else None
    ctx.rpeaks_raw = np.asarray(signal.get("rpeaks"), dtype=int) if signal.get("rpeaks") is not None else None
    ctx.age = float(signal.get("age", 0)) if signal.get("age") is not None else None
    ctx.sbp_label = float(signal.get("sbp", 0)) if signal.get("sbp") is not None else None
    ctx.dbp_label = float(signal.get("dbp", 0)) if signal.get("dbp") is not None else None
    ctx.has_filtered = bool(signal.get("has_filtered", False))
    ctx.fs = int(signal.get("fs", 125))
    ctx.signal_ref = SignalRef(
        source=source, subject_id=subject_id, seg_idx=seg_idx,
        fs=ctx.fs, duration_sec=float(len(ctx.ppg_signal)) / ctx.fs,
    )
    extra = ""
    if ctx.rpeaks_raw is not None:
        extra += f", {len(ctx.rpeaks_raw)} R-peaks"
    if ctx.abp_signal is not None:
        extra += ", +ABP"
    ctx.add_log(f"[foundation] Loaded {source}/{subject_id}/{seg_idx}: "
                f"{len(ctx.ppg_signal)} samples @ {ctx.fs} Hz "
                f"({'PPG_F' if ctx.has_filtered else 'raw'}){extra}")
    return {**result, "loaded": True, "fs": ctx.fs, "has_filtered": ctx.has_filtered}


# ---------------------------------------------------------------------------
# Foundation node: normalize
# ---------------------------------------------------------------------------

def _normalize_node(state: dict) -> dict:
    """Min-max normalize to [0, 1]. Idempotent if already in range."""
    ctx = get_context()
    result: dict = {"messages": state.get("messages", [])}
    if ctx.ppg_signal is None:
        return {**result, "error": "load step failed"}

    pmin = float(np.min(ctx.ppg_signal))
    pmax = float(np.max(ctx.ppg_signal))
    if pmax - pmin < 1e-8:
        return {**result, "error": "Signal is flat (zero range)."}

    if abs(pmin) < 0.01 and abs(pmax - 1.0) < 0.01:
        return {**result, "normalized": True, "signal_range": [pmin, pmax]}

    ctx.ppg_signal = (ctx.ppg_signal - pmin) / (pmax - pmin)
    ctx.add_log(f"[foundation] Normalized: [{pmin:.4f},{pmax:.4f}] → [0,1]")
    return {**result, "normalized": True, "signal_range": [0.0, 1.0]}


# ---------------------------------------------------------------------------
# Foundation node: signal_scan
# ---------------------------------------------------------------------------

def _scan_node(state: dict) -> dict:
    """Quick diagnostic scan of the loaded signal."""
    ctx = get_context()
    result: dict = {"messages": state.get("messages", [])}
    if ctx.ppg_signal is None:
        return {**result, "error": "no signal"}

    sig = ctx.ppg_signal; fs = ctx.fs
    amp_min, amp_max = float(np.min(sig)), float(np.max(sig))
    amp_range = amp_max - amp_min
    t = np.arange(len(sig))
    slope = float(np.polyfit(t, sig, 1)[0]) * len(sig)

    hf_energy = 0.0
    if len(sig) > 20:
        freqs = np.fft.rfftfreq(len(sig), 1 / fs)
        fft = np.abs(np.fft.rfft(sig))
        total = float(np.sum(fft ** 2))
        hf_energy = float(np.sum(fft[freqs > 10] ** 2)) / total * 100 if total > 0 else 0

    is_clipped = (np.sum(sig > 0.99 * amp_max) + np.sum(sig < 0.01 * amp_min)) / len(sig) > 0.05
    is_flatline = float(np.std(sig)) < 1e-6

    if is_flatline: assessment = "flatline — unusable"
    elif is_clipped: assessment = "clipped — reduced reliability"
    elif abs(slope) > 0.2: assessment = "significant baseline drift"
    elif hf_energy > 5: assessment = "high-frequency noise detected"
    else: assessment = "clean — ready for segmentation"

    scan = {
        "amplitude_range": [round(amp_min, 4), round(amp_max, 4)],
        "total_drift": round(slope, 4), "hf_noise_pct": round(hf_energy, 2),
        "is_clipped": is_clipped, "is_flatline": is_flatline, "assessment": assessment,
    }
    ctx.add_log(f"[foundation] Scan: {assessment}")
    return {**result, "scan": scan}


# ---------------------------------------------------------------------------
# Foundation node: segment (NEW — uses process_ppg_segment)
# ---------------------------------------------------------------------------

def _resolve_params(source: str) -> dict:
    """Resolve processing params for a source via the preset chain.

    Same resolution as the web pipeline (analysis_service._resolve_params):
    SOURCE_PRESET_MAP → load_preset, missing preset falls back to 'default'
    (which holds the engine defaults).  Explicit ``state["params"]`` (passed
    by the web executor) wins when present — dataset adaptations would be
    silently lost otherwise.
    """
    from ppg_agent.processing_params import preset_for_source, load_preset
    name = preset_for_source(source)
    try:
        return load_preset(name)
    except FileNotFoundError:
        return load_preset("default")


def _segment_node(state: dict) -> dict:
    """Beat segmentation with onset correction via process_ppg_segment."""
    ctx = get_context()
    result: dict = {"messages": state.get("messages", [])}
    sig = ctx.ppg_signal
    if sig is None:
        return {**result, "error": "no signal"}

    if ctx.is_segmented:
        return {**result, "segmented": True, "n_beats": ctx.num_beats}

    # Resample to 500 Hz for VitalPy compatibility
    if ctx.fs != VITALPY_FS:
        target_len = int(len(sig) * VITALPY_FS / ctx.fs)
        ppg_500 = scipy_resample(sig, target_len)
    else:
        ppg_500 = sig.copy()

    ppg_result = process_ppg_segment(
        ppg_500, VITALPY_FS, merge_fragments=True, trim_to_onset=True,
        params=state.get("params") or _resolve_params(state.get("source", "vital")))

    ctx.beats = ppg_result['beats']
    ctx.beat_starts = ppg_result['beat_starts']
    ctx.beat_valid_mask = ppg_result['valid_mask']
    ctx.beat_keypoints = ppg_result['keypoints']
    ctx.num_beats = ppg_result['n_total']
    ctx.n_valid_beats = ppg_result['n_valid']
    ctx.ppg_quality = ppg_result['quality']
    ctx.ppg_quality_tier = ppg_result['quality_tier']

    ctx.add_log(f"[foundation] Segmented: {ppg_result['n_total']} beats, "
                f"{ppg_result['n_valid']} valid, Q={ppg_result['quality']:.2f} ({ppg_result['quality_tier']})")
    return {**result, "segmented": True, "n_beats": ctx.num_beats,
            "n_valid": ctx.n_valid_beats, "quality": ctx.ppg_quality}


# ---------------------------------------------------------------------------
# Foundation node: extract_features (NEW — uses extract_segment_features)
# ---------------------------------------------------------------------------

def _extract_features_node(state: dict) -> dict:
    """Full segment feature extraction: PPG + ABP + ECG → segment-level + beat-level."""
    ctx = get_context()
    result: dict = {"messages": state.get("messages", [])}

    if ctx.is_features_extracted:
        return {**result, "features_extracted": True}

    sig = ctx.ppg_signal
    if sig is None:
        return {**result, "error": "no signal"}

    source = state.get("source", "vital")
    subject_id = state.get("subject_id", "?")
    seg_idx = state.get("seg_idx", 0)

    # Resample to 500 Hz
    if ctx.fs != VITALPY_FS:
        target_len = int(len(sig) * VITALPY_FS / ctx.fs)
        ppg_500 = scipy_resample(sig, target_len)
        if ctx.abp_signal is not None:
            abp_500 = scipy_resample(ctx.abp_signal, target_len)
        else:
            abp_500 = np.zeros(target_len)
    else:
        ppg_500 = sig.copy()
        abp_500 = ctx.abp_signal.copy() if ctx.abp_signal is not None else np.zeros_like(sig)

    seg_result = extract_segment_features(
        ppg_500, abp_500, ctx.rpeaks_raw,
        subject_id, seg_idx, source,
        sbp_label=ctx.sbp_label, dbp_label=ctx.dbp_label,
        fs=VITALPY_FS,
        params=state.get("params") or _resolve_params(source),
    )

    ctx.segment_features_raw = seg_result['segment']
    ctx.beat_records = seg_result['beats']

    # Summary for LLM
    seg = seg_result['segment']
    summary = {
        "n_beats_total": seg.get("n_beats_ppg", 0),
        "n_beats_valid": seg.get("n_beats_valid", 0),
        "quality": f"{seg.get('quality_score', 0):.2f} ({seg.get('quality_tier', '?')})",
    }
    if seg.get('ptt_abp_mean_ms'):
        summary["ptt_abp_ms"] = seg['ptt_abp_mean_ms']
    if seg.get('ptt_ppg_mean_ms'):
        summary["ptt_ppg_ms"] = seg['ptt_ppg_mean_ms']
    if seg.get('ecg_hr_mean'):
        summary["hr_bpm"] = seg['ecg_hr_mean']
    if seg.get('abp_sbp_mean'):
        summary["abp_sbp"] = seg['abp_sbp_mean']
        summary["abp_dbp"] = seg['abp_dbp_mean']

    ctx.add_log(f"[foundation] Features: segment + {len(ctx.beat_records)} beats, "
                f"PTT ABP={seg.get('ptt_abp_mean_ms','?')}ms PPG={seg.get('ptt_ppg_mean_ms','?')}ms")
    return {**result, "features_extracted": True, "feature_summary": summary}


# ---------------------------------------------------------------------------
# Decision tools
# ---------------------------------------------------------------------------

def _visualize_signal(with_keypoints: bool = True) -> dict:
    """Plot PPG waveform with green/red beat backgrounds and fiducial points.

    Uses onset-corrected beat positions from the new pipeline.
    """
    from ppg_agent.engine.visualization import plot_signal_with_beats

    ctx = get_context()
    if ctx.ppg_signal is None or not ctx.beats:
        return {"status": "error", "message": "No signal or beats. Foundation must complete first."}

    # Resample for visualization
    sig = ctx.ppg_signal
    if ctx.fs != VITALPY_FS:
        target_len = int(len(sig) * VITALPY_FS / ctx.fs)
        sig_viz = scipy_resample(sig, target_len)
        fs_viz = VITALPY_FS
    else:
        sig_viz = sig.copy()
        fs_viz = ctx.fs

    kps_for_viz = ctx.beat_keypoints if with_keypoints and ctx.beat_keypoints else None

    sid = f"{ctx.signal_ref.source}_{ctx.signal_ref.subject_id}_{ctx.signal_ref.seg_idx}" if ctx.signal_ref else "signal"
    r = plot_signal_with_beats(
        raw=sig_viz,
        processed=None,
        beats=ctx.beats,
        beat_starts=ctx.beat_starts,
        valid_mask=ctx.beat_valid_mask,
        fs=fs_viz,
        keypoints_per_beat=kps_for_viz,
        title=f"PPG Beats — {sid} | {ctx.num_beats}B/{ctx.n_valid_beats}V | Q={ctx.ppg_quality:.2f}",
        filename=f"beats_{sid}",
    )
    ctx.add_log(f"[decision] Visualization saved: {r['path']}")
    return r


def _assess_quality(source: str = "vital", subject_id: str = "?", seg_idx: int = 0) -> dict:
    """Signal quality report using pipeline-computed quality."""
    ctx = get_context()
    if not ctx.beats:
        return {"status": "error", "message": "No beats segmented."}

    n_total = ctx.num_beats
    n_valid = ctx.n_valid_beats
    q = ctx.ppg_quality
    tier = ctx.ppg_quality_tier

    return {"status": "success", "quality_tier": tier, "quality_score": q,
            "n_beats": n_total, "n_valid": n_valid,
            "valid_ratio": round(n_valid / max(n_total, 1), 2),
            "is_reliable": tier == "good"}


# ---------------------------------------------------------------------------
# Decision prompt
# ---------------------------------------------------------------------------

DECISION_PROMPT = """You are the PPG Signal Processor — Agent 1 (process mode).

The foundation pipeline has run automatically. All features are in the context (ctx).

SIGNAL TOOLS (for current segment):
  visualize_signal  — plot waveform with onset-corrected beats + 9 keypoints
  assess_quality    — segment-level quality report (tier, score, valid beats)

For questions about the CURRENT segment, answer directly from ctx (no tools needed).
You do NOT interpret cardiovascular meaning — that is Agent 2's job.
If the user wants to ADAPT a dataset (tune parameters / save a preset), that
is a separate entry intent — re-enter the graph with the adaptation request.
"""

_DECISION_TOOLS = [
    # Signal tools
    StructuredTool.from_function(func=_visualize_signal, name="visualize_signal",
        description="Plot PPG waveform with onset-corrected beat boundaries and 9 keypoints."),
    StructuredTool.from_function(func=_assess_quality, name="assess_quality",
        description="Signal quality report (tier, score, valid beats)."),
]

# ---------------------------------------------------------------------------
# Adaptation mode (dataset-parameter tuning; full AdaptationSession preserved)
# ---------------------------------------------------------------------------

ADAPT_PROMPT = """You are the PPG Signal Processor — Agent 1 (dataset adaptation mode).

You run an interactive parameter-tuning session for a target dataset. The
session (adapt_pipeline tool) samples a few segments, proposes 2-4 concrete
parameter-change options, and the user picks one. You NEVER change
parameters directly — options are the only way to mutate params.

PROTOCOL (semantic actions; adapt_pipeline returns {status, message, state}):
  - action="start" (source=<dataset>): starts the session, samples segments
    and returns 2-4 options for the user to pick. Idempotent — if a session
    is already active it returns the current state instead of resetting.
  - action="propose" (feedback=<user's words>): the user gave negative
    feedback ("分割不好"/"特征点不对"/"参数差") — propose a fresh set of
    2-4 options based on the feedback.
  - action="apply" (choice_idx=N, 1-based): the user picked option N —
    applies that one parameter change and re-samples with figures.
  - action="sample": the user wants to see more segments (params unchanged).
  - action="save": the user is satisfied — saves a user_<source> preset
    (never overwrites system presets).
  - action="cancel": the user abandons the session.
  - action="status": re-print the current state.

The session keeps state across calls — do NOT re-start it mid-conversation.
Report only metrics/options shown in the returned message; never invent
numbers. Presets are saved to models/param_presets/.
"""

_ADAPT_TOOLS = [
    StructuredTool.from_function(
        func=adapt_pipeline, name="adapt_pipeline",
        description=("Dataset parameter-adaptation session: start / propose "
                     "(feedback) / apply (choice_idx, 1-based) / sample / "
                     "save / cancel / status. Returns {status, message, "
                     "state}. Options are the only way to change parameters.")),
]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_signal_processor_graph(llm: Any = None) -> Any:
    """Build Agent 1: intent router → (adapt session | foundation + decision).

    Graph:
      START → intent ──"adapt"──► adapt ──►(tools_adapt)──► adapt ──► END
                  │
                  └─"process"──► load → normalize → scan → segment
                                 → extract_features → decide ──►(tools_decide)
                                 → decide ──► END
    """
    if llm is None:
        llm = get_llm("signal_processor")

    decision_agent = create_agent(model=llm, tools=_DECISION_TOOLS, system_prompt=DECISION_PROMPT)
    adapt_agent = create_agent(model=llm, tools=_ADAPT_TOOLS, system_prompt=ADAPT_PROMPT)

    graph = StateGraph(dict)
    graph.add_node("load", _load_node)
    graph.add_node("normalize", _normalize_node)
    graph.add_node("scan", _scan_node)
    graph.add_node("segment", _segment_node)
    graph.add_node("extract_features", _extract_features_node)
    graph.add_node("report_error", lambda s: {"messages": [AIMessage(content=f"Error: {s.get('error', 'unknown')}")]})

    def _decide(state: dict) -> dict:
        ctx = get_context()
        msgs = list(state.get("messages", []))
        n = state.get("n_beats", 0)
        q = state.get("quality", 0)
        src = state.get("source", "vital"); sid = state.get("subject_id", "?")
        seg = state.get("seg_idx", 0)
        scan = state.get("scan", {})
        fsum = state.get("feature_summary", {})
        from langchain_core.messages import SystemMessage
        ctx_msg = SystemMessage(content=(
            f"[Foundation complete. {n} beats, {state.get('n_valid', 0)} valid, "
            f"Q={q:.2f} ({ctx.ppg_quality_tier}). "
            f"PTT: ABP={ctx.segment_features_raw.get('ptt_abp_mean_ms','?')}ms "
            f"PPG={ctx.segment_features_raw.get('ptt_ppg_mean_ms','?')}ms. "
            f"HR={ctx.segment_features_raw.get('ecg_hr_mean','?')}bpm. "
            f"Features ready in ctx. Call tools only if user asks to visualize or assess.]"))
        r = decision_agent.invoke({"messages": [ctx_msg] + msgs})
        return {"messages": r.get("messages", [])}

    def _adapt(state: dict) -> dict:
        msgs = list(state.get("messages", []))
        r = adapt_agent.invoke({"messages": msgs})
        return {"messages": r.get("messages", [])}

    graph.add_node("decide", _decide)
    graph.add_node("tools_decide", ToolNode(_DECISION_TOOLS))
    graph.add_node("adapt", _adapt)
    graph.add_node("tools_adapt", ToolNode(_ADAPT_TOOLS))

    # Entry: intent conditional edge (classification in the edge, START → adapt|load)
    graph.add_conditional_edges(
        START, lambda s: _route_entry(s, llm),
        {"adapt": "adapt", "process": "load"})

    # Adaptation loop (independent tools cycle)
    def _route_adapt_tools(state: dict) -> str:
        msgs = state.get("messages", [])
        if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
            return "tools"
        return "end"

    graph.add_conditional_edges("adapt", _route_adapt_tools, {"tools": "tools_adapt", "end": END})
    graph.add_edge("tools_adapt", "adapt")

    # Foundation edges
    graph.add_edge("load", "normalize")
    graph.add_edge("normalize", "scan")
    graph.add_edge("scan", "segment")
    graph.add_edge("segment", "extract_features")

    def _after_foundation(state: dict) -> str:
        return "report_error" if state.get("error") else "decide"

    graph.add_conditional_edges("extract_features", _after_foundation, {"report_error": "report_error", "decide": "decide"})
    graph.add_edge("report_error", END)

    # Decision tools loop
    def _route_decide_tools(state: dict) -> str:
        msgs = state.get("messages", [])
        if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
            return "tools"
        return "end"

    graph.add_conditional_edges("decide", _route_decide_tools, {"tools": "tools_decide", "end": END})
    graph.add_edge("tools_decide", "decide")
    return graph.compile()


def create_signal_processor_agent(llm: Any = None) -> Any:
    """Create Agent 1 as a compiled LangGraph graph."""
    return build_signal_processor_graph(llm)
