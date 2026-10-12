"""Tests for the Agent-2 longitudinal comparison layer (plan v3 refactor).

Agent 2 was refactored from the disproven phenotype pipeline (VAI prose,
subtypes, disease association, interpretation_rules, window_assess) into a
three-layer PERSONAL comparison analyst:

    assess (deterministic compare_segment) → interpret ⇄ tools_assess
    → capture_findings (engine-facts-only validation)

Agent 2 emits NO natural-language interpretation — the Supervisor synthesis
phase words all data.

Covers:
  1. VOLATILE_FEATURES — Phase 3 walk-forward high-false-positive features
  2. format_assessment_blocks — longitudinal blocks (legacy VAI shape removed)
  3. _capture_findings — engine-facts-only validation (Δ/threshold verbatim,
     _mean-suffix tolerance, wrapped-args tolerance, verdict gate)
  4. CARDIOVASCULAR_PROMPT — the wording discipline
  5. Graph flow with a scripted LLM (no network): submit routing, tool loop,
     message-history pairing (orphaned-tool-message regression)
  6. PPGContext.beat_summary — the Agent 1 → Agent 2 handoff
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ppg_agent.agents import cardiovascular_agent as ca
from ppg_agent.agents.cardiovascular_agent import (
    CARDIOVASCULAR_PROMPT,
    VOLATILE_FEATURES,
    _capture_findings,
    _population_position,
    build_cardiovascular_graph,
    format_assessment_blocks,
)
from ppg_agent.context import PPGContext
from ppg_agent.engine.personal_engine import PersonalEngine


# ---------------------------------------------------------------------------
# Canned engine reports (same shape as PersonalEngine.compare_segment)
# ---------------------------------------------------------------------------

def _report(verdict: str = "ok", beat_available: bool = True) -> dict:
    return {
        "verdict": verdict, "seg_idx": 5, "n_history": 30, "n_recent": 32,
        "recent_k": 32, "limited_history": False, "n_real_change": 1,
        "features": {
            "pulse_amp_mean": {
                "value": 0.43, "personal_median": 0.55,
                "delta_vs_long": -0.12, "delta_vs_recent": -0.05,
                "threshold": 0.05, "threshold_source": "personal_mad",
                "verdict_long": "real_change", "verdict_recent": "real_change"},
            "kurtosis_mean": {
                "value": 4.0, "personal_median": 3.0,
                "delta_vs_long": 1.0, "delta_vs_recent": None,
                "threshold": 0.4, "threshold_source": "personal_mad",
                "verdict_long": "real_change", "verdict_recent": "stable"},
            "hr_bpm_mean": {
                "value": 72.0, "personal_median": 72.1,
                "delta_vs_long": -0.1, "delta_vs_recent": 0.0,
                "threshold": 2.0, "threshold_source": "mdc_reference",
                "verdict_long": "stable", "verdict_recent": "stable"},
        },
        "beat_layer": ({"available": True, "n_valid_cur": 12,
                        "n_beats_cur": 16, "valid_ratio": 0.75,
                        "outlier_rate_median": 0.08,
                        "beat_bp": {"sbp_median": 75.2,
                                     "sbp_iqr": [72.2, 77.8],
                                     "dbp_median": 38.6}}
                       if beat_available
                       else {"available": False, "reason": "no beat table"}),
        "drift": [{"feature": "perfusion_index_mean",
                   "exceeded": 6, "of": 32}],
    }


# ---------------------------------------------------------------------------
# P1: volatile feature set (Phase 3 walk-forward L1)
# ---------------------------------------------------------------------------

class TestVolatileFeatures:
    def test_phase3_high_fp_set(self):
        # walk-forward test split: personal 3σ false-positive rate > 10%
        assert VOLATILE_FEATURES == {
            "kurtosis", "perfusion_index", "slope_systolic_n", "crest_factor",
            "t_dia_ms", "ni_dn", "ecg_hr", "intensity_ratio_s_d"}


# ---------------------------------------------------------------------------
# P2: evidence formatting
# ---------------------------------------------------------------------------

class TestFormatAssessmentBlocks:
    def test_longitudinal_blocks(self):
        text = format_assessment_blocks(_report())
        assert "Verdict: ok" in text
        assert "n_history=30" in text and "n_recent=32" in text
        assert "Real-change features: 1" in text or "Real-change features: 2" in text
        # changed vs stable split, engine numbers verbatim
        assert "CHANGED vs personal history" in text
        assert "pulse_amp_mean" in text and "thr=0.05" in text
        assert "Stable features" in text and "hr_bpm_mean" in text
        # beat layer + drift + narrative order
        assert "Beat layer (axis ①)" in text and "valid_ratio=0.75" in text
        assert "SBP median 75.2" in text
        assert "DRIFT" in text and "6/32" in text
        assert "Narrative order" in text and "safety note" in text

    def test_volatile_marker(self):
        # engine feature keys carry the _mean suffix; the marker must still
        # fire for the VOLATILE base name
        text = format_assessment_blocks(_report())
        assert "[VOLATILE — downgrade]" in text  # kurtosis_mean

    def test_beat_layer_unavailable_reason(self):
        text = format_assessment_blocks(_report(beat_available=False))
        assert "beat layer unavailable: no beat table" in text

    def test_legacy_vai_shape_no_longer_renders(self):
        # 证伪的 VAI 分支已从 format_assessment_blocks 删除：vascular_age
        # 形态不再被渲染为 VAI 叙事（也不会误入纵向块抛幻影）。
        out = format_assessment_blocks({
            "vascular_age": {"vai": 55, "interpretation": "偏高"},
            "confidence": "medium", "baseline_source": "vital"})
        assert "VAI" not in out


# ---------------------------------------------------------------------------
# P3: capture_findings — engine facts only
# ---------------------------------------------------------------------------

def _submit_msg(**args) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "submit_findings", "args": args, "id": "call_submit"}])


class TestCaptureFindings:
    def test_only_engine_flagged_with_verbatim_numbers(self):
        state = {
            "compare_report": _report(),
            "messages": [_submit_msg(changed_features=[
                {"feature": "pulse_amp_mean", "delta": 999.0,   # fabricated
                 "threshold": 1.0, "direction": "x",
                 "beat_evidence": "outlier rate 0.2"},
                {"feature": "hr_bpm_mean", "delta": 0.1,       # stable → dropped
                 "threshold": 2.0, "direction": "increased"},
                {"feature": "made_up_feature"},                 # unknown → dropped
            ], stable_note="稳定", bp_note="SBP 74.9",
               follow_up_suggestion="复查")],
        }
        f = _capture_findings(state)["findings"]
        assert len(f["changed_features"]) == 1
        cf = f["changed_features"][0]
        assert cf["feature"] == "pulse_amp_mean"
        assert cf["delta"] == -0.12 and cf["threshold"] == 0.05  # engine's
        assert cf["direction"] == "decreased"                    # from sign
        assert cf["beat_evidence"] == "outlier rate 0.2"
        assert f["stable_note"] is None      # change narrative supersedes
        assert f["bp_note"] == "SBP 74.9"

    def test_mean_suffix_tolerance(self):
        state = {"compare_report": _report(),
                 "messages": [_submit_msg(changed_features=[
                     {"feature": "pulse_amp", "delta": 0, "threshold": 0,
                      "direction": "x"}])]}
        f = _capture_findings(state)["findings"]
        assert [i["feature"] for i in f["changed_features"]] == ["pulse_amp_mean"]

    def test_wrapped_findings_arg_tolerance(self):
        # some LLMs wrap the whole object in one extra "findings" layer
        state = {"compare_report": _report(),
                 "messages": [_submit_msg(findings={
                     "changed_features": [
                         {"feature": "pulse_amp_mean", "delta": 0,
                          "threshold": 0, "direction": "x"}],
                     "follow_up_suggestion": "复查"})]}
        f = _capture_findings(state)["findings"]
        assert len(f["changed_features"]) == 1
        assert f["follow_up_suggestion"] == "复查"

    @pytest.mark.parametrize("verdict", ["no_history",
                                         "current_segment_low_quality"])
    def test_verdict_gate_blocks_change_claims(self, verdict):
        state = {"compare_report": _report(verdict=verdict),
                 "messages": [_submit_msg(changed_features=[
                     {"feature": "pulse_amp_mean", "delta": 0,
                      "threshold": 0, "direction": "x"}],
                     stable_note="稳定")]}
        f = _capture_findings(state)["findings"]
        assert f["changed_features"] == []
        assert f["stable_note"] is None   # below "ok" nothing is asserted

    def test_beat_evidence_dropped_when_layer_unavailable(self):
        state = {"compare_report": _report(beat_available=False),
                 "messages": [_submit_msg(changed_features=[
                     {"feature": "pulse_amp_mean", "delta": 0,
                      "threshold": 0, "direction": "x",
                      "beat_evidence": "cannot exist"}])]}
        f = _capture_findings(state)["findings"]
        assert f["changed_features"][0]["beat_evidence"] is None

    def test_no_submit_call_no_findings(self):
        assert _capture_findings({"compare_report": _report(),
                                  "messages": []})["findings"] is None


# ---------------------------------------------------------------------------
# P4: prompt discipline
# ---------------------------------------------------------------------------

class TestPromptDiscipline:
    def test_no_diagnosis_no_reversal(self):
        assert "NEVER diagnose disease" in CARDIOVASCULAR_PROMPT
        assert "NEVER reverse a group BP baseline" in CARDIOVASCULAR_PROMPT

    def test_rho_is_weak_reference(self):
        assert "WEAK reference" in CARDIOVASCULAR_PROMPT

    def test_volatile_downgrade_rule(self):
        assert "volatile_features" in CARDIOVASCULAR_PROMPT
        assert "downgrade" in CARDIOVASCULAR_PROMPT

    def test_verdict_gate(self):
        # v3.2: insufficient_history is no longer a verdict — the engine
        # reports with an insufficient_power note instead of gating
        for v in ("no_history", "insufficient_power"):
            assert v in CARDIOVASCULAR_PROMPT
        assert "insufficient_history" not in CARDIOVASCULAR_PROMPT

    def test_submission_protocol(self):
        assert "submit_findings" in CARDIOVASCULAR_PROMPT
        assert "LAST tool call" in CARDIOVASCULAR_PROMPT

    def test_agent2_writes_no_prose(self):
        # wording moved to the Supervisor — no sentence-count template here
        assert "3-5 sentences" not in CARDIOVASCULAR_PROMPT
        # disproven capabilities are gone from the prompt
        for dead in ("vascular age", "subtype", "similar_patients"):
            assert dead not in CARDIOVASCULAR_PROMPT

    def test_population_position_gated_discipline(self):
        # person layer: gated background reference, never a change/diagnosis
        # basis; suppressed features must stay silent at population level;
        # dual reference (personal-median ICC(2,k) gate + single-segment
        # fallback) and population_drift semantics are spelled out
        assert "population_position" in CARDIOVASCULAR_PROMPT
        assert "gated background reference" in CARDIOVASCULAR_PROMPT
        assert "suppressed" in CARDIOVASCULAR_PROMPT
        assert "ICC(2,k)>=0.6" in CARDIOVASCULAR_PROMPT
        assert "population_drift" in CARDIOVASCULAR_PROMPT
        assert "single_segment" in CARDIOVASCULAR_PROMPT


# ---------------------------------------------------------------------------
# P5: graph flow with a scripted LLM (no network, engine monkeypatched)
# ---------------------------------------------------------------------------

class ToolCallLLM(FakeMessagesListChatModel):
    """Scripted responses — each interpret-node invoke pops one response."""

    def bind_tools(self, *a, **kw):
        return self


def _submit_ai(**args) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "submit_findings", "args": args, "id": "call_submit"}])


def _tool_ai(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call_{name}"}])


@pytest.fixture
def patched_engine(monkeypatch):
    """compare_segment is deterministic — feed the canned report so graph
    tests never touch the parquet store."""
    monkeypatch.setattr(PersonalEngine, "compare_segment",
                        lambda self, ds, sid, seg_idx, recent_k=None: _report())
    monkeypatch.setattr(ca, "_IDENTITY", {})
    yield


def _invoke(g, **state_extra):
    return g.invoke({"source": "vital", "subject_id": "p003131",
                     "seg_idx": 5,
                     "messages": [HumanMessage(content="和历史比有什么变化？")]},
                    **state_extra)


class TestAgent2Graph:
    def test_submit_routes_straight_to_capture(self, patched_engine):
        llm = ToolCallLLM(responses=[_submit_ai(
            changed_features=[{"feature": "pulse_amp_mean", "delta": -0.12,
                               "threshold": 0.05, "direction": "decreased"}],
            follow_up_suggestion="建议复查")])
        g = build_cardiovascular_graph(llm=llm)
        stages, state = [], None
        for chunk in g.stream({"source": "vital", "subject_id": "p003131",
                               "seg_idx": 5,
                               "messages": [HumanMessage(content="分析")]},
                              stream_mode="updates"):
            stages.extend(chunk.keys())
            state = chunk
        # submit ends the run — NO tools_assess, no loop
        assert stages == ["assess", "interpret", "capture_findings"]
        f = state["capture_findings"]["findings"]
        assert f["changed_features"][0]["feature"] == "pulse_amp_mean"
        assert f["follow_up_suggestion"] == "建议复查"

    def test_assess_installs_identity_and_report(self, patched_engine):
        g = build_cardiovascular_graph(
            llm=ToolCallLLM(responses=[AIMessage(content="done")]))
        state = _invoke(g)
        assert state["compare_report"]["verdict"] == "ok"
        assert state["identity"] == {"ds": "vital", "sid": "p003131",
                                     "seg_idx": 5}
        # module-global fallback for tools running in the ToolNode pool
        assert ca._IDENTITY["sid"] == "p003131"
        assert state.get("findings") is None   # plain text → nothing captured

    def test_analysis_loop_runs_tools_then_submit(self, patched_engine):
        llm = ToolCallLLM(responses=[
            _tool_ai("compare_segment", seg_idx=4),   # re-invoke another seg
            _submit_ai(changed_features=[
                {"feature": "pulse_amp_mean", "delta": 0, "threshold": 0,
                 "direction": "decreased"}]),
        ])
        g = build_cardiovascular_graph(llm=llm)
        stages = []
        for chunk in g.stream({"source": "vital", "subject_id": "p003131",
                               "seg_idx": 5,
                               "messages": [HumanMessage(content="再看第4段")]},
                              stream_mode="updates"):
            stages.extend(chunk.keys())
        assert "tools_assess" in stages
        assert stages[-1] == "capture_findings"

    def test_tool_loop_history_pairs_for_llm(self, patched_engine):
        """Regression: after a tool turn the LLM must receive the assistant
        tool_calls message immediately before its ToolMessage — the real API
        rejects orphaned tool messages (openai 400)."""
        seen = []

        class CheckingLLM(ToolCallLLM):
            def invoke(self, messages, **kw):
                seen.append(list(messages))
                return super().invoke(messages, **kw)

        llm = CheckingLLM(responses=[
            _tool_ai("compare_segment", seg_idx=4),
            _submit_ai(changed_features=[]),
        ])
        g = build_cardiovascular_graph(llm=llm)
        state = _invoke(g)
        assert len(seen) == 2    # decision, then submission after the tool turn
        msgs2 = seen[1]
        for i, m in enumerate(msgs2):
            if getattr(m, "type", "") == "tool":
                assert i > 0, f"orphaned tool message at {i}: {m}"
                prev = msgs2[i - 1]
                assert getattr(prev, "type", "") == "ai" and prev.tool_calls
        assert state.get("findings") is not None  # loop completed

    def test_unknown_feature_submission_yields_empty(self, patched_engine):
        llm = ToolCallLLM(responses=[_submit_ai(changed_features=[
            {"feature": "hallucinated_feat", "delta": 1.0,
             "threshold": 0.1, "direction": "increased"}])])
        g = build_cardiovascular_graph(llm=llm)
        state = _invoke(g)
        assert state["findings"]["changed_features"] == []

    def test_population_position_tool_in_graph_loop(self, patched_engine,
                                                    monkeypatch):
        """Person layer wired in: the graph executes population_position via
        the ToolNode (identity travels through the RunnableConfig bridge)."""
        canned = {"verdict": "ok", "seg_idx": 5, "n_history": 30,
                  "basis": "personal_median",
                  "positions": {"pulse_amp": {"percentile": "P5-P25",
                                              "value_basis": "personal_median",
                                              "k_used": 30,
                                              "reliability_used": 0.92}},
                  "recent_positions": {}, "population_drift": [],
                  "suppressed": {"kurtosis": "no ICC decomposition "
                                              "for this feature"}}
        monkeypatch.setattr(PersonalEngine, "population_position",
                            lambda self, ds, sid, seg_idx,
                            age=None, gender=None: canned)
        llm = ToolCallLLM(responses=[
            _tool_ai("population_position"),
            _submit_ai(changed_features=[]),
        ])
        g = build_cardiovascular_graph(llm=llm)
        tool_msg = None
        for chunk in g.stream(
                {"source": "vital", "subject_id": "p003131", "seg_idx": 5,
                 "messages": [HumanMessage(content="我在人群里什么位置？")]},
                config={"configurable": {"agent2_identity": {
                    "ds": "vital", "sid": "p003131", "seg_idx": 5}}},
                stream_mode="updates"):
            for upd in chunk.values():
                for m in upd.get("messages", []):
                    if getattr(m, "type", "") == "tool":
                        tool_msg = m
        assert tool_msg is not None and "percentile" in tool_msg.content


# ---------------------------------------------------------------------------
# P5c: population_position engine semantics (dual reference, synthetic data)
# ---------------------------------------------------------------------------

class _FakePercentileEngine:
    """ICC(1,1)=icc1 per feature; ICC(2,k) via the real k-segment formula;
    band by value threshold (0.65)."""

    def __init__(self, iccs: dict):
        self.iccs = iccs

    def _icc(self, source, feature):
        return self.iccs.get(feature)

    def _icc_2k(self, source, feature, k):
        i = self.iccs.get(feature)
        if i is None or k < 1:
            return i
        return round(i * k / (i * k + (1 - i)), 4)

    def query_stratified(self, feature, value, source="vital",
                         age=None, gender=None):
        pct = "P5-P25" if value < 0.65 else "P25-P75"
        return {"feature": feature, "value": value, "percentile": pct}


def _pop_engine(segs, iccs):
    eng = PersonalEngine(percentile_engine=_FakePercentileEngine(iccs))
    eng.load_subject_segments = lambda ds, sid: segs  # synthetic, no parquet
    # no demographics file lookups in synthetic tests
    eng._demo_cache = {"vital": {}, "mimic": {}, "aurora": {}}
    return eng


class TestPopulationPositionEngine:
    def test_median_basis_unlocks_low_icc_and_flags_drift(self):
        # ICC(1,1)=0.3 fails the old single-segment gate, but ICC(2,9)=0.79
        # unlocks the personal median; recent-7 sits one band higher → drift
        import pandas as pd
        # segs 0-8 in history: median 0.3 (low band); recent segs 2-8:
        # [0.3,0.3,0.3,0.9,0.9,0.9,0.9] median 0.9 (normal band)
        vals = [0.3] * 5 + [0.9] * 5
        segs = pd.DataFrame({"seg_idx": range(10),
                             "pulse_amp_mean": vals,
                             "quality_tier": ["good"] * 10})
        eng = _pop_engine(segs, {"pulse_amp": 0.3})
        # v3.2: default recent_k = causal K (32) covers all 9 history segs →
        # recent == long-term median → no drift; use the explicit 7-seg
        # override to exercise the drift contrast on synthetic short history
        out = eng.population_position("vital", "p1", 9, recent_k=7)
        assert out["basis"] == "personal_median" and out["verdict"] == "ok"
        p = out["positions"]["pulse_amp"]
        assert p["percentile"] == "P5-P25"      # long-term median 0.3
        assert p["value_basis"] == "personal_median" and p["k_used"] == 9
        assert out["recent_positions"]["pulse_amp"]["percentile"] == "P25-P75"
        assert out["population_drift"] == [{"feature": "pulse_amp",
                                            "long_term_band": "P5-P25",
                                            "recent_band": "P25-P75"}]

    def test_default_recent_k_equals_causal_window(self):
        # 默认 recent_k = causal 流式头 K（32）：合成 9 段历史全在窗内，
        # recent median == 长期 median → 无 drift 对比差异
        import pandas as pd
        vals = [0.3] * 5 + [0.9] * 5
        segs = pd.DataFrame({"seg_idx": range(10),
                             "pulse_amp_mean": vals,
                             "quality_tier": ["good"] * 10})
        eng = _pop_engine(segs, {"pulse_amp": 0.3})
        out = eng.population_position("vital", "p1", 9)
        assert out["recent_positions"]["pulse_amp"]["percentile"] == "P5-P25"
        assert out["population_drift"] == []

    def test_short_history_single_segment_gate(self):
        import pandas as pd
        segs = pd.DataFrame({"seg_idx": [0, 1],
                             "pulse_amp_mean": [0.4, 0.9],
                             "quality_tier": ["good", "good"]})
        eng = _pop_engine(segs, {"pulse_amp": 0.3, "pulse_width_50": 0.7})
        segs["pulse_width_50_mean"] = [0.4, 0.9]
        out = eng.population_position("vital", "p1", 1)
        assert out["basis"] == "single_segment"
        # ICC(1,1)=0.3 suppressed; 0.7 positioned from the CURRENT segment
        assert "pulse_amp" in out["suppressed"]
        p = out["positions"]["pulse_width_50"]
        assert p["value_basis"] == "single_segment" and p["k_used"] == 1
        assert p["percentile"] == "P25-P75"      # current value 0.9
        assert out["recent_positions"] == {}     # no contrast without median

    def test_failed_segments_excluded_from_history(self):
        import pandas as pd
        segs = pd.DataFrame({
            "seg_idx": range(6),
            "pulse_amp_mean": [0.9, 0.9, 0.9, 0.9, 0.3, 0.5],
            "quality_tier": ["good"] * 4 + ["failed", "good"]})
        eng = _pop_engine(segs, {"pulse_amp": 0.9})
        out = eng.population_position("vital", "p1", 5)
        # failed seg 4 dropped → k=4, median still 0.9
        p = out["positions"]["pulse_amp"]
        assert p["k_used"] == 4 and p["percentile"] == "P25-P75"


# ---------------------------------------------------------------------------
# P5b: population_position tool wrapper (person layer)
# ---------------------------------------------------------------------------

class TestPopulationPositionTool:
    def test_no_subject_returns_error(self, monkeypatch):
        monkeypatch.setattr(ca, "_IDENTITY", {})
        out = _population_position()
        assert out == {"error": "No subject in context. "
                                "Agent 1 must run first."}

    def test_delegates_to_engine_with_identity(self, monkeypatch):
        monkeypatch.setattr(ca, "_IDENTITY", {})
        seen = {}

        def fake_pp(self, ds, sid, seg_idx, age=None, gender=None):
            seen.update(ds=ds, sid=sid, seg_idx=seg_idx)
            return {"verdict": "all_suppressed", "positions": {},
                    "suppressed": {"kurtosis": "ICC 0.31 < 0.6"}}

        monkeypatch.setattr(PersonalEngine, "population_position", fake_pp)
        cfg = {"configurable": {"agent2_identity": {
            "ds": "vital", "sid": "p003131", "seg_idx": 5}}}
        out = _population_position(config=cfg)
        assert out["verdict"] == "all_suppressed"
        assert seen == {"ds": "vital", "sid": "p003131", "seg_idx": 5}
        # explicit seg_idx overrides the in-context segment
        out = _population_position(seg_idx=9, config=cfg)
        assert seen["seg_idx"] == 9

    def test_population_note_passthrough_in_capture(self):
        # population_note rides findings verbatim (like bp_note) so the
        # Supervisor can word the gated population position
        state = {"compare_report": _report(), "messages": [
            AIMessage(content="", tool_calls=[{
                "name": "submit_findings", "id": "c1",
                "args": {"changed_features": [],
                         "population_note": "pulse_amp 约 P27（门控后）"}}])]}
        out = _capture_findings(state)["findings"]
        assert out["population_note"] == "pulse_amp 约 P27（门控后）"
        assert out["changed_features"] == []


# ---------------------------------------------------------------------------
# P6: Agent 1 → Agent 2 handoff (PPGContext.beat_summary)
# ---------------------------------------------------------------------------

class TestBeatSummary:
    def test_summary_with_beat_bp(self):
        ctx = PPGContext()
        ctx.num_beats, ctx.n_valid_beats = 4, 3
        ctx.ppg_quality, ctx.ppg_quality_tier = 0.8, "good"
        ctx.beat_records = [
            {"abp_sbp": 76.0, "abp_dbp": 39.0},
            {"abp_sbp": 74.0, "abp_dbp": 38.0},
            {"abp_sbp": float("nan"), "abp_dbp": None},   # invalid beat
        ]
        s = ctx.beat_summary()
        assert s["n_beats"] == 3 and s["n_valid"] == 3
        assert s["has_beat_bp"] is True
        assert s["beat_sbp_median"] == 75.0 and s["beat_dbp_median"] == 38.5

    def test_summary_without_bp(self):
        ctx = PPGContext()
        ctx.num_beats = 5
        ctx.beat_records = [{"hr_bpm": 70}] * 5
        s = ctx.beat_summary()
        assert s["has_beat_bp"] is False
        assert "beat_sbp_median" not in s
        assert s["n_beats"] == 5

    def test_empty_context(self):
        s = PPGContext().beat_summary()
        assert s["n_beats"] == 0 and s["valid_ratio"] is None


# ---------------------------------------------------------------------------
# P7: joint redundancy knowledge — narrative folding annotations
# (analysis steps 28/29; folding is wording-only, engine stays per-feature)
# ---------------------------------------------------------------------------

import json as _json

from ppg_agent.knowledge.clusters import (
    FeatureClusters,
    annotate_changes,
    annotate_positions,
)


@pytest.fixture
def synth_fc(tmp_path):
    """Synthetic KB: one joint cluster with an inverse member (mirrors the
    real t_sys_ms family) + one between-only pair."""
    kb = {
        "version": 2,
        "joint_clusters": [{
            "id": 2, "representative": "t_sys_ms",
            "members": ["pulse_width_90", "slope_systolic", "t_sys_ms"],
            "members_info": {
                "t_sys_ms": {"is_representative": True},
                "pulse_width_90": {"is_representative": False,
                                   "direction_vs_rep": "same"},
                "slope_systolic": {"is_representative": False,
                                   "direction_vs_rep": "inverse"},
            }}],
        "between_only_pairs": [
            {"a": "iqr", "b": "pulse_width_50", "direction": "same",
             "rho_between": 0.74, "rho_within_median": 0.47,
             "consistency": 0.45}],
    }
    (tmp_path / "feature_clusters.json").write_text(
        _json.dumps(kb), encoding="utf-8")
    return FeatureClusters(tmp_path)


class TestFeatureClustersKB:
    def test_missing_kb_is_inert(self, tmp_path):
        fc = FeatureClusters(tmp_path)
        assert not fc.available
        rep = {"features": {"pulse_amp_mean": {"verdict_long": "real_change",
                                                "delta_vs_long": -0.1}}}
        assert annotate_changes(rep, fc) is rep  # untouched

    def test_real_kb_shape(self):
        fc = FeatureClusters()  # knowledge/feature_clusters.json
        assert fc.available
        # step-29 outcome (方案B 重算 v3.1.1): 8 joint clusters, 52 between-only
        assert len(fc._clusters) == 8
        assert len(fc._between_pairs) == 52
        assert fc.representative("pulse_width_90") == "t_sys_ms"
        assert fc.direction_vs_rep("slope_systolic") == "inverse"
        assert fc.is_representative("t_sys_ms")
        # singletons are their own representative
        assert fc.representative("lasi") == "lasi"


class TestAnnotateChanges:
    def test_inverse_member_direction_adjusted(self, synth_fc):
        # width family expands (same members +) while slope_systolic DROPS
        # — inverse direction ⇒ both are cluster_consistent
        rep = {"features": {
            "t_sys_ms_mean": {"verdict_long": "real_change",
                              "delta_vs_long": 5.0},
            "pulse_width_90_mean": {"verdict_long": "real_change",
                                    "delta_vs_long": 0.2},
            "slope_systolic_mean": {"verdict_long": "real_change",
                                    "delta_vs_long": -0.3},
            "lasi_mean": {"verdict_long": "real_change",
                          "delta_vs_long": 0.05}}}
        out = annotate_changes(rep, synth_fc)
        f = out["features"]
        assert f["t_sys_ms_mean"]["cluster_consistency"] == "cluster_consistent"
        assert f["t_sys_ms_mean"]["is_representative"] is True
        assert f["slope_systolic_mean"]["cluster_consistency"] == "cluster_consistent"
        assert f["lasi_mean"]["cluster_consistency"] == "singleton"
        fold = out["cluster_fold"][2]
        assert fold["representative"] == "t_sys_ms"
        assert len(fold["flagged"]) == 3 and fold["consistent"]

    def test_direction_divergent_keeps_members_separate(self, synth_fc):
        # slope_systolic RISES while the width family expands: inverse member
        # moving WITH the same-signed members is a real within-family split
        rep = {"features": {
            "t_sys_ms_mean": {"verdict_long": "real_change",
                              "delta_vs_long": 5.0},
            "slope_systolic_mean": {"verdict_long": "real_change",
                                    "delta_vs_long": 0.3}}}
        out = annotate_changes(rep, synth_fc)
        f = out["features"]
        assert f["t_sys_ms_mean"]["cluster_consistency"] == "direction_divergent"
        assert f["slope_systolic_mean"]["cluster_consistency"] == "direction_divergent"
        assert out["cluster_fold"][2]["consistent"] is False

    def test_isolated_when_no_co_member_flagged(self, synth_fc):
        rep = {"features": {
            "t_sys_ms_mean": {"verdict_long": "real_change",
                              "delta_vs_long": 5.0},
            "pulse_width_90_mean": {"verdict_long": "stable",
                                    "delta_vs_long": 0.0}}}
        out = annotate_changes(rep, synth_fc)
        assert out["features"]["t_sys_ms_mean"]["cluster_consistency"] == "isolated"


class TestAnnotatePositions:
    def test_fold_band_matching_family(self, synth_fc):
        out = {"verdict": "ok",
               "positions": {
                   "t_sys_ms": {"percentile": "P25-P75"},
                   "pulse_width_90": {"percentile": "P25-P75"},
                   "slope_systolic": {"percentile": "P75-P85"},
                   "iqr": {"percentile": "P5-P25"},
                   "pulse_width_50": {"percentile": "P5-P25"}},
               "recent_positions": {"t_sys_ms": {"percentile": "P25-P75"}}}
        res = annotate_positions(out, synth_fc)
        pf = res["position_fold"][0]
        assert pf["representative"] == "t_sys_ms"
        assert sorted(pf["members_positioned"]) == [
            "pulse_width_90", "slope_systolic", "t_sys_ms"]
        assert pf["band_match"] == ["pulse_width_90", "t_sys_ms"]
        # between-only pairs exposed only at population level
        assert res["between_only_note"]["pairs"][0]["a"] == "iqr"
        assert res["positions"]["t_sys_ms"]["is_representative"] is True
        assert res["recent_positions"]["t_sys_ms"][
            "cluster_representative"] == "t_sys_ms"


class TestClusterFoldingIntegration:
    def test_prompt_fold_discipline(self):
        assert "cluster_consistency" in CARDIOVASCULAR_PROMPT
        assert "direction_divergent" in CARDIOVASCULAR_PROMPT
        assert "cluster_representative" in CARDIOVASCULAR_PROMPT
        assert "between_only_note" in CARDIOVASCULAR_PROMPT

    def test_format_blocks_render_fold(self):
        rep = _report()
        rep["cluster_fold"] = {2: {"representative": "t_sys_ms",
                                   "flagged": ["t_sys_ms", "pulse_width_90"],
                                   "consistent": True}}
        rep["features"]["pulse_amp_mean"]["cluster_consistency"] = "isolated"
        text = format_assessment_blocks(rep)
        assert "Cluster folding" in text and "rep t_sys_ms" in text
        assert "direction-consistent" in text
        assert "[isolated]" in text

    def test_capture_carries_cluster_fields(self):
        report = _report()
        report["features"]["pulse_amp_mean"][
            "cluster_representative"] = "pulse_amp"
        report["features"]["pulse_amp_mean"][
            "cluster_consistency"] = "singleton"
        state = {"compare_report": report,
                 "messages": [_submit_msg(changed_features=[
                     {"feature": "pulse_amp_mean", "delta": 0,
                      "threshold": 0, "direction": "x"}])]}
        f = _capture_findings(state)["findings"]
        item = f["changed_features"][0]
        assert item["cluster_representative"] == "pulse_amp"
        assert item["cluster_consistency"] == "singleton"


# ---------------------------------------------------------------------------
# P8: physiological interpretability knowledge (feature_interpretation tool)
# — pure PPG (no BP labels needed); FORWARD labeled-group descriptions only
# ---------------------------------------------------------------------------

from ppg_agent.knowledge.interpretation import FeatureInterpretation


@pytest.fixture
def synth_interp(tmp_path):
    kb = {
        "version": 1,
        "discipline": "FORWARD description only",
        "features": {
            "reflection_index": {
                "mechanism": {"label": "Reflection index",
                              "rationale": "reflected waves",
                              "high_direction": "more_reflection"},
                "age_patterns": [{"dataset": "mimic",
                                  "trend": "随年龄升高", "n": 1000,
                                  "rho_age_sex_adjusted": 0.13}],
                "population_patterns": [
                    {"dataset": "mimic", "label": "mimic_ckd",
                     "label_name": "慢性肾病",
                     "in_that_group": "该标签组数值偏高", "d": 0.3,
                     "q_fdr": 0.001, "n_pos": 200, "n_neg": 800,
                     "adjusted": "age+sex (rank-ANCOVA)"}]},
            "t_sys_ms": {
                "mechanism": {"label": "Systolic time"},
                "age_patterns": [], "population_patterns": []},
        }}
    (tmp_path / "feature_interpretation.json").write_text(
        _json.dumps(kb), encoding="utf-8")
    return FeatureInterpretation(tmp_path)


class TestFeatureInterpretationKB:
    def test_missing_kb_is_inert(self, tmp_path):
        fi = FeatureInterpretation(tmp_path)
        assert not fi.available
        out = fi.interpret(["reflection_index"])
        assert out["available"] is False

    def test_real_kb_shape(self):
        fi = FeatureInterpretation()  # knowledge/feature_interpretation.json
        assert fi.available
        e = fi.entry("reflection_index_mean")  # _mean tolerated
        assert e["mechanism"]["label"] == "Reflection index"
        assert e["population_patterns"]  # FDR<0.05 labeled-group patterns
        assert all(p["in_that_group"].startswith("该标签组")
                   for p in e["population_patterns"])  # forward wording
        # every pattern carries effect size + FDR evidence
        assert all("d" in p and "q_fdr" in p
                   for p in e["population_patterns"])


class TestInterpret:
    def test_explicit_features_and_caveat(self, synth_interp):
        out = synth_interp.interpret(["reflection_index"])
        assert out["available"]
        assert "reflection_index" in out["features"]
        assert "FORWARD" in out["caveat"]
        assert out["discipline"].startswith("FORWARD")

    def test_member_carries_fold_note(self, synth_interp):
        fc = FeatureClusters()  # real joint clusters
        out = synth_interp.interpret(["pulse_width_90"], fc=fc)
        assert out["missing"] == ["pulse_width_90"]  # not in the synth KB
        out2 = FeatureInterpretation().interpret(["pulse_width_90"], fc=fc)
        entry = out2["features"]["pulse_width_90"]
        assert entry["cluster_representative"] == "t_sys_ms"
        assert "family" in entry["fold_note"]

    def test_default_falls_back_to_representatives(self, synth_interp):
        out = synth_interp.interpret(None, representatives=["t_sys_ms"])
        assert list(out["features"]) == ["t_sys_ms"]
        out2 = synth_interp.interpret(None)  # no clusters → first KB feats
        assert len(out2["features"]) == 2

    def test_unknown_feature_and_cap(self, synth_interp):
        out = synth_interp.interpret(["nonexistent_xyz"])
        assert out["features"] == {} and out["missing"] == ["nonexistent_xyz"]
        many = ["reflection_index"] * 20
        assert len(synth_interp.interpret(many)["features"]) == 1  # dedup


class TestInterpretationIntegration:
    def test_tool_works_without_identity(self):
        # pure PPG: no subject context needed at all
        from ppg_agent.agents.cardiovascular_agent import (
            _feature_interpretation)
        out = _feature_interpretation(["reflection_index"])
        assert out["available"] is True
        assert out["features"]["reflection_index"]["mechanism"][
            "label"] == "Reflection index"

    def test_tool_default_returns_representatives(self):
        from ppg_agent.agents.cardiovascular_agent import (
            _feature_interpretation)
        out = _feature_interpretation(None)
        assert out["available"] is True
        assert out["features"]  # joint-cluster representatives

    def test_prompt_carries_interpretation_discipline(self):
        assert "feature_interpretation" in CARDIOVASCULAR_PROMPT
        assert "NEVER " in CARDIOVASCULAR_PROMPT
        assert "NO blood-pressure labels" in CARDIOVASCULAR_PROMPT
        # forward-description-only wording for labeled groups
        assert "FORWARD descriptions" in CARDIOVASCULAR_PROMPT

    def test_capture_carries_interpretation_note(self):
        state = {"compare_report": _report(),
                 "messages": [_submit_msg(
                     changed_features=[],
                     interpretation_note="带高血压标签的人群中反射指数倾向偏高")]}
        f = _capture_findings(state)["findings"]
        assert f["interpretation_note"].startswith("带高血压标签")


# ---------------------------------------------------------------------------
# P9: population feature×BP association layer
# — step-30 rigorous KB (per-patient medians, age+sex adjusted, FDR),
#   aurora scenarios a/o kept apart, engine prior + interpretation tier
# ---------------------------------------------------------------------------


class TestPopulationBPAssociationKB:
    def test_kb_shape_and_sources(self):
        from ppg_agent.config import KNOWLEDGE_ROOT
        kb = _json.loads((KNOWLEDGE_ROOT / "population_bp_association.json")
                         .read_text(encoding="utf-8"))
        assert set(kb["patterns"]) == {"vital", "mimic", "aurora_a",
                                       "aurora_o"}  # a/o never pooled
        assert "NEVER" in kb["discipline"]  # forward-only, no BP estimate
        for source, feats in kb["patterns"].items():
            assert feats, f"{source} has no significant BP associations"
            for feat, targets in feats.items():
                for target, v in targets.items():
                    assert target in ("sbp", "dbp")
                    assert -1 <= v["rho_partial"] <= 1
                    assert v["q_fdr"] < 0.05 and v["n"] >= 12

    def test_engine_prior_uses_scenario_source(self):
        from ppg_agent.engine.personal_engine import PersonalEngine
        eng = PersonalEngine()
        # aurora a-scenario subject → aurora_a partition, rigorous prior
        prior = eng._rho_bp_top("aurora", "a002")
        assert prior and "per-patient" in prior["_provenance"]
        feats = {k: v for k, v in prior.items() if k != "_provenance"}
        assert feats  # aurora population BP prior no longer missing
        assert all(set(v) <= {"sbp", "dbp"} for v in feats.values())
        # o-scenario stays separate
        prior_o = eng._rho_bp_top("aurora", "o001")
        assert prior_o and "per-patient" in prior_o["_provenance"]
        # vital keeps working
        assert eng._rho_bp_top("vital", "p003131")


class TestInterpretationBPPatterns:
    def test_real_kb_has_bp_tier(self):
        fi = FeatureInterpretation()
        n_bp = 0
        for name in fi._feats:
            for p in fi.entry(name).get("bp_patterns", []):
                n_bp += 1
                assert p["correlation"].startswith("与血压呈弱")
                assert p["target"] in ("sbp", "dbp") and "q_fdr" in p
        assert n_bp > 0  # fourth evidence tier populated

    def test_loader_surfaces_bp_patterns_and_caveat(self):
        out = FeatureInterpretation().interpret()
        assert all("bp_patterns" in e for e in out["features"].values())
        assert "bp_patterns" in out["caveat"]  # no-BP-estimate warning

    def test_prompt_mentions_bp_patterns(self):
        assert "bp_patterns" in CARDIOVASCULAR_PROMPT


# ---------------------------------------------------------------------------
# P10: same-age/gender-group positioning — demographics auto-filled into
# population_position so positions resolve against stratified baselines
# ---------------------------------------------------------------------------


class _CapturingPE(_FakePercentileEngine):
    """Records the age/gender the position query actually received."""

    def __init__(self, iccs):
        super().__init__(iccs)
        self.calls: list[tuple] = []

    def query_stratified(self, feature, value, source="vital",
                         age=None, gender=None):
        self.calls.append((feature, age, gender))
        return super().query_stratified(feature, value, source, age, gender)


class TestPopulationPositionDemographics:
    def test_autofill_and_stratified_query(self):
        import pandas as pd
        segs = pd.DataFrame({"seg_idx": range(6),
                             "pulse_amp_mean": [0.9] * 5 + [0.5],
                             "quality_tier": ["good"] * 6})
        pe = _CapturingPE({"pulse_amp": 0.9})
        eng = PersonalEngine(percentile_engine=pe)
        eng.load_subject_segments = lambda ds, sid: segs
        eng._demo_cache = {"vital": {"p1": (65.0, "M")}}
        out = eng.population_position("vital", "p1", 5)
        assert out["age"] == 65.0 and out["gender"] == "M"
        assert out["age_stratified"] is True
        # every position query received the demographics
        assert all(age == 65.0 and g == "M" for _, age, g in pe.calls)

    def test_explicit_args_take_precedence(self):
        import pandas as pd
        segs = pd.DataFrame({"seg_idx": range(6),
                             "pulse_amp_mean": [0.9] * 5 + [0.5],
                             "quality_tier": ["good"] * 6})
        pe = _CapturingPE({"pulse_amp": 0.9})
        eng = PersonalEngine(percentile_engine=pe)
        eng.load_subject_segments = lambda ds, sid: segs
        eng._demo_cache = {"vital": {"p1": (65.0, "M")}}
        out = eng.population_position("vital", "p1", 5, age=40.0,
                                      gender="F")
        assert out["age"] == 40.0 and out["gender"] == "F"

    def test_missing_demographics_degrades_to_global(self):
        import pandas as pd
        segs = pd.DataFrame({"seg_idx": range(6),
                             "pulse_amp_mean": [0.9] * 5 + [0.5],
                             "quality_tier": ["good"] * 6})
        pe = _CapturingPE({"pulse_amp": 0.9})
        eng = PersonalEngine(percentile_engine=pe)
        eng.load_subject_segments = lambda ds, sid: segs
        eng._demo_cache = {"vital": {}}
        out = eng.population_position("vital", "p_unknown", 5)
        assert out["age"] is None and out["age_stratified"] is False
        assert out["verdict"] == "ok"  # global baseline still resolves

    def test_real_demographics_lookup(self):
        # 人群位置 demographics 来自 gitignore 的可重生产物
        # analysis/01_data_preparation/output/{ds}_merged.parquet（非仓内数据）。
        # 干净 checkout / 磁盘清理后该文件不存在 → 引擎按设计降级为全局基线，
        # 此处跳过而非失败（跑过 step-01 管线的机器上仍会断言真实分层）。
        merged = (Path(__file__).parent.parent / "analysis"
                  / "01_data_preparation" / "output" / "vital_merged.parquet")
        if not merged.exists():
            pytest.skip("vital_merged.parquet 未物化（可经 analysis/01 重生成）")
        eng = PersonalEngine()
        age, gender = eng.subject_demographics("vital", "p003131")
        assert age is not None and age > 0  # age column fully populated
        assert gender in ("M", "F")
        assert eng.subject_demographics("vital", "no_such_subject") \
            == (None, None)

    def test_prompt_mentions_age_group_positioning(self):
        assert "same age/gender group" in CARDIOVASCULAR_PROMPT


# ---------------------------------------------------------------------------
# P11: age_direction — deterministic position-vs-age-trend annotation,
# fixed wording, per-source matching (trend sign flips across datasets)
# ---------------------------------------------------------------------------


class TestAgeDirectionAnnotation:
    def _eng(self):
        eng = PersonalEngine(percentile_engine=_FakePercentileEngine({}))
        eng._age_kb = {
            "vital": {"lasi": {"rho": -0.25, "n": 2484},
                      "pulse_width_50": {"rho": 0.29, "n": 2484}},
            "aurora_a": {"lasi": {"rho": 0.23, "n": 534}},
        }
        return eng

    def test_fixed_wording_all_branches(self):
        eng = self._eng()
        # high band + rising trend -> consistent
        ad = eng._age_direction("vital", "pulse_width_50", "P75-P85")
        assert ad["consistency"] == "consistent"
        assert ad["statement"].endswith("方向一致")
        assert "偏高" in ad["statement"] and "随年龄升高" in ad["statement"]
        # high band + falling trend -> opposite
        ad = eng._age_direction("vital", "lasi", "P95+")
        assert ad["consistency"] == "opposite"
        assert ad["statement"].endswith("方向相反")
        # low band + falling trend -> consistent
        ad = eng._age_direction("vital", "lasi", "P5-P25")
        assert ad["consistency"] == "consistent" and "偏低" in ad["statement"]
        # center band -> not applicable
        ad = eng._age_direction("vital", "lasi", "P25-P75")
        assert ad["consistency"] == "not_applicable"
        # no pattern for this source/feature -> silent skip
        assert eng._age_direction("vital", "unknown_feature", "P95+") is None
        assert eng._age_direction("mimic", "lasi", "P95+") is None

    def test_per_source_flip_respected(self):
        eng = self._eng()
        # same feature, same band, different source -> opposite verdicts
        v = eng._age_direction("vital", "lasi", "P85-P95")
        a = eng._age_direction("aurora_a", "lasi", "P85-P95")
        assert v["trend"] == "随年龄降低" and a["trend"] == "随年龄升高"
        assert v["consistency"] == "opposite"
        assert a["consistency"] == "consistent"

    def test_population_position_carries_annotation(self):
        import pandas as pd
        segs = pd.DataFrame({"seg_idx": range(6),
                             "pulse_amp_mean": [0.5] * 5 + [0.5],
                             "quality_tier": ["good"] * 6})
        eng = _pop_engine(segs, {"pulse_amp": 0.9})
        eng._age_kb = {"vital": {"pulse_amp": {"rho": -0.3, "n": 100}}}
        out = eng.population_position("vital", "p1", 5)
        ad = out["positions"]["pulse_amp"]["age_direction"]
        # median 0.5 -> P5-P25 (low band) with falling trend: consistent
        assert ad["consistency"] == "consistent"
        assert "方向一致" in ad["statement"]
        assert "age_direction_note" in out          # discipline note
        assert "CROSS-SECTIONAL" in out["age_direction_note"]

    def test_real_kb_cross_source_flip(self):
        eng = PersonalEngine()
        kb = eng._age_patterns()
        assert kb["vital"]["lasi"]["rho"] < 0 < kb["aurora_a"]["lasi"]["rho"]
        assert kb["vital"]["pulse_width_50"]["rho"] > 0
        assert kb["aurora_a"]["pulse_width_50"]["rho"] < 0

    def test_prompt_mentions_age_direction(self):
        assert "age_direction" in CARDIOVASCULAR_PROMPT
        assert "vascular-age" in CARDIOVASCULAR_PROMPT


# ---------------------------------------------------------------------------
# P12: data-quality gates — implausible BP labels masked at engine load;
# step-01 materialization carries an explicit exclusion manifest
# ---------------------------------------------------------------------------


class TestDataQualityGates:
    def test_sanitize_bp_masks_outliers(self):
        import pandas as pd
        from ppg_agent.engine.personal_engine import _sanitize_bp
        df = pd.DataFrame({
            "sbp_label": [120.0, 13.0, 300.0, 104.0, 90.0],
            "dbp_label": [80.0, 8.0, 170.0, 16.0, 95.0],
        })
        out = _sanitize_bp(df)
        # in-range values untouched
        assert out.loc[0, "sbp_label"] == 120.0
        assert out.loc[0, "dbp_label"] == 80.0
        # out-of-range masked
        assert pd.isna(out.loc[1, "sbp_label"]) and pd.isna(out.loc[1, "dbp_label"])
        assert pd.isna(out.loc[2, "sbp_label"]) and pd.isna(out.loc[2, "dbp_label"])
        assert pd.isna(out.loc[3, "dbp_label"])   # 16 < 30
        # dbp > sbp masked on the dbp side
        assert pd.isna(out.loc[4, "dbp_label"]) and out.loc[4, "sbp_label"] == 90.0

    def test_sanitize_bp_handles_beat_columns_and_absence(self):
        import pandas as pd
        from ppg_agent.engine.personal_engine import _sanitize_bp
        beats = pd.DataFrame({"abp_sbp": [110.0, 40.0], "abp_dbp": [70.0, 25.0]})
        out = _sanitize_bp(beats)
        assert pd.isna(out.loc[1, "abp_sbp"]) and pd.isna(out.loc[1, "abp_dbp"])
        # no BP columns at all → unchanged, no crash
        plain = pd.DataFrame({"lasi": [1.0, 2.0]})
        assert _sanitize_bp(plain).equals(plain)

    def test_materialization_manifests_exist(self):
        import json
        out_dir = Path(__file__).parent.parent / "analysis" \
            / "01_data_preparation" / "output"
        # exclusions.json 是 gitignore 的可重生产物（step-01 物化门记录），
        # 不在仓内；缺失时跳过而非失败。
        if not (out_dir / "vital_exclusions.json").exists():
            pytest.skip("step-01 物化清单未生成（可经 analysis/01 重生成）")
        for ds in ("vital", "mimic"):
            m = json.loads((out_dir / f"{ds}_exclusions.json").read_text())
            assert m["bp_outlier_rules"] and m["demographics_gate"]
            assert m["kept"]["subjects"] > 0
        # the known invalid-age subject was excluded, not silently kept
        m = json.loads((out_dir / "mimic_exclusions.json").read_text())
        assert "age=0_invalid" in ";".join(
            ";".join(e["reasons"]) for e in m["excluded_subjects"].values())
