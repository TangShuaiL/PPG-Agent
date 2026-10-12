"""Agent 2 — Longitudinal Personal Comparison Analyst.

Receives the current segment from Agent 1 (via PPGContext) and compares it
against the SAME SUBJECT's stored history using the deterministic
PersonalEngine (three-layer comparison: beat → segment → person).

Agent 2 emits NO natural-language interpretation — the Supervisor synthesis
phase (web/api/services/chat_supervisor.synthesize_reply) words all data.

Refactored (plan v3): the disproven capabilities (VAI reporting, phenotype
subtypes, disease association, interpretation_rules, window_assess) were
removed from this agent.  compute_vai remains only in the engine layer
(ppg_agent/engine/percentile_engine.py) for the analysis/ reproduction
scripts; the frontend VAI panel and the /reports/vai + /reports/trend
endpoints have since been deleted as well.

Architecture:
  START → assess (deterministic compare_segment — axes ①+②)
        → interpret (LLM decides follow-up analyses via tools)
        → tools_assess ⇄ interpret   (≤8 tool turns)
        → capture_findings (deterministic: validate + store structured
                            findings from the submit_findings call)
        → END
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from ppg_agent.llm_service import get_llm

# Phase 3 walk-forward validation (test split, 2.5M judgements): these
# features exceeded the 10% false-positive fence on personal 3σ thresholds
# (kurtosis 0.35, perfusion_index 0.25, …) — they fluctuate physiologically
# and must be de-weighted in any change narrative.
VOLATILE_FEATURES = {
    "kurtosis", "perfusion_index", "slope_systolic_n", "crest_factor",
    "t_dia_ms", "ni_dn", "ecg_hr", "intensity_ratio_s_d",
}

# Joint redundancy knowledge (analysis steps 28/29 — within-person ∩
# between-person gates).  Narrative folding only; the decision engine
# stays per-feature.
_CLUSTERS = None


def _clusters():
    global _CLUSTERS
    if _CLUSTERS is None:
        from ppg_agent.knowledge.clusters import FeatureClusters
        _CLUSTERS = FeatureClusters()
    return _CLUSTERS

# Physiological interpretability knowledge (train-split association tables
# + feature_rules) — pure PPG, needs no blood-pressure labels.
_INTERP = None


def _interpretation():
    global _INTERP
    if _INTERP is None:
        from ppg_agent.knowledge.interpretation import FeatureInterpretation
        _INTERP = FeatureInterpretation()
    return _INTERP

# ── System prompt ──

CARDIOVASCULAR_PROMPT = """You are the PPG Longitudinal Comparison Analyst — Agent 2.

You receive a deterministic PERSONAL comparison: the current segment vs this
subject's own history (long-term baseline + the preceding K recent
segments; K = causal streaming-head window, defaults to 32).
Your job is NOT to write the final interpretation — the Supervisor does that.
Your job is to DECIDE THE ANALYSIS PATH: whether the base comparison answers
the user's question, which follow-up tools to run, and which changed/stable
features to highlight — then submit a STRUCTURED findings object via
submit_findings.

INPUT FORMAT (from the deterministic assess node):
  - compare verdict: ok / no_history /
    current_segment_low_quality — below "ok" NO change claim may be made;
    when the report carries insufficient_power (history below the
    reliability floor), keep the claim but flag the limited evidence
  - changed features: per-feature Δ vs personal median, the threshold
    (MDC reference or personal 3σ-MAD) and threshold_source
  - beat layer: valid_ratio, per-feature beat outlier rates, within-segment
    beat BP distribution (vital/mimic only)
  - drift: >=5 of the recent 7 segments beyond the long-term fence
  - volatile_features: features whose personal false-positive rate exceeded
    10% on the held-out walk-forward test — downgrade any change claim on
    them even when flagged real_change

DECISION RULES:
1. The base comparison is complete. Run a follow-up tool ONLY when the
   user's question needs it:
   - personal_trend — the user asks about long-term tendency or the BP
     association within this person.  Theil-Sen slopes are descriptive;
     personal ρ(feature, BP) is a WEAK reference (split-half sign
     agreement ~0.64-0.70 on held-out test) — quote it hedged, never as a
     prediction
   - bp_context — the user asks about blood pressure: measured BP, personal
     ρ, and disease-stratified group baselines.  Group baselines describe a
     KNOWN diagnosis forward (label → that group's BP distribution); NEVER
     reverse them into disease detection
   - beat_bp_coupling — the user asks about beat-level BP coupling
     (vital/mimic only; the de-confounded variant is authoritative)
   - population_position — the user asks where they stand in the
     population / vs peers: dual reference — main position from the
     personal median (gated on ICC(2,k)>=0.6, unlocks as history grows),
     recent-7 contrast plus population_drift when the recent band moved;
     history<3 falls back to single-segment (low confidence).  Positions
     resolve against the subject's same age/gender group when demographics
     are known (age_stratified + per-feature note when it fell back to the
     global baseline); quote the stratified band first.  Per-feature
     age_direction entries carry a fixed-wording statement (position vs.
     the feature's age trend in this dataset, weak |ρ|≲0.3) — quote it
     as written; it is a cross-sectional description, NOT a cause for the
     subject's change and NOT a vascular-age estimate.  Suppressed
     features carry a reason and must NOT be discussed at population level
   - feature_interpretation — the user asks what a feature means
     physiologically (what high/low values are associated with, any
     "symptom" implications).  Pass the features under discussion (changed
     or positioned ones; omitted → joint-cluster representatives).
     Needs NO blood-pressure labels — pure PPG.  Its population_patterns
     are FORWARD descriptions of labeled groups ("the hypertension-labeled
     group showed higher values", age+sex adjusted, FDR<0.05, small |d|);
     quote them exactly that way, never as what THIS subject has.
     Its bp_patterns are WEAK population correlations of the feature with
     blood pressure (age+sex adjusted, per-source FDR, aurora a/o kept
     apart) — word them as "in this population the feature correlates
     weakly with BP", never as a BP estimate for this subject
   Analysis tools take only the optional seg_idx — subject and dataset come
   from the shared context (Agent 1 already loaded them).
2. changed_features may ONLY contain features the engine flagged
   real_change; quote Δ/threshold/direction verbatim from the assessment —
   never invent values.  Prefer 2-5 features most relevant to the user's
   question; de-weight volatile_features (note the instability instead).
3. Redundancy folding (knowledge/feature_clusters.json — joint
   within-person ∩ between-person gates): flagged changes carry
   cluster_consistency.  cluster_consistent members are ONE family signal
   — lead with the cluster_representative and fold the rest as
   "same family"; isolated changes stand alone; direction_divergent means
   genuine within-family divergence — report BOTH, never fold.
   In population narratives prefer representatives: position_fold lists
   band-matching families, between_only_note pairs fold ONLY at
   population level.  Folding changes wording, never statistics — quote
   the representative's Δ/threshold verbatim.
4. submit_findings MUST be the LAST tool call, alone in its message. After
   submitting, output no further text.

DISCIPLINE (facts only, no diagnosis):
- NEVER diagnose disease, and NEVER reverse a group BP baseline into a
  disease claim.
- Real-world deployments may carry NO blood-pressure labels (wearables):
  bp_context / beat_bp_coupling degrade on their own (empty measured /
  not_available); pure-PPG analysis — compare_segment, population
  position, feature_interpretation — stays fully available.  Never make
  a BP-derived claim when no BP data was returned.
- Interpretation knowledge is population physiology (mechanism + age
  trend + labeled-group patterns + weak BP correlations).  Age trends are
  CROSS-SECTIONAL and weak: they may describe where a value sits vs. the
  age-trend direction, but NEVER "your values are drifting because of
  aging" and NEVER any vascular-age / biological-age estimate.
  A high feature
  value must be worded
  as "in labeled group X this feature tends to run higher" plus the
  mechanism — NEVER "this suggests the subject has X" and NEVER as a
  symptom prediction.  Effect sizes are small; when explaining one
  person's value, list ordinary causes (posture, perfusion, sensor
  placement, momentary physiology) first.
- verdict no_history → findings stay descriptive
  (measured values only); changed_features must be empty.
- bp_note MUST be set whenever bp_context returned data (measured BP,
  personal ρ, or group baselines) and the user asked about blood pressure
  — summarize what the tool returned; never claim the subject has no BP
  context when the tool returned any.
- population_position is a gated background reference: quote only features
  present in its "positions" key; features in "suppressed" must not be
  discussed at population level; population percentile is never a basis for
  a change claim or a diagnosis.  Always state the positioning basis:
  personal median over k history segments (reliability_used given) vs the
  low-confidence single_segment fallback; population_drift entries mean the
  recent-7 band moved >=1 band from the long-term band — report them as a
  tentative state shift, not a diagnosis.
- population_note MUST be set whenever population_position returned any
  "positions" entry and the user asked about their standing in the
  population — summarize the gated percentiles; never quote a population
  position when the tool was not invoked.
- Narrative order for the Supervisor: change vs own history → beat-level
  evidence → population position (only ICC>=0.6 features) → known-label
  context → safety note.
"""


# ── Structured findings schema (the only prose-free output of Agent 2) ──

class ChangedFeature(BaseModel):
    """One engine-flagged real_change feature — facts from the assessment."""

    feature: str = Field(description="特征名（必须被引擎判为 real_change）")
    delta: float = Field(description="当前值 − 个人中位数（逐字取自评估）")
    threshold: float = Field(description="判定阈值（逐字取自评估）")
    direction: str = Field(description="increased / decreased（由 delta 符号决定）")
    beat_evidence: str | None = Field(
        default=None, description="beat 级证据（离群率/分布摘要，来自 beat_layer）")


class Findings(BaseModel):
    """Structured analysis findings — consumed by the Supervisor for wording."""

    changed_features: list[ChangedFeature] = Field(
        default_factory=list, description="引擎判为 real_change 的特征（≤5 个）")
    stable_note: str | None = Field(
        default=None, description="无变化时的一句话稳定性说明（事实性）")
    bp_note: str | None = Field(
        default=None, description="血压语境摘要（仅当 bp_context 有数据）")
    population_note: str | None = Field(
        default=None, description="人群位置摘要（仅当 population_position "
                                  "返回了 positions）")
    interpretation_note: str | None = Field(
        default=None, description="生理可解释性摘要（仅当 feature_"
                                  "interpretation 被调用且用户询问含义；"
                                  "正推描述，禁反推诊断）")
    follow_up_suggestion: str = Field(default="", description="下一步建议（简短）")


# ── Tool argument schemas ──

class SegOnlyArgs(BaseModel):
    seg_idx: int | None = Field(
        default=None, description="目标段；省略 → 当前分析段")


class InterpretArgs(BaseModel):
    features: list[str] | None = Field(
        default=None, description="要解释的特征名列表（裸名或 _mean 均可）；"
                                  "省略 → 返回联合簇代表特征的总览")


class BpCouplingArgs(BaseModel):
    seg_idx: int | None = Field(
        default=None, description="仅在该段内算耦合；省略 → 受试者池化"
                                  "（去混杂版+池化版对照）")
    max_segments: int | None = Field(
        default=None, description="池化时的段数上限（控制耗时）")


# ── Deterministic engine wrappers (tools) ──

_IDENTITY: dict = {}  # fallback installed by the assess node; tools may run
                      # in the ToolNode's thread pool where the contextvar-
                      # scoped PPGContext is NOT visible.  The authoritative
                      # bridge is RunnableConfig.configurable (propagates
                      # through the thread pool); the global is a fallback.


def _ident_from(config: RunnableConfig | None) -> dict:
    ident = ((config or {}).get("configurable") or {}).get("agent2_identity")
    return ident or _IDENTITY


def _ident_or_error(config: RunnableConfig | None = None) -> dict | tuple[str, str, int]:
    ident = _ident_from(config)
    ds, sid = ident.get("ds"), ident.get("sid")
    if not ds or not sid:
        return {"error": "No subject in context. Agent 1 must run first."}
    return ds, sid, int(ident.get("seg_idx", 0))


def _compare_segment(seg_idx: int | None = None, *,
                     config: RunnableConfig | None = None) -> dict:
    """Personal longitudinal comparison of the segment against its own past.

    Deterministic (runs automatically in the assess node): long-term
    baseline + recent-7 dual reference, MDC / personal 3σ-MAD thresholds,
    beat-layer outlier rates.  Re-invoking returns the same result.
    """
    from ppg_agent.engine.personal_engine import PersonalEngine

    got = _ident_or_error(config)
    if isinstance(got, dict):
        return got
    ds, sid, cur = got
    eng = PersonalEngine()
    from ppg_agent.knowledge.clusters import annotate_changes
    return annotate_changes(
        eng.compare_segment(ds, sid, int(seg_idx if seg_idx is not None
                                           else cur)),
        _clusters())


def _personal_trend(seg_idx: int | None = None, *,
                    config: RunnableConfig | None = None) -> dict:
    """Personal long-term tendencies (Theil-Sen per feature) and personal
    ρ(feature, BP).  ρ needs >=20 labelled history segments and is a WEAK
    reference (held-out split-half sign agreement ~0.64-0.70)."""
    from ppg_agent.engine.personal_engine import PersonalEngine

    got = _ident_or_error(config)
    if isinstance(got, dict):
        return got
    ds, sid, cur = got
    eng = PersonalEngine()
    if seg_idx is not None:
        segs = eng.load_subject_segments(ds, sid)
        hist = [int(i) for i in segs.loc[segs["seg_idx"] < int(seg_idx),
                                         "seg_idx"]]
        prof = eng.build_profile(ds, sid, history_seg_idxs=hist)
    else:
        prof = eng.build_profile(ds, sid)
    return {"n_segments": prof.get("n_segments"),
            "trends": prof.get("trends"),
            "rho_bp": prof.get("rho_bp"),
            "rho_bp_reason": prof.get("rho_bp_reason"),
            "rho_evidence_note": "personal rho is a weak reference: "
                                 "split-half sign agreement 0.64-0.70 on "
                                 "the held-out test split"}


def _bp_context(seg_idx: int | None = None, *,
                config: RunnableConfig | None = None) -> dict:
    """Blood-pressure context for the CURRENT segment of the in-context
    subject: measured BP of the segment (measured key), personal
    ρ(feature, BP), population ρ prior, and disease-stratified BP group
    baselines.  ALWAYS returns data for vital/mimic — an empty dict never
    means 'no BP'.  Group baselines describe KNOWN labels forward — never
    use them to imply a diagnosis."""
    from ppg_agent.engine.personal_engine import PersonalEngine

    got = _ident_or_error(config)
    if isinstance(got, dict):
        return got
    ds, sid, cur = got
    eng = PersonalEngine()
    out = eng.bp_context(ds, sid, int(seg_idx if seg_idx is not None
                                      else cur))
    if out.get("condition_baseline"):
        out["condition_baseline_note"] = (
            "forward description only: known diagnosis -> that group's BP "
            "distribution; never reversed into disease detection")
    return out


def _population_position(seg_idx: int | None = None, *,
                         config: RunnableConfig | None = None) -> dict:
    """Where this subject sits in the population — dual reference against
    the per-patient-median distribution (built on the train split):
    main position from the personal median over the long-term history
    (gated on ICC(2,k)>=0.6 — low-ICC features unlock as history grows),
    recent-7 median contrast with population_drift band shifts, and a
    low-confidence single-segment fallback when history<3.  Positions are
    resolved against the subject's SAME age/gender group when demographics
    are available (auto-filled; global fallback carries a note).  Per-feature
    age_direction annotations carry FIXED-wording consistency statements
    (position band vs. the feature's age trend in this dataset) — quote
    the statement verbatim; it is cross-sectional and never explains the
    subject's own drift over time.  Background reference only: never a
    basis for change claims or diagnosis."""
    from ppg_agent.engine.personal_engine import PersonalEngine

    got = _ident_or_error(config)
    if isinstance(got, dict):
        return got
    ds, sid, cur = got
    eng = PersonalEngine()
    from ppg_agent.knowledge.clusters import annotate_positions
    return annotate_positions(
        eng.population_position(ds, sid, int(seg_idx if seg_idx is not None
                                                 else cur)),
        _clusters())


def _beat_bp_coupling(seg_idx: int | None = None,
                      max_segments: int | None = None, *,
                      config: RunnableConfig | None = None) -> dict:
    """Beat×BP coupling: ρ(beat feature, beat SBP).  seg_idx given →
    within that single segment; omitted → pooled over the subject with the
    de-confounded variant (beat − segment mean) as authoritative.
    vital/mimic only (aurora has no ABP)."""
    from ppg_agent.engine.personal_engine import PersonalEngine

    got = _ident_or_error(config)
    if isinstance(got, dict):
        return got
    ds, sid, _ = got
    eng = PersonalEngine()
    return eng.beat_bp_association(ds, sid, seg_idx=seg_idx,
                                   max_segments=max_segments)


def _feature_interpretation(features: list[str] | None = None) -> dict:
    """Physiological interpretability of features — mechanism, per-dataset
    age trend (sex-adjusted partial rho), and labeled-population patterns
    (rank-ANCOVA, age+sex adjusted, FDR<0.05).  Pure PPG: needs no BP
    labels.  Patterns describe KNOWN labels forward — never reverse them
    into a condition/symptom claim for this subject.  ``features``
    omitted → joint-cluster representatives (fold-aware overview)."""
    interp = _interpretation()
    fc = _clusters()
    reps = fc.representatives() if fc.available else None
    return interp.interpret(features, representatives=reps, fc=fc)


def _submit_findings(findings: Findings) -> str:
    """Structured findings submission — captured by the graph, never executed."""
    return "findings captured"


# ── Evidence formatting ──

def format_assessment_blocks(report: dict) -> str:
    """Format longitudinal comparison evidence blocks for the decide LLM and
    the Supervisor.

    Input shape: Agent-2 compare report ("features"/"verdict") → longitudinal
    comparison blocks (changed features, beat layer, drift, guards).  The
    legacy VAI branch was removed together with its last consumer — narrative
    callers only ever pass a real compare_report.
    """
    lines = [
        f"  Verdict: {report.get('verdict')}",
        f"  History: n_history={report.get('n_history')}, "
        f"n_recent={report.get('n_recent')}"
        + (", LIMITED" if report.get("limited_history") else ""),
        f"  Real-change features: {report.get('n_real_change', 0)}",
    ]
    if report.get("hint"):
        lines.append(f"  Hint: {report['hint']}")

    chg_lines, stable_lines = [], []
    for fname, f in (report.get("features") or {}).items():
        vol = " [VOLATILE — downgrade]" if fname.removesuffix(
            "_mean") in VOLATILE_FEATURES else ""
        cons = f.get("cluster_consistency")
        fold_tag = f" [{cons}]" if cons else ""
        line = (f"    - {fname}: Δ={f.get('delta_vs_long')} vs median "
                f"{f.get('personal_median')} (thr={f.get('threshold')}, "
                f"{f.get('threshold_source')}) → {f.get('verdict_long')}"
                f"{vol}{fold_tag}")
        if f.get("delta_vs_recent") is not None:
            line += f", Δ_recent={f.get('delta_vs_recent')}"
        (chg_lines if f.get("verdict_long") == "real_change"
         else stable_lines).append(line)

    fold_lines = []
    for cl_id, fi in sorted((report.get("cluster_fold") or {}).items()):
        cons = ("direction-consistent → fold to representative"
                if fi["consistent"] else
                "direction DIVERGENT → keep members separate")
        fold_lines.append(
            f"    - cluster {cl_id} (rep {fi['representative']}): "
            f"{len(fi['flagged'])} member(s) flagged "
            f"({', '.join(fi['flagged'])}) — {cons}")

    bl = report.get("beat_layer") or {}
    beat_lines = []
    if bl.get("available"):
        beat_lines.append(
            f"    - beats: {bl.get('n_valid_cur')}/{bl.get('n_beats_cur')} "
            f"valid (valid_ratio={bl.get('valid_ratio')})")
        if bl.get("outlier_rate_median") is not None:
            beat_lines.append(
                f"    - beat outlier rate (median over features): "
                f"{bl.get('outlier_rate_median')}")
        bb = bl.get("beat_bp")
        if bb:
            beat_lines.append(
                f"    - within-segment beat BP: SBP median "
                f"{bb.get('sbp_median')} {bb.get('sbp_iqr')}, DBP median "
                f"{bb.get('dbp_median')}")
    elif bl.get("reason"):
        beat_lines.append(f"    - beat layer unavailable: {bl['reason']}")

    drift_lines = [f"    - {d['feature']}: {d['exceeded']}/{d['of']} recent "
                   f"segments beyond the fence"
                   for d in (report.get("drift") or [])]

    ho = report.get("agent1_handoff") or {}
    ho_lines = []
    if ho:
        ho_lines.append(
            f"    - Agent 1 live extraction: {ho.get('n_valid')}/"
            f"{ho.get('n_beats')} valid beats (valid_ratio="
            f"{ho.get('valid_ratio')}, quality={ho.get('quality_tier') or '?'})"
            + (f", beat BP: SBP median {ho.get('beat_sbp_median')}"
               if ho.get("has_beat_bp") else ", no beat BP"))

    blocks = []
    if chg_lines:
        blocks.append("  CHANGED vs personal history (real_change, engine "
                      "thresholds):\n" + "\n".join(chg_lines))
    if fold_lines:
        blocks.append("  Cluster folding (joint redundancy KB — one family "
                      "signal folds to its representative):\n"
                      + "\n".join(fold_lines))
    if stable_lines:
        blocks.append(f"  Stable features ({len(stable_lines)}): first 5 —\n"
                      + "\n".join(stable_lines[:5]))
    if beat_lines:
        blocks.append("  Beat layer (axis ①):\n" + "\n".join(beat_lines))
    if drift_lines:
        blocks.append("  DRIFT (recent-window beyond long-term fence):\n"
                      + "\n".join(drift_lines))
    if ho_lines:
        blocks.append("  Handoff (Agent 1 live extraction, cross-check for "
                      "the beat layer):\n" + "\n".join(ho_lines))
    blocks.append("  Narrative order: change vs own history → beat evidence "
                  "→ population position (gated: personal-median ICC(2,k)>=0.6 "
                  "or single-segment ICC(1,1)>=0.6 fallback) → "
                  "known-label context → safety note.")
    return "\n".join(lines) + ("\n" + "\n".join(blocks) if blocks else "")


# ── Graph builder ──

class Agent2State(TypedDict, total=False):
    """Graph state schema — declared keys merge across nodes instead of the
    bare-dict whole-state replacement semantics."""

    subject_id: str
    seg_idx: int
    source: str
    scenario: str | None
    # Subject identity — tools must read it from the STATE, not from the
    # shared PPGContext: the ToolNode runs tool functions in a thread pool,
    # and contextvars do not propagate into those threads.
    identity: dict
    # add_messages reducer: nodes APPEND instead of replacing — the tool loop
    # (interpret ⇄ tools_assess) must keep the full history (assistant
    # tool_calls + ToolMessages) or the LLM API rejects the tool response
    messages: Annotated[list, add_messages]
    compare_report: dict
    findings: dict | None


def _capture_findings(state: dict) -> dict:
    """Deterministic: validate the submitted findings against the compare
    report (engine facts only — only engine-flagged real_change features are
    kept, with Δ/threshold taken verbatim from the engine) and store them.

    Runs as the capture_findings node; module-level so tests can drive the
    validation directly with a hand-built state.
    """
    report = state.get("compare_report") or {}
    msgs = state.get("messages") or []
    args = None
    for m in reversed(msgs):
        calls = getattr(m, "tool_calls", None) or []
        if calls:
            for tc in calls:
                if tc["name"] == "submit_findings":
                    args = tc.get("args") or {}
                    if set(args) == {"findings"} and isinstance(
                            args["findings"], dict):
                        args = args["findings"]  # LLM wrapped one extra layer
            if args is not None:
                break
    if not args:
        return {"findings": None}

    feats = report.get("features") or {}
    usable = report.get("verdict") == "ok"
    beat_layer = report.get("beat_layer") or {}
    items = []
    for it in args.get("changed_features") or []:
        fname = (it or {}).get("feature")
        f = feats.get(fname) or feats.get(f"{fname}_mean")  # LLM may drop _mean
        if f is not None and fname not in feats:
            fname = f"{fname}_mean"
        # engine facts only: the feature must exist AND be engine-flagged
        if not usable or not f or f.get("verdict_long") != "real_change":
            continue
        delta = float(f.get("delta_vs_long") or 0.0)
        beat_ev = it.get("beat_evidence")
        if beat_ev and not beat_layer.get("available"):
            beat_ev = None  # no beat evidence to cite
        items.append({
            "feature": fname,
            "delta": round(delta, 4),
            "threshold": f.get("threshold"),
            "direction": "increased" if delta > 0 else "decreased",
            "beat_evidence": beat_ev,
            "cluster_representative": f.get("cluster_representative"),
            "cluster_consistency": f.get("cluster_consistency"),
        })
    stable_note = args.get("stable_note")
    if items:
        stable_note = None  # a change narrative supersedes the stable note
    return {"findings": {
        "changed_features": items,
        "stable_note": stable_note if usable else None,
        "bp_note": args.get("bp_note"),
        "population_note": args.get("population_note"),
        "interpretation_note": args.get("interpretation_note"),
        "follow_up_suggestion": args.get("follow_up_suggestion", ""),
    }}


def build_cardiovascular_graph(llm: Any = None) -> Any:
    """Build Agent 2: personal comparison → LLM analysis decisions →
    structured findings.

    Graph: START → assess → interpret → (tools_assess ⇄ interpret) →
           capture_findings → END
    """
    if llm is None:
        llm = get_llm("cardiovascular")  # DeepSeek API (registry), not local qwen

    tools = [
        StructuredTool.from_function(
            func=_compare_segment,
            name="compare_segment",
            args_schema=SegOnlyArgs,
            description="Personal longitudinal comparison of a segment "
                        "against its own past (already run by the assess "
                        "node for the current segment — re-invoke only for "
                        "a different seg_idx)."),
        StructuredTool.from_function(
            func=_personal_trend,
            name="personal_trend",
            args_schema=SegOnlyArgs,
            description="Personal long-term tendencies (Theil-Sen slopes) "
                        "and personal rho(feature, BP) — the rho is a WEAK "
                        "reference, quote it hedged."),
        StructuredTool.from_function(
            func=_bp_context,
            name="bp_context",
            args_schema=SegOnlyArgs,
            description="Blood-pressure context of the in-context subject: "
                        "measured BP of the segment, personal rho, "
                        "disease-stratified group baselines (known-label "
                        "forward description only, never disease "
                        "detection).  Always returns data for vital/mimic."),
        StructuredTool.from_function(
            func=_population_position,
            name="population_position",
            args_schema=SegOnlyArgs,
            description="Where the subject sits in the population: main "
                        "position from the personal median over history "
                        "(gated on ICC(2,k)>=0.6, unlocks as history "
                        "grows), recent-7 contrast with population_drift, "
                        "single-segment low-confidence fallback when "
                        "history<3.  Background reference only — never a "
                        "basis for change claims or diagnosis."),
        StructuredTool.from_function(
            func=_beat_bp_coupling,
            name="beat_bp_coupling",
            args_schema=BpCouplingArgs,
            description="Beat x BP coupling detail (vital/mimic only; "
                        "de-confounded variant authoritative)."),
        StructuredTool.from_function(
            func=_feature_interpretation,
            name="feature_interpretation",
            args_schema=InterpretArgs,
            description="Physiological meaning of features: mechanism, "
                        "age trend, labeled-group patterns (FORWARD "
                        "description only, never diagnosis/symptom "
                        "claims).  Pure PPG — works without any BP "
                        "labels.  Features omitted -> cluster "
                        "representatives overview."),
        StructuredTool.from_function(
            func=_submit_findings,
            name="submit_findings",
            description="Submit the structured analysis findings. MUST be "
                        "the final tool call."),
    ]
    # submit_findings is intercepted by capture_findings — the ToolNode only
    # ever executes analysis tools.
    analysis_tools = tools[:-1]

    graph = StateGraph(Agent2State)

    def _assess_node(state: dict) -> dict:
        """Run the deterministic personal comparison and hand the evidence
        to the LLM.  Installs the subject identity on the shared context so
        the follow-up tools need no dataset/subject arguments."""
        from ppg_agent.context import get_context

        ds = state.get("source", "vital")
        sid = state.get("subject_id")
        seg_idx = int(state.get("seg_idx", 0))
        if not sid:
            return {"messages": [AIMessage(
                content="Error: no subject_id in state.")] }
        ctx = get_context()
        ident = {"ds": ds, "sid": sid, "seg_idx": seg_idx}
        ctx.agent2_identity = ident   # convenience for same-thread readers
        _IDENTITY.update(ident)       # authoritative bridge for the tools

        from ppg_agent.engine.personal_engine import PersonalEngine
        from ppg_agent.knowledge.clusters import annotate_changes
        result = annotate_changes(
            PersonalEngine().compare_segment(ds, sid, seg_idx),
            _clusters())

        # Agent 1 handoff cross-check: live beat counts from the shared
        # context (if Agent 1 extracted in this session) vs the engine's
        # beat layer read from the parquet store.
        handoff = ctx.beat_summary() if ctx.beat_records else None
        if handoff:
            result = {**result, "agent1_handoff": handoff}

        ctx_msg = SystemMessage(
            content=(
                f"{CARDIOVASCULAR_PROMPT}\n\n"
                f"Personal comparison complete (segment {seg_idx} vs its own "
                f"history):\n{format_assessment_blocks(result)}\n"
                f"Decide the analysis path for the user's question, then "
                f"submit structured findings via submit_findings."
            )
        )
        # add_messages appends ctx_msg to the input messages — do NOT return
        # the input list again or it would be duplicated
        return {"messages": [ctx_msg], "compare_report": result,
                "identity": ident}

    def _interpret(state: dict) -> dict:
        """LLM analysis decisions: follow-up tools or the final submission."""
        msgs = state.get("messages", [])
        ai = llm.bind_tools(tools).invoke(msgs)
        return {"messages": [ai]}

    def _route_interpret_tools(state: dict) -> str:
        msgs = state.get("messages", [])
        # safety cap: at most 8 tool turns per run (protects against an LLM
        # that keeps looping on analysis tools)
        tool_turns = 0
        for m in msgs:
            tool_turns += len(getattr(m, "tool_calls", None) or [])
        if tool_turns > 8:
            return "end"
        if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
            names = {tc["name"] for tc in msgs[-1].tool_calls}
            if "submit_findings" in names:
                return "capture_findings"
            return "tools"
        return "end"

    graph.add_node("assess", _assess_node)
    graph.add_node("interpret", _interpret)
    graph.add_node("tools_assess", ToolNode(analysis_tools))
    graph.add_node("capture_findings", _capture_findings)

    graph.set_entry_point("assess")
    graph.add_edge("assess", "interpret")
    graph.add_conditional_edges(
        "interpret", _route_interpret_tools,
        {"tools": "tools_assess", "capture_findings": "capture_findings",
         "end": END})
    graph.add_edge("tools_assess", "interpret")
    graph.add_edge("capture_findings", END)

    return graph.compile()


def create_cardiovascular_agent(llm: Any = None) -> Any:
    """Create Agent 2 as a compiled LangGraph graph."""
    return build_cardiovascular_graph(llm)
