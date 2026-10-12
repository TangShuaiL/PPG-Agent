"""Supervisor orchestration for the web chat: plan → dispatch → synthesize.

The chat router previously drove the Supervisor prompt through a single-LLM
ReAct tool loop.  This module implements the real three-phase orchestration:

    (1) plan      — extract a structured SupervisorPlan (intent parse +
                    task assignment) with three tiers of robustness
    (2) dispatch  — run the sub-agent LangGraph graphs in task-queue order
                    (Agent 1 → Agent 2, sharing one PPGContext so Agent 2
                    reads Agent 1's extracted features)
    (3) synthesize— combine the sub-agent outputs + retrieved physiological
                    rules into a concise cross-evidence reply (≤4 sentences)

Robustness: DeepSeek sometimes returns the plan as XML text or plain JSON
instead of a tool call — extract_plan tries tool_calls → JSON → XML, then
keyword_plan provides a deterministic fallback.  If graph execution fails,
the router falls back to the deterministic analysis-service pipeline.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage

from ppg_agent.agents.cardiovascular_agent import (
    create_cardiovascular_agent,
    format_assessment_blocks,
)
from ppg_agent.agents.signal_processor_agent import (
    _ADAPT_KEYWORDS,
    _PROCESS_KEYWORDS,
    create_signal_processor_agent,
)
from ppg_agent.agents.supervisor import (
    AGENT_CARDIO, AGENT_ML, AGENT_SIGNAL,
    SubTask, SupervisorPlan, retrieve_knowledge,
)
from ppg_agent.context import get_context

from . import analysis_service as svc

# Non-feature metadata keys inside segment feature dicts (mirrors
# analysis_service._META_KEYS; duplicated here to avoid re-export churn).
_META_KEYS = {"subject_id", "seg_idx", "source", "sbp_label", "dbp_label",
              "age", "gender", "n_beats_ppg", "n_beats_valid"}

_PHENOTYPE_KEYWORDS = ("表型", "血管", "百分位", "血压", "年龄",
                       "心血管", "人群", "位置", "历史", "变化",
                       "含义", "生理", "代表什么", "代表", "倾向",
                       "什么情况", "说明什么", "意味着", "意思")
# 已证伪/降级指标（docs/agent_architecture_overview.md 状态表）：横断面 VAI、
# 表型亚型、疾病关联、规则依据、窗口上下文 —— 传输层 report 字段照旧（研究
# 可追溯），但禁止进入任何提示词叙事入口。
_FALSIFIED_REPORT_KEYS = ("vascular_age", "phenotype_subtype",
                          "disease_associations", "interpretation_rules",
                          "window_context")
_ML_KEYWORDS = ("预测", "风险", "高血压", "模型", "ml", "prediction", "deep learning")
# 强归因标记（Agent 3 模型依赖度视角）：只收"解释的是模型预测"的硬标记，
# 不含泛化词"为什么/解释/依据"——否则"血压为什么偏高"（生理问题）会被
# 劫持进 explain_prediction。
_MODEL_EXPLAIN_KEYWORDS = ("归因", "可解释", "哪些特征点", "特征点", "怎么得出",
                           "什么驱动", "为什么是这个", "为什么出这个", "为什么预测",
                           "参考了哪些", "依据哪些", "预测依据", "模型依据",
                           "解释这个预测", "解释这个估计")
# 生理轨触发词（Agent 2 视角）：须与表型关键词共现才生效（降级兜底）。
_PHYSIO_WHY_KEYWORDS = ("为什么", "为何", "意思", "含义", "偏高", "偏低",
                        "说明什么", "意味着", "代表")
# 复合句硬标记：只有这些词才触发"归因+生理"并派——不用泛化词"为什么+血压"
# 共现，否则前端解释按钮句（收缩压/为什么）会被误判复合、多跑一轮 Agent 2。
_PHYSIO_COMPOUND_MARKERS = ("偏高", "偏低", "什么意思", "含义", "意味着",
                            "说明什么", "代表什么", "和本人", "历史比")

# Web chat expands the graph's shared keyword sets — common user phrasings
# ("帮我调一下参数", "评估血管年龄") don't hit the agent-layer constants.
_ADAPT_WEB = _ADAPT_KEYWORDS + ("调一下", "帮我调", "改参数", "参数不好", "参数不对")
_PROCESS_WEB = _PROCESS_KEYWORDS + ("评估", "看看", "看一下", "查看")


# ---------------------------------------------------------------------------
# Plan phase
# ---------------------------------------------------------------------------

PLAN_ADDENDUM = """【Web 编排模式 — 本环境覆盖上方的工具选择规则】
你只做意图解析与任务分配：输出一个 SupervisorPlan，不调用任何工具（工具由编排器执行）。
任务分配规则：
  - 分析波形/信号质量/特征提取 → task_queue=[signal_processing(analyze_signal)]
  - 血压解读/人群位置/纵向变化/表型/指标生理含义 → 追加 cardiovascular(analyze_phenotype)
  - 数据集适配/调参（用户已选数据集、未选 segment）→ signal_processing 的
    task="dataset_adaptation"
  - 模型预测（血压估计/年龄估计/模型能预测什么）→ ml_prediction 的 task="ml_predict_risk"；输出为群体级统计估计，必须携带置信说明（MAE），不得表述为诊断或风险结论，也不得与 Agent 2 的纵向变化叙事混述
  - 可解释性（为什么出这个预测值/解释这个预测/依据哪些特征点/归因/怎么得出）→ ml_prediction 的 task="explain_prediction"（对 Agent 3 预测做三层证据叙事，**模型输入依赖视角**，非个体病因、非诊断）
  - 生理原因（为什么偏高/偏低、这个指标什么意思、和本人历史比、人群位置）→ cardiovascular(analyze_phenotype)（生理机制 + 纵向/人群证据视角）；"为什么出这个预测值"与"为什么生理上偏高"是两轨，不得混述；但一句话同时含两义（既问模型依据又问生理含义/本人比较）时应同时派发 explain_prediction 与 analyze_phenotype，综合时分视角陈述、不得互推
  - 特征点知识（waveform_keypoints）与标量特征知识（features）同源于 feature_rules.json 知识层：前者由 Agent 3 解释链消费，后者由 Agent 2 的 feature_interpretation 消费
  - 需要分析但未选择 segment → needs_dispatch=false，direct_reply 引导先选择 segment
  - 纯闲聊/问候/感谢/澄清 → needs_dispatch=false，direct_reply 直接回答（≤2 句）
task_queue 顺序即执行顺序（先信号处理，再表型分析）。"""


def _plan_from_json(content: str) -> SupervisorPlan | None:
    """Tier-2 fallback: the plan as a JSON object (maybe inside code fences)."""
    if not content:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    candidate = m.group(1) if m else content
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(candidate[start:end + 1])
    except Exception:  # noqa: BLE001 — best-effort tier
        return None
    if not isinstance(data, dict) or "thought" not in data:
        return None
    try:
        return SupervisorPlan.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


_XML_FIELD_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)


def _plan_from_xml(content: str) -> SupervisorPlan | None:
    """Tier-3 fallback: DeepSeek sometimes emits the plan as literal XML."""
    if not content:
        return None
    m = re.search(r"<SupervisorPlan>(.*?)</SupervisorPlan>", content, re.DOTALL)
    body = m.group(1) if m else content
    fields: dict = {}
    for fm in _XML_FIELD_RE.finditer(body):
        name, val = fm.group(1), fm.group(2).strip()
        if name in {"thought", "needs_dispatch", "direct_reply"}:
            fields[name] = val
    if "thought" not in fields:
        return None
    fields["needs_dispatch"] = str(fields.get("needs_dispatch", "")).lower() in {"true", "1"}
    fields.setdefault("direct_reply", None)
    tasks = []
    tq = re.search(r"<task_queue>(.*?)</task_queue>", body, re.DOTALL)
    if tq:
        for im in re.finditer(r"<item>(.*?)</item>", tq.group(1), re.DOTALL):
            f2 = {fm2.group(1): fm2.group(2).strip()
                  for fm2 in _XML_FIELD_RE.finditer(im.group(1))}
            if "target_agent" not in f2:
                continue
            params: dict = {}
            pm = re.search(r"<params>(.*?)</params>", im.group(1), re.DOTALL)
            if pm:
                params = {fm3.group(1): fm3.group(2).strip()
                          for fm3 in _XML_FIELD_RE.finditer(pm.group(1))}
            tasks.append({"target_agent": f2["target_agent"],
                          "task": f2.get("task", ""), "params": params})
    fields["task_queue"] = tasks
    try:
        return SupervisorPlan.model_validate(fields)
    except Exception:  # noqa: BLE001
        return None


def extract_plan(llm, messages: list) -> SupervisorPlan | None:
    """Three-tier structured plan extraction (tool_calls → JSON → XML).

    Returns None when every tier fails — the router then falls back to
    keyword_plan, and finally to the legacy ReAct loop.
    """
    try:
        llm_tools = llm.bind_tools([SupervisorPlan])
        ai = llm_tools.invoke(messages)
    except Exception:  # noqa: BLE001 — every tier is best-effort
        return None
    if getattr(ai, "tool_calls", None):
        try:
            return SupervisorPlan.model_validate(ai.tool_calls[0]["args"])
        except Exception:  # noqa: BLE001
            pass
    content = ai.content if isinstance(ai.content, str) else ""
    plan = _plan_from_json(content)
    if plan is not None:
        return plan
    return _plan_from_xml(content)


def keyword_plan(text: str, seg_id: int | None) -> SupervisorPlan | None:
    """Deterministic intent routing (no LLM) — fallback for plan extraction.

    Mirrors the Agent-1 graph's keyword priorities: adaptation keywords beat
    process keywords ("适配好了，提取特征" must route to adaptation).
    """
    low = (text or "").lower()
    if any(k in low for k in _ADAPT_WEB):
        return SupervisorPlan(
            thought="关键词路由: 数据集适配",
            needs_dispatch=True,
            task_queue=[SubTask(target_agent=AGENT_SIGNAL,
                                task="dataset_adaptation", params={})])
    if any(k in low for k in _MODEL_EXPLAIN_KEYWORDS):
        tasks = [SubTask(target_agent=AGENT_ML,
                         task="explain_prediction", params={})]
        # 复合句（含生理追问硬标记）→ 同步派 Agent 2，综合时分视角陈述
        if seg_id is not None and any(k in low for k in _PHYSIO_COMPOUND_MARKERS):
            tasks.append(SubTask(target_agent=AGENT_CARDIO,
                                 task="analyze_phenotype", params={}))
        return SupervisorPlan(thought="关键词路由: 模型归因 (解释 Agent 3 预测)",
                              needs_dispatch=True, task_queue=tasks)
    # 生理轨："为什么/什么意思"类提问且命中表型词 → Agent 2（不与模型归因混派）
    if (any(k in low for k in _PHYSIO_WHY_KEYWORDS)
            and any(k in low for k in _PHENOTYPE_KEYWORDS)):
        if seg_id is None:
            return SupervisorPlan(
                thought="关键词路由: 生理解读意图但未选 segment",
                needs_dispatch=False,
                direct_reply="请先在上方选择一个 segment，我才能解读这个生理问题。")
        return SupervisorPlan(
            thought="关键词路由: 生理解读 (Agent 2 纵向/人群/特征含义)",
            needs_dispatch=True,
            task_queue=[SubTask(target_agent=AGENT_SIGNAL,
                                task="analyze_signal", params={}),
                        SubTask(target_agent=AGENT_CARDIO,
                                task="analyze_phenotype", params={})])
    if any(k in low for k in _ML_KEYWORDS):
        return SupervisorPlan(
            thought="关键词路由: ML 预测",
            needs_dispatch=True,
            task_queue=[SubTask(target_agent=AGENT_ML,
                                task="ml_predict_risk", params={})])
    if any(k in low for k in _PROCESS_WEB):
        if seg_id is None:
            return SupervisorPlan(
                thought="关键词路由: 分析意图但未选 segment",
                needs_dispatch=False,
                direct_reply="请先在上方选择一个 segment，然后再告诉我你想分析什么。")
        tasks = [SubTask(target_agent=AGENT_SIGNAL,
                         task="analyze_signal", params={})]
        if any(k in low for k in _PHENOTYPE_KEYWORDS):
            tasks.append(SubTask(target_agent=AGENT_CARDIO,
                                 task="analyze_phenotype", params={}))
        return SupervisorPlan(thought="关键词路由: 信号处理",
                              needs_dispatch=True, task_queue=tasks)
    return None


# ---------------------------------------------------------------------------
# Dispatch phase (sub-agent graphs)
# ---------------------------------------------------------------------------

_AGENT1 = None
_AGENT2 = None


def _agent1_graph():
    """Lazily compiled Agent 1 graph (compiled graphs are stateless/thread-safe)."""
    global _AGENT1
    if _AGENT1 is None:
        _AGENT1 = create_signal_processor_agent()
    return _AGENT1


def _agent2_graph():
    """Lazily compiled Agent 2 graph."""
    global _AGENT2
    if _AGENT2 is None:
        _AGENT2 = create_cardiovascular_agent()
    return _AGENT2


def _last_ai_text(msgs) -> str | None:
    for m in reversed(msgs or []):
        if getattr(m, "type", None) == "ai" and getattr(m, "content", None):
            return str(m.content).strip()[:400]
    return None


def run_agent1(seg: dict, params: dict | None, msgs: list) -> tuple[str | None, list[dict]]:
    """Run the Agent 1 graph, collecting the node trace.

    ctx.session_mode is preset to 'process' so the graph's LLM intent
    classification is skipped (the Supervisor already routed this task).
    Returns (final AI text, trace entries).  Raises on graph failure — the
    router falls back to the deterministic pipeline.
    """
    ctx = get_context()
    ctx.session_mode = "process"
    trace: list[dict] = []
    last_text = None
    for step in _agent1_graph().stream(
            {"source": seg["source"], "subject_id": seg["subject_id"],
             "seg_idx": seg["seg_idx"], "params": params,
             "messages": [m for m in msgs]},
            stream_mode="updates"):
        for node, upd in step.items():
            entry: dict = {"agent": AGENT_SIGNAL, "stage": node}
            if node == "segment":
                entry["summary"] = (f"{upd.get('n_beats')} beats, "
                                    f"{upd.get('n_valid')} valid, Q={upd.get('quality')}")
            elif node == "extract_features":
                fs = upd.get("feature_summary") or {}
                parts = [f"{fs.get('n_beats_total')} beats", "features extracted"]
                if fs.get("hr_bpm"):
                    parts.append(f"HR={fs['hr_bpm']}bpm")
                if fs.get("ptt_abp_ms"):
                    parts.append(f"PTT={fs['ptt_abp_ms']}ms")
                entry["summary"] = ", ".join(parts)
            trace.append(entry)
            txt = _last_ai_text(upd.get("messages") or [])
            if txt:
                last_text = txt
    return last_text, trace


def run_agent1_adapt(source: str, msgs: list) -> tuple[str | None, list[dict]]:
    """Run the Agent 1 graph in dataset-adaptation mode.

    state["mode"]="adapt" makes the graph's entry router authoritative (the
    Supervisor already classified this turn as dataset adaptation), so the
    adapt branch drives the session via the adapt_pipeline tool (semantic
    actions) instead of running the foundation pipeline.  The adapt branch
    never touches the signal, so subject_id/seg_idx are placeholders.
    Returns (final AI text, trace entries); raises on graph failure.
    """
    ctx = get_context()
    ctx.session_mode = "adapt"
    trace: list[dict] = []
    last_text = None
    for step in _agent1_graph().stream(
            {"source": source, "subject_id": "p000001", "seg_idx": 0,
             "mode": "adapt", "messages": [m for m in msgs]},
            stream_mode="updates"):
        for node, upd in step.items():
            entry: dict = {"agent": AGENT_SIGNAL, "stage": node}
            if node == "adapt":
                entry["summary"] = "适配会话推进"
            elif node == "tools_adapt":
                entry["summary"] = "适配工具执行"
            trace.append(entry)
            txt = _last_ai_text(upd.get("messages") or [])
            if txt:
                last_text = txt
    return last_text, trace


def run_agent2(seg: dict, scenario: str | None,
               msgs: list) -> tuple[dict | None, dict | None, list[dict]]:
    """Run the Agent 2 graph; returns (structured findings, deterministic
    compare report, trace entries).

    Agent 2 emits NO natural-language text — its structured findings (LLM
    analysis decisions) and the deterministic comparison report are worded
    by the Supervisor's synthesis phase.  Trace stages: assess / interpret /
    tools_assess / capture_findings.
    """
    trace: list[dict] = []
    findings = None
    compare_report = None
    # Subject identity via RunnableConfig.configurable — the ToolNode runs
    # tools in a thread pool where the request's contextvar-scoped
    # PPGContext is invisible; config propagates through the pool.
    ident = {"ds": seg["source"], "sid": seg["subject_id"],
             "seg_idx": seg["seg_idx"]}
    for step in _agent2_graph().stream(
            {"source": seg["source"], "subject_id": seg["subject_id"],
             "seg_idx": seg["seg_idx"], "scenario": scenario,
             "messages": [m for m in msgs]},
            config={"configurable": {"agent2_identity": ident}},
            stream_mode="updates"):
        for node, upd in step.items():
            entry: dict = {"agent": AGENT_CARDIO, "stage": node}
            if node == "assess":
                report = upd.get("compare_report") or {}
                compare_report = report
                entry["summary"] = (
                    f"verdict={report.get('verdict')}, "
                    f"历史={report.get('n_history')}段, "
                    f"real_change={report.get('n_real_change', 0)}")
            elif node == "interpret":
                entry["summary"] = "分析决策"
            elif node == "tools_assess":
                entry["summary"] = "分析工具执行"
            elif node == "capture_findings":
                findings = upd.get("findings")
                n = len((findings or {}).get("changed_features") or [])
                entry["summary"] = (f"结构化发现: {n} 个变化特征"
                                    if n else "未提交结构化发现")
            trace.append(entry)
    return findings, compare_report, trace


def analysis_from_ctx(seg: dict, preset_name: str) -> dict:
    """Assemble the frontend analysis dict from the graph-populated context.

    Shape matches the deterministic _pipeline output so the existing
    frontend cards (VaiPanel, AnalysisDetails) render unchanged.  The caller
    decides whether to add the phenotype ``report``.
    """
    ctx = get_context()
    raw = ctx.segment_features_raw or {}
    means = {k[:-5]: v for k, v in raw.items()
             if k.endswith("_mean") and k[:-5] not in _META_KEYS
             and isinstance(v, (int, float))}
    beats = {"n_beats": ctx.num_beats,
             "n_valid": ctx.n_valid_beats,
             "quality": round(ctx.ppg_quality, 3) if ctx.ppg_quality is not None else None,
             "quality_tier": ctx.ppg_quality_tier,
             "preset": preset_name}
    return {"beats": beats,
            "features": {"n_features": len(means),
                         "n_beat_records": len(ctx.beat_records or []),
                         "preset": preset_name,
                         "feature_means": {k: round(float(v), 5) for k, v in means.items()}},
            "overlay": svc.beat_overlay()}


def scenario_for(seg: dict) -> str | None:
    """Aurora scenario ('a'/'o') from the subject id's first char (same as _pipeline)."""
    if seg.get("source") in ("aurora", "aurorabp"):
        return str(seg.get("subject_id", ""))[:1] or None
    return None


# ---------------------------------------------------------------------------
# Synthesis phase
# ---------------------------------------------------------------------------

def _without_falsified(report):
    """Strip falsified cross-sectional keys (VAI/subtype/disease/...) from a
    report before it enters any prompt payload.  The transport-layer analysis
    dict is untouched — only the narrative view is filtered.  Descriptive keys
    (stratified_percentiles 等) 保留；Empty/None input passes through unchanged."""
    if not report:
        return report
    return {k: v for k, v in report.items() if k not in _FALSIFIED_REPORT_KEYS}


def synthesize_reply(agent_texts: list[str], analysis: dict,
                     history_msgs: list, llm) -> str:
    """Supervisor synthesis: sub-agent outputs + knowledge → concise reply.

    Takes the sub-agents' natural-language outputs (Agent 1 pipeline
    summary), the structured analysis (engine report + Agent 2's structured
    findings — Agent 2 emits no prose), and retrieved physiological rules,
    and asks the Supervisor to write the final cross-evidence report under
    the phenotype wording discipline.  Returns "" on failure — the router
    keeps its default reply.
    """
    feats = (analysis.get("features") or {}).get("feature_means") or {}
    klines = retrieve_knowledge(list(feats.keys()))
    # When Agent 2's longitudinal compare_report exists it supersedes the
    # legacy deterministic phenotype report (VAI prose) for wording — the
    # old report stays in analysis only for the frontend panels.
    structured = {
        "beats": analysis.get("beats"),
        "report": (None if analysis.get("compare_report")
                   else _without_falsified(analysis.get("report"))),
        "features_mean": dict(list(feats.items())[:6]),
    }
    try:
        text = json.dumps(structured, ensure_ascii=False, default=str)[:2000]
    except TypeError:
        text = str(structured)[:2000]
    knowledge_txt = "\n".join(klines) if klines else "(无匹配规则)"
    agent_out = "\n".join(f"- {t}" for t in agent_texts if t) or "(无)"
    # Agent 2's structured evidence: the deterministic comparison blocks +
    # the LLM's analysis decisions (findings).  Both feed the wording.
    # format_assessment_blocks 只对真纵向 compare_report 调用——旧形态 report
    # 剥 vascular_age 后会被它当成空 compare 块渲染幻影，故不再传入。
    compare = analysis.get("compare_report")
    pheno_evidence = format_assessment_blocks(compare) if compare else "(无)"
    findings = analysis.get("findings")
    findings_txt = (json.dumps(findings, ensure_ascii=False, default=str,
                               indent=1)
                    if findings else "(Agent 2 未提交结构化发现)")
    synth = [
        SystemMessage(content=(
            "子 Agent 已返回结果。请按综合报告规则生成最终回复：\n"
            "叙事顺序：相对本人历史的变化 → beat 级证据 → 人群位置"
            "（仅 ICC≥0.6 特征）→ 已知诊断标签语境 → 安全声明；"
            "先一句话结论 → 2-3 个关键指标（具体数字）→ 一句话下一步建议；"
            "总长不超过 4 句；可引用知识库规则（注明来源）；"
            "不要复述前端已展示的表格内容。\n"
            "可解释叙事纪律（子 Agent 输出含 explain_prediction 时）："
            "必须以「特征点知识解读」中的生理含义×方向提示为叙事主体"
            "（如“重搏波低=顺应性差”），贡献数值只作佐证一笔带过；"
            "保留“提示性/群体模型依赖度/非诊断”限定；注意力 1.0 等"
            "已标注为归一化病态值时不得解读为强证据。\n"
            "缺席叙事纪律：下方「Agent 2 发现/纵向比较证据」为(无)时表示该子 Agent "
            "本轮未被派发而非故障——严禁叙述为“无…证据可用/不可用”；"
            "应如实说明“本轮仅做了模型归因解读，如需与自己历史比较/"
            "人群定位可继续追问”（按实际派发情况措辞）。\n"
            "纵向比较解读纪律（Agent 2 的证据必须按此措辞，不得违反）：\n"
            "  - 绝不诊断疾病；疾病分层血压基线只能正推描述（已知诊断 → "
            "该组血压分布），严禁反推疾病\n"
            "  - verdict 非 ok（no_history/"
            "current_segment_low_quality）时不得做任何变化声明，只做描述；"
            "报告带 insufficient_power 注记时可保留判定但须降权措辞\n"
            "  - 变化声明只能引用 changed_features 中引擎判定的特征，"
            "Δ/阈值逐字引用；标记 VOLATILE 的特征要说明其波动性并降权\n"
            "  - 个人内 ρ(feature,BP) 是弱参考（留一符号一致率 0.64-0.70），"
            "只能以「弱相关」措辞引用，不得当作预测\n"
            "  - interpretation_note 里的生理含义只能正推描述（机制 + "
            "「带 X 标签的人群中该特征倾向偏高/偏低」），严禁反推为该受试者"
            "患病或出现某症状；解释个人数值时优先列举常见原因（体位/灌注/"
            "传感器位置/瞬时生理）\n"
            "  - 无血压数据时（measured 为空 / not_available）不得做任何"
            "血压相关推断，纯 PPG 分析（变化/人群位置/含义解释）照常可用\n"
            "  - 必须包含安全声明\n"
            f"子 Agent 输出：\n{agent_out}\n"
            f"Agent 2 发现（LLM 决策的结构化结果）：\n{findings_txt}\n"
            f"纵向比较证据（Agent 2 确定性证据块）：\n{pheno_evidence}\n"
            f"知识库规则：\n{knowledge_txt}\n"
            f"结构化结果：\n{text}"))
    ]
    try:
        out = llm.invoke([*history_msgs, *synth])
    except Exception:  # noqa: BLE001 — fall back to the router's reply
        return ""
    return out.content if isinstance(out.content, str) else str(out.content)
