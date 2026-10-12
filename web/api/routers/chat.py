"""Agent-driven chat analysis — Supervisor orchestrating 3 sub-agents.

The web chat runs the real three-phase Supervisor orchestration (the LLM no
longer drives a tool loop):

    (1) plan      — the Supervisor LLM emits a structured SupervisorPlan
                    (intent parse + task assignment); keyword routing and the
                    legacy ReAct loop are fallbacks
    (2) dispatch  — sub-agent LangGraph graphs run in task-queue order inside
                    one fresh PPGContext: Agent 1 (signal processing) then
                    Agent 2 (cardiovascular).  Dataset-adaptation sessions
                    (option-list HITL) are driven directly.
    (3) synthesize— the Supervisor rewrites the final reply from the sub-agent
                    outputs + retrieved physiological rules (≤4 sentences)

Frontend quick-action buttons (保存/取消/换一批段) still execute
deterministically before anything else.  The full structured result is
returned to the frontend for rich rendering, plus an ``agent_execution``
trace for the conversation-level orchestration card.
"""
from __future__ import annotations

import re

from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from ppg_agent.agents.adaptation_session import (
    adapt_pipeline,
    active_adapt_session,
    get_adapt_session,
)
from ppg_agent.agents.ml_prediction_agent import create_ml_prediction_agent
from ppg_agent.agents.supervisor import (
    AGENT_CARDIO, AGENT_LLM_KEY, AGENT_ML, AGENT_SIGNAL,
    SUPERVISOR_SYSTEM_PROMPT,
    SubTask,
    SupervisorPlan,
)
from ppg_agent.context import fresh_async_task_context
from ppg_agent.llm_service import get_llm

from ..services import chat_supervisor as chat_sup
from ..services.analysis_service import _resolve_params
from ..deps import resolve_segment as _resolve_segment

router = APIRouter(prefix="/chat", tags=["chat"])

# The dataset-adaptation session engine lives in ppg_agent (module-level
# per-source sessions): the Agent-1 graph's adapt branch drives it via the
# adapt_pipeline tool, and this router drives the very same tool
# deterministically for quick actions / 选N.  There is ONE engine — no web
# copy, no letter protocol, no per-request session handle.


# Frontend quick-action buttons send these fixed phrases.  Save/cancel are
# deterministic HITL operations — relying on the LLM to translate them into
# tool calls is flaky (it sometimes replies with text and calls nothing),
# which left the frontend with stale 'active' state.  These phrases are
# executed server-side before the LLM loop instead.
_QUICK_SAVE = "可以了，保存吧"
_QUICK_CANCEL = "不调了，取消"
_QUICK_SAMPLE = "再看几个 segment"
_QUICK_ACTIONS = {_QUICK_SAVE, _QUICK_CANCEL, _QUICK_SAMPLE}


def _force_quick_action(text: str) -> str | None:
    """Deterministically execute a quick-action button press; return an
    instruction for the LLM (which only confirms to the user)."""
    if text not in _QUICK_ACTIONS:
        return None
    dapt = active_adapt_session()
    if dapt is None:
        return None
    if text == _QUICK_SAVE:
        res = adapt_pipeline("save", source=dapt.source)
        return (f"服务端已确定性保存用户适配预设 (用户按了\"保存参数\"按钮):\n{res['message']}"
                "请直接向用户确认保存结果, 不要调用任何工具。")
    if text == _QUICK_CANCEL:
        res = adapt_pipeline("cancel", source=dapt.source)
        return (f"服务端已确定性取消本次适配会话 (用户按了\"取消\"按钮): {res['message']}。"
                "请直接告知用户已取消、未保存任何参数, 不要调用任何工具。")
    # _QUICK_SAMPLE
    res = adapt_pipeline("sample", source=dapt.source)
    return (f"服务端已确定性更换采样 (用户按了\"换一批段\"按钮):\n{res['message']}"
            "请直接向用户汇报新采样的概况, 不要调用任何工具。")


# 选N is likewise deterministic (the option list was generated server-side,
# so picking is a pure index lookup — no LLM needed).
_PICK_RE = re.compile(r"^选\s*([A-Fa-f1-6一二三四五六])\s*$")
_PICK_IDX = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
             "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6,
             "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}


def _force_pick(text: str) -> str | None:
    """Deterministically apply a "选N" option pick against the active session;
    return an instruction for the LLM (which only reports the outcome)."""
    dapt = active_adapt_session()
    if dapt is None:
        return None
    m = _PICK_RE.match(text.strip())
    if not m:
        return None
    idx = _PICK_IDX[m.group(1)]
    res = adapt_pipeline("apply", source=dapt.source, choice_idx=idx)
    return (f"服务端已确定性应用选项 {idx} (用户输入\"选 {m.group(1)}\"):\n{res['message']}"
            "请直接向用户汇报新参数与采样概况, 不要调用任何工具。")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # [{role: "user"|"assistant", content: str}]
    source: str | None = None
    subject_id: str | None = None
    seg_id: int | None = None
    target: str = "sbp"        # 可解释目标(sbp|dbp|age)：与右侧面板当前查看目标对齐


def _pipeline(seg: dict, phenotype: bool) -> dict:
    """Deterministic Agent 1 (+2) pipeline inside a fresh context."""
    from ..services import analysis_service as svc

    svc.load_signal(seg["source"], seg["subject_id"], seg["seg_idx"])
    svc.normalize()
    beats = svc.segment_beats_step()
    feats = svc.extract_features_step(seg["source"], seg["subject_id"], seg["seg_idx"])
    overlay = svc.beat_overlay()
    out = {"beats": beats, "features": feats, "overlay": overlay}
    if phenotype:
        scenario = chat_sup.scenario_for(seg)
        out["report"] = svc.run_phenotype(seg["source"], scenario=scenario)
        out["overlay"] = svc.enrich_overlay_feats(out["overlay"], out["report"])
    return out


def _summarize(r: dict, phenotype: bool) -> str:
    b = r["beats"]
    lines = [f"信号处理完成: {b['n_valid']}/{b['n_beats']} 有效beat, "
             f"质量 {b['quality']} ({b['quality_tier']}), preset={b.get('preset')}"]
    if phenotype:
        rep = r["report"]
        strat = rep.get("stratified_percentiles") or {}
        if strat:
            # 百分位口径摘要：证伪的横断面 VAI 不再进叙事（只保留描述性统计）
            n_dev = sum(1 for e in strat.values()
                        if e.get("level") not in ("normal", "unknown", None))
            lines.append(f"表型基线评估完成: {n_dev}/{len(strat)} 项偏离参考范围"
                         f"（基线 {rep.get('baseline_source')}, "
                         f"置信度={rep.get('confidence')}）")
        else:
            lines.append(f"表型基线评估完成: 基线={rep.get('baseline_source')}, "
                         f"置信度={rep.get('confidence')}")
        if rep.get("baseline_warning"):
            lines.append(f"警告: {rep['baseline_warning']}")
    return "\n".join(lines)


# Tool results are captured here (keyed by tool_call_id) so the endpoint can
# return the full structured result to the frontend, not just the summary.
_TOOL_RESULTS: dict[str, dict] = {}


_XML_TOOL_RE = re.compile(
    r'<tool_calls>\s*<invoke name="([^"]+)">(.*?)</invoke>\s*</tool_calls>',
    re.DOTALL)
_XML_PARAM_RE = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.DOTALL)


def _parse_xml_tool_calls(content: str) -> list[dict]:
    """Fallback: deepseek sometimes emits tool calls as literal XML text."""
    calls = []
    for i, m in enumerate(_XML_TOOL_RE.finditer(content or "")):
        args = {}
        for pm in _XML_PARAM_RE.finditer(m.group(2)):
            val = pm.group(2).strip()
            if val.isdigit():
                val = int(val)
            elif val.lower() in {"true", "false"}:
                val = val.lower() == "true"
            args[pm.group(1)] = val
        calls.append({"name": m.group(1), "args": args, "id": f"xml_{i}"})
    return calls


@tool
def analyze_signal(seg_id: int) -> str:
    """运行 Agent 1 信号处理管线: 波形分段 + 特征提取, 返回质量与特征概要.

    当用户想分析波形、查看信号质量、或者进行特征提取时调用.
    """
    seg = _resolve_segment(seg_id)
    result = fresh_async_task_context().run(lambda: _pipeline(seg, phenotype=False))
    _TOOL_RESULTS[f"signal:{seg_id}"] = result
    return _summarize(result, False)


@tool
def analyze_phenotype(seg_id: int) -> str:
    """运行表型分析 (Agent 1 + Agent 2): 波形处理 + 个人纵向比较 + 人群位置 + 特征生理含义解读.

    当用户想了解相对本人历史的变化、人群位置、血压水平与特征的生理含义时调用.
    """
    seg = _resolve_segment(seg_id)
    result = fresh_async_task_context().run(lambda: _pipeline(seg, phenotype=True))
    _TOOL_RESULTS[f"phenotype:{seg_id}"] = result
    return _summarize(result, True)


# Agent 3 is a deterministic inference engine (BeatMAE + causal streaming head
# + IG attribution), NOT a LangGraph agent — its predict/explain/format logic
# lives in ppg_agent.agents.ml_prediction_agent for symmetry with Agent 1/2.
# The router just drives the facade; the Supervisor words every reply.
_AGENT3 = create_ml_prediction_agent()


def _format_explain(ex: dict) -> str:
    return _AGENT3.format_explain(ex)


@tool
def ml_predict_risk(seg_id: int) -> str:
    """Agent 3: BeatMAE 深度学习模型预测 (血压 SBP/DBP + 年龄).

    用户想得到模型预测 (如估计血压、模型估的年龄) 时调用.
    主口径：当前段波形走完整实时流程（读窗→拍检测→BeatMAE 编码→
    causal_attn 流式头）出值；不可用时降级查表（同口径）。
    附带实测标签对照与置信说明；段级群体统计估计，非诊断结论.
    """
    seg = _resolve_segment(seg_id)
    try:
        pred, eng = _AGENT3.predict(seg)
    except Exception as e:  # noqa: BLE001 — 实时+查表均失败时如实降级
        return (f"Agent 3 (ML 预测) 不可用: {str(e)[:80]}. "
                "可改用 analyze_phenotype 获取基于百分位基线的表型评估.")
    return _AGENT3.format_prediction(seg, eng, pred)


# ReAct-fallback tool → sub-agent mapping (for the execution trace).
_TOOL_AGENT = {
    "analyze_signal": AGENT_SIGNAL,
    "analyze_phenotype": AGENT_CARDIO,
    "ml_predict_risk": AGENT_ML,
}


def _run_adapt(req: ChatRequest, messages: list,
               agent_execution: list[dict]) -> tuple[str, dict | None]:
    """Adaptation turn: drive the ONE session engine via the Agent-1 graph.

    Quick actions (保存/取消/换一批) and 选N already executed deterministically
    before this function.  Everything else runs the graph's adapt branch
    (state["mode"]="adapt"): the adapt LLM drives adapt_pipeline with
    semantic actions (start/propose/apply/sample/save/cancel) and replies to
    the user.  After the graph, if no session was started at all (text-only
    reply), a deterministic start guarantees the option list exists — the
    LLM never gets to claim options that aren't selectable.
    """
    from langchain_core.messages import SystemMessage

    source = req.source
    dapt = active_adapt_session(source)
    if dapt is not None:
        source = dapt.source  # the session wins over an absent/foreign request source
    elif not source:
        return "请先在顶部选择一个数据集，再进行适配。", None
    ctx_msgs = [SystemMessage(
        content=(f"当前上下文: 数据集 {source}, 未选具体受试者/segment。"
                 f"用户意图为该数据集的参数适配。"))] + list(messages[1:])
    st: dict = {}

    def _exec() -> None:
        try:
            text, trace = chat_sup.run_agent1_adapt(source, ctx_msgs)
            st["text"] = text
            st["trace"] = trace
        except Exception as e:  # noqa: BLE001 — deterministic fallback
            st["error"] = str(e)[:120]

    fresh_async_task_context().run(_exec)
    agent_execution.extend(st.get("trace", []) or [])
    if st.get("error"):
        agent_execution.append({"agent": AGENT_SIGNAL, "stage": "adapt",
                                "summary": f"适配图执行失败: {st['error']}"})
    reply = st.get("text") or ""
    sess = get_adapt_session(source)
    if sess is None:  # graph never touched this source → deterministic start
        res = adapt_pipeline("start", source=source)
        reply = res["message"]
        sess = get_adapt_session(source)  # refresh — start created the session
        agent_execution.append({"agent": AGENT_SIGNAL, "stage": "adapt",
                                "summary": "确定性启动适配会话 (图未启动)"})
    elif sess.status == "active" and not reply:
        res = adapt_pipeline("status", source=source)
        reply = res["message"]
    return reply, (sess.state_dict() if sess is not None else None)


# SubTask.task 是"人类可读描述"（LLM 常填中文自由文本，而非固定 verb），
# 因此按标记词识别"解释"意图；keyword 路由写入的字面量 "explain_prediction" 也在其中。
_EXPLAIN_TASK_MARKERS = ("explain_prediction", "可解释", "解释", "归因",
                        "为什么", "为何", "依据", "哪些特征点", "什么驱动", "理由")


def _ml_task_is_explain(t) -> bool:
    """判定一个 ml_prediction 子任务是"解释"还是"纯预测"。"""
    text = f"{t.task or ''} {t.params or ''}"
    return any(m in text for m in _EXPLAIN_TASK_MARKERS)


def _run_graphs(req: ChatRequest, seg: dict, plan,
                messages: list, agent_execution: list[dict]) -> tuple[str, dict | None, dict | None, dict | None]:
    """Supervisor dispatch: sub-agent graphs in ONE fresh PPGContext.

    Agent 1 → Agent 2 share the context (Agent 2 reads Agent 1's features);
    a single fresh context also keeps concurrent requests isolated.  On graph
    failure the deterministic _pipeline takes over (marked in the trace).
    """
    from ..services import analysis_service as svc

    tasks = plan.task_queue
    want_phenotype = any(t.target_agent == AGENT_CARDIO for t in tasks)
    _ml_tasks = [t for t in tasks if t.target_agent == AGENT_ML]
    want_ml = any(not _ml_task_is_explain(t) for t in _ml_tasks)
    want_explain = any(_ml_task_is_explain(t) for t in _ml_tasks)
    graph_msgs = messages[1:]  # drop the Supervisor system prompt
    scenario = chat_sup.scenario_for(seg)
    st: dict = {}

    def _exec() -> None:
        preset_name, params = _resolve_params(seg["source"])
        agent_texts: list[str] = []
        analysis = None
        try:
            if want_phenotype or any(t.target_agent == AGENT_SIGNAL for t in tasks):
                txt, trace = chat_sup.run_agent1(seg, params, graph_msgs)
                if txt:
                    agent_texts.append(txt)
                agent_execution.extend(trace)
                analysis = chat_sup.analysis_from_ctx(seg, preset_name)
                if want_phenotype:
                    analysis["report"] = svc.run_phenotype(seg["source"], scenario=scenario)
                    analysis["overlay"] = svc.enrich_overlay_feats(
                        analysis["overlay"], analysis["report"])
            if want_phenotype:
                # Agent 2 emits structured findings (no prose) — the
                # Supervisor words them (plus the deterministic compare
                # report) in the synthesis phase.
                findings, compare_report, trace = chat_sup.run_agent2(
                    seg, scenario, [graph_msgs[-1]])
                if findings:
                    analysis["findings"] = findings
                if compare_report:
                    analysis["compare_report"] = compare_report
                agent_execution.extend(trace)
            if want_ml:
                try:
                    pred, _eng = _AGENT3.predict(seg)
                    agent_texts.append(_AGENT3.format_prediction(seg, _eng, pred))
                    p = pred["predictions"]
                    agent_execution.append({
                        "agent": AGENT_ML, "stage": "predict",
                        "summary": (f"SBP {p['sbp']} / DBP {p['dbp']} mmHg / "
                                    f"年龄 {p['age']} 岁 ({pred['engine']}, "
                                    f"{pred['encoder_version']})")})
                except Exception as e:  # noqa: BLE001 — 实时+查表均失败时如实降级
                    agent_texts.append(f"Agent 3 (ML 预测) 不可用: {str(e)[:80]}")
                    agent_execution.append({"agent": AGENT_ML,
                                            "stage": "predict", "summary": "预测失败/产物缺失"})
            if want_explain:
                ex = _AGENT3.explain(seg, req.target)
                st["explain"] = ex
                if ex.get("unavailable"):
                    agent_texts.append(f"Agent 3 (可解释) 不可用: {ex['unavailable']}")
                    agent_execution.append({
                        "agent": AGENT_ML, "stage": "explain_prediction",
                        "summary": "解释链不可用（需 GPU+encoder）"})
                else:
                    agent_texts.append(_format_explain(ex))
                    agent_execution.append({
                        "agent": AGENT_ML, "stage": "explain_prediction",
                        "summary": (ex.get("direct_answer", "")[:100])})
        except Exception as e:  # noqa: BLE001 — fall back to the deterministic pipeline
            agent_execution.append({"agent": AGENT_SIGNAL, "stage": "fallback",
                                    "summary": f"图执行失败，降级确定性管线: {str(e)[:80]}"})
            analysis = _pipeline(seg, phenotype=want_phenotype)
            agent_texts.append(_summarize(analysis, want_phenotype))
        st["analysis"] = analysis
        st["texts"] = agent_texts

    fresh_async_task_context().run(_exec)

    analysis = st.get("analysis")
    agent_texts = st.get("texts", [])
    reply = ""
    # explain-only 时 analysis 为 None，但同样必须过 Supervisor 综合（否则
    # 回复是 _format_explain 原文直出，丢"≤ 4 句 + 知识叙事主体"纪律）
    if analysis is not None or st.get("explain") is not None:
        reply = chat_sup.synthesize_reply(agent_texts, analysis or {},
                                          messages[-8:], get_llm("supervisor"))
    if not reply:
        # synthesis failed → deterministic fallback; Agent 2 no longer
        # contributes prose, so the phenotype summary must come from the
        # structured analysis to avoid losing the phenotype content
        reply = "\n".join(agent_texts)
        if not reply and analysis is not None and (analysis.get("report") or {}):
            reply = _summarize(analysis, True)
        if not reply:
            reply = "分析完成。"
    sess = get_adapt_session()
    return reply, analysis, (sess.state_dict() if sess is not None else None), st.get("explain")


@router.post("/message")
def chat_message(req: ChatRequest):
    llm = get_llm(AGENT_LLM_KEY[AGENT_SIGNAL])
    if req.seg_id:
        ctx_desc = (f"当前已选 segment: source={req.source}, subject={req.subject_id}, "
                    f"seg_id={req.seg_id} (用户意图应为单段分析)")
    elif req.source:
        ctx_desc = (f"当前已选数据集: source={req.source}, 未选具体受试者/segment. "
                    f"用户若提到对该数据集做特征提取适配/调参/新数据集, "
                    f"输出 task=\"dataset_adaptation\"")
        dapt = active_adapt_session(req.source)
        if dapt is not None:
            opts = dapt.state_dict().get("pending_options") or []
            if opts:
                opt_lines = "; ".join(
                    f"{i + 1}){o.get('user_desc') or o.get('rationale') or o['param']}"
                    for i, o in enumerate(opts))
                ctx_desc += (f" 当前有 {len(opts)} 个待选选项: {opt_lines}. "
                             f"用户说\"选 N\"由服务端确定性处理, 不需要 LLM 操作. "
                             f"用户给出负面反馈时输出 task=\"dataset_adaptation\" "
                             f"重新提议选项, 不要反问或猜测.")
            else:
                ctx_desc += (f" 该数据集已有进行中的适配会话. 用户给出负面反馈时输出 "
                             f"task=\"dataset_adaptation\" 重新提议选项, 不要反问细节.")
    else:
        ctx_desc = "当前未选择数据集/segment"
    adapt_rules = (
        "适配会话（signal_processing 子 Agent 的调参能力，由 Agent 1 适配分支驱动）:\n"
        "  - 数据集适配/调参（用户已选数据集、未选 segment）→ 输出 task=\"dataset_adaptation\"，"
        "由适配分支通过 adapt_pipeline 语义动作推进 (start/propose/apply/sample/save/cancel)\n"
        "  - \"选 N\"、保存参数/取消/换一批段按钮由服务端确定性处理，不需要 LLM 调用工具\n"
        "  - 用户给出负面反馈(分割不好/特征点不对/参数差)或说\"有什么可以调\"时，"
        "适配分支基于反馈重新提议 2-4 个选项，绝不直接改参数\n"
    )
    system = (SUPERVISOR_SYSTEM_PROMPT + "\n\n" + adapt_rules
              + f"当前上下文: {ctx_desc}\n\n{chat_sup.PLAN_ADDENDUM}")
    messages = [SystemMessage(system)]
    for h in req.history[-12:]:
        if h.get("role") == "user":
            messages.append(HumanMessage(content=h.get("content", "")))
        else:
            messages.append(AIMessage(content=h.get("content", "")))
    messages.append(HumanMessage(content=req.message))
    # Deterministic layer (no LLM): quick-action buttons (保存参数/取消/换一批段)
    # and 选N execute before anything else — a text-only LLM reply left the
    # session unchanged and the frontend kept stale 'active' buttons, and the
    # option list is generated server-side so picking is a pure index lookup.
    quick_note = _force_quick_action(req.message.strip())
    pick_note = _force_pick(req.message) if quick_note is None else None
    note = quick_note or pick_note
    if note:
        messages.append(SystemMessage(content=note))

    analysis = None
    adapt = None
    explain = None
    agent_execution: list[dict] = []
    reply = None
    plan = None

    # ── (1) PLAN: structured SupervisorPlan (tool_calls → JSON → XML → keywords)
    if note is None:
        plan = chat_sup.extract_plan(get_llm("supervisor"), messages)
        if plan is None:
            # An active adaptation session turns free-text feedback into an
            # adaptation turn ("特征点没标好" carries no adapt keyword, and the
            # "特征" process keyword must not route it to segment analysis).
            # Strong new-intent keywords still win.
            if (active_adapt_session(req.source) is not None
                    and not any(k in req.message.lower()
                                for k in ("分析", "评估", "看看", "查看",
                                          "提取", "管线", "可视化"))):
                plan = SupervisorPlan(
                    thought="进行中的适配会话: 反馈路由",
                    needs_dispatch=True,
                    task_queue=[SubTask(target_agent=AGENT_SIGNAL,
                                        task="dataset_adaptation", params={})])
            else:
                plan = chat_sup.keyword_plan(req.message, req.seg_id)
        if plan is not None:
            agent_execution.append({"agent": "supervisor", "stage": "plan",
                                    "summary": plan.thought})

    # ── (2) DISPATCH + SYNTHESIZE
    if plan is not None:
        if plan.needs_dispatch and plan.task_queue:
            if any(t.task == "dataset_adaptation" for t in plan.task_queue):
                reply, adapt = _run_adapt(req, messages, agent_execution)
            else:
                seg = _resolve_segment(req.seg_id) if req.seg_id is not None else None
                if seg is None:
                    reply = plan.direct_reply or (
                        "我理解你想分析，但还没有选择具体 segment —— 请先在顶部"
                        "选择 数据集 / 受试者 / segment。")
                    agent_execution.append({"agent": "supervisor", "stage": "dispatch",
                                            "summary": "缺少 segment 参数"})
                else:
                    reply, analysis, adapt, explain = _run_graphs(req, seg, plan, messages,
                                                                  agent_execution)
        elif plan.direct_reply:
            reply = plan.direct_reply

    # ── (3) FALLBACK: legacy ReAct loop (plan extraction fully failed)
    if reply is None:
        if note is not None:
            # A deterministic action (保存/取消/换一批段 / 选N) already executed —
            # the ReAct loop below only runs the LLM's confirmation reply.
            summary = ("确定性选项应用（选N），LLM 仅向用户确认"
                       if pick_note is not None
                       else "确定性快捷动作，LLM 仅向用户确认")
            agent_execution.append({"agent": "supervisor", "stage": "dispatch",
                                    "summary": summary})
        else:
            agent_execution.append({"agent": "supervisor", "stage": "fallback",
                                    "summary": "计划提取失败，降级 ReAct 工具循环"})
        tools = [analyze_signal, analyze_phenotype, ml_predict_risk]
        llm_tools = llm.bind_tools(tools)
        ai = llm_tools.invoke(messages)
        for _turn in range(8):
            if not ai.tool_calls and ai.content:
                ai.tool_calls = _parse_xml_tool_calls(ai.content)  # XML-text fallback
            if not ai.tool_calls:
                break
            tool_messages = []
            for tc in ai.tool_calls:
                name = tc["name"]
                args = tc.get("args") or {}
                try:
                    if name == "analyze_signal":
                        summary = analyze_signal.invoke(args)
                        analysis = _TOOL_RESULTS.pop(f"signal:{args['seg_id']}")
                    elif name == "analyze_phenotype":
                        summary = analyze_phenotype.invoke(args)
                        analysis = _TOOL_RESULTS.pop(f"phenotype:{args['seg_id']}")
                    elif name == "ml_predict_risk":
                        summary = ml_predict_risk.invoke(args)
                    else:
                        summary = f"未知工具 {name}"
                except Exception as e:  # noqa: BLE001 — surface tool errors to the LLM
                    summary = f"工具执行失败: {e}"
                agent_execution.append({"agent": _TOOL_AGENT.get(name, "supervisor"),
                                        "stage": name, "summary": summary[:100]})
                tool_messages.append(ToolMessage(content=summary, tool_call_id=tc["id"]))
            messages = [*messages, ai, *tool_messages]
            ai = llm_tools.invoke(messages)
        reply = ai.content if isinstance(ai.content, str) else str(ai.content)
        # Synthesis phase: if the sub-agents returned structured analysis, the
        # Supervisor rewrites the final reply from the full structured results +
        # retrieved physiological rules (concise cross-evidence report).
        if analysis is not None:
            synth = chat_sup.synthesize_reply(
                [_summarize(analysis, bool(analysis.get("report")))],
                analysis, messages[-8:], get_llm("supervisor"))
            if synth:
                reply = synth

    # Always surface the real adapt state for the touched session — even when
    # this message ran no adapt action (text-only reply), the frontend must not
    # keep rendering stale 'active' buttons.  get_adapt_session() also returns
    # saved/aborted sessions so the UI clears its buttons after save/cancel.
    if adapt is None:
        sess = get_adapt_session()
        if sess is not None:
            adapt = sess.state_dict()

    return {"reply": reply, "analysis": analysis, "adapt": adapt,
            "seg_id": req.seg_id, "agent_execution": agent_execution,
            "explain": explain}
