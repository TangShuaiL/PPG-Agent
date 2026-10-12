"""Tests for the supervisor orchestration layer (web chat): plan extraction,
keyword routing, sub-agent graph dispatch, preset-params pass-through, and
the chat_message end-to-end flow.  All network/LLM/graph calls are faked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from langchain_core.messages import AIMessage, SystemMessage

from ppg_agent.agents.supervisor import SupervisorPlan
from ppg_agent.context import reset_context
from web.api.services import chat_supervisor as cs


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLLM:
    """bind_tools → self; invoke → the first preset response (repeatable)."""

    def __init__(self, response):
        self._response = response

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        return self._response


def plan_ai(**plan_fields):
    args = {"thought": "intent", "needs_dispatch": True,
            "task_queue": [{"target_agent": "signal_processing",
                            "task": "analyze_signal", "params": {}}],
            "direct_reply": None}
    args.update(plan_fields)
    return AIMessage(content="", tool_calls=[
        {"name": "SupervisorPlan", "args": args, "id": "call_1"}])


# ---------------------------------------------------------------------------
# Plan extraction tiers
# ---------------------------------------------------------------------------

def test_plan_from_json_plain():
    plan = cs._plan_from_json(
        '{"thought": "t", "needs_dispatch": false, "task_queue": [], "direct_reply": "hi"}')
    assert plan is not None and plan.direct_reply == "hi" and not plan.needs_dispatch


def test_plan_from_json_code_fence():
    plan = cs._plan_from_json(
        '```json\n{"thought": "t", "needs_dispatch": true, '
        '"task_queue": [{"target_agent": "cardiovascular", "task": "x", "params": {}}]}\n```')
    assert plan is not None and plan.task_queue[0].target_agent == "cardiovascular"


def test_plan_from_json_garbage():
    assert cs._plan_from_json("no plan here") is None
    assert cs._plan_from_json("") is None
    assert cs._plan_from_json('{"foo": 1}') is None


def test_plan_from_xml_plain():
    xml = ("<SupervisorPlan><thought>t</thought><needs_dispatch>true</needs_dispatch>"
           "<task_queue><item><target_agent>cardiovascular</target_agent>"
           "<task>x</task><params><source>vital</source></params></item></task_queue>"
           "</SupervisorPlan>")
    plan = cs._plan_from_xml(xml)
    assert plan is not None and plan.task_queue[0].params == {"source": "vital"}


def test_plan_from_xml_garbage():
    assert cs._plan_from_xml("random text") is None
    assert cs._plan_from_xml("") is None


def test_extract_plan_tool_call_tier():
    llm = FakeLLM(plan_ai(needs_dispatch=True))
    plan = cs.extract_plan(llm, [])
    assert plan is not None and plan.needs_dispatch


def test_extract_plan_json_tier():
    llm = FakeLLM(AIMessage(content=('{"thought": "t", "needs_dispatch": false, '
                                     '"task_queue": [], "direct_reply": "hi"}')))
    assert cs.extract_plan(llm, []).direct_reply == "hi"


def test_extract_plan_xml_tier():
    llm = FakeLLM(AIMessage(content=(
        "<SupervisorPlan><thought>t</thought><needs_dispatch>false</needs_dispatch>"
        "<direct_reply>hi</direct_reply></SupervisorPlan>")))
    assert cs.extract_plan(llm, []).direct_reply == "hi"


def test_extract_plan_all_tiers_fail():
    llm = FakeLLM(AIMessage(content="I don't know what to do"))
    assert cs.extract_plan(llm, []) is None


def test_extract_plan_llm_error():
    class Boom:
        def bind_tools(self, tools, **kw):
            raise RuntimeError("llm down")
    assert cs.extract_plan(Boom(), []) is None


# ---------------------------------------------------------------------------
# Keyword routing fallback
# ---------------------------------------------------------------------------

def test_keyword_plan_adapt():
    p = cs.keyword_plan("帮我调一下参数", None)
    assert p.needs_dispatch and p.task_queue[0].task == "dataset_adaptation"


def test_keyword_plan_ml():
    p = cs.keyword_plan("预测高血压风险", 1)
    assert p.task_queue[0].target_agent == "ml_prediction"


def test_keyword_plan_phenotype():
    p = cs.keyword_plan("评估血管年龄", 1)
    assert [t.target_agent for t in p.task_queue] == ["signal_processing", "cardiovascular"]


def test_keyword_plan_analysis_needs_segment():
    p = cs.keyword_plan("分析这个波形", None)
    assert not p.needs_dispatch and p.direct_reply


def test_keyword_plan_chitchat():
    assert cs.keyword_plan("你好呀", None) is None


# ---------------------------------------------------------------------------
# Sub-agent dispatch (fake graphs)
# ---------------------------------------------------------------------------

class FakeGraph1:
    def stream(self, inp, stream_mode="updates"):
        yield {"load": {"messages": []}}
        yield {"segment": {"messages": [], "n_beats": 5, "n_valid": 3, "quality": 0.6}}
        yield {"extract_features": {"messages": [],
                                    "feature_summary": {"n_beats_total": 5, "hr_bpm": 70}}}
        yield {"decide": {"messages": [AIMessage(content="done")]}}


class FakeGraph2:
    """Stream-based Agent 2 graph: assess → interpret → capture_findings.

    Agent 2 emits no prose — the deterministic compare report and the
    structured findings come from the node updates.
    """

    def stream(self, inp, stream_mode="updates", **kwargs):
        yield {"assess": {"compare_report": {
            "verdict": "ok", "seg_idx": 0, "n_history": 30, "n_recent": 32,
            "n_real_change": 1,
            "features": {"pulse_amp_mean": {
                "verdict_long": "real_change", "delta_vs_long": -0.12,
                "delta_vs_recent": -0.05, "personal_median": 0.55,
                "threshold": 0.05, "threshold_source": "personal_mad"}}}}}
        yield {"interpret": {"messages": []}}
        yield {"capture_findings": {"findings": {
            "changed_features": [{"feature": "pulse_amp_mean",
                                   "delta": -0.12, "threshold": 0.05,
                                   "direction": "decreased"}]}}}


def test_run_agent1_trace(monkeypatch):
    monkeypatch.setattr(cs, "_agent1_graph", lambda: FakeGraph1())
    reset_context()
    text, trace = cs.run_agent1({"source": "vital", "subject_id": "p1", "seg_idx": 0},
                                {"x": 1}, [])
    assert text == "done"
    assert [t["stage"] for t in trace] == ["load", "segment", "extract_features", "decide"]
    assert "5 beats" in trace[1]["summary"]
    assert all(t["agent"] == "signal_processing" for t in trace)


def test_run_agent2_trace(monkeypatch):
    monkeypatch.setattr(cs, "_agent2_graph", lambda: FakeGraph2())
    reset_context()
    findings, compare_report, trace = cs.run_agent2(
        {"source": "vital", "subject_id": "p1", "seg_idx": 0}, None, [])
    # structured findings + deterministic compare report passthrough
    assert findings["changed_features"][0]["feature"] == "pulse_amp_mean"
    assert compare_report["verdict"] == "ok"
    assert [t["stage"] for t in trace] == ["assess", "interpret", "capture_findings"]
    assert trace[0]["summary"].startswith("verdict=ok")
    assert "结构化发现" in trace[2]["summary"]


def test_synthesize_reply_phenotype_discipline():
    """Supervisor is the only wording layer: longitudinal discipline +
    Agent 2's deterministic compare blocks + structured findings all reach
    the prompt."""

    class CaptureLLM:
        def __init__(self):
            self.captured = None
            self._response = AIMessage(content="综合结论：指标稳定。建议复查。")

        def invoke(self, messages, **kwargs):
            self.captured = [m for m in messages
                             if isinstance(m, SystemMessage)][-1]
            return self._response

    llm = CaptureLLM()
    compare_report = {
        "verdict": "ok", "seg_idx": 5, "n_history": 30, "n_recent": 32,
        "n_real_change": 1,
        "features": {
            "pulse_amp_mean": {
                "verdict_long": "real_change", "delta_vs_long": -0.12,
                "delta_vs_recent": -0.05, "personal_median": 0.55,
                "threshold": 0.05, "threshold_source": "personal_mad"},
            "kurtosis_mean": {
                "verdict_long": "real_change", "delta_vs_long": 1.0,
                "delta_vs_recent": None, "personal_median": 3.0,
                "threshold": 0.4, "threshold_source": "personal_mad"}},
        "beat_layer": {"available": True, "n_valid_cur": 12,
                        "n_beats_cur": 16, "valid_ratio": 0.75,
                        "outlier_rate_median": 0.08},
    }
    analysis = {"compare_report": compare_report,
                "findings": {"changed_features": [
                    {"feature": "pulse_amp_mean", "delta": -0.12,
                     "threshold": 0.05, "direction": "decreased"}]},
                "features": {"feature_means": {"pulse_amp": 0.43}},
                "beats": {"n_beats": 16, "n_valid": 12}}
    reply = cs.synthesize_reply(["信号处理完成"], analysis, [], llm)
    sys_msg = llm.captured.content
    # wording discipline (longitudinal)
    assert "纵向比较解读纪律" in sys_msg
    assert "绝不诊断疾病" in sys_msg
    assert "弱相关" in sys_msg and "安全声明" in sys_msg
    # deterministic compare blocks + structured findings both reach wording
    assert "Verdict: ok" in sys_msg
    assert "pulse_amp_mean" in sys_msg
    assert "VOLATILE" in sys_msg          # kurtosis is flagged for downgrade
    assert "Agent 2 发现" in sys_msg
    assert reply == "综合结论：指标稳定。建议复查。"


# ---------------------------------------------------------------------------
# 证伪指标泄漏封堵（叙事层只剥 vascular_age/亚型/疾病关联/规则/窗口）
# ---------------------------------------------------------------------------

def test_without_falsified_strips_only_falsified_keys():
    rep = {"vascular_age": {"vai": 0.3}, "phenotype_subtype": {"cluster": 1},
           "disease_associations": {"ri": [...]}, "interpretation_rules": {"ri": {}},
           "window_context": {"is_short_window": False},
           "stratified_percentiles": {"ri_mean": {"level": "high"}},
           "baseline_source": "vital", "confidence": "medium"}
    out = cs._without_falsified(rep)
    assert not (set(cs._FALSIFIED_REPORT_KEYS) & set(out))
    assert out["stratified_percentiles"] and out["baseline_source"] == "vital"
    # 空/None 原样返回
    assert cs._without_falsified(None) is None
    assert cs._without_falsified({}) == {}


def test_synthesize_reply_falsified_fallback_leak_sealed():
    """compare_report 缺失、只有旧形态 report 时，证伪字段不得进提示词。"""

    class CaptureLLM:
        def __init__(self):
            self.captured = None
            self._response = AIMessage(content="综合结论。")

        def invoke(self, messages, **kwargs):
            self.captured = [m for m in messages
                             if isinstance(m, SystemMessage)][-1]
            return self._response

    llm = CaptureLLM()
    analysis = {"compare_report": None,
                "report": {"vascular_age": {"vai": 0.42,
                                            "interpretation": "血管年龄偏大"},
                           "phenotype_subtype": {"available": True, "cluster": 2},
                           "disease_associations": {"ri_mean": []},
                           "stratified_percentiles": {"ri_mean": {"level": "high"}},
                           "baseline_source": "vital", "confidence": "low"},
                "features": {"feature_means": {"ri": 0.8}},
                "beats": {"n_beats": 10, "n_valid": 8}}
    cs.synthesize_reply(["表型基线评估完成"], analysis, [], llm)
    sys_msg = llm.captured.content
    assert "vascular_age" not in sys_msg
    assert "血管年龄" not in sys_msg
    assert "VAI" not in sys_msg
    assert "phenotype_subtype" not in sys_msg
    assert "disease_associations" not in sys_msg
    # 描述性百分位键不在剥除集：旧形态 report 的 JSON dump 仍可见
    assert "stratified_percentiles" in sys_msg
    # 旧形态 report 不再喂 format_assessment_blocks（否则渲染全 None 幻影纵向块）
    assert "Verdict: None" not in sys_msg


def test_summarize_phenotype_percentile_wording():
    """_summarize 降级摘要改百分位口径：不再直出 VAI 数值。"""
    from web.api.routers import chat as chat_mod
    r = {"beats": {"n_beats": 16, "n_valid": 12, "quality": 0.75,
                   "quality_tier": "good", "preset": "default"},
         "report": {"vascular_age": {"vai": 0.42},
                    "stratified_percentiles": {
                        "ri_mean": {"level": "high"},
                        "lasi_mean": {"level": "normal"},
                        "auc_mean": {"level": "low"},
                        "slope_mean": {"level": "unknown"}},
                    "baseline_source": "vital", "confidence": "medium"}}
    out = chat_mod._summarize(r, True)
    assert "VAI" not in out
    assert "偏离参考范围" in out
    assert "2/4" in out          # ri(high) + auc(low)；normal/unknown 不计
    assert chat_mod._summarize(r, False).startswith("信号处理完成")


def test_keyword_plan_no_compound_dispatch_on_vascular_age():
    """"血管年龄"已从复合硬标记移除：归因+血管年龄句不并派 Agent 2。"""
    assert "血管年龄" not in cs._PHYSIO_COMPOUND_MARKERS
    p = cs.keyword_plan("依据哪些特征点，评估血管年龄", 1)
    assert [(t.target_agent, t.task) for t in p.task_queue] == [
        ("ml_prediction", "explain_prediction")]


def test_analysis_from_ctx_shape():
    reset_context()
    ctx = cs.get_context()
    ctx.num_beats, ctx.n_valid_beats = 10, 8
    ctx.ppg_quality, ctx.ppg_quality_tier = 0.75, "good"
    ctx.beat_records = [{}] * 5
    ctx.beat_starts, ctx.beat_valid_mask, ctx.beat_keypoints = [0, 1], [True, False], [{}, {}]
    ctx.segment_features_raw = {"ri_mean": 0.8, "subject_id": "p1", "seg_idx": 0}
    out = cs.analysis_from_ctx({"source": "vital"}, "default")
    assert out["beats"]["n_beats"] == 10 and out["beats"]["preset"] == "default"
    assert out["features"]["feature_means"] == {"ri": 0.8}
    assert out["overlay"]["beats"][0]["valid"] is True


# ---------------------------------------------------------------------------
# Preset params pass-through (Step 0b — the graph must honor the preset chain)
# ---------------------------------------------------------------------------

def _fake_ppg_result():
    return {"beats": [], "beat_starts": [], "valid_mask": [], "keypoints": [],
            "features": [], "n_total": 0, "n_valid": 0, "quality": 0.5,
            "quality_tier": "fair", "n_rejected_mad": 0}


def test_segment_node_passes_state_params(monkeypatch):
    from ppg_agent.agents import signal_processor_agent as spa
    reset_context()
    ctx = spa.get_context()
    ctx.ppg_signal = np.linspace(0, 1, 1000)
    ctx.fs = 500
    captured = {}

    def fake_process(ppg, fs, **kw):
        captured["params"] = kw.get("params")
        return _fake_ppg_result()

    monkeypatch.setattr(spa, "process_ppg_segment", fake_process)
    spa._segment_node({"source": "vital", "messages": [], "params": {"x": 1}})
    assert captured["params"] == {"x": 1}


def test_segment_node_resolves_params_when_absent(monkeypatch):
    from ppg_agent.agents import signal_processor_agent as spa
    reset_context()
    ctx = spa.get_context()
    ctx.ppg_signal = np.linspace(0, 1, 1000)
    ctx.fs = 500
    captured = {}

    def fake_process(ppg, fs, **kw):
        captured["params"] = kw.get("params")
        return _fake_ppg_result()

    monkeypatch.setattr(spa, "process_ppg_segment", fake_process)
    monkeypatch.setattr(spa, "_resolve_params", lambda source: {"y": 2})
    spa._segment_node({"source": "vital", "messages": []})
    assert captured["params"] == {"y": 2}


def test_extract_features_node_passes_params(monkeypatch):
    from ppg_agent.agents import signal_processor_agent as spa
    reset_context()
    ctx = spa.get_context()
    ctx.ppg_signal = np.linspace(0, 1, 1000)
    ctx.fs = 500
    captured = {}

    def fake_extract(ppg, abp, rpeaks, subject_id, seg_idx, dataset, **kw):
        captured["params"] = kw.get("params")
        return {"segment": {"n_beats_ppg": 3, "n_beats_valid": 2,
                            "quality_score": 0.6, "quality_tier": "good"},
                "beats": []}

    monkeypatch.setattr(spa, "extract_segment_features", fake_extract)
    spa._extract_features_node({"source": "vital", "subject_id": "p1",
                                "seg_idx": 0, "messages": [], "params": {"x": 1}})
    assert captured["params"] == {"x": 1}


# ---------------------------------------------------------------------------
# chat_message end-to-end (plan → dispatch → synthesize), faked sub-agents
# ---------------------------------------------------------------------------

def test_chat_message_end_to_end(monkeypatch):
    from web.api.routers import chat as chat_mod
    from web.api.routers.chat import ChatRequest, chat_message

    fake = FakeLLM(plan_ai())
    monkeypatch.setattr(chat_mod, "get_llm", lambda key: fake)
    monkeypatch.setattr(cs, "run_agent1",
                        lambda seg, params, msgs: ("agent1", [{"agent": "signal_processing", "stage": "load"}]))
    monkeypatch.setattr(cs, "analysis_from_ctx",
                        lambda seg, preset: {"beats": {"n_beats": 5, "n_valid": 3,
                                                       "quality": 0.6, "quality_tier": "good",
                                                       "preset": "default"},
                                             "features": {"feature_means": {"ri": 0.5}},
                                             "overlay": {}})
    monkeypatch.setattr(cs, "synthesize_reply", lambda *a, **k: "综合回复")

    r = chat_message(ChatRequest(message="分析这个波形的信号质量",
                                 source="vital", subject_id="p000051", seg_id=1))
    assert r["reply"] == "综合回复"
    assert r["analysis"]["beats"]["n_beats"] == 5
    assert r["seg_id"] == 1
    assert r["agent_execution"][0]["agent"] == "supervisor"
    assert r["agent_execution"][0]["stage"] == "plan"


def test_chat_message_keyword_fallback_when_plan_fails(monkeypatch):
    from web.api.routers import chat as chat_mod
    from web.api.routers.chat import ChatRequest, chat_message

    class BoomLLM:
        def bind_tools(self, tools, **kw):
            raise RuntimeError("down")

    monkeypatch.setattr(chat_mod, "get_llm", lambda key: BoomLLM())
    monkeypatch.setattr(chat_mod, "_resolve_params", lambda source: ("default", {}))
    monkeypatch.setattr(cs, "run_agent1",
                        lambda seg, params, msgs: ("agent1", [{"agent": "signal_processing", "stage": "load"}]))
    monkeypatch.setattr(cs, "analysis_from_ctx",
                        lambda seg, preset: {"beats": {"n_beats": 5, "n_valid": 3,
                                                       "quality": 0.6, "quality_tier": "good",
                                                       "preset": "default"},
                                             "features": {"feature_means": {"ri": 0.5}},
                                             "overlay": {}})
    monkeypatch.setattr(cs, "synthesize_reply", lambda *a, **k: "综合回复")

    r = chat_message(ChatRequest(message="分析这个波形的信号质量",
                                 source="vital", subject_id="p000051", seg_id=1))
    assert r["reply"] == "综合回复"
    assert r["agent_execution"][0]["summary"].startswith("关键词路由") or \
        r["agent_execution"][0]["stage"] == "plan"


def test_chat_message_missing_segment_guides_user(monkeypatch):
    from web.api.routers import chat as chat_mod
    from web.api.routers.chat import ChatRequest, chat_message

    fake = FakeLLM(plan_ai())
    monkeypatch.setattr(chat_mod, "get_llm", lambda key: fake)
    r = chat_message(ChatRequest(message="分析一下这个波形", source="vital"))
    assert "segment" in r["reply"]
    assert r["analysis"] is None
    assert any(e["stage"] == "dispatch" for e in r["agent_execution"])


def test_chat_message_direct_reply(monkeypatch):
    from web.api.routers import chat as chat_mod
    from web.api.routers.chat import ChatRequest, chat_message

    fake = FakeLLM(plan_ai(needs_dispatch=False, direct_reply="你好，有什么可以帮你？"))
    monkeypatch.setattr(chat_mod, "get_llm", lambda key: fake)
    r = chat_message(ChatRequest(message="你好"))
    assert r["reply"] == "你好，有什么可以帮你？"
    assert r["analysis"] is None


def test_chat_message_react_fallback_when_no_plan(monkeypatch):
    from web.api.routers import chat as chat_mod
    from web.api.routers.chat import ChatRequest, chat_message

    class TextOnlyLLM:
        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, messages, **kw):
            return AIMessage(content="直接回复")

    monkeypatch.setattr(chat_mod, "get_llm", lambda key: TextOnlyLLM())
    r = chat_message(ChatRequest(message="随便聊聊"))
    assert r["reply"] == "直接回复"
    assert any(e["stage"] == "fallback" for e in r["agent_execution"])


# ---------------------------------------------------------------------------
# Agent 3 可解释工具 (explain_prediction): 路由 / 降级 / 载荷 / 分发
# ---------------------------------------------------------------------------

def test_keyword_plan_explain_routing():
    p = cs.keyword_plan("为什么是这个血压值", 1)
    assert p.task_queue[0].target_agent == "ml_prediction"
    assert p.task_queue[0].task == "explain_prediction"
    # 纯预测意图仍走 ml_predict_risk
    q = cs.keyword_plan("预测血压", 1)
    assert q.task_queue[0].task == "ml_predict_risk"


def test_keyword_plan_physio_routing():
    """生理轨："为什么偏高/什么意思" → Agent 2 表型组合，不被模型归因劫持。"""
    p = cs.keyword_plan("我血压为什么偏高", 1)
    assert [t.target_agent for t in p.task_queue] == ["signal_processing",
                                                      "cardiovascular"]
    q = cs.keyword_plan("这个 RI 是什么意思", 1)
    assert [t.task for t in q.task_queue] == ["analyze_signal",
                                              "analyze_phenotype"]
    # 未选 segment → 引导选段，不分发
    r = cs.keyword_plan("我血压为什么偏高", None)
    assert r.needs_dispatch is False and r.direct_reply


def test_keyword_plan_no_hijack():
    """强归因分支优先于生理轨：含"特征点/归因"的"为什么"仍属模型解释。"""
    p = cs.keyword_plan("为什么特征点是这样分布的", 1)
    assert p.task_queue[0].task == "explain_prediction"
    # 前端两处便捷按钮预设指令回归：必须仍命中强归因轨，且**不**触发复合并派
    # （按钮句含"为什么+收缩压"但无生理硬标记，不得多跑一轮 Agent 2）
    for btn in ("请解释这个段当前的收缩压预测为什么是这个值，主要看哪些波形特征点",
                "展开这个段 Top-3 历史段的特征点归因"):
        b = cs.keyword_plan(btn, 1)
        assert len(b.task_queue) == 1, btn
        assert b.task_queue[0].target_agent == "ml_prediction", btn
        assert b.task_queue[0].task == "explain_prediction", btn


def test_keyword_plan_compound_dispatch():
    """复合句（归因硬标记 + 生理追问硬标记）→ 双轨并派，顺序：归因在前。"""
    p = cs.keyword_plan("我这个段血压为什么偏高？模型出这个值是依据哪些特征点？", 1)
    assert [(t.target_agent, t.task) for t in p.task_queue] == [
        ("ml_prediction", "explain_prediction"),
        ("cardiovascular", "analyze_phenotype")]
    # 未选 segment → 不并派（并派只加 phenotype 任务；缺段引导由端点统一守卫）
    q = cs.keyword_plan("血压偏高依据哪些特征点", None)
    assert len(q.task_queue) == 1


def test_ml_task_is_explain_freetext():
    """LLM 将 SubTask.task 填为中文自由描述（非固定 verb）时仍能识别解释意图。"""
    from web.api.routers import chat as chat_mod
    from ppg_agent.agents.supervisor import SubTask
    assert chat_mod._ml_task_is_explain(
        SubTask(target_agent="ml_prediction", task="解释舒张压是怎么得出的", params={}))
    assert not chat_mod._ml_task_is_explain(
        SubTask(target_agent="ml_prediction", task="预测血压和年龄", params={}))


def test_agent3_explain_degrades_without_gpu(monkeypatch):
    import ppg_agent.engine.explain as explain_mod
    import ppg_agent.agents.ml_prediction_agent as ml3_mod

    monkeypatch.setattr(ml3_mod, "predict_segment",
                        lambda seg: ({"predictions": {"sbp": 120, "dbp": 80, "age": 50}}, None))

    def _boom():
        raise RuntimeError("需要 GPU+encoder")

    monkeypatch.setattr(explain_mod, "get_live_explainer", _boom)
    out = ml3_mod.explain_segment(
        {"source": "vital", "subject_id": "p1", "seg_idx": 0}, "sbp")
    assert "unavailable" in out and out["target"] == "sbp"


def test_agent3_explain_payload(monkeypatch):
    import ppg_agent.engine.explain as explain_mod
    import ppg_agent.agents.ml_prediction_agent as ml3_mod

    fake_expl = {
        "direct_answer": "模型对该收缩压的段级估计为 120 mmHg",
        "current_segment": {"top_keypoints": [{"label": "收缩峰", "contrib": 12.3}]},
        "history": {"segments": [{"seg_idx": 3, "attention_weight": 0.7}]},
        "knowledge_interpretation": {
            "statements": ["当前段 · 收缩峰：动脉顺应性下降时收缩峰抬高；"
                           "模型敏感度上把 sbp 往上推（提示性）。"],
            "caveat": "群体模型依赖度，非个体病因"},
        "uncertainty": ["近窗注意力 + 当前段 IG 完备性均在阈值内"],
    }

    class FakeExplainer:
        def explain_segment(self, source, subject_id, seg_idx, target,
                           top_history=3, predict=None):
            assert predict is not None  # 复用已算好的 predict
            return dict(fake_expl)

    monkeypatch.setattr(ml3_mod, "predict_segment",
                        lambda seg: ({"predictions": {"sbp": 120, "dbp": 80, "age": 50}}, None))
    monkeypatch.setattr(explain_mod, "get_live_explainer", lambda: FakeExplainer())
    out = ml3_mod.explain_segment(
        {"source": "vital", "subject_id": "p1", "seg_idx": 0}, "sbp")
    assert out["direct_answer"] == fake_expl["direct_answer"]
    assert out["current_segment"]["top_keypoints"][0]["label"] == "收缩峰"
    assert out["target"] == "sbp"
    text = ml3_mod.format_explain(out)
    assert "收缩峰" in text and "#3" in text
    # Agent 2 知识层必须进入给 Supervisor 的概要（防回复退化成数字回读）
    assert "动脉顺应性" in text and "特征点知识解读" in text
    assert "不确定性" in text
    # 近窗仅一条历史段且注意力归一到 1.0 → 必须标注为病态值
    deg = ml3_mod.format_explain({
        "direct_answer": "x",
        "history": {"segments": [{"seg_idx": 0, "attention_weight": 1.0}]}})
    assert "病态值" in deg


def test_chat_message_explain_dispatch(monkeypatch):
    from web.api.routers import chat as chat_mod
    from web.api.routers.chat import ChatRequest, chat_message
    import ppg_agent.engine.explain as explain_mod

    fake = FakeLLM(plan_ai(needs_dispatch=True, task_queue=[
        {"target_agent": "ml_prediction", "task": "explain_prediction", "params": {}}]))
    monkeypatch.setattr(chat_mod, "get_llm", lambda key: fake)
    monkeypatch.setattr(chat_mod, "_resolve_params", lambda source: ("default", {}))
    monkeypatch.setattr(chat_mod, "_resolve_segment",
                        lambda seg_id: {"source": "vital", "subject_id": "p1", "seg_idx": 0})
    import ppg_agent.agents.ml_prediction_agent as ml3_mod
    monkeypatch.setattr(ml3_mod, "predict_segment",
                        lambda seg: ({"predictions": {"sbp": 120, "dbp": 80, "age": 50}}, None))

    class FakeExplainer:
        def explain_segment(self, *a, **k):
            # 画布历史 Tab 复用本次 Agent 解释载荷：必须带逐拍归因点位
            return {"direct_answer": "解释叙事", "current_segment": {"top_keypoints": []},
                    "history": {"segments": [{"seg_idx": 3, "attention_weight": 0.31,
                                             "beat_points": [[{"x": 1, "ig": 0.2}]],
                                             "ig_total": 0.2, "n_beats_used": 8}]}}

    monkeypatch.setattr(explain_mod, "get_live_explainer", lambda: FakeExplainer())
    monkeypatch.setattr(cs, "synthesize_reply", lambda *a, **k: "综合解释回复")

    r = chat_message(ChatRequest(message="为什么出这个值", source="vital",
                                 subject_id="p1", seg_id=7, target="sbp"))
    assert r["explain"]["direct_answer"] == "解释叙事"
    assert r["explain"]["target"] == "sbp"
    # L2.5 历史段与 L3 叙事同源：前端不再直连 live-attrib-history 也能渲染历史 Tab
    hist0 = r["explain"]["history"]["segments"][0]
    assert hist0["beat_points"] and hist0["n_beats_used"] == 8
    assert any(e["stage"] == "explain_prediction" for e in r["agent_execution"])
