"""Tests for the unified dataset-adaptation engine (adapt_pipeline).

The ONE session engine lives in ppg_agent/agents/adaptation_session.py and is
shared by the Agent-1 graph's adapt branch and the web chat's deterministic
layer (quick actions / 选N).  All DB/LLM/pipeline calls are faked — no
network, no real .mat data, no phenotype DB.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from langchain_core.messages import AIMessage

from ppg_agent.agents import adaptation_session as A
from ppg_agent.context import reset_context


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeProposeLLM:
    """bind_tools → self; invoke → a tool call proposing two param changes."""

    def bind_tools(self, tools, **kw):
        return self

    def invoke(self, messages, **kw):
        return AIMessage(content="", tool_calls=[{
            "name": "AdaptProposeOptions",
            "args": {"summary": "s",
                     "options": [
                         {"param": "peak_distance_ms", "value": 450,
                          "rationale": "r", "expected_effect": "e",
                          "user_desc": "分割更齐"},
                         {"param": "onset_search_ms", "value": 160,
                          "rationale": "r2", "expected_effect": "e2",
                          "user_desc": "o点更准"},
                     ]},
            "id": "c1"}])


@pytest.fixture(autouse=True)
def fake_env(monkeypatch, tmp_path):
    """Isolated sessions + faked data pipeline for every test."""
    reset_context()
    A._SESSIONS.clear()
    A._ORDER.clear()
    (tmp_path / "raw" / "vital").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "aurora").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(A, "raw_data_dir", lambda source: tmp_path / "raw" / source)
    monkeypatch.setattr(A, "_subject_pool", lambda source: ["p1", "p2", "p3"])
    monkeypatch.setattr(A, "load_sample_segment",
                        lambda source, subj, seg_idx: {"ppg_500": np.ones(5000)})
    monkeypatch.setattr(A, "process_ppg_segment",
                        lambda ppg, fs, **kw: {"beats": [], "beat_starts": [],
                                               "valid_mask": [], "keypoints": [],
                                               "n_total": 5, "n_valid": 3, "quality": 0.8})
    monkeypatch.setattr(A, "segment_metrics",
                        lambda result: {"n_valid": 3, "n_total": 5,
                                        "quality": 0.8, "tier": "good"})
    monkeypatch.setattr(A, "diagnose", lambda result: {"issues": []})
    monkeypatch.setattr(A, "_build_waveform_payload",
                        lambda ppg, result: {"fs": 500, "n_samples": 5000,
                                             "duration_sec": 10.0,
                                             "points": [0.0, 0.5, 1.0],
                                             "overlay": {"fs": 500, "beats": [
                                                 {"beat_idx": 0, "start": 100,
                                                  "end": 600, "valid": True,
                                                  "keypoints": {"o": 100}}]}})
    monkeypatch.setattr(A, "get_llm", lambda key: FakeProposeLLM())
    yield
    A._SESSIONS.clear()
    A._ORDER.clear()


# ---------------------------------------------------------------------------
# adapt_pipeline semantic actions
# ---------------------------------------------------------------------------

def test_start_proposes_options_and_is_idempotent():
    res = A.adapt_pipeline("start", source="vital")
    assert res["status"] == "active"
    assert res["state"]["source"] == "vital"
    assert len(res["state"]["pending_options"]) == 2
    assert "选项1" in res["message"] and "采样概况" in res["message"]

    # unified ECharts payload replaces the PNG: waveform points + overlay
    sample = res["state"]["rounds"][0]["samples"][0]
    assert sample["waveform"]["fs"] == 500
    assert sample["waveform"]["points"] == [0.0, 0.5, 1.0]
    assert sample["waveform"]["overlay"]["beats"][0]["end"] == 600  # start+len

    # idempotent: a second start must NOT reset the running session
    res2 = A.adapt_pipeline("start", source="vital")
    assert res2["status"] == "active"
    assert len(res2["state"]["rounds"]) == 1
    assert "已在运行" in res2["message"]


def test_start_errors():
    assert A.adapt_pipeline("start", source="").get("status") == "error"
    res = A.adapt_pipeline("start", source="no_such_dataset")
    assert res.get("status") == "error"
    assert "原始数据目录不存在" in res.get("message", "")


def test_apply_choice_changes_params_and_resamples():
    A.adapt_pipeline("start", source="vital")

    # invalid index and missing idx are errors, never edits
    res = A.adapt_pipeline("apply", source="vital", choice_idx=9)
    assert "不合法" in res["message"]
    res = A.adapt_pipeline("apply", source="vital", choice_idx=None)
    assert res["status"] == "error"

    res = A.adapt_pipeline("apply", source="vital", choice_idx=2)
    assert res["status"] == "active"
    assert res["state"]["params"]["onset_search_ms"] == 160
    assert res["state"]["rounds"][1]["change"]["param"] == "onset_search_ms"
    assert "已应用选项 2" in res["message"]

    # options were consumed → further applies are refused, never stale edits
    res = A.adapt_pipeline("apply", source="vital", choice_idx=1)
    assert "没有待选选项" in res["message"]
    assert res["state"]["params"]["onset_search_ms"] == 160


def test_sample_more_adds_round_without_changing_params():
    A.adapt_pipeline("start", source="vital")
    before = A.active_adapt_session("vital").state_dict()["params"]
    res = A.adapt_pipeline("sample", source="vital")
    assert len(res["state"]["rounds"]) == 2
    assert res["state"]["params"] == before


def test_propose_with_feedback_never_edits_params():
    A.adapt_pipeline("start", source="vital")
    before = dict(A.active_adapt_session("vital").state_dict()["params"])
    res = A.adapt_pipeline("propose", source="vital", feedback="beat 太碎了")
    assert len(res["state"]["pending_options"]) == 2
    assert res["state"]["params"] == before


def test_save_hygiene_and_status_transition(monkeypatch):
    monkeypatch.setattr(A, "save_preset",
                        lambda name, params: {"path": f"/tmp/{name}.yaml"})
    A.adapt_pipeline("start", source="vital")
    res = A.adapt_pipeline("save", source="vital")
    assert res["status"] == "saved"
    assert "user_vital" in res["message"]

    # after save the session is no longer active → further actions are idle
    assert A.active_adapt_session("vital") is None
    assert A.adapt_pipeline("apply", source="vital", choice_idx=1).get("status") == "idle"
    assert A.adapt_pipeline("save", source="vital").get("status") == "idle"


def test_cancel_aborts_and_removes_preset(monkeypatch):
    monkeypatch.setattr(A, "save_preset",
                        lambda name, params: {"path": f"/tmp/{name}.yaml"})
    fp = A.MODELS_ROOT / "param_presets" / "vital" / "user.yaml"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text("params: {}")
    # saved before → re-adapt (new active session) → cancel with cleanup
    A.adapt_pipeline("start", source="vital")
    A.adapt_pipeline("save", source="vital")
    A.adapt_pipeline("start", source="vital")
    res = A.adapt_pipeline("cancel", source="vital", remove_preset=True)
    assert res["status"] == "aborted"
    assert res["state"]["status"] == "aborted"
    assert not fp.exists()
    # cancel on a saved/aborted session is refused (idle), never double-cancel
    assert A.adapt_pipeline("cancel", source="vital").get("status") == "idle"


def test_status_before_and_after():
    assert A.adapt_pipeline("status", source="vital").get("status") == "idle"
    A.adapt_pipeline("start", source="vital")
    res = A.adapt_pipeline("status", source="vital")
    assert res["status"] == "active"
    assert res["state"]["rounds"]


def test_unknown_action():
    assert A.adapt_pipeline("foo", source="vital").get("status") == "idle"
    A.adapt_pipeline("start", source="vital")
    assert A.adapt_pipeline("foo", source="vital").get("status") == "error"


def test_rule_fallback_when_llm_fails(monkeypatch):
    def boom(key):
        raise RuntimeError("llm down")
    monkeypatch.setattr(A, "get_llm", boom)
    A.adapt_pipeline("start", source="vital")
    res = A.adapt_pipeline("propose", source="vital", feedback="beat 太多太碎了")
    opts = res["state"]["pending_options"]
    assert opts and opts[0]["param"] == "peak_distance_ms"
    assert opts[0]["value"] == 350  # default 300 + 50


# ---------------------------------------------------------------------------
# Session lookup semantics (strict per-source, MRU)
# ---------------------------------------------------------------------------

def test_session_lookup_strict_source():
    A.adapt_pipeline("start", source="vital")
    A.adapt_pipeline("start", source="aurora")
    # both sources keep independent active sessions
    assert A.active_adapt_session("vital").source == "vital"
    assert A.active_adapt_session("aurora").source == "aurora"
    # MRU lookup returns the last touched
    assert A.active_adapt_session().source == "aurora"

    A.adapt_pipeline("save", source="vital")
    assert A.active_adapt_session("vital") is None          # no foreign fallback
    assert A.active_adapt_session("aurora") is not None
    assert A.get_adapt_session("vital") is not None         # saved still surfacable


# ---------------------------------------------------------------------------
# Web chat deterministic layer (quick actions / 选N) — same engine
# ---------------------------------------------------------------------------

def test_force_pick_applies_option(monkeypatch):
    from web.api.routers import chat as chat_mod
    A.adapt_pipeline("start", source="vital")
    note = chat_mod._force_pick("选 2")
    assert note is not None and "选项 2" in note
    assert A.active_adapt_session("vital").state_dict()["params"]["onset_search_ms"] == 160


def test_force_pick_requires_active_session():
    from web.api.routers import chat as chat_mod
    assert chat_mod._force_pick("选 1") is None
    assert chat_mod._force_pick("随便聊聊") is None


def test_force_quick_actions(monkeypatch):
    from web.api.routers import chat as chat_mod
    monkeypatch.setattr(A, "save_preset",
                        lambda name, params: {"path": f"/tmp/{name}.yaml"})
    # no session → no-op
    assert chat_mod._force_quick_action("可以了，保存吧") is None
    A.adapt_pipeline("start", source="vital")
    assert "保存" in chat_mod._force_quick_action("可以了，保存吧")
    assert A.active_adapt_session("vital") is None            # saved
    # sample on the aborted/saved session → no-op
    assert chat_mod._force_quick_action("再看几个 segment") is None


# ---------------------------------------------------------------------------
# _run_adapt: graph adapt branch + deterministic start fallback
# ---------------------------------------------------------------------------

class FakeAdaptRunner:
    def __init__(self, text="适配中"):
        self.text = text
        self.calls = []

    def __call__(self, source, msgs):
        self.calls.append(source)
        return self.text, [{"agent": "signal_processing", "stage": "adapt",
                            "summary": "适配会话推进"}]


def test_run_adapt_graph_starts_session(monkeypatch):
    from web.api.routers.chat import _run_adapt, ChatRequest
    runner = FakeAdaptRunner()
    monkeypatch.setattr("web.api.services.chat_supervisor.run_agent1_adapt", runner)
    reply, adapt = _run_adapt(ChatRequest(message="帮我调下参数", source="vital"),
                              [], [])
    assert runner.calls == ["vital"]
    assert reply and "选项1" in reply          # deterministic start fallback message
    assert adapt is not None and adapt["status"] == "active"
    assert adapt["rounds"]


def test_run_adapt_no_source_no_session():
    from web.api.routers.chat import _run_adapt, ChatRequest
    reply, adapt = _run_adapt(ChatRequest(message="调参数", source=None), [], [])
    assert "数据集" in reply and adapt is None


def test_run_adapt_saved_session_not_restarted(monkeypatch):
    from web.api.routers.chat import _run_adapt, ChatRequest

    def runner(source, msgs):  # the graph saves the session itself
        A.adapt_pipeline("start", source=source)
        A.adapt_pipeline("save", source=source)
        return "已保存 user_vital", [{"agent": "signal_processing", "stage": "adapt",
                                      "summary": "保存"}]

    monkeypatch.setattr("web.api.services.chat_supervisor.run_agent1_adapt", runner)
    reply, adapt = _run_adapt(ChatRequest(message="适配好了，保存", source="vital"),
                              [], [])
    assert adapt["status"] == "saved"
    assert "选项" not in reply                # no auto-restart after save


def test_run_adapt_graph_failure_still_starts(monkeypatch):
    from web.api.routers.chat import _run_adapt, ChatRequest

    def boom(source, msgs):
        raise RuntimeError("graph broke")

    monkeypatch.setattr("web.api.services.chat_supervisor.run_agent1_adapt", boom)
    reply, adapt = _run_adapt(ChatRequest(message="帮我调下参数", source="vital"),
                              [], [])
    assert adapt is not None and adapt["status"] == "active"


# ---------------------------------------------------------------------------
# chat_message end-to-end
# ---------------------------------------------------------------------------

def test_chat_message_pick_n_deterministic(monkeypatch):
    from web.api.routers import chat as chat_mod
    from web.api.routers.chat import ChatRequest, chat_message

    class TextOnlyLLM:
        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, messages, **kw):
            return AIMessage(content="好的，已应用选项 2")

    A.adapt_pipeline("start", source="vital")
    monkeypatch.setattr(chat_mod, "get_llm", lambda key: TextOnlyLLM())
    r = chat_message(ChatRequest(message="选 2", source="vital"))
    assert r["reply"]
    assert r["adapt"]["status"] == "active"
    assert r["adapt"]["params"]["onset_search_ms"] == 160     # applied deterministically
    assert any(e["stage"] == "dispatch" for e in r["agent_execution"])


def test_chat_message_adapt_intent_runs_graph(monkeypatch):
    from web.api.routers import chat as chat_mod
    from web.api.routers.chat import ChatRequest, chat_message

    class PlanLLM:
        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, messages, **kw):
            return AIMessage(content="", tool_calls=[{
                "name": "SupervisorPlan",
                "args": {"thought": "适配", "needs_dispatch": True,
                         "task_queue": [{"target_agent": "signal_processing",
                                         "task": "dataset_adaptation",
                                         "params": {}}],
                         "direct_reply": None},
                "id": "p1"}])

    monkeypatch.setattr(chat_mod, "get_llm", lambda key: PlanLLM())
    monkeypatch.setattr("web.api.services.chat_supervisor.run_agent1_adapt",
                        lambda source, msgs: ("开始适配", [{"agent": "signal_processing",
                                                           "stage": "adapt"}]))
    r = chat_message(ChatRequest(message="帮我调下参数", source="vital"))
    assert r["reply"]
    assert r["adapt"]["status"] == "active" and r["adapt"]["rounds"]
    assert any(e["agent"] == "signal_processing" and e["stage"] == "adapt"
               for e in r["agent_execution"])


def test_chat_message_active_session_feedback_routes_to_adapt(monkeypatch):
    from web.api.routers import chat as chat_mod
    from web.api.routers.chat import ChatRequest, chat_message

    class BoomLLM:
        def bind_tools(self, tools, **kw):
            raise RuntimeError("down")

    # active session + plan extraction fails → feedback routes to adaptation
    # ("特征点没标好" must not fall through to segment analysis)
    A.adapt_pipeline("start", source="vital")
    monkeypatch.setattr(chat_mod, "get_llm", lambda key: BoomLLM())
    monkeypatch.setattr("web.api.services.chat_supervisor.run_agent1_adapt",
                        lambda source, msgs: ("重新提议", [{"agent": "signal_processing",
                                                           "stage": "adapt"}]))
    r = chat_message(ChatRequest(message="特征点没标好", source="vital"))
    assert r["adapt"] is not None
    assert any(e["agent"] == "signal_processing" and e["stage"] == "adapt"
               for e in r["agent_execution"])
