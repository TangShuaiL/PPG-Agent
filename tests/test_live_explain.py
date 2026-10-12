"""L3 解释叙事单测（无需 GPU：用 duck-typed 假 live 引擎喂三层合成证据）。

覆盖：
- 特征点 → waveform_keypoints 知识映射（9 点全覆盖 + 未知点降级不崩）；
- 叙事结构字段齐备（direct_answer/current_segment/history/
  knowledge_interpretation/uncertainty/safety_note）；
- 安全纪律：群体/非诊断措辞存在，禁诊断词不出现；
- 复用注入：传入已算好的 current/history/predict 不再回调 live 链。
"""
from __future__ import annotations

import pytest

from ppg_agent.engine.explain import LiveExplainer
from ppg_agent.knowledge import RuleEngine

ALL_KPS = ["o", "a", "md", "b", "s", "dn", "ip", "d", "v"]


def _fps(names_and_contrib):
    return [{"name": n, "label": n, "contrib": c, "importance": abs(c)}
            for n, c in names_and_contrib]


class FakeLive:
    """duck-typed live_stream：只实现 explain_segment 会调到的四个方法。"""

    def __init__(self):
        self.calls: list[str] = []

    def predict_segment_stream(self, source, subject_id, seg_idx):
        self.calls.append("predict")
        return {
            "predictions": {"sbp": 125.1, "dbp": 78.0, "age": 55.0},
            "engine": "live_stream",
            "history_source": "table",
            "confidence": {"kind": "population_test_mae_seg",
                           "sbp_mae": 8.3, "dbp_mae": 5.1, "age_mae": 6.2},
            "history_attention": {"sbp": [{"seg_idx": 1, "weight": 0.91}]},
        }

    def segment_waveform_attribution(self, source, subject_id, seg_idx,
                                     target, *, steps=32):
        self.calls.append("current")
        return {
            "feature_points": _fps([("a", 6.1), ("o", 3.2), ("md", -2.0),
                                    ("dn", 1.1), ("s", 0.5)]),
            "ig_total": 9.8, "pred_actual": 125.1, "pred_baseline": 113.5,
            "completeness_gap": 1.8, "n_beats_used": 12, "n_beats_total": 12,
            "windowed": False, "beats_with_keypoints": 11,
        }

    def segment_history_attribution(self, source, subject_id, seg_idx,
                                    target, *, top=3, steps=24):
        self.calls.append("history")
        return {
            "segments": [
                {"seg_idx": 1, "attention_weight": 0.911, "ig_total": 9.79,
                 "feature_points": _fps([("a", 4.0), ("o", 2.1), ("v", 1.0)])},
            ],
            "note": "注意力加权二阶效应，非生理病因。",
        }

    def measured_bp(self, source, subject_id, seg_idx):
        return {"sbp": 122.0, "dbp": 76.0}


@pytest.fixture()
def explainer():
    return LiveExplainer(live=FakeLive())


def test_knowledge_bridge_covers_all_nine_keypoints():
    rules = RuleEngine()
    for name in ALL_KPS:
        kp = rules.waveform_keypoint(name)
        assert kp.label, f"{name} 应有非空 label"
        assert set(kp.up.keys()) >= {"sbp", "dbp", "age"}, f"{name} 缺方向语"
    # 未知点降级为 stub（label 兜底为名字），不抛异常
    unk = rules.waveform_keypoint("zzz")
    assert unk.label == "zzz"
    assert unk.up == {}


def test_narrative_structure_and_fields(explainer):
    out = explainer.explain_segment("mimic", "p000160", 8, "sbp")
    for key in ("direct_answer", "current_segment", "history",
                "knowledge_interpretation", "uncertainty", "safety_note",
                "layers"):
        assert key in out, f"缺字段 {key}"
    assert out["prediction"] == 125.1
    assert "125.1" in out["direct_answer"]
    assert out["current_segment"]["top_keypoints"], "当前段应有已标注特征点"
    assert out["knowledge_interpretation"]["statements"], "应有知识一致性语句"


def test_keypoint_annotation_resolves_to_knowledge(explainer):
    out = explainer.explain_segment("mimic", "p000160", 8, "sbp")
    kp0 = out["current_segment"]["top_keypoints"][0]
    assert kp0["name"] == "a"
    assert kp0["direction"] == "up"
    assert kp0["physiology"], "a 点应带生理含义"
    assert "提示" in kp0["phrase"] or "反射" in kp0["phrase"]
    # a → 相关 Agent 2 标量特征（RI/LASI）
    shorts = {r["short_name"] for r in kp0["related_features"]}
    assert shorts & {"RI", "LASI"}
    # md 负贡献 → direction down
    md = next(f for f in out["current_segment"]["top_keypoints"] if f["name"] == "md")
    assert md["direction"] == "down"


def test_history_segment_annotation(explainer):
    out = explainer.explain_segment("mimic", "p000160", 8, "sbp")
    segs = out["history"]["segments"]
    assert segs and segs[0]["seg_idx"] == 1
    assert segs[0]["top_keypoints"][0]["name"] == "a"


def test_safety_discipline_no_diagnostic_phrases(explainer):
    out = explainer.explain_segment("mimic", "p000160", 8, "sbp")
    blob = " ".join([out["direct_answer"], out["safety_note"],
                     out["knowledge_interpretation"]["caveat"],
                     " ".join(out["knowledge_interpretation"]["statements"]),
                     out["note"]]).lower()
    # 群体/非诊断框架措辞存在
    assert ("群体" in blob) or ("population" in blob)
    assert ("非诊断" in blob) or ("not a medical diagnosis" in blob)
    # 禁硬诊断断言词（英文诊断动词 / 中文确诊）
    for forbidden in ("diagnose you", "you have", "确诊", "诊断为"):
        assert forbidden not in blob


def test_reuse_injected_avoids_recompute():
    fake = FakeLive()
    ex = LiveExplainer(live=fake)
    predict = fake.predict_segment_stream("m", "s", 1)
    current = fake.segment_waveform_attribution("m", "s", 1, "sbp")
    history = fake.segment_history_attribution("m", "s", 1, "sbp")
    fake.calls.clear()
    out = ex.explain_segment("m", "s", 1, "sbp",
                             predict=predict, current=current, history=history)
    assert fake.calls == ["measured_bp"] or fake.calls == [] or "measured" in str(fake.calls)
    # 关键：不应再触发 predict/current/history 三条重算
    assert "predict" not in fake.calls
    assert "current" not in fake.calls
    assert "history" not in fake.calls
    assert out["prediction"] == 125.1


def test_first_segment_no_history_note():
    class NoHistFake(FakeLive):
        def predict_segment_stream(self, source, subject_id, seg_idx):
            d = super().predict_segment_stream(source, subject_id, seg_idx)
            d["history_source"] = "none"
            d["history_attention"] = {}
            return d

        def segment_history_attribution(self, *a, **k):
            raise AssertionError("首段不应触发历史归因")

    out = LiveExplainer(live=NoHistFake()).explain_segment("vital", "p1", 0, "sbp")
    assert out["history"]["segments"] == []
    assert any("首段" in u or "无可用历史" in u for u in out["uncertainty"])
