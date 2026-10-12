"""Agent 3 — ML Prediction (BeatMAE risk / attribution).

Symmetry note: Agent 1 (signal_processing) and Agent 2 (cardiovascular) are
LangGraph state machines driven by an LLM tool loop, created via
``create_signal_processor_agent`` / ``create_cardiovascular_agent``.  Agent 3
is a **deterministic inference engine** — BeatMAE encoder + causal streaming
head + IG attribution — with NO LLM and NO graph loop; all natural-language
wording happens in the Supervisor's synthesis phase.  This module gives the
third agent a matching entry point (``create_ml_prediction_agent``) without
pretending Agent 3 runs an agent loop: the facade just wraps the deterministic
predict / explain / fixed-format-narrative helpers that the web router used to
inline.

Dispatch (unchanged from the pre-refactor chat router):
  predict_segment — 段级实时流式主口径（波形→读窗→拍检测→编码→流式头出值）；
                    失败降级 = 流式头查表段级（同头同嵌入，口径一致）。
  explain_segment — 对当前段预测做三层证据叙事（L1 注意力 × L2 当前段特征点 ×
                    L2.5 历史段特征点），复用 predict_segment 免重算。
  format_prediction / format_explain — 确定性固定格式叙事（不进 LLM）。
"""

from __future__ import annotations

from typing import Any


class _Agent3EngShim:
    """查表降级路径的 measured_bp 适配（CausalStreamEngine 无该方法）。"""

    @staticmethod
    def measured_bp(source: str, subject_id: str, seg_idx: int) -> dict:
        from ppg_agent.engine.ml_predict import measured_bp
        return measured_bp(source, subject_id, seg_idx)


def predict_segment(seg: dict) -> tuple[dict, Any]:
    """Agent 3 主口径 = 段级实时流式（波形→读窗→拍检测→编码→流式头出值）；
    失败降级 = 流式头查表段级（同头同嵌入，口径一致）。返回 (pred, eng)。"""
    live_err = None
    try:
        from ppg_agent.engine.live_stream import get_live_stream_engine
        eng = get_live_stream_engine()
        return (eng.predict_segment_stream(seg["source"], seg["subject_id"],
                                           seg["seg_idx"]), eng)
    except Exception as e:  # noqa: BLE001 — torch/ckpt/波形缺失 → 降级
        live_err = str(e)[:80]
    from ppg_agent.engine.causal_stream import get_causal_stream_engine
    eng = get_causal_stream_engine()
    pred = eng.predict_segment_lookup(seg["source"], seg["subject_id"],
                                      seg["seg_idx"])
    pred["degraded_from_live"] = live_err
    return pred, _Agent3EngShim()


def explain_segment(seg: dict, target: str) -> dict:
    """Agent 3 可解释：对当前段预测做三层证据叙事（L1 注意力 × L2 当前段
    特征点 × L2.5 历史段特征点），复用 predict_segment 的 predict 免重算。

    特征点级证据需 GPU+encoder；不可用（或预测本身失败）时返回
    ``{"unavailable": reason}`` 而非抛出，保证对话轮次不崩。载荷为三层证据
    形状（附 ``target``），经 /chat/message 的 explain 字段返回，前端直接写入
    右侧 expl 态。输出为群体模型依赖度 + 文献方向提示，非个体病因、非诊断。"""
    try:
        pred, _eng = predict_segment(seg)
    except Exception as e:  # noqa: BLE001 — 预测都失败则解释无从谈起
        return {"unavailable": f"预测不可用: {str(e)[:80]}", "target": target}
    try:
        from ppg_agent.engine.explain import get_live_explainer
        expl = get_live_explainer().explain_segment(
            seg["source"], seg["subject_id"], seg["seg_idx"], target,
            top_history=3, predict=pred)
        expl["target"] = target
        return expl
    except Exception as e:  # noqa: BLE001 — LiveStreamUnavailable 等 → 降级
        return {"unavailable": str(e)[:120], "target": target}


def format_explain(ex: dict) -> str:
    """L3 解释叙事的确定性概要（供 Supervisor 综合成 ≤ 4 句回复，不进前端渲染）。

    必须带上 Agent 2 知识层（特征点生理含义×方向提示）——否则回复
    退化成了数字回读，失去"结合特征点特性分析"的解释意义。"""
    lines = [ex.get("direct_answer", "")]
    kps = (ex.get("current_segment") or {}).get("top_keypoints") or []
    if kps:
        lines.append("当前段主要特征点: " + "、".join(
            f"{k.get('label')}({k.get('contrib'):+})" for k in kps[:4]))
    # Agent 2 知识层：生理含义 + 模型敏感度方向语（提示性，非个体病因）
    stmts = (ex.get("knowledge_interpretation") or {}).get("statements") or []
    if stmts:
        lines.append("特征点知识解读(Agent 2，提示性): " + "；".join(
            s[:140] for s in stmts[:4]))
    hist = (ex.get("history") or {}).get("segments") or []
    if hist:
        note = ""
        # 近窗仅一条历史段时 softmax 必然归一到 1.0：是病态值，非"高度聚焦"
        if len(hist) == 1 and float(hist[0].get("attention_weight") or 0) >= 0.99:
            note = "（近窗仅一条前序段，注意力 1.0 为归一化病态值，不代表高聚焦）"
        lines.append("参考历史段: " + "、".join(
            f"#{h.get('seg_idx')}(注意力 {h.get('attention_weight')})" for h in hist[:3]) + note)
    unc = ex.get("uncertainty") or []
    if unc:
        lines.append("不确定性: " + " ".join(str(u)[:100] for u in unc[:2]))
    return "\n".join(x for x in lines if x)


def format_prediction(seg: dict, eng, pred: dict) -> str:
    """Agent 3 预测的确定性叙事（固定格式，不进 LLM）。"""
    p = pred["predictions"]
    lines = [f"Agent 3 (BeatMAE 模型预测，{pred['encoder_version']}）："]
    if pred["segment_level"]:
        lines.append(f"- 当前段（seg_idx={seg['seg_idx']}）预测血压："
                     f"SBP ≈ {p['sbp']} mmHg，DBP ≈ {p['dbp']} mmHg")
        if pred.get("history_note"):
            lines.append(f"- 历史上下文：{pred['history_note']}")
        elif pred.get("n_history") is not None:
            lines.append(f"- 历史上下文：近 {pred['n_history']} 段前序嵌入"
                         f"（来源：{pred.get('history_source', 'table')}）")
        if pred.get("degraded_from_live"):
            lines.append(f"- 注意：实时链不可用，本次为查表降级"
                         f"（{pred['degraded_from_live']}）")
    else:
        lines.append(f"- 预测血压（受试者级，{pred['n_segments']} 段嵌入均值池化）："
                     f"SBP ≈ {p['sbp']} mmHg，DBP ≈ {p['dbp']} mmHg")
        try:
            seg_pred = eng.predict_segment(seg["source"], seg["subject_id"],
                                           seg["seg_idx"])
            sp = seg_pred["predictions"]
            lines.append(f"- 当前段（seg_idx={seg['seg_idx']}）预测："
                         f"SBP ≈ {sp['sbp']} / DBP ≈ {sp['dbp']} mmHg")
        except Exception:  # noqa: BLE001 — 段级查表失败不影响主输出
            pass
    measured = eng.measured_bp(seg["source"], seg["subject_id"], seg["seg_idx"])
    if measured:
        diff_s = (round(p["sbp"] - measured["sbp"], 1)
                  if "sbp" in measured and not pred["segment_level"] else None)
        m = "、".join(f"{k.upper()} {v}" for k, v in measured.items())
        extra = f"（当前段预测与实测差 {diff_s:+} mmHg）" if diff_s is not None else ""
        lines.append(f"- 该段实测标签（袖带/ABP）：{m} {extra}")
    lines.append(f"- 模型预测年龄：{p['age']} 岁")
    lines.append(f"- 置信度：{pred['confidence']['note']}")
    return "\n".join(lines)


class MlPredictionAgent:
    """Agent 3 facade — a deterministic predictor/attributer, not a graph.

    Stateless: the engines (live stream / causal stream / explainer) are
    resolved lazily inside each method, so importing this module (or building
    the agent) pulls in no torch / checkpoint / GPU dependencies.  The web
    router calls ``predict`` / ``explain`` and renders the fixed-format
    narratives via ``format_prediction`` / ``format_explain``; the Supervisor
    does all prose.
    """

    def predict(self, seg: dict) -> tuple[dict, Any]:
        return predict_segment(seg)

    def explain(self, seg: dict, target: str) -> dict:
        return explain_segment(seg, target)

    def format_prediction(self, seg: dict, eng: Any, pred: dict) -> str:
        return format_prediction(seg, eng, pred)

    def format_explain(self, ex: dict) -> str:
        return format_explain(ex)


_AGENT3_SINGLETON: MlPredictionAgent | None = None


def create_ml_prediction_agent() -> MlPredictionAgent:
    """Create the Agent 3 facade (a cached, stateless singleton).

    Mirrors ``create_signal_processor_agent`` / ``create_cardiovascular_agent``
    in name so the three agents are addressed symmetrically; unlike those two
    it returns a plain deterministic-engine facade, not a compiled LangGraph.
    """
    global _AGENT3_SINGLETON
    if _AGENT3_SINGLETON is None:
        _AGENT3_SINGLETON = MlPredictionAgent()
    return _AGENT3_SINGLETON
