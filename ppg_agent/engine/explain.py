"""explain.py — L3 可解释性：三层层级证据 × Agent 2 特征知识 → "为什么出这个值"叙事。

把已建好的三层证据融合成一段可读的解释叙事（不引入新模型/新依赖）：
- **L1** 近窗注意力：当前段本次估计参考了哪几条历史段（权重）；
- **L2** 当前段特征点级 IG：模型最依赖当前段波形的哪些生理关键点、
  各自把 target 往上/下拉；
- **L2.5**（L1×L2）历史段特征点级 IG：被参考的历史段里又更读哪些关键点。

每条特征点证据都下钻到 ``knowledge/feature_rules.json`` 的
``waveform_keypoints`` 词汇（生理含义 + 相关 Agent 2 标量特征 + 方向性
提示语），把"模型敏感度"翻译成"特征点语言"。所有方向性表述均取自
文献规则层（该层未通过留出复现，见 evidence_basis），只做**提示性**
一致性说明，绝不表述为个体生理病因或诊断。

纪律：输出是群体模型对输入的依赖度 + 文献方向性提示的合成叙事，非个
体生理因果、非诊断。特征点级证据需 GPU+encoder（复用 live_stream 的
L2/L2.5 归因）；不可用时上层抛 LiveStreamUnavailable → 端点降级 503。
"""
from __future__ import annotations

from .live_stream import LiveStreamUnavailable, get_live_stream_engine

_TARGET_UNIT = {"sbp": "mmHg", "dbp": "mmHg", "age": "岁"}
TARGET_LABEL = {"sbp": "收缩压", "dbp": "舒张压", "age": "年龄"}


def _direction_phrase(kp_up: dict, target: str, contrib: float) -> tuple[str, str]:
    """把某特征点的签名贡献映成 (方向, 提示性方向语)。

    正贡献 → 模型在该点把 target 往上推，取文献方向语；负贡献 →
    往下拉，措辞为"反向"。两者都显式标注为提示性、非个体病因。"""
    base = kp_up.get(target, "")
    if contrib >= 0:
        return "up", (base or "模型在该特征点上把该 target 往上推（提示性）")
    return "down", (f"反向：模型在该特征点上把 {TARGET_LABEL.get(target, target)} "
                    f"往低推" + (f"（与“{base}”相反的方向，提示性）" if base else ""))


class LiveExplainer:
    """L3 叙事合成器（只读；依赖 live_stream 的三层归因 + knowledge 规则层）。

    ``live`` 注入点供测试使用（duck-typed：需实现 predict_segment_stream /
    segment_waveform_attribution / segment_history_attribution / measured_bp）。
    生产路径用进程内 live_stream 单例。"""

    def __init__(self, live=None):
        self._live = live if live is not None else get_live_stream_engine()

    # ────────────────────────── 特征点 → 知识 ──────────────────────────
    def _annotate_keypoints(self, feature_points: list[dict], target: str,
                            rules, top: int = 4) -> list[dict]:
        """把归因输出的 feature_points（按重要度排序）逐条下钻到
        waveform_keypoints 词汇 + 相关 Agent 2 标量特征的方向语。取前 top 条。"""
        out = []
        for fp in (feature_points or [])[:top]:
            name = fp.get("name")
            contrib = float(fp.get("contrib", 0.0))
            kp = rules.waveform_keypoint(name)
            direction, phrase = _direction_phrase(kp.up, target, contrib)
            related = []
            for feat in kp.related_features:
                fr = rules.get(feat)
                related.append({
                    "feature": feat,
                    "short_name": fr.short_name or fr.label or feat,
                    "label": fr.label,
                    "supports_elevated_sbp_when": fr.supports_elevated_sbp_when,
                })
            out.append({
                "name": name,
                "label": kp.label or fp.get("label", name),
                "contrib": round(contrib, 3),
                "importance": round(abs(contrib), 3),
                "direction": direction,
                "physiology": kp.physiology,
                "phrase": phrase,
                "related_features": related,
            })
        return out

    # ────────────────────────── 主入口 ──────────────────────────
    def explain_segment(self, source: str, subject_id: str, seg_idx: int,
                        target: str, *, top_history: int = 3,
                        steps_current: int = 32, steps_history: int = 24,
                        predict: dict | None = None,
                        current: dict | None = None,
                        history: dict | None = None) -> dict:
        """融合 L1/L2/L2.5 证据 × Agent 2 知识 → "为什么出这个值"叙事。

        可注入已算好的 ``predict``/``current``/``history`` 复用（前端已展开
        某层时无需重算）；缺省时按需调用 live_stream 三层链（需 GPU+encoder，
        不可用抛 LiveStreamUnavailable）。"""
        from ..knowledge import RuleEngine

        live = self._live
        rules = RuleEngine()
        seg_idx = int(seg_idx)

        if predict is None:
            predict = live.predict_segment_stream(source, subject_id, seg_idx)
        pred_val = (predict.get("predictions") or {}).get(target)
        conf = predict.get("confidence") or {}
        mae = conf.get(f"{target}_mae")
        hist_src = predict.get("history_source")

        if current is None:
            current = live.segment_waveform_attribution(
                source, subject_id, seg_idx, target, steps=steps_current)
        cur_fps = self._annotate_keypoints(
            current.get("feature_points"), target, rules, top=4)

        want_history = bool(predict.get("history_attention", {}).get(target))
        if history is None and want_history:
            try:
                history = live.segment_history_attribution(
                    source, subject_id, seg_idx, target,
                    top=top_history, steps=steps_history)
            except LiveStreamUnavailable:
                history = None
        hist_segs = []
        for s in (history or {}).get("segments", []) or []:
            hist_segs.append({
                "seg_idx": s.get("seg_idx"),
                "attention_weight": s.get("attention_weight"),
                "net_contrib": s.get("ig_total"),
                # 逐拍归因透传：前端统一画布的“展开历史段”Tab 直接复用本次
                # Agent 解释的载荷渲染（L2.5 不再单独直连 GPU 端点）。
                "beat_points": s.get("beat_points"),
                "ig_total": s.get("ig_total"),
                "n_beats_used": s.get("n_beats_used"),
                "top_keypoints": self._annotate_keypoints(
                    s.get("feature_points"), target, rules, top=3),
            })

        unit = _TARGET_UNIT.get(target, "")
        measured = None
        try:
            measured = live.measured_bp(source, subject_id, seg_idx)
        except Exception:  # noqa: BLE001 — 无标签不影响叙事主体
            measured = None
        meas_val = (measured or {}).get(target) if isinstance(measured, dict) else None

        # ── direct_answer：一句话回答"为什么出这个值" ──
        drivers = [f"{fp['label']}（{'+拉高' if fp['direction'] == 'up' else '↓拉低'}）"
                   for fp in cur_fps[:3]]
        parts = [f"模型对该{TARGET_LABEL.get(target, target)}的段级估计为 "
                 f"{pred_val}{(' ' + unit) if unit else ''}"]
        if drivers:
            parts.append(f"；从可解释性看主要由当前段波形的 {('、'.join(drivers))} 驱动")
        if hist_segs:
            h0 = hist_segs[0]
            attn = h0.get('attention_weight')
            # 近窗仅一条前序段时 softmax 必然归一到 1.0：如实标注病态性，
            # 避免叙事把"注意力 1.0"误导为"模型高度依赖该段"。
            solo = (len(hist_segs) == 1 and isinstance(attn, (int, float)) and attn >= 0.99)
            parts.append(f"，并参考了历史段 #{h0.get('seg_idx')}"
                         f"（近窗注意力 {attn}"
                         f"{'，近窗仅此一条，归一化病态值非高聚焦' if solo else ''}）")
        direct_answer = "".join(parts) + "。"

        # ── knowledge_interpretation：把 top 特征点方向语汇成一致性说明 ──
        statements = []
        for fp in cur_fps[:3]:
            if fp["physiology"] or fp["phrase"]:
                statements.append(
                    f"当前段 · {fp['label']}：{fp['physiology']}；"
                    f"模型敏感度上 {fp['phrase']}。")
        for s in hist_segs[:2]:
            for fp in s["top_keypoints"][:2]:
                statements.append(
                    f"历史段 #{s['seg_idx']}（注意力 {s['attention_weight']}）· "
                    f"{fp['label']}：{fp['physiology']}；{fp['phrase']}。")
        knowledge_interpretation = {
            "statements": statements,
            "caveat": rules.keypoint_note or (
                "特征点级贡献是群体模型输入依赖度，非个体生理病因；"
                "方向性提示来自未通过留出复现的文献规则层，仅作提示不作证据。"),
        }

        # ── uncertainty：如实暴露近似/退化 ──
        uncertainty = []
        span = abs(float(current.get("pred_actual", 0.0))
                   - float(current.get("pred_baseline", 0.0))) or 1.0
        gap = abs(float(current.get("completeness_gap", 0.0)))
        if gap > 0.15 * span:
            uncertainty.append(
                f"当前段 IG 完备性 gap={current.get('completeness_gap')} 相对净变化"
                f"偏大，特征点划分为近似（最近关键点归并）。")
        if current.get("windowed"):
            uncertainty.append(
                f"当前段拍数超过 {current.get('n_beats_used')}/"
                f"{current.get('n_beats_total')} 上限，取首窗，"
                f"归因仅覆盖被编码的前 K 拍。")
        nok, nuse = current.get("beats_with_keypoints"), current.get("n_beats_used")
        if isinstance(nok, int) and isinstance(nuse, int) and nuse and nok < nuse:
            uncertainty.append(
                f"{nuse} 拍中 {nuse - nok} 拍关键点标定失败或过短，"
                f"退化为 (o,s,v) 三分点归属。")
        if hist_src == "live":
            uncertainty.append(
                "受试者不在离线嵌入表，历史为实时补算（近窗全算 + 前缀近似），"
                "注意力/历史归因随窗口略有抖动。")
        elif hist_src == "none":
            uncertainty.append("首段/无可用历史：仅当前段证据（L1/L2.5 缺席）。")
        if not uncertainty:
            uncertainty.append("近窗注意力 + 当前段 IG 完备性均在阈值内，无额外退化提示。")

        layers = {
            "L1": "近窗注意力：本次估计参考了哪些历史段（权重）。",
            "L2": "当前段特征点级 IG：模型最读当前段波形的哪些生理关键点、往哪个方向拉。",
            "L2.5": "历史段特征点级 IG：被参考的历史段里又更读哪些关键点。",
            "L3": "把上述三层证据 × Agent 2 特征知识（waveform_keypoints + 标量规则）"
                  "融合成'为什么出这个值'的提示性叙事。",
        }

        return {
            "source": source, "subject_id": subject_id, "seg_idx": seg_idx,
            "target": target, "target_label": TARGET_LABEL.get(target, target),
            "unit": unit,
            "prediction": pred_val,
            "measured": meas_val,
            "confidence": {"kind": conf.get("kind"), f"{target}_mae": mae},
            "engine": predict.get("engine"),
            "history_source": hist_src,
            "direct_answer": direct_answer,
            "current_segment": {
                "ig_total": current.get("ig_total"),
                "pred_actual": current.get("pred_actual"),
                "pred_baseline": current.get("pred_baseline"),
                "completeness_gap": current.get("completeness_gap"),
                "n_beats_used": nuse, "n_beats_total": current.get("n_beats_total"),
                "windowed": current.get("windowed"),
                "beats_with_keypoints": nok,
                "top_keypoints": cur_fps,
            },
            "history": {"segments": hist_segs,
                        "note": (history or {}).get("note")},
            "knowledge_interpretation": knowledge_interpretation,
            "uncertainty": uncertainty,
            "safety_note": rules.safety_note or (
                "This explanation describes population-model input dependency and "
                "literature-derived directional hints. It is not a medical diagnosis."),
            "alternative_explanations": rules.alternative_explanations,
            "layers": layers,
            "note": ("L3 合成叙事：把'参考哪些历史段（L1）× 当前段/历史段的哪些生理"
                     "特征点被模型 leaned on 且往哪拉（L2/L2.5）'翻译到特征点语言。"
                     "属群体模型依赖度 + 提示性方向语，非个体生理病因、非诊断。"),
        }


_EXPLAINER: "LiveExplainer | None" = None


def get_live_explainer() -> "LiveExplainer":
    """进程内单例（线程安全复用 live_stream 单例；缺 GPU 时由调用方降级）。"""
    global _EXPLAINER
    if _EXPLAINER is None:
        _EXPLAINER = LiveExplainer()
    return _EXPLAINER
