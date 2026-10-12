"""L1 近窗注意力归因单测（causal_stream，纯 numpy、无 torch 依赖）。

覆盖 collect_history_attention / predict_segment_lookup 的 history_attention：
- 因果：被关注历史段 seg_idx 必须 < 当前段；
- 范围：仅最近 K 段内的段有非零权（更早段被 c_recent 丢弃）；
- 归一：全窗口权重和 ≈ 1（softmax × fold/seed 平均保持）；
- 结构：按权重降序、非负、截断到 top；
- 降级：首段（无历史）不产出 history_attention。
"""
from __future__ import annotations

import pytest

from ppg_agent.engine.causal_stream import (CausalStreamEngine,
                                            get_causal_stream_engine)


@pytest.fixture(scope="module")
def engine() -> CausalStreamEngine:
    try:
        return get_causal_stream_engine()
    except FileNotFoundError:
        pytest.skip("固化头/嵌入产物缺失（需 build_causal_heads + build_ml_embeddings）")


def _pick_small_history_subject(engine: CausalStreamEngine):
    """返回 (source, subject_id, cur_seg_idx, min_seg_idx)：选一个段数适中
    (3..20) 的非 anker 受试者，用其最大 seg_idx 作当前段（有历史）。"""
    m = engine.meta
    m = m[m["source"] != "anker"]
    cnt = m.groupby(["source", "subject_id"])["seg_idx"].agg(["size", "min", "max"])
    cand = cnt[(cnt["size"] >= 3) & (cnt["size"] <= 20)]
    if cand.empty:
        pytest.skip("嵌入表无 3~20 段规模的受试者可用于轻量测试")
    (src, sid), row = next(iter(cand.iterrows()))
    return src, sid, int(row["max"]), int(row["min"])


def test_lookup_returns_causal_normalized_attention(engine):
    src, sid, cur, _min = _pick_small_history_subject(engine)
    out = engine.predict_segment_lookup(src, sid, cur)
    assert out["n_history"] >= 2, "该用例需要至少 2 段历史"
    att = out.get("history_attention")
    assert att, "有历史时应产出 history_attention"
    assert set(att) == set(engine.targets), "每个 target 都应有注意力"

    for t, pairs in att.items():
        assert pairs, f"{t}: 非空"
        # 因果：全部严格早于当前段
        assert all(p["seg_idx"] < cur for p in pairs), f"{t}: 注意力越界到未来/自身"
        # 非负 + 降序 + 截断
        ws = [p["weight"] for p in pairs]
        assert all(w >= 0 for w in ws), f"{t}: 权重非负"
        assert ws == sorted(ws, reverse=True), f"{t}: 应按权重降序"
        # 近窗约束：被关注段只能落在最近 K 段内
        hist_idx = sorted(
            int(s) for s in engine.meta.loc[
                engine._subject_rows(src, sid), "seg_idx"
            ].tolist() if s < cur
        )
        recent_window = set(hist_idx[-engine.K:])
        assert all(p["seg_idx"] in recent_window for p in pairs), \
            f"{t}: 出现超出近 K={engine.K} 窗的历史段"


def test_attention_full_window_sums_to_one(engine):
    src, sid, cur, _min = _pick_small_history_subject(engine)
    hist, hist_idx = engine.subject_history(src, sid, cur)
    e_cur = hist[-1:]  # 任意一行嵌入即可（结构测试不关心具体值）
    import numpy as np
    x = np.concatenate([hist, e_cur], axis=0)
    domain = "anker" if src == "anker" else "full"
    fold_ids = engine._fold_ids(domain, None)
    att = engine.collect_history_attention(
        x, domain, fold_ids, hist_idx, top=len(hist_idx) + 5)
    for t, pairs in att.items():
        total = sum(p["weight"] for p in pairs)
        # 全窗口 softmax×均值 ≈ 1（1e-4 阈值丢掉极小尾权，容差放宽）
        assert 0.90 <= total <= 1.02, f"{t}: 全窗口权重和应≈1，实得 {total:.3f}"


def test_no_history_segment_has_no_attention(engine):
    src, sid, _cur, mn = _pick_small_history_subject(engine)
    out = engine.predict_segment_lookup(src, sid, mn)
    assert out["n_history"] == 0, "最小 seg_idx 应无历史"
    assert "history_attention" not in out, "无历史不应产出注意力归因"


def test_live_and_lookup_share_attention_api(engine):
    """predict_segment_from_sequence 透传 hist_seg_idxs 才产注意力：
    不传时不应有 history_attention（保持旧调用零影响）。"""
    import numpy as np
    src, sid, cur, _min = _pick_small_history_subject(engine)
    hist, hist_idx = engine.subject_history(src, sid, cur)
    x = np.concatenate([hist, hist[-1:]], axis=0)
    no_idx = engine.predict_segment_from_sequence(x, src, fold=None)
    with_idx = engine.predict_segment_from_sequence(
        x, src, fold=None, hist_seg_idxs=hist_idx)
    assert "history_attention" not in no_idx
    assert "history_attention" in with_idx
    # 预测数值与是否归因无关
    assert no_idx["predictions"] == with_idx["predictions"]
