"""L2 波形梯度归因单测。

头侧数学（纯 numpy，双解释器可跑）：
- ``_gelu_grad`` 对齐 ``_gelu`` 的有限差分；
- ``current_embed_grad``（∂(真实值预测)/∂e_k 解析）对齐方向导数有限差分。
端到端 IG（需 torch + GPU + encoder）：
- ``segment_waveform_attribution`` 满足完备性公理 Σ归因 ≈ pred(实际)−pred(baseline)。
无固化产物 / 无 GPU 时相应跳过。
"""
from __future__ import annotations

import numpy as np
import pytest

from ppg_agent.engine import causal_stream as cs
from ppg_agent.engine.causal_stream import (CausalStreamEngine,
                                            get_causal_stream_engine)


@pytest.fixture(scope="module")
def engine() -> CausalStreamEngine:
    try:
        return get_causal_stream_engine()
    except FileNotFoundError:
        pytest.skip("固化头/嵌入产物缺失（需 build_causal_heads + build_ml_embeddings）")


# ────────────────────────── 头侧解析梯度 ──────────────────────────
def test_gelu_grad_matches_finite_difference():
    xs = np.linspace(-4.0, 4.0, 41)
    eps = 1e-4
    fd = (cs._gelu(xs + eps) - cs._gelu(xs - eps)) / (2 * eps)
    ana = cs._gelu_grad(xs)
    assert np.max(np.abs(fd - ana)) < 1e-4, "GELU 解析导数与有限差分应一致"


def _raw_pred(engine, x, domain, fold_ids, target):
    """未取整的末行预测均值：Σ_head (z·sd+mu)/cnt。"""
    acc, cnt = 0.0, 0
    for k in fold_ids:
        for s in engine.infer_seeds:
            h = engine._head(domain, k, target, s)
            mu, sd = engine._mu_sd(domain, k, target, s)
            acc += engine._forward_std_last_row(x, h) * sd + mu
            cnt += 1
    return acc / cnt


def test_current_embed_grad_matches_directional_fd(engine):
    """∂pred/∂e_k 解析梯度应等于对当前段嵌入（末行）的方向导数。"""
    rng = np.random.default_rng(7)
    d = engine.d_model
    for N in (1, 3, 40):
        x = (rng.standard_normal((N, d)) * 0.5).astype(np.float32)
        g = engine.current_embed_grad(x, "full", [0], "sbp")
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        eps = 1e-3
        xp, xm = x.copy(), x.copy()
        xp[-1] += eps * v            # 仅扰当前段（历史窗与 e_k 无关）
        xm[-1] -= eps * v
        fd = (_raw_pred(engine, xp, "full", [0], "sbp")
              - _raw_pred(engine, xm, "full", [0], "sbp")) / (2 * eps)
        assert abs(fd - float(g @ v)) <= 1e-2 * (abs(fd) + 1.0), \
            f"N={N}: 解析梯度与有限差分不一致 fd={fd} g·v={g @ v}"


def test_embed_grad_ignores_history_rows(engine):
    """∂pred/∂e_k 只反映当前段：历史 c_recent/c_long 与 e_k 无关，
    故解析梯度对 x[:-1] 的扰动的方向导数只应经 e_k 通道（此处验证末行
    梯度形状与数值有限，不 NaN）。"""
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((12, engine.d_model))).astype(np.float32)
    g = engine.current_embed_grad(x, "full", [0], "dbp")
    assert g.shape == (engine.d_model,)
    assert np.all(np.isfinite(g))


# ────────────────────────── 端到端 IG（需 GPU） ──────────────────────────
def _pick_live_subject(engine):
    m = engine.meta
    m = m[m["source"] != "anker"]
    cnt = m.groupby(["source", "subject_id"])["seg_idx"].agg(["size", "max"])
    cand = cnt[(cnt["size"] >= 4) & (cnt["size"] <= 40)]
    if cand.empty:
        return None
    (src, sid), row = next(iter(cand.iterrows()))
    return src, sid, int(row["max"])


def test_segment_waveform_attribution_completeness(engine):
    """IG 完备性：Σ 归因 ≈ pred(实际段) − pred(baseline)，相对误差应很小。"""
    ls = pytest.importorskip("torch")  # noqa: F841
    from ppg_agent.engine.live_stream import (LiveStreamUnavailable,
                                              get_live_stream_engine)
    pick = _pick_live_subject(engine)
    if pick is None:
        pytest.skip("嵌入表无 4~40 段规模受试者可用于 IG 测试")
    src, sid, cur = pick
    try:
        r = get_live_stream_engine().segment_waveform_attribution(
            src, sid, cur, "sbp", steps=16)
    except LiveStreamUnavailable as e:
        pytest.skip(f"实时归因链不可用（无 GPU/波形/encoder）: {e}")
    assert len(r["feature_points"]) == 9, "特征点应为 9 个生理关键点"
    # 划分完整性：各特征点签名贡献求和应等于 ig_total（逐点 IG 无遗漏）
    fp_sum = sum(fp["contrib"] for fp in r["feature_points"])
    assert abs(fp_sum - r["ig_total"]) <= 0.02, \
        f"特征点划分未完整覆盖 IG: Σ={fp_sum} ig_total={r['ig_total']}"
    span = abs(r["pred_actual"] - r["pred_baseline"]) or 1.0
    assert abs(r["completeness_gap"]) / span < 0.05, \
        f"IG 完备性差过大: gap={r['completeness_gap']} span={span}"


# ────────────────────── 特征点划分：逐拍定位（纯 numpy） ──────────────────────
def test_feature_point_partition_localizes_per_beat():
    """_feature_point_partition 除聚合 feature_points 外, 还产出逐 (拍,关键点)
    的 beat_points(带绝对时间), 且 Σ beat_points = Σ feature_points = ig_total
    ——把梯度归因定位到具体拍, 而非用聚合值把每个 beat 同色高亮。
    用 blen<8 触发 (o,s,v) 三分点退化路径, 不依赖 GPU。"""
    from ppg_agent.engine.live_stream import LiveStreamEngine

    ig = np.array([[1, 1, 0, 0, 0, 0, 0, 2],
                   [0, 3, 0, 0, 0, 0, 0, -4]], dtype=float)
    beats_np = np.array([[0, 0, 5, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 6, 0, 0, 0]], dtype=float)
    bounds = np.array([[100, 104], [200, 206]])
    seg125 = np.zeros(300)
    fps, n_ok, bps = LiveStreamEngine._feature_point_partition(
        ig, beats_np, seg125, bounds, 8)
    ig_total = float(ig.sum())
    assert abs(sum(f["contrib"] for f in fps) - ig_total) < 1e-6
    assert abs(sum(b["contrib"] for b in bps) - ig_total) < 1e-6
    assert n_ok == 0  # 全部走三分点退化（未调 vitalpy）
    by_beat = {}
    for b in bps:
        by_beat.setdefault(b["beat_idx"], set()).add(b["name"])
    assert set(by_beat) == {0, 1}  # 两拍都定位到各自关键点
    t0o = next(b for b in bps if b["beat_idx"] == 0 and b["name"] == "o")
    t1o = next(b for b in bps if b["beat_idx"] == 1 and b["name"] == "o")
    assert abs(t0o["t"] - 100 / 125) < 1e-6
    assert abs(t1o["t"] - 200 / 125) < 1e-6
    assert t0o["contrib"] != t1o["contrib"]  # 非“每拍一样”的直接证据


# ────────────────────────── 历史段（近窗）特征点归因 ──────────────────────
def test_embed_grads_for_rows_structure(engine):
    """embed_grads_for_rows 应逐行返回 (d,) 有限梯度。"""
    rng = np.random.default_rng(5)
    d = engine.d_model
    x = (rng.standard_normal((10, d)) * 0.5).astype(np.float32)
    gmap = engine.embed_grads_for_rows(x, "full", [0], "age", [0, 3, 8])
    assert set(gmap) == {0, 3, 8}
    for r, g in gmap.items():
        assert g.shape == (d,) and np.all(np.isfinite(g))


def test_embed_grads_outside_window_matches_fd(engine):
    """窗外历史行（仅经 c_long 影响预测）：解析梯度应精确等于方向导数 FD。

    窗外行 α_r=0，梯度只剩 (1/M)·∂pred/∂c_long（无 softmax 二阶项）
    → 注意力权重“视为常数”的近似在此退化为精确，可严格对 FD。"""
    rng = np.random.default_rng(11)
    d, K = engine.d_model, engine.K
    N = K + 5
    x = (rng.standard_normal((N, d)) * 0.3).astype(np.float32)
    r = 0                                   # 窗外：r < N-1-K
    g = engine.embed_grads_for_rows(x, "full", [0], "sbp", [r])[r]
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    eps = 1e-3
    xp, xm = x.copy(), x.copy()
    xp[r] += eps * v
    xm[r] -= eps * v
    fd = (_raw_pred(engine, xp, "full", [0], "sbp")
          - _raw_pred(engine, xm, "full", [0], "sbp")) / (2 * eps)
    assert abs(fd - float(g @ v)) <= 1e-2 * (abs(fd) + 1.0), \
        f"窗外行 FD 不一致 fd={fd} g·v={g @ v}"


def test_segment_history_attribution_structure(engine):
    """历史段 Top-N 特征点划分完整性：每段 Σ特征点 = 该段 ig_total。"""
    pytest.importorskip("torch")
    from ppg_agent.engine.live_stream import (LiveStreamUnavailable,
                                              get_live_stream_engine)
    pick = _pick_live_subject(engine)
    if pick is None:
        pytest.skip("嵌入表无 4~40 段规模受试者可用于历史归因测试")
    src, sid, cur = pick
    try:
        r = get_live_stream_engine().segment_history_attribution(
            src, sid, cur, "sbp", top=2, steps=12)
    except LiveStreamUnavailable as e:
        pytest.skip(f"实时归因链不可用（需 GPU/波形/encoder）: {e}")
    assert r["attrib_method"] == "history_attention_feature_point_ig"
    for seg in r["segments"]:
        assert len(seg["feature_points"]) == 9, "特征点应为 9 个关键点"
        fp_sum = sum(f["contrib"] for f in seg["feature_points"])
        assert abs(fp_sum - seg["ig_total"]) <= 0.02, \
            f"seg{seg['seg_idx']} 划分未完整: Σ={fp_sum} ig_total={seg['ig_total']}"
