"""Agent 3 实时流式推理链测试（合成头产物 + 注入假编码器，torch-free）。

零权重头技巧（同 test_agent3_causal_stream）：标准化预测恒为末层偏置 b →
段级预测 = mu + sd*b，可精确断言。live 侧注入固定向量假编码器并
monkeypatch 波形路径/拍检测/前序段查询，不触真实 DB 与 ckpt。

五步链覆盖：读窗/拍检测/表示层由 embed 注入点旁路（其真实口径由
scripts/validate_live_stream.py 对离线产物逐位校验，不属单测范畴），
此处验证历史上下文组装、出值契约与降级语义。
"""
import json

import numpy as np
import pandas as pd
import pytest

from ppg_agent.engine import live_stream as ls
from ppg_agent.engine.causal_stream import CausalStreamEngine
from ppg_agent.engine.live_stream import LiveStreamEngine, LiveStreamUnavailable
from ppg_agent.engine.personal_engine import PersonalEngine, _causal_recent_k

D, HID, K = 8, 4, 32
PARAM_NAMES = ["a", "W.weight", "W.bias", "mlp.0.weight", "mlp.0.bias",
               "mlp.2.weight", "mlp.2.bias", "mlp.4.weight", "mlp.4.bias"]
PARAM_SHAPES = [(HID,), (HID, D), (HID,), (HID, 3 * D), (HID,),
                (HID, HID), (HID,), (1, HID), (1,)]
TARGETS = ["sbp", "dbp", "age"]
VER = "test_enc@cafe"


def _zero_head(b):
    """零权重 + 末层偏置 b → 标准化预测恒为 b。"""
    arrs = {}
    for name, shape in zip(PARAM_NAMES, PARAM_SHAPES):
        arrs[name] = np.full(int(np.prod(shape)), 0.0, np.float32) \
            .reshape(shape)
    arrs["mlp.4.bias"] = np.full((1,), b, np.float32)
    return np.concatenate([arrs[n].ravel() for n in PARAM_NAMES])


def _make_env(tmp_path):
    """合成 causal_heads.npz + 嵌入产物（anker fold=4 / vital full）。"""
    P = len(_zero_head(0.0))
    anker = np.stack([np.stack([
        np.stack([_zero_head(0.0) for _ in range(5)])
        for _ in TARGETS])
        for _ in range(5)])
    full = np.stack([np.stack([_zero_head(0.0) for _ in range(5)])
                     for _ in TARGETS])
    z = {"targets": np.array(TARGETS),
         "anker_params": anker,
         "anker_mu": np.stack([np.full((5, 5), 100.0 + k, np.float64)
                               for k in range(5)]),
         "anker_sd": np.full((5, 3, 5), 2.0),
         "full_params": full,
         "full_mu": np.full((3, 5), 110.0), "full_sd": np.full((3, 5), 2.0)}
    np.savez(tmp_path / "causal_heads.npz", **z)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "kind": "causal_stream_heads", "encoder_version": VER,
        "data_version": "v2c", "K": K, "d_model": D, "hid": HID,
        "param_names": PARAM_NAMES,
        "param_shapes": [list(s) for s in PARAM_SHAPES],
        "targets": TARGETS,
        "official_metrics": {
            "anker": {f"f{k}": {t: {"seg_mean": 9.0, "sub_mean": 6.0 + k}
                                for t in TARGETS} for k in range(5)},
            "full": {t: {"seg_mean": 12.0, "sub_mean": 10.0}
                     for t in TARGETS}}}))
    vdir = tmp_path / VER
    vdir.mkdir()
    rng = np.random.default_rng(0)
    n1, n2 = 6, 4
    emb = rng.normal(0, 1, (n1 + n2, D)).astype(np.float32)
    np.save(vdir / "embeddings.npy", emb)
    pd.DataFrame({
        "row": range(n1 + n2),
        "subject_id": ["a1"] * n1 + ["v1"] * n2,
        "source": ["anker"] * n1 + ["vital"] * n2,
        "fold": [4] * n1 + [3] * n2,
        "seg_idx": [30, 10, 20, 50, 40, 60] + [5, 3, 1, 4],
    }).to_parquet(vdir / "meta.parquet", index=False)
    return tmp_path


@pytest.fixture
def heads(tmp_path):
    return CausalStreamEngine(_make_env(tmp_path))


# ---------------------------------------------------------------------------
# 段级原语（CausalStreamEngine）
# ---------------------------------------------------------------------------

class TestSegmentPrimitives:
    def test_segment_embedding_and_keyerror(self, heads):
        e = heads.segment_embedding("vital", "v1", 4)
        assert e.shape == (D,) and e.dtype == np.float32
        with pytest.raises(KeyError):
            heads.segment_embedding("vital", "v1", 99)   # 段不在表
        with pytest.raises(KeyError):
            heads.segment_embedding("vital", "nobody", 0)  # 受试者不在表

    def test_history_embeddings_causal_order(self, heads):
        # v1 seg_idx 乱序 [5,3,1,4]：seg4 的历史 = [1,3]，必须保序
        h = heads.subject_history_embeddings("vital", "v1", 4)
        assert h.shape == (2, D)
        assert np.allclose(h[0], heads.segment_embedding("vital", "v1", 1))
        assert np.allclose(h[1], heads.segment_embedding("vital", "v1", 3))
        # 首段无历史 → (0,d) 空返回
        assert heads.subject_history_embeddings(
            "vital", "v1", 1).shape == (0, D)
        # 受试者不在表 → (0,d)，不抛
        assert heads.subject_history_embeddings(
            "vital", "nobody", 0).shape == (0, D)

    def test_predict_segment_from_sequence_zero_head(self, heads):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (5, D)).astype(np.float32)
        out = heads.predict_segment_from_sequence(
            x, "vital", engine_label="live_stream")
        # 零头 → z=0 → full 域 mu=110
        assert out["predictions"] == {"sbp": 110.0, "dbp": 110.0,
                                      "age": 110.0}
        assert out["n_history"] == 4 and out["n_segments"] == 5
        assert out["pool"] == "segment_stream"
        assert out["engine"] == "live_stream"
        assert out["segment_level"] is True

    def test_predict_segment_lookup_matches_sequence(self, heads):
        ref = heads.predict_segment_lookup("vital", "v1", 4)
        x = np.concatenate([
            heads.subject_history_embeddings("vital", "v1", 4),
            heads.segment_embedding("vital", "v1", 4)[None]], axis=0)
        manual = heads.predict_segment_from_sequence(
            x, "vital", fold=3, engine_label="x")
        assert ref["predictions"] == manual["predictions"]
        assert ref["history_source"] == "table"
        assert ref["n_history"] == 2
        # anker fold=4 → pf=2 → mu=102（单折头，非 5 折 ensemble）
        ak = heads.predict_segment_lookup("anker", "a1", 20)
        assert ak["predictions"]["sbp"] == pytest.approx(102.0, abs=0.05)
        assert ak["fold_used"] == 2

    def test_lookup_keyerror_paths(self, heads):
        with pytest.raises(KeyError):
            heads.predict_segment_lookup("vital", "nobody", 0)
        with pytest.raises(KeyError):
            heads.predict_segment_lookup("vital", "v1", 99)


# ---------------------------------------------------------------------------
# LiveStreamEngine（假编码器 + monkeypatch，不触 DB/ckpt/torch）
# ---------------------------------------------------------------------------

class TestLiveStreamEngine:
    def test_stream_with_table_history(self, heads, monkeypatch):
        """五步链主干：表内历史 + 假编码器=表内嵌入 → 与查表逐位一致。"""
        monkeypatch.setattr(ls.LiveStreamEngine, "_waveform_path",
                            lambda self, s, sid, k: f"fake/{sid}/{k}")
        # 假编码器：实时编码结果 == 离线表（校验等价后的理想情形）
        monkeypatch.setattr(
            ls.LiveStreamEngine, "embed_segment_live",
            lambda self, s, sid, k: heads.segment_embedding(s, sid, k))
        eng = LiveStreamEngine(heads=heads, encoder=lambda *a: None)
        out = eng.predict_segment_stream("vital", "v1", 4)
        ref = heads.predict_segment_lookup("vital", "v1", 4)
        assert out["predictions"] == ref["predictions"]
        assert out["history_source"] == "table"
        assert out["n_history"] == 2
        assert out["engine"] == "live_stream"
        assert "history_note" not in out          # 表内历史无需注记

    def test_stream_first_segment_e_k_only(self, heads, monkeypatch):
        """首段/不在表：无历史 → e_k-only 降级模式，如实标注。"""
        monkeypatch.setattr(
            ls.LiveStreamEngine, "embed_segment_live",
            lambda self, s, sid, k: np.zeros(D, np.float32))
        monkeypatch.setattr(ls.LiveStreamEngine, "_fold_of",
                            lambda self, s, sid: (None, "db"))
        monkeypatch.setattr(ls.LiveStreamEngine, "_previous_seg_idxs",
                            staticmethod(lambda s, sid, k: []))
        eng = LiveStreamEngine(heads=heads, encoder=lambda *a: None)
        out = eng.predict_segment_stream("vital", "new_user", 0)
        assert out["history_source"] == "none" and out["n_history"] == 0
        assert "e_k-only" in out["history_note"]
        assert out["predictions"]["sbp"] == pytest.approx(110.0, abs=0.05)

    def test_stream_dim_mismatch(self, heads, monkeypatch):
        """实时嵌入维度 != 头 d_model → LiveStreamUnavailable（上层降级查表）。"""
        monkeypatch.setattr(
            ls.LiveStreamEngine, "embed_segment_live",
            lambda self, s, sid, k: np.zeros(7, np.float32))
        eng = LiveStreamEngine(heads=heads, encoder=lambda *a: None)
        with pytest.raises(LiveStreamUnavailable):
            eng.predict_segment_stream("vital", "v1", 4)

    def test_fold_of_prefers_meta(self, heads, monkeypatch):
        eng = LiveStreamEngine(heads=heads, encoder=lambda *a: None)
        fold, src = eng._fold_of("anker", "a1")
        assert fold == 4 and src == "table"

    def test_window_index_identity_for_clinical_sources(self):
        """mimic/vital/aurora：窗号 = seg_idx（一一对应，无 DB 查询）。"""
        eng = LiveStreamEngine(heads=None, encoder=lambda *a: None)
        assert eng.window_index("vital", "p000051", 42) == 42
        assert eng.window_index("mimic", "p003131", 0) == 0
        assert eng.window_index("aurora", "a000", 6) == 6

    def test_window_index_anker_rank_mapping(self, monkeypatch):
        """ankera：窗号 = seg_idx 在该受试者有效段升序中的 rank。

        DB mock：SZTU-ABPM001 段集 [2,3,4,...]（head_trim 跳过 0,1）→
        seg_idx=2 → 窗 0；seg_idx=4 → 窗 2。"""
        class _FakeCon:
            def execute(self, sql, params):
                assert params[0] == "SZTU-ABPM001"
                rank = int(params[1]) - 2          # seg_idx 从 2 起连续
                class _Cur:
                    def fetchone(self):
                        return (max(rank, 0),)
                return _Cur()

            def close(self):
                pass
        import sqlite3
        monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: _FakeCon())
        eng = LiveStreamEngine(heads=None, encoder=lambda *a: None)
        assert eng.window_index("anker", "SZTU-ABPM001", 2) == 0
        assert eng.window_index("anker", "SZTU-ABPM001", 4) == 2


# ---------------------------------------------------------------------------
# Agent 2 历史口径对齐（v3.2）
# ---------------------------------------------------------------------------

class _FakePercentileEngine:
    def _icc(self, source, feature):
        return None

    def _icc_2k(self, source, feature, k):
        return None

    def _mdc(self, source, feature):
        return None

    def query_stratified(self, feature, value, source="vital",
                         age=None, gender=None):
        return {"feature": feature, "value": value,
                "percentile": "P25-P75"}


def test_causal_recent_k_defaults_to_manifest():
    # 真实 manifest K=32；产物缺失时回退 32 —— 两侧均应为 32
    assert _causal_recent_k() == 32


class TestCompareSegmentV32:
    def _engine(self, segs, monkeypatch):
        # beat 层不属本组测试范畴（生产路径由真实 beat 表驱动）
        monkeypatch.setattr(
            PersonalEngine, "_compare_beats",
            lambda self, *a, **k: {"available": True, "n_valid_cur": 10})
        eng = PersonalEngine(percentile_engine=_FakePercentileEngine())
        eng.load_subject_segments = lambda ds, sid: segs
        return eng

    def test_short_history_still_judges_with_notes(self, monkeypatch):
        """v3.2：2 段历史不再硬门拦截 —— verdict=ok 照常判定 + 注记。"""
        segs = pd.DataFrame({
            "seg_idx": [0, 1, 2],
            "pulse_amp_mean": [0.51, 0.51, 0.90],   # 历史全同 → MAD=0 地板
            "quality_tier": ["good"] * 3})
        out = self._engine(segs, monkeypatch).compare_segment("vital", "p1", 2)
        assert out["verdict"] == "ok"
        assert out["insufficient_history"] is True   # 注记，非拦截
        assert out["insufficient_power"] is True
        assert out["power_note"] and "2" in out["power_note"]
        f = out["features"]["pulse_amp_mean"]
        assert f["verdict_long"] == "real_change"    # Δ=0.4 > 1% 地板
        assert f["threshold"] == pytest.approx(0.01 * 0.51, rel=1e-3)
        assert f.get("insufficient_power") is True

    def test_recent_k_uses_causal_window(self, monkeypatch):
        segs = pd.DataFrame({
            "seg_idx": range(40),
            "pulse_amp_mean": np.linspace(0.3, 0.9, 40),
            "quality_tier": ["good"] * 40})
        out = self._engine(segs, monkeypatch).compare_segment("vital", "p1", 39)
        assert out["recent_k"] == 32
        assert out["n_recent"] == 32 and out["n_history"] == 39
        assert out["limited_history"] is False
        assert out["insufficient_history"] is False
