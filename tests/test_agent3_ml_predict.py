"""Agent 3 查表推理引擎测试（LinearHeads / MlPredictEngine / 确定性叙事）。"""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from ppg_agent.engine._ml_heads import LinearHeads


def _make_heads_npz(tmp_path, d=8):
    """按 ppg_model/serve/heads.py 固化格式构造小三头（带偏置主导演示）。"""
    rng = np.random.default_rng(0)
    z = {"targets": np.array(["sbp", "dbp", "age"]),
         "mu": rng.normal(0, 1, d),
         "sd": np.abs(rng.normal(1, 0.1, d)) + 1e-3,
         "test_folds": np.array([1, 2]), "n_subjects": np.int64(3)}
    for t, scale in (("sbp", 120.0), ("dbp", 80.0), ("age", 50.0)):
        z[f"{t}_alpha"] = np.float64(1.0)
        z[f"{t}_w"] = rng.normal(0, 0.01, d)
        z[f"{t}_b"] = np.float64(scale)
        z[f"{t}_MAE"] = np.float64(11.7 if t == "sbp" else 7.2 if t == "dbp" else 8.9)
        z[f"{t}_r"] = np.float64(0.5)
    path = tmp_path / "heads_src.npz"
    np.savez(path, **z)
    return path


def _make_emb_artifacts(tmp_path, heads_path, d=8):
    """在 tmp_path 下铺好嵌入产物（2 受试者 × 3 段）。"""
    shutil.copy(heads_path, tmp_path / "linear_heads_subject.npz")
    ver = "test_enc@deadbeefcafe"
    vdir = tmp_path / ver
    vdir.mkdir()
    pd.DataFrame({
        "row": [0, 1, 2, 3, 4, 5],
        "subject_id": ["p1"] * 3 + ["p2"] * 3,
        "source": ["vital"] * 6,
        "fold": [3, 3, 3, 4, 4, 4],
        "seg_idx": [10, 11, 12, 20, 21, 22],
    }).to_parquet(vdir / "meta.parquet", index=False)
    emb = np.linspace(-1.0, 1.0, 6 * d, dtype=np.float32).reshape(6, d)
    np.save(vdir / "embeddings.npy", emb)
    (vdir / "build_info.json").write_text(json.dumps({"encoder_version": ver}))
    return vdir, emb


class TestLinearHeads:
    def test_predict_offset_by_bias(self, tmp_path):
        h = LinearHeads(_make_heads_npz(tmp_path))
        # 输入 = mu 时标准化后为 0，预测应退化为偏置
        out = h.predict(h.mu)
        assert out["sbp"] == pytest.approx(120.0)
        assert out["dbp"] == pytest.approx(80.0)
        assert out["age"] == pytest.approx(50.0)

    def test_metrics_carried(self, tmp_path):
        h = LinearHeads(_make_heads_npz(tmp_path))
        assert h.metrics["sbp"]["MAE"] == pytest.approx(11.7)
        assert h.n_subjects == 3


class TestMlPredictEngine:
    def test_subject_prediction_matches_manual_pool(self, tmp_path):
        from ppg_agent.engine.ml_predict import MlPredictEngine

        vdir, emb = _make_emb_artifacts(tmp_path, _make_heads_npz(tmp_path))
        eng = MlPredictEngine(root=tmp_path)
        pred = eng.predict_subject("vital", "p1")

        heads = eng.heads
        pooled = emb[:3].astype(np.float64).mean(axis=0)
        expected = heads.predict(pooled)
        assert pred["predictions"]["sbp"] == pytest.approx(
            round(expected["sbp"], 1))
        assert pred["pool"] == "subject" and pred["n_segments"] == 3
        assert pred["encoder_version"] == vdir.name
        assert "非诊断" in pred["confidence"]["note"]

    def test_segment_prediction(self, tmp_path):
        from ppg_agent.engine.ml_predict import MlPredictEngine

        _make_emb_artifacts(tmp_path, _make_heads_npz(tmp_path))
        eng = MlPredictEngine(root=tmp_path)
        pred = eng.predict_segment("vital", "p2", 21)
        assert pred["segment_level"] is True
        assert pred["n_segments"] == 1

    def test_missing_subject_and_segment_raise(self, tmp_path):
        from ppg_agent.engine.ml_predict import MlPredictEngine

        _make_emb_artifacts(tmp_path, _make_heads_npz(tmp_path))
        eng = MlPredictEngine(root=tmp_path)
        with pytest.raises(KeyError):
            eng.predict_subject("vital", "ghost")
        with pytest.raises(KeyError):
            eng.predict_segment("vital", "p1", 999)

    def test_deterministic_narrative(self, tmp_path):
        from ppg_agent.engine.ml_predict import MlPredictEngine

        _make_emb_artifacts(tmp_path, _make_heads_npz(tmp_path))
        eng = MlPredictEngine(root=tmp_path)
        pred = eng.predict_subject("vital", "p1")
        seg = {"source": "vital", "subject_id": "p1", "seg_idx": 10}
        from ppg_agent.agents.ml_prediction_agent import format_prediction
        text = format_prediction(seg, eng, pred)
        assert "SBP ≈" in text and "DBP ≈" in text
        assert "置信度" in text
        assert "群体级统计估计" in text
        # 纪律：不得出现诊断/风险结论式措辞
        for banned in ("高血压风险", "心血管事件", "诊断患有"):
            assert banned not in text


class TestAgent3Wiring:
    def test_supervisor_prompt_declares_ready_agent3(self):
        from ppg_agent.agents.supervisor import SUPERVISOR_SYSTEM_PROMPT

        assert "BeatMAE" in SUPERVISOR_SYSTEM_PROMPT
        assert "非诊断结论" in SUPERVISOR_SYSTEM_PROMPT
        assert "训练中" not in SUPERVISOR_SYSTEM_PROMPT

    def test_plan_addendum_discipline(self):
        from web.api.services.chat_supervisor import PLAN_ADDENDUM

        assert "置信说明" in PLAN_ADDENDUM
        assert "不得表述为诊断或风险结论" in PLAN_ADDENDUM
