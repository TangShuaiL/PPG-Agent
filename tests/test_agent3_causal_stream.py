"""Agent 3 主口径 CausalStreamEngine 测试（合成 npz，torch-free）。

确定性技巧：把头的 W/a/mlp 隐层权重全部置零、末层偏置 b、权重置零，
则任何输入的标准化预测恒为 b → 段预测 = mu + sd*b。
据此精确校验 fold//2 选头、5 折 ensemble、median 池化与版本一致性。
"""
import json

import numpy as np
import pandas as pd
import pytest

from ppg_agent.engine.causal_stream import CausalStreamEngine

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
    P = len(_zero_head(0.0))
    # mu 按折区分（0..4 → 100..104），sd 恒 2；full 域用 mu=110
    anker = np.stack([np.stack([
        np.stack([_zero_head(0.0) for _ in range(5)])          # seed 维
        for _ in TARGETS])                                     # target 维
        for _ in range(5)])                                    # fold 维
    full = np.stack([np.stack([_zero_head(0.0) for _ in range(5)])
                     for _ in TARGETS])
    z = {"targets": np.array(TARGETS),
         "anker_params": anker, "anker_mu": np.stack([
             np.full((5, 5), 100.0 + k, np.float64) for k in range(5)]),
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
    # 嵌入产物：2 受试者（anker fold=4 → pf=2；vital full）
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
        "seg_idx": [30, 10, 20, 50, 40, 60] + [5, 3, 1, 4],  # 乱序给排序用
    }).to_parquet(vdir / "meta.parquet", index=False)
    return tmp_path


def test_anker_fold2_select_and_median(tmp_path):
    root = _make_env(tmp_path)
    eng = CausalStreamEngine(root)
    out = eng.predict_subject("anker", "a1")
    # fold=4 → pf=2 → mu=102, sd=2, z 恒 0 → 每段预测恒 102 → median=102
    assert out["predictions"]["sbp"] == pytest.approx(102.0, abs=0.05)
    assert out["fold_used"] == 2 and out["domain"] == "anker"
    assert out["n_segments"] == 6 and out["n_heads"] == 1  # 最优单 seed (seed=4) × 1 折
    assert out["confidence"]["sbp_mae"] == pytest.approx(6.0 + 2)


def test_full_domain_single_head(tmp_path):
    root = _make_env(tmp_path)
    eng = CausalStreamEngine(root)
    out = eng.predict_subject("vital", "v1")
    assert out["predictions"]["sbp"] == pytest.approx(110.0, abs=0.05)
    assert out["domain"] == "full" and out["n_segments"] == 4
    assert out["confidence"]["sbp_mae"] == pytest.approx(10.0)


def test_version_consistency(tmp_path):
    root = _make_env(tmp_path)
    eng = CausalStreamEngine(root)
    assert eng.encoder_version == VER
    bad = dict(json.loads((root / "manifest.json").read_text()))
    bad["encoder_version"] = "other@123"
    (root / "manifest.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError):
        CausalStreamEngine(root)


def test_missing_subject(tmp_path):
    root = _make_env(tmp_path)
    eng = CausalStreamEngine(root)
    with pytest.raises(KeyError):
        eng.predict_subject("vital", "nobody")
