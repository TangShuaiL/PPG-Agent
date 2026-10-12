"""ml_predict.py — Agent 3 查表推理引擎。

当前主力 = CausalStreamEngine（v2.5c 嵌入 + causal_attn 流式头，2026-09 起）；
LinearHeads（subject/linear 岭回归头）保留为降级路径。

数据面（scripts/build_ml_embeddings.py 离线预计算，data_version=v2c）：
    $PPG_DATA_ROOT/ppg_agent/embeddings/
    ├── causal_heads.npz + manifest.json  # serve/heads_causal.py 固化的流式头
    ├── linear_heads_subject.npz          # 降级：serve/heads.py 固化的三头
    └── <encoder_version>/embeddings.npy  # (N, 256) fp32 段嵌入
        meta.parquet                      # row/subject_id/source/fold (+seg_idx)

主口径（CausalStreamEngine，encoder_version=
beatmae_v25c_traverse_v2new@cb89b83cf320 × data_version=v2c）：
- 逐段 causal_attn（e_k + c_recent[K=32] + c_long）→ 最优单 seed(seed=4) × 折集成
  → 受试者级 median（先预测后池化，sub 主口径）
- anker（评估口径 = fold2 测试集 + f2 头）：
  SBP MAE 2.94 / DBP 3.55 / Age 1.36（sub 口径官方数）
- full（三库 serving，按 28 号折结构选头）：
  SBP MAE 10.14 / DBP 6.90 / Age 7.78（sub 口径官方数）
- 段级口径作附注（full）：SBP 12.42 / DBP 7.98 / Age 8.33（seg_mean）

降级路径（LinearHeads，历史 v1 口径，仅当流式产物缺失时使用）：
- 受试者级：全部段嵌入均值 → LinearHeads
  （probe subject/linear 历史 v1 数字：SBP 11.72 / DBP 7.23 / Age 8.87）
- 段级：单段嵌入直接过头（历史 v1：14.42/8.66/9.82）

纪律：模型输出是群体级统计估计（置信度见 metrics），不作诊断、不作
心血管风险结论；与 Agent 2 的纵向变化叙事严格分离。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_ROOT

# 嵌入产物根目录（可用环境变量 PPG_EMB_ROOT 覆盖）
EMB_ROOT = Path(DATA_ROOT) / "ppg_agent" / "embeddings"


def _resolve_version(root: Path) -> Path:
    """取唯一版本目录；多个时要求 PPG_ENCODER_VERSION 指定。"""
    import os

    want = os.environ.get("PPG_ENCODER_VERSION")
    versions = sorted(p for p in root.iterdir() if p.is_dir())
    if want:
        d = root / want
        if not d.is_dir():
            raise FileNotFoundError(f"PPG_ENCODER_VERSION={want} 不存在于 {root}")
        return d
    if len(versions) == 1:
        return versions[0]
    names = [v.name for v in versions]
    raise RuntimeError(f"嵌入版本不唯一 {names}，请设置 PPG_ENCODER_VERSION")


class MlPredictEngine:
    """BeatMAE 嵌入查表 + 三头预测（只读、惰性加载、进程内单例）。"""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else EMB_ROOT
        self.version_dir = _resolve_version(self.root)
        self.heads_npz = self.root / "linear_heads_subject.npz"
        self.meta: pd.DataFrame = pd.read_parquet(self.version_dir / "meta.parquet")
        self.emb: np.ndarray = np.load(self.version_dir / "embeddings.npy",
                                       mmap_mode="r")
        info_path = self.version_dir / "build_info.json"
        self.build_info = json.loads(info_path.read_text()) if info_path.exists() else {}
        self.encoder_version = str(self.build_info.get("encoder_version")
                                   or self.version_dir.name)
        self._heads = None
        # (source, subject_id) → 行索引数组（惰性）
        self._idx_cache: dict[tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    @property
    def heads(self):
        if self._heads is None:
            from ._ml_heads import LinearHeads

            self._heads = LinearHeads(self.heads_npz)
        return self._heads

    def _subject_rows(self, source: str, subject_id: str) -> np.ndarray:
        key = (source, subject_id)
        if key not in self._idx_cache:
            m = self.meta
            idx = m.index[(m["source"] == source) & (m["subject_id"] == subject_id)]
            self._idx_cache[key] = idx.to_numpy()
        return self._idx_cache[key]

    # ------------------------------------------------------------------
    def predict_subject(self, source: str, subject_id: str) -> dict:
        """受试者级预测（主口径：全部段嵌入均值池化 → 三头）。"""
        rows = self._subject_rows(source, subject_id)
        if len(rows) == 0:
            raise KeyError(f"受试者不在嵌入表: {source}/{subject_id} "
                           f"(可能未过语料质量/拍数过滤)")
        pooled = np.asarray(self.emb[rows], dtype=np.float64).mean(axis=0)
        preds = self.heads.predict(pooled)
        return self._format(preds, pool="subject", n_segments=int(len(rows)),
                            segment_level=False)

    def predict_segment(self, source: str, subject_id: str, seg_idx: int) -> dict:
        """段级预测（segment 口径：单段嵌入直接过头）。"""
        m = self.meta
        sel = m[(m["source"] == source) & (m["subject_id"] == subject_id)
                & (m["seg_idx"] == int(seg_idx))]
        if len(sel) == 0:
            raise KeyError(f"段不在嵌入表: {source}/{subject_id}/seg{seg_idx}")
        emb = np.asarray(self.emb[sel.index[0]], dtype=np.float64)
        preds = self.heads.predict(emb)
        return self._format(preds, pool="segment", n_segments=1,
                            segment_level=True)

    # ------------------------------------------------------------------
    def measured_bp(self, source: str, subject_id: str, seg_idx: int) -> dict:
        """该段实测血压标签（同样过离谱闸门），供对照展示；无标签返回空。"""
        return measured_bp(source, subject_id, seg_idx)

    # ------------------------------------------------------------------
    def _format(self, preds: dict, pool: str, n_segments: int,
                segment_level: bool) -> dict:
        metrics = self.heads.metrics
        mae = {"sbp_mae": metrics.get("sbp", {}).get("MAE"),
               "dbp_mae": metrics.get("dbp", {}).get("MAE"),
               "age_mae": metrics.get("age", {}).get("MAE")}
        return {
            "predictions": {"sbp": round(preds["sbp"], 1),
                            "dbp": round(preds["dbp"], 1),
                            "age": round(preds["age"], 1)},
            "pool": pool,
            "n_segments": n_segments,
            "segment_level": segment_level,
            "encoder_version": self.encoder_version,
            "confidence": {"kind": "population_test_mae", **mae,
                           "note": ("预测为群体级统计估计；主口径（受试者池化）"
                                    "测试集 MAE：SBP±11.7 / DBP±7.2 mmHg、"
                                    "年龄±8.9 年（r=0.72/0.67/0.50）。"
                                    "非诊断结论，仅供横向参考。")},
        }



def measured_bp(source: str, subject_id: str, seg_idx: int) -> dict:
    """该段实测血压标签（模块级：live_stream 与各引擎共用）。"""
    from .personal_engine import PersonalEngine  # noqa: PLC0415

    eng = PersonalEngine()
    try:
        segs = eng.load_subject_segments(source, subject_id)
    except Exception:  # noqa: BLE001 — 表缺失/列缺失时静默降级
        return {}
    row = segs[segs["seg_idx"] == int(seg_idx)]
    if len(row) == 0:
        return {}
    r = row.iloc[0]  # 已过 _sanitize_bp 闸门
    out = {}
    if pd.notna(r.get("sbp_label")):
        out["sbp"] = float(r["sbp_label"])
    if pd.notna(r.get("dbp_label")):
        out["dbp"] = float(r["dbp_label"])
    return out
