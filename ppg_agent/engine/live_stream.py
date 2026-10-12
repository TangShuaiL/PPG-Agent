"""live_stream.py — Agent 3 段级实时流式推理（波形 → 完整流程 → 出值）。

主口径 = causal_attn 流式头 × 实时编码：当前段从 raw 波形走完整链
（读窗 → 拍检测 → 语料同源表示 → BeatMAE 编码 → 流式头出值），
历史上下文 = 该受试者前序段嵌入（因果约定：当前段只看 <seg_idx 的历史）。

与 corpus_v2 的逐位对齐契约（口径依据，改前必读）：
- 读窗：raw .mat Subj_Wins['PPG_F'][0, seg_idx]（h5py ref / vlen 内联 /
  直接数组三态兼容，与 scripts/batch_extract_v2.py、ppg_model
  preprocess/02_build_corpus.read_windows 同源）；1250 样点 @125Hz 原值
  [0,1]。fp16 转换复现 corpus 分片的存储量化（逐位对齐需要）。
- 拍检测：scipy resample → 5000 点 @500Hz → signal_engine.process_ppg_segment
  （merge_fragments=True, trim_to_onset=True，官方 preset，user preset
  不参与——beats_full parquet 即该口径产物）→ beat_start_sample(500Hz)
  ÷4 → 125Hz 起点（corpus KP_SRC_FACTOR 约定）。
- 表示层：ppg_model/data/beat_ops（zscore / extract_beats 拍长 [20,250]
  @125Hz / beat_bounds / resample_beat(128) / beat_side_info）；>32 拍时
  corpus 用 rng(row) 随机连续窗，实时链无 row 号 → 确定性取首窗并标注
  windowed（逐位校验仅覆盖 ≤32 拍子集，见 scripts/validate_live_stream.py）。
- 编码：Agent 侧 beatmae_encoder.BeatmaeEncoder.embed_arrays（CUDA fp16
  autocast / CPU fp32，与 probe 口径逐位一致）；ckpt 由 weights.resolve_encoder_ckpt
  解析（env > 数据盘缓存 > Zenodo 下载），版本由 manifest.encoder_version 锁定。
- 出值：CausalStreamEngine.predict_segment_from_sequence（头参数 numpy
  前向，K=32 近窗注意力 + 全量前缀均值，最优单 seed（seed=4）× 折头 ensemble）。

历史嵌入来源：优先离线嵌入表（embeddings.npy——与实时链做逐位校验后
视为等价）；受试者不在表内时对前序段实时编码（预算内近窗全算 + 前缀
近似，附覆盖率注记）。降级路径：torch/ckpt 缺失 →
CausalStreamEngine.predict_segment_lookup（同头同嵌入查表）。

纪律：模型输出是群体级统计估计（段级置信度见 manifest seg_mean 指标），
不作诊断、不作心血管风险结论；与 Agent 2 的纵向变化叙事严格分离。
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..config import DATA_ROOT, PHENOTYPE_DB, resolve_beatmae_root
from .causal_stream import CausalStreamEngine, get_causal_stream_engine

EMB_ROOT = Path(DATA_ROOT) / "ppg_agent" / "embeddings"
SEG_LEN = 1250          # 10 s @ 125 Hz（corpus 约定）
MAX_BEATS = 32          # encoder 输入拍数上限（= corpus max_beats）
MAX_LIVE_HISTORY = 128  # 实时补算历史段的预算上限（近窗优先）

# 特征点级归因的生理关键点（vitalpy extract_key_points 九点）：
# 顺序用于最近邻划分时的排序参照，label 供前端/解释对齐 Agent 2 术语。
FP_ORDER = ["o", "a", "md", "b", "s", "dn", "ip", "d", "v"]
FP_LABELS = {
    "o": "起点 onset", "a": "叩诊波(二阶导峰)", "md": "最大斜率上升支",
    "b": "b 点", "s": "收缩峰", "dn": "重搏切迹", "ip": "拐点",
    "d": "潮波/舒张峰", "v": "谷点(结束)",
}

# aurora beats_full（v3.1 方案B全漏斗重跑，2026-09-02）的生成口径：
# batch_extract_v2 --extra-params（td 0.4 + corr 0.6），逐位校验实证
# （scripts/validate_live_stream.py check A）；漏掉则首拍 onset 漂移。
AURORA_PLAN_B_PARAMS = {"trough_diff_max": 0.4, "template_corr_min": 0.6}

# anker 检测口径：corpus 信号不是 mat 内 fp16，而是 build_anker_corpus 的
# processed 链 fp32 输出（拍检测发生在 fp32 上，fp16 只是存储量化）——
# 要逐位复现 beats_full 必须重走 processed 链（build_info.json 同口径）。
ANKER_PROCESSED_ROOT = Path("<DATA_ROOT>/data_anker/processed")
_ANKER_FS_IN = 100       # processed 段库采样率 @100Hz
_ANKER_CH_MAIN = 22      # SLOT07_ADC01
_ANKER_POLARITY = -1.0   # 设备极性校正（build_info.polarity）
_ANKER_WIN_SAMPLES = 1000    # 10 s @100Hz
_ANKER_HEAD_TRIM = 2000      # run 头 20 s 滤波启动瞬态弃用区

_LOCK = threading.Lock()
_LIVE: "LiveStreamEngine | None" = None


class LiveStreamUnavailable(RuntimeError):
    """实时链不可用（torch 缺失 / ckpt 缺失 / 波形读取失败）→ 上层降级查表。"""


def _load_rep_ops():
    """BeatMAE 表示层算子（惰性；缺仓库即 LiveStreamUnavailable）。"""
    try:
        import sys
        root = str(resolve_beatmae_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        from data.beat_ops import (beat_bounds, beat_side_info, extract_beats,
                                   resample_beat, zscore)
        return zscore, extract_beats, beat_bounds, resample_beat, beat_side_info
    except Exception as e:  # noqa: BLE001 — 仓库/依赖缺失统一降级
        raise LiveStreamUnavailable(f"BeatMAE 表示层不可用: {e}") from e


class LiveStreamEngine:
    """段级实时流式推理（只读；头参数/离线嵌入复用 CausalStreamEngine 单例）。

    ``encoder`` 注入点供测试使用（callable(beats, pad, onset, side) -> (d,)）；
    生产路径惰性加载 engine.beatmae_encoder.BeatmaeEncoder（torch 依赖，缺失时抛
    LiveStreamUnavailable 由上层降级查表）。
    """

    def __init__(self, heads: CausalStreamEngine | None = None,
                 encoder=None, device: str | None = None):
        self._heads = heads
        self._encoder = encoder          # 测试注入；None = 惰性 BeatmaeEncoder
        self._enc_loaded = encoder is not None
        self.device = device
        self._emb_cache: dict[tuple, np.ndarray] = {}

    # ────────────────────────── 依赖 ──────────────────────────
    @property
    def heads(self) -> CausalStreamEngine:
        if self._heads is None:
            self._heads = get_causal_stream_engine()
        return self._heads

    @staticmethod
    def _ig_from_baseline(enc, dev, steps, beats_np, beats_t, pad, onset_t,
                          side_t, embed_grad_fn) -> np.ndarray:
        """沿 baseline=0 → 实际 beats 直线的 Integrated-Gradients 积分（L2/L2.5 共用）。

        ``embed_grad_fn(e_np) -> (d,)`` 提供某嵌入处对当前预测的头侧梯度
        ∂pred/∂e（当前段 current_embed_grad / 历史段 embed_grads_for_rows）；
        经 encoder autograd 反传到 ∂pred/∂beats，取 steps 个中点均值×beats
        得每拍每点贡献 (K,L)。两路归因共用，保证完备性口径逐位一致。"""
        import torch
        grads = []
        for i in range(int(steps)):
            alpha = (i + 0.5) / steps
            beats_a = (alpha * beats_t).detach().clone().requires_grad_(True)
            e_a = enc.embed_arrays_grad(beats_a, pad, onset_t, side_t)
            e_a_np = e_a[0].detach().float().cpu().numpy()
            g = np.asarray(embed_grad_fn(e_a_np))
            g_t = torch.from_numpy(np.ascontiguousarray(g[None])).to(dev).float()
            gb, = torch.autograd.grad(e_a, beats_a, grad_outputs=g_t)
            grads.append(gb[0].detach().float().cpu().numpy())
        return beats_np * np.mean(grads, axis=0)

    def _get_encoder(self):
        if self._enc_loaded:
            return self._encoder
        try:
            from .beatmae_encoder import BeatmaeEncoder
            from .weights import resolve_encoder_ckpt
        except Exception as e:  # noqa: BLE001
            raise LiveStreamUnavailable(f"torch/BeatmaeEncoder 不可用: {e}") from e
        try:
            ckpt = resolve_encoder_ckpt()
        except Exception as e:  # noqa: BLE001 — 无缓存/下载失败/完整性不符
            raise LiveStreamUnavailable(f"encoder 权重不可得: {e}") from e
        try:
            enc = BeatmaeEncoder(ckpt, device=self.device)
        except Exception as e:  # noqa: BLE001
            raise LiveStreamUnavailable(f"encoder 装配失败: {e}") from e
        # 运行时对账（Phase F）：装载的 encoder 版本 vs manifest 烙印。
        # 不一致不静默降级，而是显式告警（防上游 ckpt/嵌入/头三者的版本漂移）。
        declared = self.heads.manifest.get("encoder_version")
        if declared and enc.info["encoder_version"] != declared:
            import warnings
            warnings.warn(
                f"实时链 encoder={enc.info['encoder_version']} 与 manifest "
                f"烙印 {declared} 不一致：嵌入/头可能非同一模型产出。")
        self._encoder = enc
        self._enc_loaded = True
        return self._encoder

    # ────────────────────────── ① 读窗 ──────────────────────────
    def segment_signal(self, source: str, subject_id: str,
                       seg_idx: int) -> np.ndarray:
        """检测口径信号 (1250,) float64 @125Hz —— 与 beats_full 生成输入同精度。

        mimic/vital/aurora: raw mat 窗（fp64 原值）；
        anker: 重走 build_anker_corpus.process_subject 信号链（processed
        ppg_ppg.npy → run 划分 → 整条 sosfilt → to_corpus min-max+降采样），
        返回 fp32 精度（build 拍检测的输入；fp16 只是 corpus 存储量化，
        从 fp16 无法复现 fp32 检测结果，见本文件头部说明）。"""
        if source != "anker":
            path = self._waveform_path(source, subject_id, seg_idx)
            return self.load_window(path, self.window_index(source,
                                                             subject_id,
                                                             seg_idx))
        return self._anker_build_signals(subject_id, seg_idx)

    @staticmethod
    def _anker_build_signals(subject_id: str, seg_idx: int) -> np.ndarray:
        """ankera corpus 信号链逐位复刻（口径 = _staging_anker/
        build_anker_corpus.py + preprocess_unified.py，2026-09-02 定稿）。"""
        import pandas as pd
        from scipy.signal import butter, resample as _resample, sosfilt

        root = ANKER_PROCESSED_ROOT / subject_id
        segs = np.load(root / "ppg_ppg.npy", mmap_mode="r")
        man = pd.read_parquet(root / "manifest_ppg.parquet")
        si = int(seg_idx)                       # corpus seg_idx = man 行号
        if not 0 <= si < len(man):
            raise LiveStreamUnavailable(
                f"anker seg_idx 越界: {subject_id}/seg{si}")
        end = man["end_ts_ms"].to_numpy(np.int64)
        start = man["start_ts_ms"].to_numpy(np.int64)
        rid = np.zeros(len(man), dtype=int)
        for bi in np.where(np.abs(end[:-1] - start[1:]) > 50)[0]:
            rid[bi + 1:] += 1                   # gap>50ms 切 run
        members = np.where(rid == rid[si])[0]
        k = int(np.searchsorted(members, si))   # run 内序号 → off
        x = np.asarray(segs[members[0]:members[-1] + 1, _ANKER_CH_MAIN],
                       dtype=np.float64).reshape(-1)
        sos = butter(4, [0.35, 10], btype="band", fs=_ANKER_FS_IN,
                     output="sos")
        xf = sosfilt(sos, x * _ANKER_POLARITY - x[0] * _ANKER_POLARITY)
        off = k * _ANKER_WIN_SAMPLES
        if off < _ANKER_HEAD_TRIM:
            raise LiveStreamUnavailable(
                f"anker 段在滤波启动瞬态区（不应入库）: {subject_id}/seg{si}")
        mid100 = xf[off:off + _ANKER_WIN_SAMPLES]
        m = ((mid100 - mid100.min())
             / (mid100.max() - mid100.min() + 1e-12))
        return _resample(m, SEG_LEN).astype(np.float32)   # to_corpus 同款

    def window_index(self, source: str, subject_id: str, seg_idx: int) -> int:
        """raw .mat 内的窗号。

        mimic/vital/aurora: 窗号 = seg_idx（一一对应）；
        anker: corpus seg_idx 是 processed 全局索引（head_trim 20s 与 run
        间隙造成跳段，如 SZTU-ABPM001 seg_idx 从 2 起、8639 止而仅 8636 窗），
        mat 窗序为紧凑行序 → 窗号 = 该受试者有效段升序中的 rank
        （DB segments 即 corpus 全量行集，COUNT(seg_idx<k) 即 rank）。"""
        if source != "anker":
            return int(seg_idx)
        import sqlite3

        con = sqlite3.connect(str(PHENOTYPE_DB))
        try:
            row = con.execute(
                "SELECT COUNT(*) FROM segments "
                "WHERE source = 'anker' AND subject_id = ? AND seg_idx < ?",
                (subject_id, int(seg_idx))).fetchone()
        finally:
            con.close()
        return int(row[0])

    @staticmethod
    def load_window(waveform_path: str, seg_idx: int) -> np.ndarray:
        """raw .mat 单窗 (1250,) float64 @125Hz 原值 [0,1]。

        返回 float64：拍检测必须在原值精度上做（beats_full 即 float64
        路径产物；fp16 量化在表示层复现 corpus 分片存储，见 represent）。"""
        import h5py

        try:
            with h5py.File(waveform_path, "r") as f:
                ref = f["Subj_Wins"]["PPG_F"][0, int(seg_idx)]
                if isinstance(ref, np.ndarray):
                    v = ref.flatten()               # vlen 内联 / aurora 直接数组
                else:
                    v = f[ref][:].flatten()         # h5py object ref（PulseDB）
        except Exception as e:  # noqa: BLE001
            raise LiveStreamUnavailable(
                f"波形读取失败 {waveform_path}#win{seg_idx}: {e}") from e
        if v.size != SEG_LEN:
            raise LiveStreamUnavailable(
                f"窗口长度 {v.size} != {SEG_LEN} ({waveform_path}#win{seg_idx})")
        return np.asarray(v, dtype=np.float64)

    # ────────────────────────── ② 拍检测 ──────────────────────────
    @staticmethod
    def detect_beat_starts_125(seg_raw: np.ndarray, source: str) -> np.ndarray:
        """拍起点（125Hz 网格，int64）——与 beats_full parquet 同源口径。

        float64 原值 500Hz 重采样 → process_ppg_segment（官方 preset）→
        beat_start_sample(500Hz) ÷4；无有效拍返回空数组（调用方处理）。"""
        from scipy.signal import resample as _resample

        from .signal_engine import process_ppg_segment
        from ..processing_params import load_preset, preset_for_source

        ppg500 = _resample(np.asarray(seg_raw, dtype=np.float64), 5000)
        params = load_preset(preset_for_source(source, user_ok=False))
        if source == "aurora":
            params.update(AURORA_PLAN_B_PARAMS)   # beats_full 生成口径，见上
        res = process_ppg_segment(ppg500, fs=500, merge_fragments=True,
                                  trim_to_onset=True, params=params)
        starts500 = np.asarray(res["beat_starts"], dtype=np.float64)
        return np.round(starts500 / 4.0).astype(np.int64)

    # ────────────────────────── ③ 表示层 ──────────────────────────
    @staticmethod
    def represent(seg_raw: np.ndarray, starts125: np.ndarray,
                  max_beats: int = MAX_BEATS) -> dict:
        """CorpusDataset(beat_resampled) 同源表示 → encoder 输入三元组。

        信号先做 fp16 量化（复现 corpus 分片存储），再逐段 z-score。
        返回 {beats (K,128), onset (K,), side (K,2), n_beats_total,
        windowed, bounds (K,2), lens (K,)}；无有效拍抛
        LiveStreamUnavailable。bounds/lens 为 125Hz 网格上的真实拍窗，
        供特征点级归因在其上标定关键点。"""
        zscore, extract_beats, beat_bounds, resample_beat, beat_side_info = \
            _load_rep_ops()
        # corpus 信号以 fp16 分片存储 —— 实时链复现该量化以保证逐位对齐
        seg = zscore(np.asarray(seg_raw, dtype=np.float64)
                     .astype(np.float16).astype(np.float32))
        keep = extract_beats(np.asarray(starts125, dtype=np.int64))
        n_total = int(keep.size)
        windowed = n_total > max_beats
        if windowed:
            # corpus 为 rng(row) 随机连续窗；实时无 row 号 → 确定性首窗
            keep = keep[:max_beats]
        if keep.size == 0:
            raise LiveStreamUnavailable("无有效拍（拍长过滤后为空）")
        bounds = beat_bounds(np.asarray(starts125, dtype=np.int64), keep)
        lens = (bounds[:, 1] - bounds[:, 0]).astype(np.int64)
        beats = np.stack([resample_beat(seg[int(a):int(b)], 128)
                          for a, b in bounds])
        onset = (np.asarray(starts125, dtype=np.int64)[keep] / 125.0
                 ).astype(np.float32)
        side = beat_side_info(np.asarray(starts125, dtype=np.int64)[keep], lens)
        return {"beats": beats, "onset": onset, "side": side,
                "n_beats_total": n_total, "windowed": windowed,
                "bounds": bounds, "lens": lens}

    # ────────────────────────── ④ 编码 ──────────────────────────
    def embed_segment_live(self, source: str, subject_id: str,
                           seg_idx: int) -> np.ndarray:
        """单段实时嵌入 (d,)：读窗 → 拍检测 → 表示 → BeatMAE（带缓存）。"""
        key = (source, subject_id, int(seg_idx))
        if key in self._emb_cache:
            return self._emb_cache[key]
        seg125 = self.segment_signal(source, subject_id, seg_idx)
        starts125 = self.detect_beat_starts_125(seg125, source)
        rep = self.represent(seg125, starts125)
        emb = self._embed_arrays(rep)
        if self._emb_cache and len(self._emb_cache) >= 4096:
            self._emb_cache.clear()      # 简单防膨胀（进程内缓存，可丢弃）
        self._emb_cache[key] = emb
        return emb

    def _embed_arrays(self, rep: dict) -> np.ndarray:
        import torch

        enc = self._get_encoder()
        beats = torch.from_numpy(np.ascontiguousarray(rep["beats"][None]))
        pad = torch.zeros((1, rep["beats"].shape[0]), dtype=torch.bool)
        onset = torch.from_numpy(np.ascontiguousarray(rep["onset"][None]))
        side = torch.from_numpy(np.ascontiguousarray(rep["side"][None]))
        return enc.embed_arrays(beats, pad, onset, side).numpy()[0]

    # ────────────────────────── ⑤ 历史上下文 ──────────────────────────
    def _waveform_path(self, source: str, subject_id: str, seg_idx: int) -> str:
        import sqlite3

        con = sqlite3.connect(str(PHENOTYPE_DB))
        try:
            row = con.execute(
                "SELECT waveform_path FROM segments "
                "WHERE source = ? AND subject_id = ? AND seg_idx = ?",
                (source, subject_id, int(seg_idx))).fetchone()
        finally:
            con.close()
        if not row or not row[0]:
            raise LiveStreamUnavailable(
                f"DB 无波形路径: {source}/{subject_id}/seg{seg_idx}")
        return row[0]

    def _fold_of(self, source: str, subject_id: str) -> tuple:
        """(fold, fold_source)：离线 meta 优先，DB subjects 兜底。"""
        heads = self.heads
        rows = heads._subject_rows(source, subject_id)
        if len(rows) and "fold" in heads.meta.columns:
            f = heads.meta.loc[rows, "fold"].iloc[0]
            if f is not None and f == f:            # not NaN
                return f, "table"
        import sqlite3

        con = sqlite3.connect(str(PHENOTYPE_DB))
        try:
            row = con.execute(
                "SELECT fold_id FROM subjects WHERE source = ? AND subject_id = ?",
                (source, subject_id)).fetchone()
        finally:
            con.close()
        return (row[0] if row else None), "db"

    def _history_embeddings(self, source: str, subject_id: str,
                            seg_idx: int) -> tuple[np.ndarray, list[int], str, float | None]:
        """(hist (M,d), hist_seg_idxs (M,), 来源, 覆盖率)。

        来源 table = 离线嵌入表（与实时链逐位校验等价）；
        live = 不在表内时实时补算（近 MAX_LIVE_HISTORY 段全算，
        c_long 为该子集前缀均值 → 覆盖率 = M/可用前序段数）。
        hist_seg_idxs 与 hist 行一一对应（seg_idx 升序），供近窗注意力归因。"""
        heads = self.heads
        hist, hist_idx = heads.subject_history(source, subject_id, seg_idx)
        if len(hist_idx) or heads._subject_rows(source, subject_id).size:
            return hist, hist_idx, "table", None
        # 受试者不在嵌入表 → 实时补算前序段（预算内，近窗优先）
        segs = self._previous_seg_idxs(source, subject_id, int(seg_idx))
        if not segs:
            return (np.zeros((0, heads.d_model), dtype=np.float32),
                    [], "none", None)
        use = segs[-MAX_LIVE_HISTORY:]
        embs = [self.embed_segment_live(source, subject_id, s) for s in use]
        cov = len(use) / len(segs)
        return (np.asarray(embs, dtype=np.float32), [int(s) for s in use],
                "live", round(cov, 3))

    @staticmethod
    def _previous_seg_idxs(source: str, subject_id: str,
                           seg_idx: int) -> list[int]:
        """DB 中该受试者 seg_idx < k 的非 failed 段号（升序）。"""
        import sqlite3

        con = sqlite3.connect(str(PHENOTYPE_DB))
        try:
            rows = con.execute(
                "SELECT seg_idx FROM segments "
                "WHERE source = ? AND subject_id = ? AND seg_idx < ? "
                "AND quality_tier != 'failed' ORDER BY seg_idx",
                (source, subject_id, int(seg_idx))).fetchall()
        finally:
            con.close()
        return [int(r[0]) for r in rows]

    # ────────────────────────── 主入口 ──────────────────────────
    def predict_segment_stream(self, source: str, subject_id: str,
                               seg_idx: int) -> dict:
        """段级实时流式预测（主口径）：当前段实时编码 + 历史上下文 → 出值。"""
        seg_idx = int(seg_idx)
        hist, hist_idx, hist_src, cov = self._history_embeddings(
            source, subject_id, seg_idx)
        e_cur = self.embed_segment_live(source, subject_id, seg_idx)
        d = self.heads.d_model
        if e_cur.shape[0] != d:
            raise LiveStreamUnavailable(
                f"实时嵌入维度 {e_cur.shape[0]} != 头 d_model {d} "
                "(encoder 版本与固化头不一致)")
        x = np.concatenate([hist, e_cur[None].astype(np.float32)], axis=0)
        fold, _fold_src = self._fold_of(source, subject_id)
        out = self.heads.predict_segment_from_sequence(
            x, source, fold=fold, engine_label="live_stream",
            hist_seg_idxs=hist_idx)
        out["history_source"] = hist_src
        if cov is not None:
            out["history_coverage"] = cov
        out["seg_idx"] = seg_idx
        if hist_src == "live":
            out["history_note"] = (
                f"受试者不在离线嵌入表，历史为实时补算（近 {len(hist)} 段，"
                f"覆盖率 {cov}；c_long 为子集前缀均值近似）")
        elif hist_src == "none":
            out["history_note"] = "首段/无可用历史：e_k-only 降级模式"
        return out

    def measured_bp(self, source: str, subject_id: str, seg_idx: int) -> dict:
        """该段实测血压标签（复用 MlPredictEngine 口径，供对照展示）。"""
        from .ml_predict import measured_bp as _measured_bp

        return _measured_bp(source, subject_id, seg_idx)

    # ────────────────────── L2 波形归因（IG） ──────────────────────
    def segment_waveform_attribution(self, source: str, subject_id: str,
                                     seg_idx: int, target: str,
                                     *, steps: int = 32) -> dict:
        """当前段波形的特征点级 Integrated-Gradients 归因（L2）。

        回答“模型对当前段波形的哪些生理特征点（onset/收缩峰/重搏
        切迹/潮波…）最敏感”：头侧解析 ∂pred/∂e_k（历史固定）→ 作为
        grad_outputs 喂 encoder 可微通道→ ∂pred/∂beats；沿 baseline=0
        （z 空间即段均值）→ 实际 beats 的直线路径中点积分得逐点 IG。
        再对每拍的 125Hz 真实拍窗跑 extract_beat_keypoints 标 9 个关键点，
        按“最近关键点”将该拍逐点 IG 划分到各特征点（保持完备：
        Σ特征点贡献 = ig_total）。

        需 GPU+encoder（实时链）；不可用时抛 LiveStreamUnavailable（由
        上层降级/提示）。输出为群体模型对输入的敏感度，非生理因果。"""
        import torch

        heads = self.heads
        seg_idx = int(seg_idx)
        if target not in heads.targets:
            raise ValueError(f"未知 target {target}（可选 {heads.targets}）")
        enc = self._get_encoder()          # 不可用则抛 LiveStreamUnavailable
        dev = enc.device
        # ① 当前段表示（与实时链同源）
        seg125 = self.segment_signal(source, subject_id, seg_idx)
        starts125 = self.detect_beat_starts_125(seg125, source)
        rep = self.represent(seg125, starts125)
        beats_np = np.ascontiguousarray(rep["beats"], dtype=np.float32)  # (K,L)
        K, L = beats_np.shape
        # ② 当前段表示→组装 x=[hist; e_k]；每步重新取 e_α 处的头梯度
        pad = torch.zeros((1, K), dtype=torch.bool, device=dev)
        onset_t = torch.from_numpy(np.ascontiguousarray(rep["onset"][None])).to(dev).float()
        side_t = torch.from_numpy(np.ascontiguousarray(rep["side"][None])).to(dev).float()
        beats_base = torch.from_numpy(beats_np[None]).to(dev).float()
        hist, _hi, hist_src, _cov = self._history_embeddings(
            source, subject_id, seg_idx)
        domain = "anker" if source == "anker" else "full"
        fold, _fs = self._fold_of(source, subject_id)
        fold_ids = heads._fold_ids(domain, fold)

        def _x_of(e_np):
            return (np.concatenate([hist, e_np[None]], axis=0)
                    if len(hist) else e_np[None])

        # ③ IG：沿 baseline=0 → beats 中点路积分。每个 α 处重算
        #   ∂pred/∂e_k|_{e_α}（头 MLP 自身非线性），再经 encoder autograd 得
        #   ∂pred/∂beats——保证复合函数 f(beats)=pred 的梯度，使完备性成立。
        ig = self._ig_from_baseline(
            enc, dev, steps, beats_np, beats_base, pad, onset_t, side_t,
            lambda e_np: heads.current_embed_grad(_x_of(e_np), domain,
                                                  fold_ids, target))  # (K,L)
        # ④ 完备性公理：Σ ig ≈ pred(beats) - pred(baseline)
        with torch.no_grad():
            zero = torch.zeros_like(beats_base)
            e_zero = enc.embed_arrays_grad(zero, pad, onset_t, side_t)[0].float().cpu().numpy()
            e_full = enc.embed_arrays_grad(beats_base, pad, onset_t, side_t)[0].float().cpu().numpy()
        pred_a = self._head_pred_scalar(_x_of(e_full), domain, fold_ids, target)
        pred_b = self._head_pred_scalar(_x_of(e_zero), domain, fold_ids, target)
        # ⑤ 特征点级划分（与历史段归因共用同一 helper）
        feature_points, n_ok, beat_points = self._feature_point_partition(
            ig, beats_np, seg125, rep["bounds"], L)
        return {
            "target": target,
            "attrib_method": "feature_point_integrated_gradients",
            "baseline": "zero (z 空间→段均值参考)",
            "steps": int(steps),
            "n_beats_used": int(K), "beat_len": int(L),
            "n_beats_total": int(rep["n_beats_total"]),
            "windowed": bool(rep["windowed"]),
            "beats_with_keypoints": int(n_ok),
            "keypoint_tool": "extract_beat_keypoints@125Hz (vitalpy)",
            "feature_points": feature_points,
            "beat_points": beat_points,
            "ig_total": round(float(ig.sum()), 3),
            "pred_actual": round(float(pred_a), 2),
            "pred_baseline": round(float(pred_b), 2),
            "completeness_gap": round(float(pred_a - pred_b - ig.sum()), 3),
            "history_source": hist_src,
            "note": ("当前段波形对该 target 的特征点级 IG 敏感度：每拍逐点 IG 按"
                     "最近生理关键点划分（Σ=ig_total，已沿 0→实际积分）；属群体"
                     "模型输入依赖度，非个体生理病因，不作诊断。"),
        }

    # ────────────────── 历史段特征点级归因（L1×L2） ──────────────────
    def _segment_representation(self, source: str, subject_id: str,
                                seg_idx: int) -> tuple[np.ndarray, dict]:
        """(seg125, rep)：读窗→拍检测→表示（与实时链同源），
        供当前/历史段归因复用。"""
        seg125 = self.segment_signal(source, subject_id, seg_idx)
        starts125 = self.detect_beat_starts_125(seg125, source)
        rep = self.represent(seg125, starts125)
        return seg125, rep

    @staticmethod
    def _feature_point_partition(ig, beats_np, seg125, bounds, L):
        """逐点 IG → 特征点级划分：每拍在其 125Hz 真实拍窗上标 9 关键点，
        按“最近关键点”把该拍逐点 IG 完整划分到各生理特征点（Σ = ig.sum()）。
        关键点失败/过短拍退化为 (o,s,v) 三分点，保证不遗漏。

        返回 ``(feature_points, n_ok, beat_points)``：
          - ``feature_points``：按特征点名**跨全段聚合**的签名贡献（Σ=ig_total，
            按重要度排序）——回答“整体最关注哪个特征点”；
          - ``beat_points``：逐 (拍, 关键点) 的签名贡献 + 该关键点在本段内的
            **绝对时间(秒)**，供前端把梯度归因**定位到具体拍的具體关键点**
            （而非用聚合值把每个 beat 同色高亮）；某关键点在该拍 IG≈0 则贡献小、
            自动不显著。
        """
        from .signal_engine import extract_beat_keypoints
        K = int(ig.shape[0])
        fs_beat = 125.0  # seg125 / bounds 的采样率口径
        fp_total = {n: 0.0 for n in FP_ORDER}
        beat_points: list[dict] = []
        n_ok = 0
        for k in range(K):
            a_i, b_i = int(bounds[k, 0]), int(bounds[k, 1])
            blen = b_i - a_i
            centers = None
            if blen >= 8:
                try:
                    _v, kp = extract_beat_keypoints(
                        np.asarray(seg125[a_i:b_i], dtype=np.float64), fs=125)
                    pts = []
                    for name in FP_ORDER:
                        idx125 = kp.get(name)
                        if idx125 is None:
                            continue
                        i128 = int(round(float(idx125) * (L - 1) / (blen - 1)))
                        pts.append((min(max(i128, 0), L - 1), name, float(idx125)))
                    if pts:
                        pts.sort()
                        centers = pts
                        n_ok += 1
                except Exception:  # noqa: BLE001 — 关键点失败退化为三分点
                    centers = None
            if centers is None:            # 保证该拍 IG 完整归属
                s_idx = int(np.argmax(beats_np[k]))
                sc = min(max(s_idx, 1), L - 2)
                centers = [(0, "o", 0.0),
                           (sc, "s", sc * (blen - 1) / max(L - 1, 1)),
                           (L - 1, "v", float(blen - 1))]
            names = [nm for _c, nm, _i in centers]
            cidx = [c for c, _nm, _i in centers]
            row = ig[k]
            per_beat: dict[str, float] = {}
            for j in range(L):
                u = min(range(len(cidx)), key=lambda w: abs(j - cidx[w]))
                nm = names[u]
                val = float(row[j])
                fp_total[nm] += val
                per_beat[nm] = per_beat.get(nm, 0.0) + val
            # 逐 (拍, 关键点) 贡献 + 绝对时间(秒)：前端据此定位高亮
            for (_c, nm, idx125) in centers:
                contrib = per_beat.get(nm)
                if contrib is None or abs(contrib) < 1e-6:
                    continue
                beat_points.append({
                    "beat_idx": int(k), "name": nm, "label": FP_LABELS[nm],
                    "contrib": round(contrib, 3),
                    "t": round((a_i + float(idx125)) / fs_beat, 4),
                })
        feature_points = [
            {"name": n, "label": FP_LABELS[n],
             "contrib": round(fp_total[n], 3),
             "importance": round(abs(fp_total[n]), 3)} for n in FP_ORDER]
        feature_points.sort(key=lambda r: r["importance"], reverse=True)
        return feature_points, n_ok, beat_points

    def segment_history_attribution(self, source: str, subject_id: str,
                                    seg_idx: int, target: str, *,
                                    top: int = 3, steps: int = 24) -> dict:
        """近窗注意力 Top-`top` 历史段的特征点级归因（L1×L2 融合，二阶）。

        当前段预测对某历史段 j 的敏感度经两路：∂pred/∂e_j =
        α_j·∂pred/∂c_recent + (1/M)·∂pred/∂c_long（embed_grads_for_rows，
        注意力权重视为常数）。以此作为 grad_outputs 沿 baseline=0→实际
        beats_j 积分得 IG，再划分到 j 的生理关键点。基线=将 e_j 置零，
        ig_total ≈ pred − pred_without_j。需 GPU+encoder。"""
        import torch

        heads = self.heads
        seg_idx = int(seg_idx)
        if target not in heads.targets:
            raise ValueError(f"未知 target {target}（可选 {heads.targets}）")
        if not 1 <= int(top) <= 5:
            raise ValueError("top ∈ [1,5]")
        enc = self._get_encoder()
        dev = enc.device
        hist, hist_idx, hist_src, _cov = self._history_embeddings(
            source, subject_id, seg_idx)
        if not len(hist_idx):
            return {"target": target, "top": int(top), "segments": [],
                    "history_source": hist_src, "note": "首段/无可用历史：无近窗可归因"}
        e_cur = self.embed_segment_live(source, subject_id, seg_idx)
        x_base = np.concatenate(
            [hist, e_cur[None].astype(np.float32)], axis=0)   # (M+1, d)
        domain = "anker" if source == "anker" else "full"
        fold, _fs = self._fold_of(source, subject_id)
        fold_ids = heads._fold_ids(domain, fold)
        att = heads.collect_history_attention(
            x_base, domain, fold_ids, hist_idx, top=int(top))
        pairs = (att.get(target) or [])[:int(top)]
        if not pairs:
            return {"target": target, "top": int(top), "segments": [],
                    "history_source": hist_src,
                    "note": "近窗注意力无可归因历史段（权重过低）"}
        hist_arr = np.asarray(hist_idx)
        results = []
        for item in pairs:
            sidx = int(item["seg_idx"])
            weight = float(item["weight"])
            r = int(np.where(hist_arr == sidx)[0][0])
            seg125_j, rep_j = self._segment_representation(
                source, subject_id, sidx)
            beats_j = np.ascontiguousarray(rep_j["beats"], dtype=np.float32)
            Kj, Lj = beats_j.shape
            pad = torch.zeros((1, Kj), dtype=torch.bool, device=dev)
            onset_t = torch.from_numpy(np.ascontiguousarray(rep_j["onset"][None])).to(dev).float()
            side_t = torch.from_numpy(np.ascontiguousarray(rep_j["side"][None])).to(dev).float()
            bb = torch.from_numpy(beats_j[None]).to(dev).float()

            def _xs_with(e_r, _r=r):
                xs = x_base.copy()
                xs[_r] = e_r
                return xs

            ig = self._ig_from_baseline(
                enc, dev, steps, beats_j, bb, pad, onset_t, side_t,
                lambda e_np, _r=r: heads.embed_grads_for_rows(
                    _xs_with(e_np), domain, fold_ids, target, [_r])[_r])
            feature_points, n_ok, beat_points = self._feature_point_partition(
                ig, beats_j, seg125_j, rep_j["bounds"], Lj)
            # IG 基线：零波形过 encoder 的嵌入 e_zero（≠ 0 向量，encoder 有
            #   bias/LayerNorm），与当前段同口径；ig_total ≈ pred_actual − pred_baseline。
            with torch.no_grad():
                zero = torch.zeros_like(bb)
                e_full = enc.embed_arrays_grad(
                    bb, pad, onset_t, side_t)[0].float().cpu().numpy()
                e_zero = enc.embed_arrays_grad(
                    zero, pad, onset_t, side_t)[0].float().cpu().numpy()
            pred_a = self._head_pred_scalar(_xs_with(e_full), domain, fold_ids, target)
            pred_b = self._head_pred_scalar(_xs_with(e_zero), domain, fold_ids, target)
            results.append({
                "seg_idx": sidx, "attention_weight": round(weight, 4),
                "n_beats_used": int(Kj),
                "n_beats_total": int(rep_j["n_beats_total"]),
                "windowed": bool(rep_j["windowed"]),
                "beats_with_keypoints": int(n_ok),
                "feature_points": feature_points,
                "beat_points": beat_points,
                "ig_total": round(float(ig.sum()), 3),
                "pred_actual": round(float(pred_a), 2),
                "pred_baseline": round(float(pred_b), 2),
                "completeness_gap": round(float(pred_a - pred_b - ig.sum()), 3),
            })
        return {
            "target": target, "top": int(top), "current_seg_idx": seg_idx,
            "attrib_method": "history_attention_feature_point_ig",
            "order": "by_attention_weight", "steps": int(steps),
            "segments": results, "history_source": hist_src,
            "note": ("当前段预测对近窗注意力 Top-N 历史段的特征点级敏感度"
                     "（注意力加权二阶效应：经 c_recent·α + c_long·1/M 两路，"
                     "α 视为常数）。基线=零波形过 encoder 的嵌入 e_zero，"
                     "ig_total ≈ pred_actual − pred_baseline（completeness_gap 因 α"
                     "常数近似可非零）。属群体模型依赖度，非个体生理病因，不作诊断。"),
        }

    @staticmethod
    def _head_pred_scalar(x, domain, fold_ids, target) -> float:
        """末行当前段预测均值（未取整）：Σ_head (z·sd+mu)/cnt，与头口径一致。"""
        heads = get_causal_stream_engine()
        acc, cnt = 0.0, 0
        for k in fold_ids:
            for s in heads.infer_seeds:
                h = heads._head(domain, k, target, s)
                mu, sd = heads._mu_sd(domain, k, target, s)
                acc += heads._forward_std_last_row(x, h) * sd + mu
                cnt += 1
        return acc / max(cnt, 1)


def get_live_stream_engine() -> "LiveStreamEngine":
    """进程内单例（线程安全；依赖缺失时抛 LiveStreamUnavailable 由上层降级）。"""
    global _LIVE
    if _LIVE is None:
        with _LOCK:
            if _LIVE is None:
                _LIVE = LiveStreamEngine()
    return _LIVE
