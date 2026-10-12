"""corpus_dataset：corpus_v2c 语料的内存映射 Dataset 与 collate。

语料布局（根路径由调用方传入，候选见 data/paths.py 的 CORPUS_V2C）：
    signals/{ppg,ecg,meta}_shard_{0000..0049}.npy   (100000, 1, 1250) fp16
    keypoints.npz   {offsets(N+1,), keypoints(T,10) int16, kp_valid(T,) u8}  CSR
    beat_features.npz {offsets, features(T,40) fp16, feat_valid(T,) u8}      CSR
    manifest.parquet  row/source/fold/shard/row_in_shard/sig_ok/...
    seg_features.npy (N,61) fp16 / seg_feat_valid.npy (N,) u8
        （preprocess/03 建库产物，登记在 04；训练/评估链不读，留作事后分析）

设计：
- 全部走只读 memmap（npz 成员按 zip 内偏移直接映射，不解压不占常驻内存）；
  DataLoader fork 出的 worker 通过 COW 共享。
- 初始化时一次性向量化预筛：拍长过滤后的"有效拍数 ≥ min_beats"，
  运行期 __getitem__ 不再丢段。
- 输出单位 = 1 段：拍波形已定长重采样、关键点已换算到 L 尺度、逐段 z-score。
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .beat_ops import (FS, KP_COLS, SEG_LEN, START_COL, beat_bounds,
                       beat_side_info, extract_beats, resample_beat,
                       rescale_keypoints, zscore)

N_FEATURES = 40       # 拍级形态特征维数（features_meta.json）

# 段输入窗形状 —— 全仓**唯一权威**（2026-09-11 收敛；此前 5 处各写一份字面量）。
# MAX_BEATS 拍 × BEAT_LEN 定长 = 4096 点拼接波形（对比模型 seg_len）；
# BeatMAE 走拍级 token 不拼接，但同样受 MAX_BEATS 约束。
# 连带面：train/probe.py、train/pretrain_baseline.py、scripts/extract_embeddings.py
# 全部 import 本处；train/pretrain.py 与 train/pretrain_baseline.py
# 另从 yaml 的 data.max_beats 取生效值（缺省即本常量）。
# ⚠ 改这两个数**任何路径都不会报错**：超上限段按 rng 随机连续窗截取（见 __getitem__），
# 而 BeatMAE 无定长位置编码 → 拿旧 ckpt 提取嵌入会静默换输入拍。改前先核对 run_meta.json。
MAX_BEATS = 32
BEAT_LEN = 128

# ⚠ 坐标基准修正（2026-08-25 实测确认）：
# beats_full 的 beat_start_sample 与拍内关键点坐标均为源采样率 500Hz 网格，
# 而语料信号为 125Hz（1250 样本/10s）。三库统一 ÷4 换算（HR 对齐误差 ≤2bpm，
# 关键点形态与波形峰谷对齐验证通过）。features_meta.json 中“125Hz”表述有误。
KP_SRC_FACTOR = 4

__all__ = ['memmap_npz_member', 'CorpusDataset', 'beat_collate', 'raw_collate',
           'N_FEATURES', 'KP_SRC_FACTOR', 'MAX_BEATS', 'BEAT_LEN']


def memmap_npz_member(npz_path: str, member: str) -> np.memmap:
    """对 npz 内未压缩成员建立只读 memmap（避免整包加载）。

    注：corpus_v2 的 keypoints/features 成员以 Fortran 序存储（物化时
    保留），按 header 标志自动选择 order；行切片语义与 np.load 一致。
    """
    name = member if member.endswith('.npy') else member + '.npy'
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(name)
    assert info.compress_type == zipfile.ZIP_STORED, \
        f'{npz_path}:{name} 非 ZIP_STORED，无法 memmap'
    with open(npz_path, 'rb') as f:
        f.seek(info.header_offset)
        hdr = f.read(30)
        assert hdr[:4] == b'PK\x03\x04'
        fname_len = int.from_bytes(hdr[26:28], 'little')
        extra_len = int.from_bytes(hdr[28:30], 'little')
        f.seek(info.header_offset + 30 + fname_len + extra_len)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        return np.memmap(npz_path, dtype=dtype, shape=shape,
                         offset=f.tell(), mode='r',
                         order='F' if fortran else 'C')


class CorpusDataset(Dataset):
    """corpus_v2 段级 Dataset。

    Args:
        root:         语料根目录
        exclude_folds: 排除折（严格模式预训练用 {1,2}）
        include_folds: 只保留指定折（验证集用 {1,2}，与 exclude 二选一）
        sources:      限定源（None=全部 mimic/vital/aurora）
        min_beats/max_beats: 每段有效拍数下限 / 返回拍数上限（超上限随机连续窗）。
                           max_beats 缺省 = 模块常量 MAX_BEATS（全仓唯一权威）
        min_beat_len/max_beat_len: 拍长过滤（样本 @125Hz）
        beat_len:     拍定长重采样长度，缺省 = 模块常量 BEAT_LEN
        input_mode:   'beat_resampled'=拍定长重采样视图（BeatMAE 用）；
                      'raw_waveform'=原始 125Hz 整段波形直喂（对比模型公平口径 v2）。
                      两模式共享同一段池预筛/折/标签，拍重采样是模型自有表示选择。
        max_rows:     调试用，限制段数
    """

    def __init__(self, root: str, exclude_folds=(), include_folds=None,
                 sources=None,
                 min_beats: int = 1, max_beats: int = MAX_BEATS,
                 min_beat_len: int = 20, max_beat_len: int = 250,
                 beat_len: int = BEAT_LEN, max_rows: int | None = None,
                 input_mode: str = 'beat_resampled'):
        assert input_mode in ('beat_resampled', 'raw_waveform'), \
            f'未知 input_mode: {input_mode}'
        self.input_mode = input_mode
        self.root = Path(root)
        self.beat_len = beat_len
        self.max_beats = max_beats
        self.min_beat_len = min_beat_len
        self.max_beat_len = max_beat_len

        # ---------- manifest 过滤 ----------
        man = pd.read_parquet(
            self.root / 'manifest.parquet',
            columns=['row', 'source', 'fold', 'shard', 'row_in_shard',
                     'sig_ok', 'subject_id', 'quality_tier'])
        if max_rows is not None:
            # 均匀抽样（避免 head 只取到单一源：manifest 按源排序）
            idx = np.linspace(0, len(man) - 1, min(max_rows, len(man))).astype(int)
            man = man.iloc[idx]
        m = man['sig_ok'] == 1
        if include_folds is not None:
            m &= man['fold'].isin(list(include_folds))
        elif exclude_folds:
            m &= ~man['fold'].isin(list(exclude_folds))
        if sources:
            m &= man['source'].isin(list(sources))
        man = man[m].reset_index(drop=True)

        # ---------- CSR memmap ----------
        kp_path = str(self.root / 'keypoints.npz')
        self.offsets = np.asarray(memmap_npz_member(kp_path, 'offsets'))
        self.keypoints = memmap_npz_member(kp_path, 'keypoints')
        self.kp_valid = memmap_npz_member(kp_path, 'kp_valid')
        bf_path = str(self.root / 'beat_features.npz')
        off_f = np.asarray(memmap_npz_member(bf_path, 'offsets'))
        assert np.array_equal(off_f, self.offsets), 'keypoints/beat_features offsets 不一致'
        self.features = memmap_npz_member(bf_path, 'features')
        self.feat_valid = memmap_npz_member(bf_path, 'feat_valid')

        # ---------- PPG 分片 memmap ----------
        self.ppg_shards = {}
        for sh in man['shard'].unique():
            self.ppg_shards[int(sh)] = np.load(
                self.root / 'signals' / f'ppg_shard_{int(sh):04d}.npy',
                mmap_mode='r')

        # ---------- 向量化预筛：拍长过滤后有效拍数 ----------
        rows = man['row'].to_numpy()
        n_seg = self.offsets.shape[0] - 1
        seg_len_beats = np.diff(self.offsets)
        # 空段（0 拍，语料中约 0.5%）直接排除，避免 CSR 跨界链式计算错位：
        # 若空段夹在两段之间，全局 ends 会把"下一段首拍"错链给"上一段末拍"
        seg_has_beat = seg_len_beats > 0
        beat_seg = np.repeat(np.arange(n_seg), seg_len_beats)   # 每拍→所属段
        starts_all = np.round(np.asarray(self.keypoints[:, START_COL],
                                         dtype=np.float64) / KP_SRC_FACTOR
                              ).astype(np.int64)
        n_total = starts_all.shape[0]
        ends_all = np.empty(n_total, dtype=np.int64)
        ends_all[:-1] = starts_all[1:]
        ends_all[-1] = SEG_LEN
        # 段边界修正：每段末拍的 end = SEG_LEN（按真实拍→段归属，含空段安全）
        is_seg_last = np.ones(n_total, dtype=bool)
        is_seg_last[:-1] = beat_seg[:-1] != beat_seg[1:]
        ends_all[is_seg_last] = SEG_LEN
        lens_all = ends_all - starts_all
        valid_beat = (lens_all >= min_beat_len) & (lens_all <= max_beat_len) \
            & (starts_all >= 0) & (ends_all <= SEG_LEN)
        counts = np.bincount(beat_seg[valid_beat], minlength=n_seg)
        keep = (counts[rows] >= min_beats) & seg_has_beat[rows]
        man = man[keep].reset_index(drop=True)
        self.n_valid_beats = counts[rows][keep]

        # ---------- 定稿索引 ----------
        self.rows = man['row'].to_numpy(np.int64)
        self.shard = man['shard'].to_numpy(np.int64)
        self.row_in_shard = man['row_in_shard'].to_numpy(np.int64)
        self.source = man['source'].to_numpy()
        self.fold = man['fold'].to_numpy(np.int64)
        self.subject_id = man['subject_id'].to_numpy()
        self.offsets_kept = self.offsets[self.rows]       # (N,)

    # ------------------------------------------------------------
    def __len__(self):
        return self.rows.shape[0]

    def __getitem__(self, i):
        row = int(self.rows[i])
        seg = np.asarray(self.ppg_shards[int(self.shard[i])][
            self.row_in_shard[i], 0], dtype=np.float32)

        if self.input_mode == 'raw_waveform':
            # 公平口径 v2：125Hz 原始整段波形直喂，不经拍重采样，
            # 心率的绝对时间尺度完整保留；段池与 beat 模式逐段一致
            return {'waveform': zscore(seg), 'row': row,
                    'source': self.source[i], 'fold': int(self.fold[i]),
                    'subject_id': self.subject_id[i]}

        seg = zscore(seg)

        o0, o1 = int(self.offsets_kept[i]), int(self.offsets[row + 1])
        kp_raw = np.asarray(self.keypoints[o0:o1])
        starts = np.round(kp_raw[:, START_COL].astype(np.float64)
                          / KP_SRC_FACTOR).astype(np.int64)   # 500Hz → 125Hz
        keep = extract_beats(starts, self.min_beat_len, self.max_beat_len)

        # 超上限：随机连续窗（按 row 固定种子，跨 epoch 确定性可再叠加）
        if keep.shape[0] > self.max_beats:
            rng = np.random.default_rng(row)
            s = int(rng.integers(0, keep.shape[0] - self.max_beats + 1))
            keep = keep[s:s + self.max_beats]

        bounds = beat_bounds(starts, keep)
        lens = (bounds[:, 1] - bounds[:, 0]).astype(np.int64)
        K = keep.shape[0]

        beats = np.stack([
            resample_beat(seg[int(a):int(b)], self.beat_len)
            for a, b in bounds])

        kpv = np.asarray(self.kp_valid[o0:o1], dtype=bool)[keep]
        # 拍内关键点同样从 500Hz 网格换算到 125Hz，再缩放到定长 L（无效保持 -1）
        kp_src = kp_raw[keep, :KP_COLS].astype(np.float64)
        kp_src125 = np.where(kp_src >= 0, np.round(kp_src / KP_SRC_FACTOR), -1)
        kp_r = rescale_keypoints(kp_src125.astype(np.int64),
                                 lens, self.beat_len, kpv)
        kp_out = np.concatenate(
            [kp_r, starts[keep, None].astype(np.int64)], axis=1)

        onset = (starts[keep] / FS).astype(np.float32)
        side = beat_side_info(starts[keep], lens)

        feat = np.asarray(self.features[o0:o1][keep], dtype=np.float32)
        # 非有限值 → 0（NaN 是既有语义；±Inf 是 corpus 早期构建的残留，现役语料
        # 已在建库时原地 clip，此处仅兜底。修复脚本是当时的一次性件，已出仓）
        feat = np.where(np.isfinite(feat), feat, 0.0)
        fv = np.asarray(self.feat_valid[o0:o1], dtype=bool)[keep]

        return {
            'beats': beats,                       # (K, L)
            'keypoints': kp_out,                  # (K, 10) 9 换算坐标 + start_sample
            'kp_valid': kpv,                      # (K,)
            'onset': onset,                       # (K,) 秒
            'side': side,                         # (K, 2) [beat_len_ms, hr_bpm]
            'feat': feat,                         # (K, 40)
            'feat_valid': fv,                     # (K,)
            'row': row,
            'source': self.source[i],
            'fold': int(self.fold[i]),
            'subject_id': self.subject_id[i],
        }


def raw_collate(batch: list) -> torch.Tensor:
    """raw_waveform 模式 collate：固定 1250 点原始波形，无需 padding。

    返回裸张量 (B, 1250) float32（与 waveform_collate 约定一致，训练循环直接 .to(device)）。
    窗口时间区间与 beat 模式一致（同一 10s 段），差异仅在表示层（无拍重采样）。
    """
    return torch.from_numpy(
        np.stack([np.asarray(it['waveform'], dtype=np.float32) for it in batch]))


def beat_collate(batch: list) -> dict:
    """变拍数 padding → batch 张量。

    返回：
        beats (B,K,L) float32 / keypoints (B,K,10) long / kp_valid (B,K) bool
        onset (B,K) / side (B,K,2) / feat (B,K,F) / feat_valid (B,K) bool
        beat_pad_mask (B,K) bool，True=padding 拍
        meta: dict(source/fold/subject_id/row 列表)
    """
    K = max(item['beats'].shape[0] for item in batch)
    B, L, F = len(batch), batch[0]['beats'].shape[1], batch[0]['feat'].shape[1]

    beats = torch.zeros(B, K, L)
    keypoints = torch.full((B, K, 10), -1, dtype=torch.long)
    kp_valid = torch.zeros(B, K, dtype=torch.bool)
    onset = torch.zeros(B, K)
    side = torch.zeros(B, K, 2)
    feat = torch.zeros(B, K, F)
    feat_valid = torch.zeros(B, K, dtype=torch.bool)
    pad = torch.ones(B, K, dtype=torch.bool)

    for b, it in enumerate(batch):
        k = it['beats'].shape[0]
        beats[b, :k] = torch.from_numpy(it['beats'])
        keypoints[b, :k] = torch.from_numpy(it['keypoints'])
        kp_valid[b, :k] = torch.from_numpy(it['kp_valid'])
        onset[b, :k] = torch.from_numpy(it['onset'])
        side[b, :k] = torch.from_numpy(it['side'])
        feat[b, :k] = torch.from_numpy(it['feat'])
        feat_valid[b, :k] = torch.from_numpy(it['feat_valid'])
        pad[b, :k] = False

    meta = {
        'source': [it['source'] for it in batch],
        'fold': [it['fold'] for it in batch],
        'subject_id': [it['subject_id'] for it in batch],
        'row': [it['row'] for it in batch],
    }
    return {'beats': beats, 'keypoints': keypoints, 'kp_valid': kp_valid,
            'onset': onset, 'side': side, 'feat': feat,
            'feat_valid': feat_valid, 'beat_pad_mask': pad, 'meta': meta}
