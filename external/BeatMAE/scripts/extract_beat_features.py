#!/usr/bin/env python3
"""extract_beat_features.py — 从 PPG 段提取 9 个关键点 + 40 维形态学特征。

背景
    BeatMAE 的预训练语料以 `features_v2/*_beats_full.parquet` 为起点（见
    `preprocess/README.md`），而该表的**生产者原先不在本仓**。本脚本补上这一环，
    让「原始 PPG → 40 维特征」这条链在本仓内闭合。

    40 维特征的核心计算派生自 VitalPy（GPL-3.0），见 `third_party/vitalpy/`。

⚠ 输入前提（最容易踩的坑）
    本管线**不做滤波**（上游把它放在读数据那一步）。送进来的 PPG 必须**已经是
    0.25–10 Hz 带通 + 基线校正过的 500 Hz 信号**。给未滤波的信号会得到错误结果，
    而且不会报错。

    `--fs` 给的是输入采样率，脚本会重采样到 500 Hz（10 s 段 → 5000 点）。

⚠ 关于 ptt_ppg_ms
    40 维之一，需要 ECG R 峰。不给 `--rpeaks` 时该列恒为 None，其余 39 维不受影响。

用法
    # 零数据试跑（合成信号，用于确认环境装对了）
    python scripts/extract_beat_features.py --demo --out /tmp/beats.parquet

    # 真实数据：segments.npy 形状 (N, L)，或 .npz 里键名为 'ppg'
    python scripts/extract_beat_features.py --input segs.npy --fs 500 --out beats.parquet
    python scripts/extract_beat_features.py --input segs.npy --fs 125 --sbp sbp.npy --dbp dbp.npy ...

产物
    parquet，列 = 标识 (subject_id, seg_idx, beat_idx, beat_start_sample, dataset)
    + 9 个 kp_* + 40 维特征，**与 `preprocess/corpus_build/03_build_features.py` 的
    `KP_COLS + BEAT_FEAT_COLS` 完全一致**，可直接喂给该脚本建语料。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess.features import extract_segment_features      # noqa: E402

FS = 500                 # 管线内部采样率（上游约定）
SEG_LEN = FS * 10        # 每段 10 秒
ID_COLS = ['subject_id', 'seg_idx', 'beat_idx', 'beat_start_sample']
KP_COLS = ['kp_o', 'kp_a', 'kp_md', 'kp_b', 'kp_s', 'kp_dn', 'kp_ip', 'kp_d', 'kp_v']
# 40 维：与 preprocess/corpus_build/03_build_features.py::BEAT_FEAT_COLS 必须逐字一致
FEAT_COLS = ['lasi', 'slope_systolic', 'slope_systolic_n', 'slope_md', 'slope_md_n',
             'pulse_width_50', 'pulse_width_90', 'bwr_50', 'reflection_index',
             'ni_dn', 'ni_ip', 't_dn', 'nt_dn', 'systolic_width_50', 'systolic_width_75',
             'diastolic_width_50', 'diastolic_width_75', 'perfusion_index',
             'normalized_pulse_vol', 'ppg_characteristic', 'mean_intensity', 'pulse_amp',
             'skewness', 'kurtosis', 'rms', 'crest_factor', 'iqr',
             'freq_dominant_1', 'freq_dominant_2', 'freq_dominant_3',
             'spectral_entropy', 'wavelet_entropy', 'spectral_centroid', 'fsqi',
             'intensity_ratio_s_d', 'beat_len_ms', 'hr_bpm', 't_sys_ms', 't_dia_ms',
             'ptt_ppg_ms']


def demo_segments(n=3, fs=FS):
    """合成 PPG 段（含 systolic 峰 + 重搏切迹 + 呼吸基线漂移）。

    仅供**确认环境装对**——数值本身没有生理意义，不要用于任何分析。
    """
    out = []
    for k in range(n):
        t = np.arange(SEG_LEN) / fs
        hr = 1.1 + 0.15 * k
        sig = np.zeros_like(t)
        for b in range(int(10 * hr)):
            off = b / hr
            m = (t >= off) & (t < off + 1.0 / hr)
            x = t[m] - off
            sig[m] += ((np.sin(2 * np.pi * hr * x) + 0.35 * np.sin(2 * np.pi * 2.6 * hr * x + 0.7))
                       * np.exp(-3.0 * x))
        sig += 0.15 * np.sin(2 * np.pi * 0.25 * t)          # 呼吸基线
        sig += np.random.default_rng(k).normal(0, 0.01, len(t))
        out.append(sig / np.std(sig))
    return np.asarray(out), fs


def load_input(args):
    if args.demo:
        ppg, fs = demo_segments()
        return ppg, fs, None, None, None
    a = np.load(args.input)
    ppg = a['ppg'] if isinstance(a, np.lib.npyio.NpzFile) else a
    if ppg.ndim == 1:
        ppg = ppg[None, :]
    rp = np.load(args.rpeaks) if args.rpeaks else None
    sbp = np.load(args.sbp) if args.sbp else None
    dbp = np.load(args.dbp) if args.dbp else None
    return ppg, args.fs, rp, sbp, dbp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--input', help='.npy / .npz（键名 ppg），形状 (N, L) 或 (L,)')
    src.add_argument('--demo', action='store_true', help='用合成信号跑通环境，不需要任何数据')

    ap.add_argument('--fs', type=int, default=FS, help='输入采样率（默认 500）')
    ap.add_argument('--rpeaks', default=None, help='.npy，每段的 R 峰样本位置（可选）')
    ap.add_argument('--sbp', default=None, help='.npy，每段的 SBP 标签（可选）')
    ap.add_argument('--dbp', default=None, help='.npy，每段的 DBP 标签（可选）')
    ap.add_argument('--dataset', default='demo', help='数据集名（写进 dataset 列）')
    ap.add_argument('--out', default='beats.parquet', help='输出 parquet 路径')
    ap.add_argument('--limit', type=int, default=None, help='只处理前 N 段（试跑用）')
    a = ap.parse_args()

    ppg, fs_in, rpeaks, sbp, dbp = load_input(a)
    abp = None          # 见下方调用处注释：ABP 不影响 40 维
    n = len(ppg) if a.limit is None else min(a.limit, len(ppg))
    print(f'输入: {ppg.shape[1]} 点 @ {fs_in} Hz × {n} 段'
          f'{"（合成 demo 数据，仅供环境自检）" if a.demo else ""}')

    if fs_in != FS:                                     # 重采样到内部 500 Hz
        from scipy.signal import resample
        L = int(round(ppg.shape[1] * FS / fs_in))
        ppg = np.stack([resample(np.asarray(s, dtype=np.float64), L) for s in ppg])
        print(f'  已重采样 {fs_in} → {FS} Hz，段长 {L} 点')
    if ppg.shape[1] != SEG_LEN:
        print(f'  ⚠ 段长 {ppg.shape[1]} ≠ {SEG_LEN}（10 s @ {FS} Hz）。'
              f'管线按段处理，长度不匹配不必然出错，但请确认这是你想要的。')

    rows, n_fail = [], 0
    for i in range(n):
        seg = np.asarray(ppg[i], dtype=np.float64)
        try:
            o = extract_segment_features(
                seg,
                # abp 只被 segment 级的 abp_sbp/abp_dbp 与 ptt 用到，**都不在 40 维里**
                # （ptt_ppg_ms 依赖的是 rpeaks 而非 abp）。故缺省用同长占位数组，
                # 使管线可跑通；那些非 40 维的输出本脚本也不写出。
                seg if abp is None else np.asarray(abp[i], dtype=np.float64),
                None if rpeaks is None else np.asarray(rpeaks[i], dtype=int),
                subject_id=f's{i:04d}', seg_idx=i, dataset=a.dataset,
                sbp_label=None if sbp is None else float(sbp[i]),
                dbp_label=None if dbp is None else float(dbp[i]), fs=FS)
            rows.extend(o['beats'])
        except Exception as e:                          # 单段失败不该中断整批
            n_fail += 1
            print(f'  [跳过] 段 {i}: {type(e).__name__}: {str(e)[:120]}')

    if not rows:
        sys.exit('没有产出任何拍——检查输入信号是否满足「已滤波 500 Hz」前提')

    df = pd.DataFrame(rows)
    # 列序与 preprocess/corpus_build/03_build_features.py 的 KP_COLS + BEAT_FEAT_COLS 对齐，
    # 使产物成为该脚本的**直接替代输入**；dataset 是对外附加列，放最后。
    cols = ID_COLS + KP_COLS + FEAT_COLS
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f'提取器未产出这些列: {missing}'

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df[cols + ['dataset']].to_parquet(out, index=False)

    print(f'\n产出 {len(df)} 拍 × {len(cols)} 列 → {out}')
    print(f'  受试者 {df.subject_id.nunique()}，段 {df.groupby(["subject_id","seg_idx"]).ngroups}'
          f'，失败段 {n_fail}')
    kp_ok = df[KP_COLS].notna().all(axis=1).mean()
    print(f'  9 关键点齐全的拍: {kp_ok:.1%}')


if __name__ == '__main__':
    main()
