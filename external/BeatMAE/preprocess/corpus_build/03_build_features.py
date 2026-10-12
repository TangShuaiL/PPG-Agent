#!/usr/bin/env python
"""
03_build_features.py — corpus_v2 特征/关键点物化（纯搬运，不重算）

产出（追加到 corpus_v2/）：
- seg_features.npy     : (n_segs, 61) fp16，行号对齐 manifest；列序见 features_meta.json
- keypoints.npz        : CSR。offsets(n_segs+1) + 拼接数组（每拍 10×int16：
                         o,a,md,b,s,dn,ip,d,v,start_sample；缺失坐标=-1）+ kp_valid
- beat_features.npz    : CSR。offsets + 拼接数组（每拍 40×fp16，统一列序）
- features_meta.json   : 列定义、来源、缺失编码说明

来源（全部现成，零重算）：
- 段级：DB segment_features 表（62 列，去掉 seg_id 后 61 维）
- 拍级：mimic3/vitaldb/aurora 各自 features_v2/*_beats_full.parquet（坐标+特征）
- aurora 无 PPG 通道的拍特征为 NaN（补 0，beat_feat_valid=0 标记）
"""
import os
import json
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = '/data/phenotype/ppg_phenotype.db'
CORPUS = '/data/corpus_v2'
BEATS_FULL = {
    # 顺序必须与 manifest 的 source 分块顺序一致（uid 排序：aurora < mimic < vital）
    'aurora': '/data/aurora_dataset/features_v2/aurorabp_beats_full.parquet',
    'mimic': '/data/mimic3/features_v2/mimic_beats_full.parquet',
    'vital': '/data/vitaldb/features_v2/vital_beats_full.parquet',
}
KP_COLS = ['kp_o', 'kp_a', 'kp_md', 'kp_b', 'kp_s', 'kp_dn', 'kp_ip', 'kp_d', 'kp_v']
# 统一拍级特征列序（两库列序不同，取共有的 40 维）
BEAT_FEAT_COLS = ['lasi', 'slope_systolic', 'slope_systolic_n', 'slope_md', 'slope_md_n',
                  'pulse_width_50', 'pulse_width_90', 'bwr_50', 'reflection_index',
                  'ni_dn', 'ni_ip', 't_dn', 'nt_dn', 'systolic_width_50', 'systolic_width_75',
                  'diastolic_width_50', 'diastolic_width_75', 'perfusion_index',
                  'normalized_pulse_vol', 'ppg_characteristic', 'mean_intensity', 'pulse_amp',
                  'skewness', 'kurtosis', 'rms', 'crest_factor', 'iqr',
                  'freq_dominant_1', 'freq_dominant_2', 'freq_dominant_3',
                  'spectral_entropy', 'wavelet_entropy', 'spectral_centroid',
                  'fsqi', 'intensity_ratio_s_d', 'beat_len_ms', 'hr_bpm',
                  't_sys_ms', 't_dia_ms', 'ptt_ppg_ms']


def clip_nonfinite(arr):
    """Inf → 该列有限值 p99.9，-Inf → p0.1，NaN → 0；返回 {列号: (n_posinf, n_neginf, n_nan, hi, lo)}。
    背景：intensity_ratio_s_d 等比值类特征在分母≈0 时产生 Inf（源表即如此）。"""
    info = {}
    for c in range(arr.shape[1]):
        col = arr[:, c]
        pos, neg, nan = np.isposinf(col), np.isneginf(col), np.isnan(col)
        if not (pos.any() or neg.any() or nan.any()):
            continue
        fvals = col[np.isfinite(col)]
        lo, hi = (np.quantile(fvals, [0.001, 0.999]) if len(fvals) else (0.0, 0.0))
        col[pos] = hi
        col[neg] = lo
        col[nan] = 0.0
        info[str(c)] = {'posinf': int(pos.sum()), 'neginf': int(neg.sum()),
                        'nan': int(nan.sum()), 'clip_hi': float(hi), 'clip_lo': float(lo)}
    return info


def load_manifest():
    mf = pd.read_parquet(os.path.join(CORPUS, 'manifest.parquet'))
    return mf


# ------------------------------------------------------------- 1. 段级特征
def build_seg_features(mf):
    con = sqlite3.connect(DB_PATH)
    sf = pd.read_sql_query("SELECT * FROM segment_features", con)
    seg = pd.read_sql_query("SELECT id, source, subject_id, seg_idx FROM segments", con)
    con.close()
    feat_cols = [c for c in sf.columns if c != 'seg_id']
    sf = sf.merge(seg, left_on='seg_id', right_on='id').drop(columns=['id', 'seg_id'])
    m = mf[['row', 'source', 'subject_id', 'seg_idx']].merge(
        sf, on=['source', 'subject_id', 'seg_idx'], how='left')
    assert m['row'].is_monotonic_increasing
    # 对齐完整；缺失来自源表本身的 NULL（~2.2% 段特征提取无结果）
    no_match = m[feat_cols].isna().all(axis=1)
    missing = no_match.mean()
    arr = m[feat_cols].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float16)
    seg_clip = clip_nonfinite(arr)
    np.save(os.path.join(CORPUS, 'seg_features.npy'), arr)
    np.save(os.path.join(CORPUS, 'seg_feat_valid.npy'), (~no_match).to_numpy().astype(np.uint8))
    print(f'seg_features: {arr.shape} | 特征空段（源表 NULL，已补 0）: {missing:.4%} | '
          f'Inf 裁剪列: {seg_clip}', flush=True)
    return feat_cols, missing, seg_clip


# ------------------------------------------------------------- 2. 拍级（CSR）
def build_beats(mf):
    n_segs = len(mf)
    # CSR 拼接顺序依赖：各 source 的 manifest 行号必须是连续块，且块序与 BEATS_FULL 一致
    blk_order = mf['source'].ne(mf['source'].shift()).cumsum()
    src_order = mf.groupby(blk_order)['source'].first().tolist()
    assert src_order == [s for s in BEATS_FULL if (mf['source'] == s).any()], \
        f'manifest source 块序 {src_order} 与 BEATS_FULL {list(BEATS_FULL)} 不一致'
    offsets = np.zeros(n_segs + 1, dtype=np.int64)
    kp_chunks, feat_chunks, valid_chunks, start_chunks, fvalid_chunks = [], [], [], [], []

    for src, path in BEATS_FULL.items():
        print(f'读取 {src} beats_full ...', flush=True)
        cols = ['subject_id', 'seg_idx', 'beat_idx', 'beat_start_sample'] + KP_COLS + BEAT_FEAT_COLS
        # 内存充足（754GB），直接全量载入
        t = pd.read_parquet(path, columns=cols)
        t['source'] = src
        t = t.merge(mf[['row', 'source', 'subject_id', 'seg_idx']],
                    on=['source', 'subject_id', 'seg_idx'], how='inner')  # 顺带剔除孤儿段
        print(f'{src}: {len(t)} 拍对齐 manifest', flush=True)
        t = t.sort_values(['row', 'beat_idx'])

        kp = t[KP_COLS].to_numpy(dtype=np.float32)
        kp_valid_row = ~np.isnan(kp).any(axis=1)
        kp_int = np.nan_to_num(kp, nan=-1).astype(np.int16)
        start = np.nan_to_num(t['beat_start_sample'].to_numpy(dtype=np.float32),
                              nan=-1).astype(np.int16)
        feat_raw = t[BEAT_FEAT_COLS].to_numpy(dtype=np.float32)
        feat_valid_row = np.isfinite(feat_raw).all(axis=1)  # NaN 拍（aurora 无 PPG 通道）补 0 后标记 0
        feat = np.nan_to_num(feat_raw, nan=0.0).astype(np.float16)
        rows_arr = t['row'].to_numpy()
        del t

        # CSR 段计数
        cnt = np.bincount(rows_arr, minlength=n_segs)
        kp_chunks.append(kp_int); start_chunks.append(start)
        feat_chunks.append(feat); valid_chunks.append(kp_valid_row.astype(np.uint8))
        fvalid_chunks.append(feat_valid_row.astype(np.uint8))
        offsets[1:] += cnt
        print(f'{src}: 关键点缺失拍 {(~kp_valid_row).mean():.2%} | 特征 NaN 拍 {(~feat_valid_row).mean():.2%}',
              flush=True)

    offsets = np.cumsum(offsets)  # 修正：先累加各源 cnt 再前缀和
    kp_arr = np.concatenate([np.concatenate([k, s[:, None]], axis=1)
                             for k, s in zip(kp_chunks, start_chunks)]).astype(np.int16)
    feat_arr = np.concatenate(feat_chunks)
    valid_arr = np.concatenate(valid_chunks)
    feat_valid_arr = np.concatenate(fvalid_chunks)
    beat_clip = clip_nonfinite(feat_arr)
    print(f'拍总数: {offsets[-1]} | kp {kp_arr.shape} | feat {feat_arr.shape} | '
          f'特征 NaN 拍占比 {(~feat_valid_arr.astype(bool)).mean():.2%} | '
          f'Inf 裁剪列: {beat_clip}', flush=True)

    np.savez(os.path.join(CORPUS, 'keypoints.npz'),
             offsets=offsets, keypoints=kp_arr, kp_valid=valid_arr)
    np.savez(os.path.join(CORPUS, 'beat_features.npz'),
             offsets=offsets, features=feat_arr, feat_valid=feat_valid_arr)
    return beat_clip, float((~feat_valid_arr.astype(bool)).mean())


def main():
    mf = load_manifest()
    print(f'manifest: {len(mf)} 段', flush=True)
    feat_cols, missing, seg_clip = build_seg_features(mf)
    beat_clip, feat_nan_ratio = build_beats(mf)

    inf_clip_doc = 'Inf → 该列有限值 p99.9，-Inf → p0.1（见各自 inf_clip 明细）'
    meta = {
        'seg_features': {'source': 'DB segment_features 表', 'n_cols': len(feat_cols),
                         'columns': feat_cols, 'missing_fill': 0.0,
                         'null_seg_ratio': float(missing),
                         'valid_flag': 'seg_feat_valid.npy（0=源表 NULL 补 0 段）',
                         'inf_clip': {'policy': inf_clip_doc, 'detail': seg_clip}},
        'keypoints': {'layout': 'CSR', 'per_beat_cols': KP_COLS + ['beat_start_sample'],
                      'coord_basis': '拍内相对样本坐标（125Hz），缺失=-1',
                      'kp_valid': '9 关键点全部非空=1，否则 0（区域 mask 只用 valid 拍）'},
        'beat_features': {'layout': 'CSR', 'columns': BEAT_FEAT_COLS,
                          'source': '*_beats_full.parquet', 'nan_fill': 0.0,
                          'feat_valid': 'npz 内同名数组：1=源表特征完整，0=含 NaN 补 0（aurora 无 PPG 通道的拍）',
                          'feat_nan_ratio': float(feat_nan_ratio),
                          'inf_clip': {'policy': inf_clip_doc, 'detail': beat_clip}},
        'aurora': 'beats_full 已补入（aurorabp_beats_full.parquet）；无 PPG 通道的拍特征补 0 且 '
                  'feat_valid=0，关键点同样可能缺失（kp_valid=0）',
    }
    with open(os.path.join(CORPUS, 'features_meta.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('== 完成 ==', flush=True)


if __name__ == '__main__':
    main()
