#!/usr/bin/env python
"""
02_build_corpus.py — corpus_v2 物化（Phase 0，异常排除版）

产出（/data/corpus_v2/）：
- manifest.parquet : 全局行号 → (uid, subject_id, source, seg_idx, shard, row,
                     fold, quality_score, quality_tier, has_ecg, sig_ok)
- labels.parquet   : 全局行号 → (sbp, dbp, age, gender)
- signals/ppg_shard_*.npy : (SHARD_ROWS, 1, 1250) fp16，内容 = PPG_F 原值 [0,1]
- signals/ecg_shard_*.npy : 同形状；缺失窗口补零，以 manifest.has_ecg 区分

设计要点：
- 全量物化（含 fair/failed），训练时按 quality_tier 过滤——保留消融自由度
- 行序 = 按 (source, subject_id, seg_idx) 排序，同受试者连续，便于按人读写
- PPG/ECG 独立分片，互不牵连；归一化不进缓存（[0,1] 原值）
- 断点续传：已完成受试者记录在 progress.json，重跑自动跳过
- aurora mat 为直接数组、无引用单元格，加载分支单独处理
"""
import os
import json
import sqlite3
import threading
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd

# ------------------------------------------------------------- 配置
DB_PATH = '/data/phenotype/ppg_phenotype.db'
FOLDS_PATH = 'folds/subject_folds_v2.parquet'
BP_MIMIC = '/data/mimic3/features_v2/mimic_segments.parquet'
BP_VITAL = '/data/vitaldb/features_v2/vital_segments.parquet'
BP_AURORA = '/data/aurora_dataset/segments_meta.csv'
OUT_ROOT = '/data/corpus_v2'

SHARD_ROWS = 100_000        # 每分片行数 ≈ 238 MB
WIN_LEN = 1250              # 10 s @ 125 Hz
N_WORKERS = 24              # 并发进程数：HDF5 全局锁会串行化线程，必须多进程；
                            # 磁盘随机读仍是上限，进程数过高反而造成磁头抖动
TIER_CODE = {'good': 0, 'fair': 1, 'failed': 2}

SIG_DIR = os.path.join(OUT_ROOT, 'signals')
PROGRESS_PATH = os.path.join(OUT_ROOT, 'progress.json')
_lock = threading.Lock()


# ------------------------------------------------------------- 1. manifest
def build_manifest():
    """从 DB segments + 分折结果 + 血压标签构建全局表。
    v2 排除规则：
    - 段级：segments.anomaly_note 非空（血压标签超阈/缺失，已备注）
    - 受试者级：异常受试者未进折（见 01_make_folds.py v2），其段无折号自然落入剔除"""
    con = sqlite3.connect(DB_PATH)
    seg = pd.read_sql_query(
        "SELECT subject_id, source, seg_idx, quality_score, quality_tier, anomaly_note "
        "FROM segments", con)
    con.close()
    seg['uid'] = seg['source'] + ':' + seg['subject_id']

    # 段级排除：血压标签异常（备注已由 05_annotate_anomalies.py 写入）
    bp_bad = seg['anomaly_note'].fillna('').str.strip().ne('')
    print(f'段级排除（血压异常，已备注）: {bp_bad.sum()} 段 / '
          f'{seg.loc[bp_bad, "uid"].nunique()} 受试者', flush=True)
    seg = seg[~bp_bad].reset_index(drop=True)

    seg = seg.sort_values(['source', 'subject_id', 'seg_idx']).reset_index(drop=True)

    folds = pd.read_parquet(FOLDS_PATH)[['uid', 'fold_id']]
    seg = seg.merge(folds, on='uid', how='left')
    # 无折号段：孤儿段 + 异常受试者（年龄/性别缺失、未成年）的段，剔除并记录
    orphan = seg['fold_id'].isna()
    if orphan.any():
        print(f'[WARN] 剔除无折号段 {orphan.sum()} 行（{seg.loc[orphan, "uid"].nunique()} 个受试者：'
              f'孤儿段或异常受试者）', flush=True)
        seg = seg[~orphan].reset_index(drop=True)

    # 血压标签（段级）
    frames = []
    for path, src in [(BP_MIMIC, 'mimic'), (BP_VITAL, 'vital')]:
        t = pd.read_parquet(path, columns=['subject_id', 'seg_idx', 'sbp_label', 'dbp_label'])
        t['source'] = src
        frames.append(t)
    au = pd.read_csv(BP_AURORA)
    au['source'] = 'aurora'
    frames.append(au[['source', 'subject_id', 'seg_idx', 'sbp_label', 'dbp_label']])
    bp = pd.concat(frames, ignore_index=True)
    seg = seg.merge(bp, on=['source', 'subject_id', 'seg_idx'], how='left')

    seg['row'] = np.arange(len(seg), dtype=np.int64)
    seg['shard'] = seg['row'] // SHARD_ROWS
    seg['row_in_shard'] = seg['row'] % SHARD_ROWS
    return seg


def build_labels(seg, out_path):
    """段级标签：sbp/dbp + 受试者级 age/gender。"""
    con = sqlite3.connect(DB_PATH)
    sub = pd.read_sql_query("SELECT subject_id, source, age, gender FROM subjects", con)
    con.close()
    sub['uid'] = sub['source'] + ':' + sub['subject_id']
    lab = seg[['row', 'uid', 'sbp_label', 'dbp_label']].merge(
        sub[['uid', 'age', 'gender']], on='uid', how='left')
    lab = lab.rename(columns={'sbp_label': 'sbp', 'dbp_label': 'dbp',
                              'age': 'age', 'gender': 'gender'})
    lab = lab.sort_values('row')[['row', 'sbp', 'dbp', 'age', 'gender']]
    lab.to_parquet(out_path, index=False)
    return lab


# ------------------------------------------------------------- 2. 信号提取
def make_shards(n_rows):
    """计算分片路径；冒烟模式下不预分配，由 ensure_shard 懒创建。"""
    n_shards = int(np.ceil(n_rows / SHARD_ROWS))
    paths = {}
    for s in range(n_shards):
        for mod in ['ppg', 'ecg']:
            paths[(mod, s)] = os.path.join(SIG_DIR, f'{mod}_shard_{s:04d}.npy')
        paths[('meta', s)] = os.path.join(SIG_DIR, f'meta_shard_{s:04d}.npy')
    if not os.environ.get('CORPUS_SMOKE'):
        for s in range(n_shards):
            ensure_shard(s, paths, n_rows)
    return n_shards, paths


def ensure_shard(s, paths, n_rows):
    with _lock:
        size = min(SHARD_ROWS, n_rows - s * SHARD_ROWS)
        for mod, dtype, shape in [('ppg', np.float16, (size, 1, WIN_LEN)),
                                  ('ecg', np.float16, (size, 1, WIN_LEN)),
                                  ('meta', np.uint8, (size, 2))]:
            p = paths[(mod, s)]
            if not os.path.exists(p):
                arr = np.lib.format.open_memmap(p, mode='w+', dtype=dtype, shape=shape)
                del arr


def read_windows(group, name, idxs, direct):
    """从 Subj_Wins 读取指定窗口的信号，返回 (n_idx, WIN_LEN) 数组与有效掩码。"""
    out = np.zeros((len(idxs), WIN_LEN), dtype=np.float32)
    ok = np.zeros(len(idxs), dtype=bool)
    ds = group[name]
    for k, i in enumerate(idxs):
        try:
            if direct:
                v = np.asarray(ds[0, i], dtype=np.float32).ravel()
            else:
                ref = ds[0, i]
                v = np.asarray(group.file[ref][()], dtype=np.float32).ravel()
            if v.size == WIN_LEN:
                out[k] = v
                ok[k] = True
        except Exception:
            pass
    return out, ok


def extract_subject(args):
    """子进程：只读 mat，返回数组（避免 HDF5 全局锁串行化）。"""
    uid, source, rows, shard_ids, rows_in, mat_path = args
    try:
        with h5py.File(mat_path, 'r') as f:
            g = f['Subj_Wins']
            direct = (source == 'aurora')
            n_win = g['PPG_F'].shape[1]
            idxs = [r if r < n_win else -1 for r in rows]
            valid_pos = [k for k, r in enumerate(idxs) if r >= 0]
            ppg = np.zeros((len(rows), WIN_LEN), dtype=np.float32)
            ecg = np.zeros((len(rows), WIN_LEN), dtype=np.float32)
            has_ecg = np.zeros(len(rows), dtype=np.uint8)
            sig_ok = np.zeros(len(rows), dtype=np.uint8)
            if valid_pos:
                real = [idxs[k] for k in valid_pos]
                p, pok = read_windows(g, 'PPG_F', real, direct)
                if 'ECG_Raw' in g:
                    e, eok = read_windows(g, 'ECG_Raw', real, direct)
                else:
                    e, eok = np.zeros_like(p), np.zeros(len(real), dtype=bool)
                for j, k in enumerate(valid_pos):
                    ppg[k] = p[j]
                    ecg[k] = e[j]
                    # ECG 有效性：读取成功且非全零
                    has_ecg[k] = int(eok[j] and np.abs(e[j]).sum() > 1e-6)
                    sig_ok[k] = int(pok[j])
    except Exception as exc:
        print(f'[WARN] {uid} 读取失败: {exc}', flush=True)
        ppg = np.zeros((len(rows), WIN_LEN), dtype=np.float32)
        ecg = np.zeros_like(ppg)
        has_ecg = np.zeros(len(rows), dtype=np.uint8)
        sig_ok = np.zeros(len(rows), dtype=np.uint8)
    return uid, ppg, ecg, has_ecg, sig_ok, shard_ids, rows_in


def write_subject(ppg, ecg, has_ecg, sig_ok, shard_ids, rows_in, shard_paths, n_rows):
    """主进程：统一写分片（同一受试者的行必然落入少量连续分片）。"""
    for s in set(shard_ids):
        ensure_shard(s, shard_paths, n_rows)
        sel = np.array([k for k, x in enumerate(shard_ids) if x == s])
        rks = np.array([rows_in[k] for k in sel])
        mm_p = np.load(shard_paths[('ppg', s)], mmap_mode='r+')
        mm_e = np.load(shard_paths[('ecg', s)], mmap_mode='r+')
        mm_m = np.load(shard_paths[('meta', s)], mmap_mode='r+')
        mm_p[rks, 0, :] = ppg[sel].astype(np.float16)
        mm_e[rks, 0, :] = ecg[sel].astype(np.float16)
        mm_m[rks, 0] = has_ecg[sel]
        mm_m[rks, 1] = sig_ok[sel]
        del mm_p, mm_e, mm_m


def apply_out_override():
    """冒烟测试用 CORPUS_OUT 重定向输出目录。"""
    global OUT_ROOT, SIG_DIR, PROGRESS_PATH
    if os.environ.get('CORPUS_OUT'):
        OUT_ROOT = os.environ['CORPUS_OUT']
        SIG_DIR = os.path.join(OUT_ROOT, 'signals')
        PROGRESS_PATH = os.path.join(OUT_ROOT, 'progress.json')


def extract_signals(seg):
    os.makedirs(SIG_DIR, exist_ok=True)
    n_shards, shard_paths = make_shards(len(seg))
    print(f'分片数: {n_shards} × 3（ppg/ecg/meta）× {SHARD_ROWS} 行', flush=True)

    # 断点续传；冒烟模式用 CORPUS_SMOKE 限制每库受试者数、CORPUS_OUT 重定向输出（main 入口已生效）
    done = set()
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as fp:
            done = set(json.load(fp))
        print(f'已完成受试者: {len(done)}，跳过', flush=True)

    groups = []
    for uid, gdf in seg.groupby('uid', sort=False):
        if uid in done:
            continue
        src = gdf['source'].iloc[0]
        groups.append((uid, src, gdf['seg_idx'].tolist(),
                       gdf['shard'].tolist(), gdf['row_in_shard'].tolist()))
    if os.environ.get('CORPUS_SMOKE'):
        k = int(os.environ['CORPUS_SMOKE'])
        kept, cnt = [], {}
        for t in groups:
            cnt[t[1]] = cnt.get(t[1], 0) + 1
            if cnt[t[1]] <= k:
                kept.append(t)
        groups = kept
        print(f'[SMOKE] 每库只处理 {k} 个受试者，共 {len(groups)}', flush=True)

    # mat 路径映射
    con = sqlite3.connect(DB_PATH)
    wp = pd.read_sql_query(
        "SELECT DISTINCT subject_id, source, waveform_path FROM segments", con)
    con.close()
    wp['uid'] = wp['source'] + ':' + wp['subject_id']
    path_map = dict(zip(wp['uid'], wp['waveform_path']))

    tasks = [(*t, path_map[t[0]]) for t in groups]
    n_done_before = len(done)
    print(f'待处理受试者: {len(tasks)}', flush=True)
    # 多进程：HDF5 库全局锁会串行化线程内的读写，必须进程级并行；
    # 结果统一由主进程写入分片，避免多进程同时 mmap 写同一分片
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(extract_subject, t) for t in tasks]
        for i, fut in enumerate(futures):
            uid, ppg, ecg, has_ecg, sig_ok, shard_ids, rows_in = fut.result()
            write_subject(ppg, ecg, has_ecg, sig_ok, shard_ids, rows_in, shard_paths, len(seg))
            with _lock:
                done.add(uid)
                if len(done) % 200 == 0:
                    with open(PROGRESS_PATH, 'w') as fp:
                        json.dump(list(done), fp)
                    print(f'进度: {len(done) - n_done_before}/{len(tasks)}', flush=True)
    with open(PROGRESS_PATH, 'w') as fp:
        json.dump(list(done), fp)


def read_flags(seg):
    """从 meta 分片按行读回 (has_ecg, sig_ok)，天然支持断点续跑。"""
    has_ecg = np.zeros(len(seg), dtype=np.uint8)
    sig_ok = np.zeros(len(seg), dtype=np.uint8)
    for s, gdf in seg.groupby('shard'):
        p = os.path.join(SIG_DIR, f'meta_shard_{s:04d}.npy')
        if not os.path.exists(p):
            print(f'[WARN] meta_shard_{s:04d} 不存在（冒烟模式？）', flush=True)
            continue
        mm = np.load(p, mmap_mode='r')
        rks = gdf['row_in_shard'].values
        has_ecg[gdf.index] = mm[rks, 0]
        sig_ok[gdf.index] = mm[rks, 1]
        del mm
    return has_ecg, sig_ok


def main():
    apply_out_override()
    os.makedirs(OUT_ROOT, exist_ok=True)
    print('== 1. manifest ==', flush=True)
    seg = build_manifest()
    print(f'总段数: {len(seg)} | 折号缺失: {seg["fold_id"].isna().sum()}', flush=True)

    print('== 2. labels ==', flush=True)
    lab = build_labels(seg, os.path.join(OUT_ROOT, 'labels.parquet'))
    print(f'标签行: {len(lab)} | sbp 填充: {lab["sbp"].notna().mean():.4f}', flush=True)

    print('== 3. 信号提取 ==', flush=True)
    extract_signals(seg)

    print('== 4. manifest ==', flush=True)
    has_ecg, sig_ok = read_flags(seg)
    tier = seg['quality_tier'].map(TIER_CODE).astype(np.int8)
    manifest = pd.DataFrame({
        'row': seg['row'], 'uid': seg['uid'], 'subject_id': seg['subject_id'],
        'source': seg['source'], 'seg_idx': seg['seg_idx'],
        'shard': seg['shard'].astype(np.int32),
        'row_in_shard': seg['row_in_shard'].astype(np.int32),
        'fold': seg['fold_id'].astype(np.int8),
        'quality_score': seg['quality_score'].astype(np.float32),
        'quality_tier': tier,
        'has_ecg': has_ecg,
        'sig_ok': sig_ok,
    })
    manifest.to_parquet(os.path.join(OUT_ROOT, 'manifest.parquet'), index=False)
    print('== 完成 ==', flush=True)
    print(f'输出: {OUT_ROOT}')


if __name__ == '__main__':
    main()
