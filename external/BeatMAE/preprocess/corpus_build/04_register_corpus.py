#!/usr/bin/env python
"""
04_register_corpus.py — 分折结果写回 DB + 语料注册（v2，guest 主库）

1. subjects 表新增 fold_id 列（复合主键 source+subject_id 匹配，幂等可重跑；
   异常受试者（anomaly_note 非空）保持 NULL）
2. corpus_registry 表注册语料版本信息与配置指纹（v2 覆盖旧版需另行清理）
"""
import os
import json
import hashlib
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = '/data/phenotype/ppg_phenotype.db'
FOLDS_PATH = 'folds/subject_folds_v2.parquet'
CORPUS_ROOT = '/data/corpus_v2'
VERSION = 'corpus_v2'
FOLD_SEED = 42


def config_hash():
    """关键配置指纹：参数 + 生成脚本内容。"""
    cfg = {
        'shard_rows': 100_000, 'win_len': 1250, 'fs': 125,
        'signals': ['PPG_F', 'ECG_Raw'], 'dtype': 'float16',
        'fold_seed': FOLD_SEED,
        'scripts': {},
    }
    # 以本文件所在目录为准：01–05 同处 preprocess/（原写死 .../ppg_model/scripts，
    # 那是 2026-09 目录大整理**之前**的位置，整理后 scripts/ 下已无 01–03，
    # 走到这里必 FileNotFoundError）。顺带免掉一个硬编码本机路径。
    base = os.path.dirname(os.path.abspath(__file__))
    for name in ['01_make_folds.py', '02_build_corpus.py', '03_build_features.py']:
        with open(os.path.join(base, name), 'rb') as f:
            cfg['scripts'][name] = hashlib.sha256(f.read()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:32]


def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # ---------------- 1. fold_id 写入 ----------------
    cols = [r[1] for r in cur.execute('PRAGMA table_info(subjects)')]
    if 'fold_id' not in cols:
        cur.execute('ALTER TABLE subjects ADD COLUMN fold_id INTEGER')
        print('subjects 新增 fold_id 列')

    folds = pd.read_parquet(FOLDS_PATH)
    rows = [(int(f), src, sid) for f, src, sid in
            zip(folds['fold_id'], folds['source'], folds['subject_id'])]
    cur.executemany(
        'UPDATE subjects SET fold_id=? WHERE source=? AND subject_id=?', rows)
    con.commit()

    # 验证：无折号者必须全部是异常受试者（anomaly_note 非空）
    stat = cur.execute(
        'SELECT fold_id, COUNT(*) FROM subjects WHERE fold_id IS NOT NULL '
        'GROUP BY fold_id ORDER BY fold_id').fetchall()
    null_rows = cur.execute(
        "SELECT COUNT(*), SUM(CASE WHEN anomaly_note IS NOT NULL AND "
        "anomaly_note != '' THEN 1 ELSE 0 END) FROM subjects "
        "WHERE fold_id IS NULL").fetchone()
    print(f'fold 分布: {dict(stat)} | fold_id 为空: {null_rows[0]} '
          f'(其中已备注异常: {null_rows[1]})')
    assert len(stat) == 10 and null_rows[0] == null_rows[1]

    # 与旧 split 交叉检查（确认两套划分独立且非退化）
    cross = cur.execute(
        'SELECT split, COUNT(DISTINCT fold_id) FROM subjects GROUP BY split').fetchall()
    print('split × fold 交叉:', cross)

    # ---------------- 2. corpus_registry ----------------
    cur.execute('''
        CREATE TABLE IF NOT EXISTS corpus_registry (
            version     TEXT PRIMARY KEY,
            root_path   TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now')),
            n_segments  INTEGER,
            n_subjects  INTEGER,
            n_beats     INTEGER,
            fold_seed   INTEGER,
            config_hash TEXT,
            notes       TEXT
        )''')

    mf = pd.read_parquet(os.path.join(CORPUS_ROOT, 'manifest.parquet'),
                         columns=['uid', 'source', 'fold'])
    kp = np.load(os.path.join(CORPUS_ROOT, 'keypoints.npz'))
    n_beats = int(kp['offsets'][-1])
    notes = json.dumps({
        'files': ['manifest.parquet', 'labels.parquet', 'signals/(ppg|ecg|meta)_shard_*.npy',
                  'seg_features.npy', 'seg_feat_valid.npy', 'keypoints.npz',
                  'beat_features.npz', 'features_meta.json'],
        'signals': 'PPG_F + ECG 独立分片, (N,1,1250) fp16, [0,1] 原值',
        'keypoints': 'CSR, 每拍 9 关键点 + beat_start_sample (int16), 缺失=-1, kp_valid 标志',
        'beat_features': 'CSR, 每拍 40 维 fp16, 列序见 features_meta.json',
        'exclusions': '受试者级: 334 人(年龄/性别缺失 306 + age=0 1 + 未成年 28, 见 subjects.anomaly_note); '
                      '段级: 9485 段血压标签超阈/缺失(见 segments.anomaly_note); '
                      '孤儿段 351; aurora 无拍级数据(二期补); ~2% 段级特征源表 NULL 补 0',
        'data_root': '唯一数据根 /data; 已备份至 data_backup',
    }, ensure_ascii=False)
    cur.execute('''
        INSERT OR REPLACE INTO corpus_registry
        (version, root_path, created_at, n_segments, n_subjects, n_beats,
         fold_seed, config_hash, notes)
        VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)''',
                (VERSION, CORPUS_ROOT, len(mf), mf['uid'].nunique(), n_beats,
                 FOLD_SEED, config_hash(), notes))
    con.commit()

    print()
    for r in cur.execute('SELECT version, root_path, n_segments, n_subjects, n_beats, '
                         'fold_seed, config_hash FROM corpus_registry').fetchall():
        print('注册:', r)
    con.close()
    print('== 完成 ==')


if __name__ == '__main__':
    main()
