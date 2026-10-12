#!/usr/bin/env python
"""
05_annotate_anomalies.py — 数据异常备注写回 DB（幂等可重跑）

1. subjects.anomaly_note：给未成年人（age<18）追加 'age<18_excluded'
   （年龄/性别缺失、age=0 的备注已由用户入库时写好，本脚本不动）
2. segments 新增 anomaly_note 列：逐段血压标签异常备注，来源为标签本体
   （mimic/vital 的 features_v2 parquet、aurora 的 segments_meta.csv）：
   sbp<50 / sbp>250 / dbp<30 / dbp>160 / dbp>sbp / bp_label_missing
"""
import sqlite3
import pandas as pd

DB_PATH ='/data/phenotype/ppg_phenotype.db'
DATA_ROOT = '/data'
BP_SOURCES = {
    'mimic': f'{DATA_ROOT}/mimic3/features_v2/mimic_segments.parquet',
    'vital': f'{DATA_ROOT}/vitaldb/features_v2/vital_segments.parquet',
    'aurora': f'{DATA_ROOT}/aurora_dataset/segments_meta.csv',
}
SBP_RANGE = (50, 250)
DBP_RANGE = (30, 160)


def bp_notes(t):
    """对含 sbp_label/dbp_label 的 DataFrame 逐行生成备注列表。"""
    sbp, dbp = t['sbp_label'], t['dbp_label']
    notes = []
    for i in range(len(t)):
        parts = []
        s, d = sbp.iloc[i], dbp.iloc[i]
        if pd.isna(s) or pd.isna(d):
            parts.append('bp_label_missing')
        else:
            if s < SBP_RANGE[0]:
                parts.append('sbp<50')
            if s > SBP_RANGE[1]:
                parts.append('sbp>250')
            if d < DBP_RANGE[0]:
                parts.append('dbp<30')
            if d > DBP_RANGE[1]:
                parts.append('dbp>160')
            if d > s:
                parts.append('dbp>sbp')
        notes.append(';'.join(parts))
    return notes


def main():
    # ---------------- 1. 收集三库标签异常段 ----------------
    frames = []
    for src, path in BP_SOURCES.items():
        if path.endswith('.csv'):
            t = pd.read_csv(path)[['subject_id', 'seg_idx', 'sbp_label', 'dbp_label']]
        else:
            t = pd.read_parquet(path, columns=['subject_id', 'seg_idx', 'sbp_label', 'dbp_label'])
        t['source'] = src
        t['anomaly_note'] = bp_notes(t)
        frames.append(t[t['anomaly_note'] != ''])
    bad = pd.concat(frames, ignore_index=True)
    print(f'血压异常段合计: {len(bad)} | 按 source: {bad.groupby("source").size().to_dict()}')
    print('备注类型分布:', bad['anomaly_note'].value_counts().to_dict())

    # ---------------- 2. 写回 DB ----------------
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # 2a. segments.anomaly_note 列（不存在则建）
    cols = [r[1] for r in cur.execute('PRAGMA table_info(segments)')]
    if 'anomaly_note' not in cols:
        cur.execute("ALTER TABLE segments ADD COLUMN anomaly_note TEXT DEFAULT ''")
        print('segments 新增 anomaly_note 列')

    # 2b. 段级备注：按 (source, subject_id, seg_idx) 定位段主键
    seg_keys = pd.read_sql_query(
        'SELECT id, source, subject_id, seg_idx FROM segments', con)
    m = bad.merge(seg_keys, on=['source', 'subject_id', 'seg_idx'], how='left')
    miss = m['id'].isna().sum()
    if miss:
        print(f'[WARN] {miss} 个异常段在 segments 表中不存在（跳过）')
        m = m.dropna(subset=['id'])
    cur.executemany('UPDATE segments SET anomaly_note=? WHERE id=?',
                    [(r.anomaly_note, int(r.id)) for r in m.itertuples()])
    con.commit()
    n_note = cur.execute(
        "SELECT COUNT(*) FROM segments WHERE anomaly_note != ''").fetchone()[0]
    print(f'segments 已备注: {n_note} 行')

    # 2c. subjects：未成年人追加备注（保留已有备注）
    cur.execute("UPDATE subjects SET anomaly_note = "
                "CASE WHEN anomaly_note IS NULL OR anomaly_note = '' THEN 'age<18_excluded' "
                "ELSE anomaly_note || ';age<18_excluded' END "
                "WHERE age < 18 AND (anomaly_note IS NULL "
                "OR anomaly_note NOT LIKE '%age<18_excluded%')")
    print(f'subjects 追加 age<18_excluded 备注: {cur.rowcount} 行')
    con.commit()

    # ---------------- 3. 汇总核对 ----------------
    print('\nsubjects.anomaly_note 分布:')
    for r in cur.execute("SELECT anomaly_note, COUNT(*) FROM subjects "
                         "WHERE anomaly_note IS NOT NULL AND anomaly_note != '' "
                         "GROUP BY anomaly_note ORDER BY 2 DESC").fetchall():
        print(' ', r)
    print('\nsegments.anomaly_note 分布:')
    for r in cur.execute("SELECT anomaly_note, COUNT(*) FROM segments "
                         "WHERE anomaly_note != '' GROUP BY anomaly_note "
                         "ORDER BY 2 DESC LIMIT 10").fetchall():
        print(' ', r)
    con.close()
    print('== 完成 ==')


if __name__ == '__main__':
    main()
