#!/usr/bin/env python
"""
01_make_folds.py — 合格受试者分层 10 折划分（v2）

规则（与 v1_plan.md §3.6 一致）：
- 三个数据集（mimic / vital / aurora）各自独立分层切 10 折，同号合并为全局折
- 分层变量：年龄（库内分位 4 箱）× 性别 × 段体量（3 箱）× 中位 SBP（3 箱，aurora 无标签单独一档）
- 分配算法：迭代分层（Sechidis 2011 风格），稀有组合优先；并列时按各折段总量最均衡者
- 种子 42，可复现；不预设任何折的角色（测试折后续指定）
- 产出：folds/subject_folds_v2.parquet(.csv) + 折质检报告；不写 DB（审查通过后另行写入）

v2 变更（相对 v1）：
- 数据根改为唯一规范根 /data（v1 误用滞后备份盘）
- 排除异常受试者：anomaly_note 非空（年龄/性别缺失、age=0）与 age<18（未成年人）
"""
import os
import sqlite3
import numpy as np
import pandas as pd

SEED = 42
N_FOLDS = 10
DATA_ROOT = '/data'          # 唯一数据根（README: PPG_DATA_ROOT）
DB_PATH = os.path.join(DATA_ROOT, 'phenotype/ppg_phenotype.db')
FEAT_MIMIC = os.path.join(DATA_ROOT, 'mimic3/features_v2/mimic_segments.parquet')
FEAT_VITAL = os.path.join(DATA_ROOT, 'vitaldb/features_v2/vital_segments.parquet')
AURORA_META = os.path.join(DATA_ROOT, 'aurora_dataset/segments_meta.csv')
# 仓库根（本文件在 preprocess/corpus_build/ 下，故上溯三层）
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'folds')


# ---------------------------------------------------------------- 1. 数据加载
def load_subjects():
    """载入受试者并排除异常：anomaly_note 非空（年龄/性别缺失等，入库时已备注）
    与 age<18（未成年人，与成人血压/年龄任务域不符）。"""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT subject_id, source, age, gender, n_segments, anomaly_note FROM subjects", con)
    con.close()
    n_all = len(df)
    bad_note = df['anomaly_note'].fillna('').str.strip().ne('')
    underage = df['age'].notna() & (df['age'] < 18)
    excluded = df[bad_note | underage]
    df = df[~(bad_note | underage)].reset_index(drop=True)
    print(f'受试者: 全库 {n_all} | 排除 {len(excluded)} '
          f'(anomaly_note {bad_note.sum()} + 未成年 {underage.sum()}) | 保留 {len(df)}')
    print('  排除按 source:', excluded.groupby('source').size().to_dict())
    return df


def load_median_sbp():
    """每个受试者的中位 SBP：mimic/vital 来自各自 features_v2，
    aurora 来自 aurora_dataset/segments_meta.csv（袖带听诊标签）。"""
    frames = []
    for path in [FEAT_MIMIC, FEAT_VITAL]:
        t = pd.read_parquet(path, columns=['dataset', 'subject_id', 'sbp_label'])
        frames.append(t.dropna(subset=['sbp_label']))
    au = pd.read_csv(AURORA_META)
    au['dataset'] = 'aurora'
    frames.append(au[['dataset', 'subject_id', 'sbp_label']].dropna())
    sbp = pd.concat(frames, ignore_index=True)
    out = (sbp.groupby(['dataset', 'subject_id'])['sbp_label'].median()
           .rename('med_sbp').reset_index())
    return out.rename(columns={'dataset': 'source'})


def load_median_dbp():
    """每个受试者的中位 DBP（三库，来源同 load_median_sbp）。"""
    frames = []
    for path in [FEAT_MIMIC, FEAT_VITAL]:
        t = pd.read_parquet(path, columns=['dataset', 'subject_id', 'dbp_label'])
        frames.append(t.dropna(subset=['dbp_label']))
    au = pd.read_csv(AURORA_META)
    au['dataset'] = 'aurora'
    frames.append(au[['dataset', 'subject_id', 'dbp_label']].dropna())
    dbp = pd.concat(frames, ignore_index=True)
    out = (dbp.groupby(['dataset', 'subject_id'])['dbp_label'].median()
           .rename('med_dbp').reset_index())
    return out.rename(columns={'dataset': 'source'})


# ---------------------------------------------------------------- 2. 分箱
def make_bins(df):
    """库内分位分箱，返回带分层键的 DataFrame。"""
    rows = []
    for src, g in df.groupby('source'):
        g = g.copy()
        # 年龄：非法值（<=0）单独一档
        age = g['age'].copy()
        bad = age.isna() | (age <= 0)
        age_q = pd.qcut(age[~bad], 4, labels=False, duplicates='drop')
        g['age_bin'] = -1
        g.loc[age_q.index, 'age_bin'] = age_q.astype(int)
        # 性别：缺失单独一档（M=0, F=1, 其他/缺失=2）
        g['sex_bin'] = g['gender'].map({'M': 0, 'F': 1}).fillna(2).astype(int)
        # 段体量：库内 3 分位
        g['nseg_bin'] = pd.qcut(g['n_segments'].rank(method='first'), 3,
                                labels=False, duplicates='drop')
        # 中位 SBP：库内 3 分位，无标签（aurora / 缺失）= -1
        if 'med_sbp' in g.columns and g['med_sbp'].notna().sum() > 10:
            s_q = pd.qcut(g['med_sbp'].dropna(), 3, labels=False, duplicates='drop')
            g['sbp_bin'] = -1
            g.loc[s_q.index, 'sbp_bin'] = s_q.astype(int)
        else:
            g['sbp_bin'] = -1
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out['stratum'] = (out['age_bin'].astype(str) + '_' + out['sex_bin'].astype(str)
                      + '_' + out['nseg_bin'].astype(str) + '_' + out['sbp_bin'].astype(str))
    return out


# ---------------------------------------------------------------- 3. 迭代分层分配
def iterative_stratified_assign(df, n_folds, seed):
    """
    Sechidis 风格迭代分层：
    - 分层组合按稀有度升序处理（稀有组合优先）
    - 同层内段数大的受试者优先放置（难放的先放）
    - 每个受试者放入"本层剩余需求最大"的折；并列时取段总量累计最小的折
    """
    rng = np.random.default_rng(seed)
    fold_seg_load = np.zeros(n_folds)
    fold_stratum_cnt = {s: np.zeros(n_folds, dtype=int) for s in df['stratum'].unique()}
    fold_cnt = np.zeros(n_folds, dtype=int)
    assign = {}

    strata = df.groupby('stratum')
    order = sorted(strata.groups.keys(), key=lambda s: len(strata.groups[s]))
    for s in order:
        members = df.loc[strata.groups[s]].sort_values('n_segments', ascending=False)
        per_fold_need = len(members) / n_folds  # 该层每折期望数量
        for _, sub in members.iterrows():
            need = per_fold_need - fold_stratum_cnt[s]
            best = np.flatnonzero(need == need.max())
            if len(best) > 1:
                best = best[np.argmin(fold_seg_load[best])]
            best = np.atleast_1d(best)[0]
            f = int(best)
            assign[sub['uid']] = f
            fold_stratum_cnt[s][f] += 1
            fold_seg_load[f] += sub['n_segments']
            fold_cnt[f] += 1
    return assign, fold_cnt, fold_seg_load


# ---------------------------------------------------------------- 4. 质检报告
def qc_report(df):
    lines = []
    lines.append('# 10 折质检报告')
    lines.append('')
    lines.append(f'全局折 = 三库同号折合并；种子 {SEED}；折号 0-9，角色未指定。')
    lines.append('')
    # 全局概览
    lines.append('## 1. 全局（三库合并，按折）')
    g = df.groupby('fold_id').agg(
        n_sub=('subject_id', 'count'),
        n_seg=('n_segments', 'sum'),
        age_mean=('age', 'mean'), age_p50=('age', 'median'),
        female_ratio=('sex_bin', lambda x: (x == 1).mean()),
        sbp_p50=('med_sbp', 'median'),
    )
    lines.append('```')
    lines.append(g.round(2).to_string())
    lines.append('```')
    lines.append('')
    # 分库
    for src in ['mimic', 'vital', 'aurora']:
        d = df[df['source'] == src]
        lines.append(f'## 2.{src}')
        gg = d.groupby('fold_id').agg(
            n_sub=('subject_id', 'count'),
            n_seg=('n_segments', 'sum'),
            age_mean=('age', 'mean'), age_p50=('age', 'median'),
            female_ratio=('sex_bin', lambda x: (x == 1).mean()),
            sbp_p50=('med_sbp', 'median'),
        )
        lines.append('```')
        lines.append(gg.round(2).to_string())
        lines.append('```')
        lines.append('')
    # 分层键分布一致性：每折的 stratum 计数占比
    lines.append('## 3. 分层组合占比（全局，每折前 8 大组合）')
    for f in range(N_FOLDS):
        d = df[df['fold_id'] == f]
        top = d['stratum'].value_counts(normalize=True).head(8)
        lines.append(f'- fold {f}: ' + ', '.join(f'{k}:{v:.3f}' for k, v in top.items()))
    return '\n'.join(lines), g


def plot_qc(df, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib 不可用，跳过绘图')
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for src, ax in zip(['mimic', 'vital', 'aurora'], axes.flatten()[:3]):
        d = df[df['source'] == src]
        g = d.groupby('fold_id').agg(n_sub=('subject_id', 'count'),
                                     n_seg=('n_segments', 'sum'),
                                     age_p50=('age', 'median'),
                                     female_ratio=('sex_bin', lambda x: (x == 1).mean()))
        x = g.index
        ax2 = ax.twinx()
        ax.bar(x - 0.2, g['n_sub'], width=0.4, label='n_subjects', color='tab:blue')
        ax2.bar(x + 0.2, g['n_seg'], width=0.4, label='n_segments', color='tab:orange', alpha=0.6)
        ax.set_title(f'{src}: subjects/segments per fold')
        ax.set_xlabel('fold_id')
        ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax = axes.flatten()[3]
    d = df[df['med_sbp'].notna()]
    for f in sorted(d['fold_id'].unique()):
        data = d[d['fold_id'] == f]['med_sbp'].values
        ax.hist(data, bins=30, alpha=0.45, label=f'fold {f}')
    ax.set_title('median SBP distribution per fold (all sources)')
    ax.set_xlabel('median SBP (mmHg)')
    ax = axes.flatten()[4]
    d = df[df['med_dbp'].notna()]
    for f in sorted(d['fold_id'].unique()):
        data = d[d['fold_id'] == f]['med_dbp'].values
        ax.hist(data, bins=30, alpha=0.45, label=f'fold {f}')
    ax.set_title('median DBP distribution per fold (all sources)')
    ax.set_xlabel('median DBP (mmHg)')
    axes.flatten()[5].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fold_qc_v2.png'), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_subjects()
    print(f'受试者总数: {len(df)}')
    df = df.merge(load_median_sbp(), on=['source', 'subject_id'], how='left')
    df = df.merge(load_median_dbp(), on=['source', 'subject_id'], how='left')
    # 注意：两库存在 28 个 subject_id 碰撞（年龄/性别均不同，非同一人），
    # 真实主键 = (source, subject_id)
    df['uid'] = df['source'] + ':' + df['subject_id']
    print(f'有中位 SBP 的受试者: {df["med_sbp"].notna().sum()} | 唯一 uid: {df["uid"].nunique()}')
    df = make_bins(df)

    results = []
    for src, g in df.groupby('source'):
        assign, fc, sl = iterative_stratified_assign(g, N_FOLDS, SEED)
        g = g.copy()
        g['fold_id'] = g['uid'].map(assign)
        results.append(g)
        print(f'{src}: {len(g)} 人 | 每折人数 {fc.tolist()} | 每折段数 {sl.astype(int).tolist()}')
    df = pd.concat(results, ignore_index=True)

    # 保存
    cols = ['uid', 'subject_id', 'source', 'fold_id', 'age', 'gender', 'n_segments',
            'med_sbp', 'med_dbp', 'age_bin', 'sex_bin', 'nseg_bin', 'sbp_bin', 'stratum']
    df[cols].to_parquet(os.path.join(OUT_DIR, 'subject_folds_v2.parquet'), index=False)
    df[cols].to_csv(os.path.join(OUT_DIR, 'subject_folds_v2.csv'), index=False)

    # 质检
    report, _ = qc_report(df)
    with open(os.path.join(OUT_DIR, 'fold_report_v2.md'), 'w') as f:
        f.write(report)
    plot_qc(df, OUT_DIR)
    print(f'输出目录: {OUT_DIR}')
    print('subject_folds_v2.parquet / .csv / fold_report_v2.md / fold_qc_v2.png')


if __name__ == '__main__':
    main()
