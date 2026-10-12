"""One-time migration: parquet → SQLite database — chunked for memory safety."""
import os, sys, sqlite3
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ppg_agent.config import PHENOTYPE_DB, DATA_ROOT  # noqa: E402
DB_PATH = Path(PHENOTYPE_DB)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
CHUNK = 50000
TEST_FOLDS = {1, 2}                                    # ppg_model 统一测试折: split=test iff fold∈{1,2}
FOLD_CSV = DATA_ROOT / "ppg_agent/v2c_subject_folds.csv"  # v2c 权威折号(subject_id,source,fold)

if DB_PATH.exists():
    DB_PATH.unlink()

conn = sqlite3.connect(str(DB_PATH))
schema_path = Path(__file__).parent / "schema.sql"
conn.executescript(schema_path.read_text())
conn.commit()
print("Schema created.")

FEATURE_COLS = [
    'lasi_mean','slope_systolic_mean','slope_systolic_n_mean','slope_md_mean','slope_md_n_mean',
    'pulse_width_50_mean','pulse_width_90_mean','bwr_50_mean','reflection_index_mean',
    'ni_dn_mean','ni_ip_mean','t_dn_mean','nt_dn_mean','intensity_ratio_s_d_mean',
    'systolic_width_50_mean','systolic_width_75_mean','diastolic_width_50_mean','diastolic_width_75_mean',
    'perfusion_index_mean','normalized_pulse_vol_mean','ppg_characteristic_mean','mean_intensity_mean',
    'pulse_amp_mean','skewness_mean','kurtosis_mean','rms_mean','crest_factor_mean','iqr_mean',
    'freq_dominant_1_mean','freq_dominant_2_mean','freq_dominant_3_mean','spectral_entropy_mean',
    'wavelet_entropy_mean','spectral_centroid_mean','fsqi_mean','beat_len_ms_mean','hr_bpm_mean',
    't_sys_ms_mean','t_dia_ms_mean','ptt_abp_mean_ms','ptt_abp_std_ms','ptt_abp_min_ms',
    'ptt_abp_max_ms','ptt_ppg_mean_ms','ptt_ppg_std_ms','ptt_ppg_min_ms','ptt_ppg_max_ms',
    'hrv_sdnn','hrv_rmssd','hrv_pnn50','hrv_mean_rr','hrv_std_rr','ecg_hr_mean','ecg_hr_std',
    'abp_sbp_mean','abp_dbp_mean','abp_pp_mean','n_beats_ecg','beat_ratio_ppg_ecg','ptt_hr_product',
]


def migrate_dataset(source, parquet_dir, labels_csv, raw_data_subdir, parquet_prefix=None):
    print(f"\n{'='*60}\nMigrating {source.upper()}\n{'='*60}")

    prefix = parquet_prefix or source
    seg_path = Path(parquet_dir) / f'{prefix}_segments.parquet'
    if not seg_path.exists():
        print(f"  Not found: {seg_path}")
        return

    # ── Subjects (from merged parquet) ──
    merged_path = Path(f'<PROJECT_ROOT>/analysis/01_data_preparation/output/{source}_merged.parquet')
    if merged_path.exists():
        demo = pd.read_parquet(merged_path, columns=['subject_id','age','gender'])
        subj_df = demo.drop_duplicates(subset='subject_id').copy()
    else:
        demo = pd.read_parquet(seg_path, columns=['subject_id'])
        subj_df = demo[['subject_id']].drop_duplicates()
        subj_df['age'] = None; subj_df['gender'] = None

    subj_df['source'] = source
    subj_df['n_segments'] = 0  # will update after

    # v2c fold → fold_id/split (权威口径, 与 scripts/migrate_db_v3_1.py 一致);
    # aurora scenario a/o 由 subject_id 前缀推导 (a=实验室 / o=动态)。
    _fold_map = {}
    if Path(FOLD_CSV).exists():
        _fm = pd.read_csv(FOLD_CSV)
        _fold_map = {(str(r.source), str(r.subject_id)): int(r.fold) for r in _fm.itertuples()}
    subj_df['fold_id'] = subj_df['subject_id'].map(lambda s: _fold_map.get((source, str(s))))
    subj_df['split'] = subj_df['fold_id'].map(
        lambda f: None if f is None else ('test' if f in TEST_FOLDS else 'train'))
    subj_df['scenario'] = (subj_df['subject_id'].map(lambda s: 'a' if str(s).startswith('a') else 'o')
                           if source == 'aurora' else None)
    subj_df['person_group'] = None
    subj_df.to_sql('subjects', conn, if_exists='append', index=False)
    print(f"  Subjects: {len(subj_df)}")

    # ── Chunked: segments + features ──
    seg_cols_avail = ['subject_id','seg_idx','quality_score','quality_tier','n_beats_valid']
    seg_cols_avail += [c for c in ['sbp_label','dbp_label','n_beats_ppg'] if c in
                       pd.read_parquet(seg_path, columns=['subject_id']).columns]
    feat_avail = [c for c in FEATURE_COLS if c in pd.read_parquet(seg_path).columns]

    n_seg_total, n_feat_total = 0, 0
    reader = pd.read_parquet(seg_path, columns=list(set(seg_cols_avail + feat_avail)))

    for i in range(0, len(reader), CHUNK):
        chunk = reader.iloc[i:i + CHUNK].copy()
        chunk['source'] = source
        if 'waveform_path' not in chunk.columns:
            chunk['waveform_path'] = chunk['subject_id'].apply(
                lambda x: f"{raw_data_subdir}/{x}.mat" if not x.startswith(source[:3]) else
                f"{raw_data_subdir}/{x}.mat"
            )
        if 'n_beats_total' not in chunk.columns and 'n_beats_ppg' in chunk.columns:
            chunk['n_beats_total'] = chunk['n_beats_ppg']

        # Write segments
        seg_write = chunk[['subject_id','source','seg_idx','quality_score','quality_tier',
                           'waveform_path']].copy()
        for c in ['sbp_label','dbp_label','n_beats_valid','n_beats_total']:
            if c in chunk.columns:
                seg_write[c] = chunk[c]
        seg_write.to_sql('segments', conn, if_exists='append', index=False)

        # Get inserted seg_ids
        seg_ids = pd.read_sql(
            f"SELECT id, subject_id, seg_idx FROM segments WHERE source='{source}' "
            f"ORDER BY id DESC LIMIT {CHUNK}", conn
        )
        seg_ids = seg_ids.sort_values('id')

        # Write features (match by subject_id + seg_idx)
        feat_write = chunk[['subject_id','seg_idx']].copy()
        feat_write = feat_write.merge(seg_ids, on=['subject_id','seg_idx'], how='inner')
        for fc in feat_avail:
            feat_write[fc] = chunk[fc].values
        feat_write = feat_write.rename(columns={'id': 'seg_id'})
        feat_write = feat_write[['seg_id'] + feat_avail]
        feat_write.to_sql('segment_features', conn, if_exists='append', index=False)

        n_seg_total += len(seg_write)
        n_feat_total += len(feat_write)
        if (i // CHUNK) % 10 == 0:
            print(f"  [{source}] {i}/{len(reader)} ({100*i/len(reader):.0f}%)", flush=True)

    conn.commit()
    print(f"  Segments: {n_seg_total:,}, Features: {n_feat_total:,} × {len(feat_avail)} cols")

    # Update n_segments
    conn.execute(f"""
        UPDATE subjects SET n_segments = (
            SELECT COUNT(*) FROM segments
            WHERE segments.subject_id = subjects.subject_id AND segments.source = '{source}'
        )
    """)
    conn.commit()

    # ── Labels ──
    label_path = Path(labels_csv)
    if label_path.exists():
        labels = pd.read_csv(label_path)
        id_col = 'filename' if 'filename' in labels.columns else 'subject_id'
        labels = labels.rename(columns={id_col: 'subject_id'})
        labels['subject_id'] = labels['subject_id'].apply(
            lambda x: f'p{int(x):06d}' if isinstance(x, (int, float)) and not pd.isna(x) else str(x)
        )
        labels['source'] = source
        # Only insert columns that exist in BOTH schema and labels
        schema_cols = {c[0] for c in conn.execute("PRAGMA table_info(disease_labels)")}
        valid_cols = ['subject_id', 'source'] + [c for c in labels.columns if c in schema_cols and c not in ('subject_id','source')]
        label_subset = labels[valid_cols].copy()
        # Drop rows where subject_id is null
        label_subset = label_subset.dropna(subset=['subject_id'])
        if len(label_subset) > 0:
            label_subset.to_sql('disease_labels', conn, if_exists='append', index=False,
                               method='multi', chunksize=1000)
        conn.execute("DELETE FROM disease_labels WHERE rowid NOT IN (SELECT MIN(rowid) FROM disease_labels GROUP BY subject_id, source)")
        print(f"  Labels: {len(label_subset)} subjects, {len(valid_cols)} cols")

    conn.commit()


# ── Migrate (data_v2 统一布局: DATA_ROOT/<dataset>/features_v2 + labels.csv + raw) ──
migrate_dataset('vital', str(DATA_ROOT / 'vital/features_v2'),
    str(DATA_ROOT / 'vital/labels.csv'), str(DATA_ROOT / 'vital/raw'))

migrate_dataset('mimic', str(DATA_ROOT / 'mimic/features_v2'),
    str(DATA_ROOT / 'mimic/labels.csv'), str(DATA_ROOT / 'mimic/raw'))

migrate_dataset('aurora', str(DATA_ROOT / 'aurora/features_v2'),
    str(DATA_ROOT / 'aurora/labels.csv'), str(DATA_ROOT / 'aurora/raw'),
    parquet_prefix='aurorabp')

migrate_dataset('anker', str(DATA_ROOT / 'anker/features_v2'),
    str(DATA_ROOT / 'anker/labels.csv'), str(DATA_ROOT / 'anker/raw'))

# ── Baselines ──
baseline_path = Path(__file__).parent.parent.parent / "analysis/05_reference_baselines/output/tables/vital_percentile_baselines.csv"
if baseline_path.exists():
    pd.read_csv(baseline_path).to_sql('baselines', conn, if_exists='append', index=False)
    print(f"\nBaselines: imported")

conn.commit()
db_size = os.path.getsize(str(DB_PATH)) / 1024**3
print(f"\nDatabase: {DB_PATH} ({db_size:.1f} GB)")
conn.close()
print("Done!")
