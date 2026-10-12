-- PPG Phenotype Database Schema
-- Three datasets: VITAL (surgery), MIMIC (ICU), Aurora (wearable)
-- Waveforms stored as file references, not raw data.

-- ============================================================
-- Subjects
-- ============================================================
CREATE TABLE IF NOT EXISTS subjects (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id    TEXT NOT NULL,
    source        TEXT NOT NULL CHECK(source IN ('vital','mimic','aurora','anker')),
    age           REAL,
    gender        TEXT,
    n_segments    INTEGER DEFAULT 0,
    -- data-quality registry note; subjects flagged here are EXCLUDED from
    -- the analysis materialization (step-01 demographics gate) but kept
    -- in this registry.  NULL = clean.
    anomaly_note  TEXT,
    created_at    TEXT DEFAULT (datetime('now')),
    split         TEXT,          -- train|test|excluded_v2c  (v2c fold; test iff fold∈{1,2})
    scenario      TEXT,          -- aurora only: a=lab, o=dynamic
    fold_id       INTEGER,       -- v2c 权威折号
    person_group  TEXT,          -- anker 人组分组
    UNIQUE(subject_id, source)
);

CREATE INDEX idx_subjects_source ON subjects(source);
CREATE INDEX idx_subjects_age ON subjects(age);

-- ============================================================
-- Segments
-- ============================================================
CREATE TABLE IF NOT EXISTS segments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id      TEXT NOT NULL,
    source          TEXT NOT NULL,
    seg_idx         INTEGER NOT NULL,
    waveform_path   TEXT,
    quality_score   REAL,
    quality_tier    TEXT CHECK(quality_tier IN ('good','fair','failed')),
    sbp_label       REAL,
    dbp_label       REAL,
    n_beats_valid   INTEGER,
    n_beats_total   INTEGER,
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(subject_id, source) REFERENCES subjects(subject_id, source),
    UNIQUE(subject_id, source, seg_idx)
);

CREATE INDEX idx_segments_subject ON segments(subject_id, source);
CREATE INDEX idx_segments_quality ON segments(quality_tier);
CREATE INDEX idx_segments_source ON segments(source);

-- ============================================================
-- Segment Features (wide table: 1 row per segment, ~147 cols)
-- ============================================================
CREATE TABLE IF NOT EXISTS segment_features (
    seg_id          INTEGER PRIMARY KEY,
    -- Arterial stiffness
    lasi_mean               REAL, slope_systolic_mean     REAL,
    slope_systolic_n_mean   REAL, slope_md_mean           REAL,
    slope_md_n_mean         REAL, pulse_width_50_mean     REAL,
    pulse_width_90_mean     REAL, bwr_50_mean             REAL,
    -- Wave reflection
    reflection_index_mean   REAL, ni_dn_mean              REAL,
    ni_ip_mean              REAL, t_dn_mean               REAL,
    nt_dn_mean              REAL, intensity_ratio_s_d_mean REAL,
    -- Pulse morphology
    systolic_width_50_mean  REAL, systolic_width_75_mean  REAL,
    diastolic_width_50_mean REAL, diastolic_width_75_mean REAL,
    -- Perfusion
    perfusion_index_mean    REAL, normalized_pulse_vol_mean REAL,
    ppg_characteristic_mean REAL, mean_intensity_mean     REAL,
    pulse_amp_mean          REAL,
    -- Statistical
    skewness_mean           REAL, kurtosis_mean           REAL,
    rms_mean                REAL, crest_factor_mean       REAL,
    iqr_mean                REAL,
    -- Spectral
    freq_dominant_1_mean    REAL, freq_dominant_2_mean    REAL,
    freq_dominant_3_mean    REAL, spectral_entropy_mean   REAL,
    wavelet_entropy_mean    REAL, spectral_centroid_mean  REAL,
    fsqi_mean               REAL,
    -- Beat metadata
    beat_len_ms_mean        REAL, hr_bpm_mean             REAL,
    t_sys_ms_mean           REAL, t_dia_ms_mean           REAL,
    -- PTT (ABP-derived, only available for VITAL/MIMIC)
    ptt_abp_mean_ms         REAL, ptt_abp_std_ms          REAL,
    ptt_abp_min_ms          REAL, ptt_abp_max_ms          REAL,
    ptt_ppg_mean_ms         REAL, ptt_ppg_std_ms          REAL,
    ptt_ppg_min_ms          REAL, ptt_ppg_max_ms          REAL,
    -- HRV
    hrv_sdnn                REAL, hrv_rmssd               REAL,
    hrv_pnn50               REAL, hrv_mean_rr             REAL,
    hrv_std_rr              REAL,
    ecg_hr_mean             REAL, ecg_hr_std              REAL,
    -- ABP derived (only VITAL/MIMIC)
    abp_sbp_mean            REAL, abp_dbp_mean            REAL, abp_pp_mean       REAL,
    -- Quality
    n_beats_ecg             INTEGER, beat_ratio_ppg_ecg   REAL,
    ptt_hr_product          REAL,
    -- Timestamps (Aurora only)
    timestamp               TEXT,
    FOREIGN KEY(seg_id) REFERENCES segments(id)
);

CREATE INDEX idx_features_seg ON segment_features(seg_id);

-- ============================================================
-- Disease Labels (wide table, one row per subject)
-- ============================================================
CREATE TABLE IF NOT EXISTS disease_labels (
    subject_id      TEXT NOT NULL,
    source          TEXT NOT NULL,
    -- VITAL surgical labels
    preop_htn       INTEGER, preop_dm       INTEGER,
    asa             REAL,    emop           INTEGER,
    death_inhosp    INTEGER,
    -- MIMIC ICU labels
    gender_label            TEXT,
    comorbidity_count       INTEGER,
    hypertension_essential  INTEGER, heart_failure          INTEGER,
    acute_kidney_failure    INTEGER, diabetes_type2         INTEGER,
    sepsis                  INTEGER, coronary_artery_disease INTEGER,
    myocardial_infarction   INTEGER, hypertensive_ckd       INTEGER,
    atrial_fibrillation     INTEGER, hyperlipidemia         INTEGER,
    ckd                     INTEGER, copd                   INTEGER,
    stroke                  INTEGER, obesity                INTEGER,
    hypertensive_heart_ckd  INTEGER, diabetes_complicated   INTEGER,
    hypertensive_heart      INTEGER,
    -- Aurora self-report labels
    self_report_htn         TEXT,    high_blood_pressure     INTEGER,
    aurora_cad              INTEGER, aurora_diabetes        INTEGER,
    arrhythmia              INTEGER, previous_heart_attack  INTEGER,
    previous_stroke         INTEGER, aurora_heart_failure   INTEGER,
    aortic_stenosis         INTEGER, valvular_heart_disease INTEGER,
    other_cv_diseases       INTEGER, cvd_meds               INTEGER,
    -- Aurora demographics
    height                  REAL,    weight                 REAL,
    FOREIGN KEY(subject_id, source) REFERENCES subjects(subject_id, source),
    UNIQUE(subject_id, source)
);

CREATE INDEX idx_labels_subject ON disease_labels(subject_id, source);

-- ============================================================
-- Reference Baselines (percentile lookup table for Agent 2)
-- ============================================================
CREATE TABLE IF NOT EXISTS baselines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    feature     TEXT NOT NULL,
    age_group   TEXT,
    gender      TEXT,
    source      TEXT DEFAULT 'vital',
    n           INTEGER,
    p05         REAL, p20         REAL, p25         REAL,
    p50         REAL, p75         REAL, p85         REAL, p95         REAL,
    mean        REAL, std         REAL,
    p05_ci_low  REAL, p05_ci_high REAL,
    p50_ci_low  REAL, p50_ci_high REAL,
    p95_ci_low  REAL, p95_ci_high REAL
);

CREATE INDEX idx_baselines_feature ON baselines(feature, age_group, gender);

-- ============================================================
-- Processing Log (track data upload and processing)
-- ============================================================
CREATE TABLE IF NOT EXISTS processing_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id  TEXT,
    seg_idx     INTEGER,
    source      TEXT,
    action      TEXT,           -- 'upload', 'feature_extraction', 'phenotype_assessment'
    status      TEXT DEFAULT 'pending',  -- 'pending','running','success','failed'
    error_msg   TEXT,
    started_at  TEXT,
    completed_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_log_status ON processing_log(status);
CREATE INDEX idx_log_subject ON processing_log(subject_id, source);
