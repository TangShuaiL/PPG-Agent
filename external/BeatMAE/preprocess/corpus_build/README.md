# corpus_build/ — corpus construction pipeline (**a record, not runnable**)

> ⚠ **This directory is a record of how the corpus was built, not a runnable tool.** It
> depends on a private 4.8 GB phenotype database and on gated raw data, and it is missing two
> post-hoc fix stages (see "Full lineage" below).
> **Why it is kept:** so readers and reviewers can see what processing the corpus went through.
> The method overview is one level up in `preprocess/README.md`; **the only runnable part is
> `../features/`**.

The corpus-build stages, executed in numeric order. Apart from `compute_feat_stats.py` (which
depends on the project's `data.memmap_npz_member`), each is a standalone script depending only
on `sqlite3`/`pandas`/`numpy`/`h5py`, and can be reused wholesale for a future corpus v3.

## Pipeline order

| Stage | Script | Purpose | Main output |
|---|---|---|---|
| 1 | `01_make_folds.py` | Stratified 10-fold split of eligible subjects | `folds/subject_folds_v2.parquet` (+ fold QC report) |
| 2 | `02_build_corpus.py` | Corpus materialization, anomaly-excluded | Segment-level corpus + DB (`manifest.parquet`, `labels.parquet`, `signals/ppg_shard_*.npy`, `signals/ecg_shard_*.npy`) |
| 3 | `03_build_features.py` | Feature/keypoint materialization (pure transport, no recomputation) | `seg_features.npy` (n_segs × 61, fp16) + `keypoints.npz` / `beat_features.npz` (CSR; 10 × int16 and 40 × fp16 per beat) + `features_meta.json` |
| 4 | `04_register_corpus.py` | Write fold assignment back to the DB and register the corpus version | DB `fold_id` + `corpus_registry` |
| 5 | `05_annotate_anomalies.py` | Annotate data anomalies back to the DB (idempotent, rerunnable) | DB `anomaly_note` |
| 6 | `compute_feat_stats.py` | Per-beat 40-dim mean/std (2000 segments drawn corpus-wide with `np.random.default_rng(0)`; reproducible) | `config/feat_stats_v2c.npz`, used by `train/pretrain.py` for the auxiliary task's z-parameterization |

Stage 1's stratification variables: age (4 within-corpus quantile bins) × gender × segment
volume (3 bins) × median SBP (3 bins; Aurora has no labels and gets its own bin), assigned by
iterative stratification (Sechidis-style) with a fixed seed of 42. The three datasets are split
independently and same-numbered folds are merged into global folds.

## Full lineage (important)

The corpus as it exists = the five stages above **plus two post-hoc fix stages** whose scripts
have since been removed from the tree (`22_corpus_v2new_fix.py`, three corpus fixes;
`23_prune_corpus_v2new.py`, physical cleanup driven by `sig_ok`). Their scripts are not in the tree.

**Therefore running 01–05 alone is not sufficient to rebuild the corpus** — the fix semantics
have to be restored from the lineage above.

## Status

The corpus is frozen; the anomaly-exclusion rules are documented in `../README.md` and in
`05_annotate_anomalies.py`. Day-to-day training and evaluation only read the corpus and do not
go through this directory. This directory would only become active again when extending to a
corpus v3.
