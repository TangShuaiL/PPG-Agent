# Knowledge Artifacts

This directory contains lightweight knowledge products used by the Evidence
Engine and Agent tools.

Phase C generated files:

- `pulse_db_reference_stats.json`: per-dataset feature distributions for core
  PPG/PTT/ABP features, split by `all_success` and `quality_ok`.
- `pulse_db_correlations.json`: Spearman correlations between features and
  `sbp_label` / `dbp_label` on quality-ok segments.  Aurora scenarios a/o
  are stored separately (`aurora_a`/`aurora_o`), never pooled.  Coarse
  prior — the engine prefers `population_bp_association.json` when present.
- `population_bp_association.json`: rigorous population feature×BP layer
  (analysis step 30): per-patient medians, rank partial correlation
  controlling age+sex, BH-FDR per source×target (q<0.05 kept).  Consumed
  by Agent 2's `bp_context` as `population_rho_prior` and surfaced in the
  `feature_interpretation` KB as the `bp_patterns` tier.  Forward-only:
  weak population correlations, never a per-individual BP estimate.
- `pulse_db_condition_stats.json`: SBP/DBP distributions for predefined RI,
  LASI, and PTT feature patterns.
- `feature_clusters.json`: joint feature-redundancy map (within-person ∩
  between-person gates, analysis steps 28/29) consumed by Agent 2's
  narrative folding.  Rebuild with `scripts/build_feature_clusters_kb.py`
  after re-running the analysis tables.
- `feature_interpretation.json`: physiological interpretability knowledge
  for Agent 2's `feature_interpretation` tool — per-feature mechanism
  (feature_rules), dataset age trends (sex-adjusted partial ρ),
  labeled-group patterns (rank-ANCOVA, age+sex adjusted, FDR<0.05) and
  population BP correlations (step 30, weak ρ, FDR<0.05).
  Forward descriptions only (labeled group → feature pattern); never
  reversed into diagnosis/symptom claims, and BP correlations are never
  used to estimate an individual's blood pressure.  Pure PPG — no BP
  labels needed.  Rebuild with `scripts/build_feature_interpretation_kb.py`.

The raw `.mat` files and large parquet caches remain outside this code
directory and are accessed through `ppg_agent.config`.

Patient-level or case-level clinical tables derived from MIMIC/VitalDB are
treated as local artifacts and are not tracked in git.
