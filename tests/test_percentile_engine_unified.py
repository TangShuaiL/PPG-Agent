"""Tests for the unified-baseline changes to PercentileEngine (step 29).

Covers the four engine changes from the unified-analysis plan:
  1. 4-step fallback ladder in _get_baseline
  2. resolve_source aurora a/o mapping
  3. per-source reliability weights (data-driven, hardcode fallback)
  4. VAI sign alignment (aurora only; vital/mimic path unchanged)

All tests use synthetic baseline CSVs / mocked knowledge maps — no real DB.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ppg_agent.engine.percentile_engine import (
    FEATURE_RELIABILITY,
    VAI_FEATURES,
    PercentileEngine,
)

FEATS = ["reflection_index", "beat_len_ms", "t_sys_ms", "diastolic_width_50"]


def _row(fc: str, ag: str = "all", g: str = "all") -> dict:
    """One baseline row: p50=20, p25=10, p75=30 → std_est = 20/1.35 ≈ 14.8."""
    return {"feature": fc, "age_group": ag, "gender": g, "n": 100,
            "p05": 5, "p10": 8, "p20": 15, "p25": 10, "p50": 20,
            "p75": 30, "p85": 40, "p90": 45, "p95": 50, "mean": 21, "std": 8,
            "p05_ci_low": 4, "p05_ci_high": 6, "p50_ci_low": 18,
            "p50_ci_high": 22, "p95_ci_low": 48, "p95_ci_high": 52}


@pytest.fixture
def engine(tmp_path) -> PercentileEngine:
    """Engine over synthetic baselines: vital (dense), aurora_a/o (sparse)."""
    bdir = tmp_path / "baselines"
    bdir.mkdir()
    for src in ("vital", "aurora_a", "aurora_o"):
        rows = [_row(fc) for fc in FEATS]
        if src == "vital":
            rows += [_row(fc, "60-70", "M") for fc in FEATS]   # exact cell
            rows += [_row(fc, "30-40", "F") for fc in FEATS]   # exact cell
        else:
            rows += [_row(fc, "60-70", "all") for fc in FEATS]  # gender collapsed
        # aurora_o lacks even the global-per-gender tier
        if src != "aurora_o":
            rows += [_row(fc, "all", "M") for fc in FEATS]
        pd.DataFrame(rows).to_csv(bdir / f"{src}_percentile_baselines.csv",
                                  index=False)
    return PercentileEngine(baseline_dir=str(bdir),
                            db_path=str(tmp_path / "none.db"))


# ---------------------------------------------------------------------------
# 1. Fallback ladder
# ---------------------------------------------------------------------------

class TestFallbackLadder:
    def test_exact_cell_when_dense(self, engine):
        """vital has (60-70, M) → exact hit, n=100 baseline."""
        q = engine.query("reflection_index", 25, "vital", "60-70", "M")
        assert q["n_baseline"] == 100

    def test_gender_collapse_for_sparse_aurora(self, engine):
        """aurora_a lacks (60-70, M) → falls back to (60-70, 'all')."""
        q = engine.query("reflection_index", 25, "aurora_a", "60-70", "M")
        assert q["n_baseline"] == 100  # the (60-70, all) row

    def test_age_collapse_then_global(self, engine):
        """aurora_o lacks (30-40, F) and (all, F) → global (all, all)."""
        q = engine.query("reflection_index", 25, "aurora_o", "30-40", "F")
        assert q["n_baseline"] == 100  # (all, all) row

    def test_global_last_resort(self, engine):
        """Unknown feature falls through all tiers → None → error dict."""
        q = engine.query("no_such_feature", 25, "aurora_o")
        assert "error" in q

    def test_aurora_tail_bin_uses_ladder(self, engine):
        """80+ bin (absent from aurora baselines) must still query cleanly."""
        q = engine.query("t_sys_ms", 25, "aurora_a", "80+", "F")
        assert q["n_baseline"] == 100  # (all, all)


# ---------------------------------------------------------------------------
# 2. Scenario resolution
# ---------------------------------------------------------------------------

class TestResolveSource:
    def test_aurora_scenario_a(self):
        assert PercentileEngine.resolve_source("aurora", "a") == "aurora_a"
        assert PercentileEngine.resolve_source("aurora", "A") == "aurora_a"

    def test_aurora_scenario_o(self):
        assert PercentileEngine.resolve_source("aurora", "o") == "aurora_o"
        assert PercentileEngine.resolve_source("aurorabp", "oscillometric") == "aurora_o"

    def test_aurora_unknown_scenario_passthrough(self):
        assert PercentileEngine.resolve_source("aurora", None) == "aurora"

    def test_non_aurora_ignores_scenario(self):
        assert PercentileEngine.resolve_source("vital", "a") == "vital"
        assert PercentileEngine.resolve_source("mimic", "o") == "mimic"


# ---------------------------------------------------------------------------
# 3. Per-source reliability weights
# ---------------------------------------------------------------------------

class TestReliabilityWeights:
    def test_per_source_wins_over_hardcode(self, engine):
        engine._reliability_map = {"aurora_a": {"reflection_index": 0.95}}
        q = engine.query("reflection_index", 25, "aurora_a")
        assert q["reliability"] == pytest.approx(0.95)

    def test_missing_feature_falls_back_to_hardcode(self, engine):
        engine._reliability_map = {"aurora_a": {"reflection_index": 0.95}}
        q = engine.query("beat_len_ms", 25, "aurora_a")
        assert q["reliability"] == pytest.approx(FEATURE_RELIABILITY["beat_len_ms"])

    def test_missing_source_falls_back_to_hardcode(self, engine):
        engine._reliability_map = {"vital": {"reflection_index": 0.95}}
        q = engine.query("reflection_index", 25, "aurora_o")
        assert q["reliability"] == pytest.approx(FEATURE_RELIABILITY["reflection_index"])

    def test_report_uses_per_source_top10(self, engine):
        engine._reliability_map = {"aurora_a": {fc: 0.99 for fc in FEATS}}
        rep = engine.generate_report(
            {fc: 25.0 for fc in FEATS}, source="aurora_a")
        stratified = rep["stratified_percentiles"]
        assert len(stratified) > 0
        assert all(round(v["reliability"], 2) == 0.99
                   for v in stratified.values())


# ---------------------------------------------------------------------------
# 4. VAI sign alignment
# ---------------------------------------------------------------------------

class TestSignAlignment:
    N_MOCK_VAI_FEATS = 3  # FEATS ∩ VAI_FEATURES (reflection_index, t_sys_ms, diastolic_width_50)

    def _features(self) -> dict[str, float]:
        # value 40 >> p50=20 → positive z for every feature
        return {fc: 40.0 for fc in VAI_FEATURES}

    def _set_aurora_map(self, engine, signs: dict, rho: float = 0.3):
        """Install a sign map + rho table (rho above the 0.15 selection cut)."""
        engine._sign_maps = {"aurora_a": signs}
        engine._sign_rhos = {"aurora_a": {fc: rho for fc in signs}}

    def test_vital_no_map_keeps_positive_vai(self, engine):
        engine._sign_maps = {}
        vai = engine.compute_vai(self._features(), "vital")
        assert vai["vai"] > 0 and "error" not in vai

    def test_aurora_map_flips_direction(self, engine):
        signs = {fc: -1 for fc in VAI_FEATURES}
        self._set_aurora_map(engine, signs)
        vai = engine.compute_vai(self._features(), "aurora_a")
        assert vai["vai"] < 0  # all values flipped negative
        assert vai["n_features_used"] == self.N_MOCK_VAI_FEATS

    def test_sign_zero_skips_feature(self, engine):
        signs = {fc: 1 for fc in VAI_FEATURES}
        signs["t_sys_ms"] = 0  # excluded
        self._set_aurora_map(engine, signs)
        vai = engine.compute_vai(self._features(), "aurora_a")
        assert vai["n_features_used"] == self.N_MOCK_VAI_FEATS - 1
        assert "t_sys_ms" not in [c["feature"] for c in vai["top_contributors"]]

    def test_sign_zero_alone_yields_zero_vai(self, engine):
        signs = {fc: 0 for fc in VAI_FEATURES}
        self._set_aurora_map(engine, signs)
        vai = engine.compute_vai(self._features(), "aurora_a")
        assert vai["n_features_used"] == 0 and vai["vai"] == 0.0

    def test_low_rho_excluded_by_selection(self, engine):
        """|ρ| ≤ 0.15 → feature dropped even with sign≠0 (data-driven choice)."""
        signs = {fc: 1 for fc in VAI_FEATURES}
        self._set_aurora_map(engine, signs, rho=0.10)
        vai = engine.compute_vai(self._features(), "aurora_a")
        assert vai["n_features_used"] == 0  # all below the 0.15 cut

    def test_missing_map_file_keeps_sign_one(self, engine):
        """vital (no map JSON) keeps the original sign=1 path."""
        engine._sign_maps = {}
        vai = engine.compute_vai(self._features(), "vital")
        assert vai["vai"] > 0


# ── Agent-2 knowledge layer regression (steps 16-27) ───────────────────────

class TestAgent2KnowledgeRegression:
    def test_default_args_keep_legacy_shape(self, engine):
        """New parameters are all optional — the legacy report shape must not
        change when the caller passes nothing new."""
        rep = engine.generate_report({fc: 25.0 for fc in FEATS}, source="vital")
        for key in ("vascular_age", "stratified_percentiles",
                    "interpretation_rules", "beat_consistency", "confidence",
                    "confidence_note", "safety_note", "baseline_source"):
            assert key in rep
        assert rep["confidence"] in ("low", "medium", "high")

    def test_missing_knowledge_degrades_safely(self, engine):
        """With every knowledge layer empty, the five new fields must degrade
        to empty/available:False — the engine never raises on missing
        optional knowledge."""
        engine._disease_map = {}
        engine._cluster_center_maps = {"vital": None}
        engine._scale_warn = {}
        engine._window_map = {}
        engine._mdc_map = {}
        rep = engine.generate_report({fc: 25.0 for fc in FEATS}, source="vital",
                                     n_beats_valid=14)
        assert rep["disease_associations"] == {}
        assert rep["phenotype_subtype"]["available"] is False
        assert rep["scale_warnings"]["available"] is False
        assert rep["change_reference"]["available"] is False
        assert rep["window_context"]["is_short_window"] is True  # beats axis
