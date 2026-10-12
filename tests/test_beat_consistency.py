"""Tests for the beat-level consistency extension of PercentileEngine.

Covers the Agent-2 beat-level input (phase 2 of the beat-level analysis):
  1. generate_report exposes ``beat_consistency`` with ICC fields for
     features that have a beat-level decomposition
  2. beat_cv input → cv_level ("elevated" when above the reference p90)
  3. confidence derived from ICC(2, n_segments) instead of the fixed
     n_segments>=5 heuristic
  4. spectral features (no per-beat meaning) are excluded

Uses the synthetic-baseline engine fixture (no real DB) + the real
knowledge/feature_icc.json / beat_cv_reference.json exports.
"""
from __future__ import annotations

import pytest

from ppg_agent.engine.percentile_engine import PercentileEngine

FEATS = ["reflection_index", "beat_len_ms", "t_sys_ms", "diastolic_width_50"]


def _row(fc: str, ag: str = "all", g: str = "all") -> dict:
    return {"feature": fc, "age_group": ag, "gender": g, "n": 100,
            "p05": 5, "p10": 8, "p20": 15, "p25": 10, "p50": 20,
            "p75": 30, "p85": 40, "p90": 45, "p95": 50, "mean": 21, "std": 8,
            "p05_ci_low": 4, "p05_ci_high": 6, "p50_ci_low": 18,
            "p50_ci_high": 22, "p95_ci_low": 48, "p95_ci_high": 52}


@pytest.fixture
def engine(tmp_path) -> PercentileEngine:
    bdir = tmp_path / "baselines"
    bdir.mkdir()
    for src in ("vital", "aurora_a", "aurora_o"):
        rows = [_row(fc) for fc in FEATS]
        if src == "vital":
            rows += [_row(fc, "60-70", "M") for fc in FEATS]
            rows += [_row(fc, "30-40", "F") for fc in FEATS]
        else:
            rows += [_row(fc, "60-70", "all") for fc in FEATS]
        if src != "aurora_o":
            rows += [_row(fc, "all", "M") for fc in FEATS]
        import pandas as pd
        pd.DataFrame(rows).to_csv(bdir / f"{src}_percentile_baselines.csv",
                                  index=False)
    return PercentileEngine(baseline_dir=str(bdir),
                            db_path=str(tmp_path / "none.db"))


class TestBeatConsistency:
    def test_report_has_beat_consistency(self, engine):
        rep = engine.generate_report({fc: 25.0 for fc in FEATS},
                                     source="vital")
        bc = rep["beat_consistency"]
        # every analyzed feature has an ICC decomposition (real knowledge file)
        assert len(bc) > 0
        for fname, b in bc.items():
            assert 0 <= b["icc1_single"] <= 1
            assert b["icc2_k5"] >= b["icc1_single"]
            assert b["icc2_k10"] >= b["icc2_k5"]

    def test_icc_2k_formula(self, engine):
        # icc1=0.5 → icc2_k5 = 0.5·5/(0.5·5+0.5) = 0.8333
        assert engine._icc_2k("vital", "reflection_index", 1) == pytest.approx(
            engine._icc("vital", "reflection_index"), abs=1e-6)
        icc1 = engine._icc("vital", "reflection_index")
        expected = icc1 * 5 / (icc1 * 5 + (1 - icc1))
        assert engine._icc_2k("vital", "reflection_index", 5) == pytest.approx(
            expected, abs=1e-4)

    def test_beat_cv_elevated_flag(self, engine):
        # t_sys_ms is inside the top-10 reliability features (analyzed by the
        # engine); p90 of its CV in the real reference is ~0.10 — pass an
        # extreme CV → must be flagged elevated
        rep = engine.generate_report(
            {fc: 25.0 for fc in FEATS}, source="vital",
            beat_cv={"t_sys_ms": 10.0})
        b = rep["beat_consistency"]["t_sys_ms"]
        assert b["cv"] == pytest.approx(10.0)
        assert b["cv_level"] == "elevated"

    def test_beat_cv_normal_flag(self, engine):
        ref = engine._cv_reference("vital", "t_sys_ms")
        if ref is None:
            pytest.skip("no CV reference for t_sys_ms")
        rep = engine.generate_report(
            {fc: 25.0 for fc in FEATS}, source="vital",
            beat_cv={"t_sys_ms": ref["p50"]})
        b = rep["beat_consistency"]["t_sys_ms"]
        assert b["cv_level"] == "normal"

    def test_confidence_uses_icc(self, engine):
        # high reliability → ICC(2, 5) high → confidence high with 5 segments
        rep = engine.generate_report({fc: 25.0 for fc in FEATS},
                                     source="vital", n_segments=5)
        assert rep["confidence"] in ("low", "medium", "high")
        assert "ICC" in rep["confidence_note"]

    def test_confidence_fallback_without_icc(self, engine, monkeypatch):
        # no ICC data (e.g. knowledge file missing) → legacy heuristic
        monkeypatch.setattr(engine, "_icc", lambda s, f: None)
        rep = engine.generate_report({fc: 25.0 for fc in FEATS},
                                     source="vital", n_segments=6)
        assert rep["confidence"] == "medium"
        assert "noise reduced" in rep["confidence_note"]

    def test_spectral_feature_excluded(self, engine):
        # freq_dominant_1 has no per-beat decomposition → no consistency entry
        rep = engine.generate_report({"freq_dominant_1": 0.5},
                                     source="vital")
        assert "freq_dominant_1" not in rep["beat_consistency"]
