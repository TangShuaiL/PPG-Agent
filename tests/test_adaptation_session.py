"""AdaptationSession tests — guardrails + scripted end-to-end flow."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from ppg_agent.agents.adaptation_session import (
    AdaptationSession,
    ParamChange,
    create_adaptation_session,
)
from ppg_agent.processing_params import PARAM_SPEC, load_preset

SID = "a000"  # aurora subject with 16 segments


def make_session(**kwargs):
    kwargs.setdefault("source", "aurora")
    kwargs.setdefault("subject_id", SID)
    kwargs.setdefault("seg_idx", 0)
    kwargs.setdefault("preset", "default")
    kwargs.setdefault("mock_llm", True)
    kwargs.setdefault("max_rounds", 3)
    kwargs.setdefault("gate_n_subjects", 12)
    return AdaptationSession(**kwargs)


class TestGuardrails:
    def test_invalid_param_option_dropped(self, monkeypatch):
        s = make_session()
        s.start()
        crafted = [
            ParamChange(param="bogus_param", value=1, rationale="x", expected_effect="y"),
            ParamChange(param="peak_distance_ms", value=99999, rationale="x", expected_effect="y"),
            ParamChange(param="peak_prominence", value=0.08, rationale="x", expected_effect="y"),
        ]
        monkeypatch.setattr("ppg_agent.agents.adaptation_session._llm_propose",
                                      lambda session: crafted)
        rec = s.rounds[-1]
        opts = s._propose_options(rec)
        assert len(opts) == 1, f"only the valid option should survive: {opts}"
        assert opts[0]["param"] == "peak_prominence"

    def test_duplicate_params_deduped(self, monkeypatch):
        s = make_session()
        s.start()
        crafted = [
            ParamChange(param="peak_prominence", value=0.08, rationale="x", expected_effect="y"),
            ParamChange(param="peak_prominence", value=0.03, rationale="x", expected_effect="y"),
            ParamChange(param="peak_distance_ms", value=350, rationale="x", expected_effect="y"),
        ]
        monkeypatch.setattr("ppg_agent.agents.adaptation_session._llm_propose",
                                      lambda session: crafted)
        opts = s._propose_options(s.rounds[-1])
        params = [o["param"] for o in opts]
        assert len(params) == len(set(params)), f"no duplicate params: {params}"

    def test_bclass_option_tagged(self, monkeypatch):
        s = make_session()
        s.start()
        crafted = [
            ParamChange(param="onset_search_ms", value=160, rationale="x", expected_effect="y"),
        ]
        monkeypatch.setattr("ppg_agent.agents.adaptation_session._llm_propose",
                                      lambda session: crafted)
        opts = s._propose_options(s.rounds[-1])
        assert opts[0]["tag"] == "[B-class]"

    def test_no_repeat_of_tried_params(self, monkeypatch):
        s = make_session()
        s.start()
        s.apply("a")  # applies first mock option (some param)
        tried = s.rounds[-1].changed["param"]
        crafted = [
            ParamChange(param=tried, value=s.params[tried], rationale="x", expected_effect="y"),
            ParamChange(param="extra_bandpass", value=True, rationale="x", expected_effect="y"),
        ]
        monkeypatch.setattr("ppg_agent.agents.adaptation_session._llm_propose",
                                      lambda session: crafted)
        opts = s._propose_options(s.rounds[-1])
        params = [o["param"] for o in opts]
        assert tried not in params, f"tried param must not be re-proposed: {tried}"

    def test_round_limit_actions_only(self):
        s = make_session(max_rounds=0)
        s.start()
        assert s.rounds[-1].options == []

    def test_quality_drop_offers_revert(self):
        s = make_session()
        s.start()
        # Find an option that degrades quality, or simulate: directly construct
        # a round with degraded quality then check proposal includes revert
        s.rounds.append(type(s.rounds[-1])(
            round_idx=1, params=s.params, metrics={**s.rounds[0].metrics, "quality": 0.01},
            diagnosis=s.rounds[0].diagnosis, changed={"param": "peak_prominence",
                                                       "old_value": 0.05, "new_value": 0.3}))
        opts = s._propose_options(s.rounds[-1])
        assert any(o.get("action") == "revert" for o in opts), \
            "degraded quality must offer a revert option"


class TestScriptedFlow:
    def test_full_session_saves_preset(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ppg_agent.agents.adaptation_session.save_preset",
                            lambda name, params: {"preset": name, "params": params})
        s = make_session(max_rounds=4)
        panel = s.start()
        assert "Round 0" in panel
        # apply a param option
        letters = "abcd"
        first = s.rounds[-1].options[0]
        letter = letters[s.rounds[-1].options.index(first)].upper()
        panel = s.apply(letter.lower())
        assert "Round 1" in panel
        # validate
        panel = s.apply("v")
        assert "Validation" in panel
        # finalize — monkeypatched gate to avoid slow real extraction
        monkeypatch.setattr(s, "run_acceptance_gate",
                            lambda: {"median_split_half_rho": 0.8,
                                     "frac_split_half_gt_05": 1.0,
                                     "vai_rho_age": 0.4, "vai_p": 0.01,
                                     "n_vai_subjects": 12, "n_features": 30,
                                     "passed": True})
        panel = s.apply("f")
        assert "Gate PASSED" in panel
        assert s.status == "done"

    def test_quit_aborts(self):
        s = make_session()
        s.start()
        out = s.apply("q")
        assert s.status == "aborted"
        assert "aborted" in out

    def test_revert_restores_previous(self):
        s = make_session(max_rounds=4)
        s.start()
        prev_params = dict(s.params)
        s.apply("a")
        assert s.params != prev_params
        out = s.apply("r")
        assert "Round" in out
        # after revert, params should equal the pre-change round's params
        assert s.params == prev_params


class TestParamSpec:
    def test_spec_covers_engine_defaults(self):
        from ppg_agent.engine.signal_engine import process_ppg_segment
        r = process_ppg_segment(__import__("numpy").zeros(5000), 500)
        assert r["params_used"] is not None
        assert r["n_rejected_mad"] == 0

    def test_preset_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ppg_agent.processing_params.PRESET_DIR", tmp_path)
        from ppg_agent.processing_params import save_preset, load_preset
        save_preset("test_rt", {"peak_prominence": 0.08, "extra_bandpass": True})
        loaded = load_preset("test_rt")
        assert loaded["peak_prominence"] == 0.08
        assert loaded["extra_bandpass"] is True
        assert loaded["onset_search_ms"] == 120  # default fill


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
