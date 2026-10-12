"""Tests for the shared waveform-overlay payload builders (ppg_agent/engine/overlay.py).

Contract under test:
  - build_beat_overlay: per-beat {beat_idx, start, end?, valid, keypoints?};
    end precedence = beat_lengths → kp['v'] → next start → absent
  - attach_beat_feats: raw per-beat features on valid beats, NaN cleaned
  - waveform_payload: {fs, n_samples, duration_sec, points, overlay}
  - enrich_overlay_feats (web analysis_service): per-beat percentile/level
    annotation for abnormal report features — same query as the report
"""
import numpy as np

from ppg_agent.engine.overlay import (
    attach_beat_feats,
    build_beat_overlay,
    downsample,
    waveform_payload,
)

STARTS = [100, 600, 1200, 1900]
VALID = [True, True, False, True]
KPS = [
    {"o": 0, "s": 120, "d": 300, "v": 500},
    {"o": 0, "s": 110, "d": 290, "v": 600},
    {},
    {"o": 0, "s": 130, "d": 310, "v": 700},
]
LENGTHS = [500, 600, 700, 800]


def test_end_precedence_beat_lengths():
    ov = build_beat_overlay(STARTS, VALID, KPS, LENGTHS, fs=500)
    assert [b["end"] for b in ov["beats"]] == [600, 1200, 1900, 2700]
    # invalid beat keeps start/end/valid but no keypoints
    b2 = ov["beats"][2]
    assert b2["valid"] is False and "keypoints" not in b2
    assert b2["start"] == 1200 and b2["end"] == 1900


def test_end_precedence_keypoint_v_fallback():
    # lengths shorter than kp['v'] would — drop lengths for beat 1, keep v
    ov = build_beat_overlay(STARTS, VALID, KPS, None, fs=500)
    # beat 0: kp v=500 → end=100+500; beat 3: kp v=700 → end=1900+700
    assert ov["beats"][0]["end"] == 600
    assert ov["beats"][3]["end"] == 2600


def test_end_precedence_next_start_and_missing():
    # no lengths, no kp v on some beats → next start; last beat → absent
    kps = [{"o": 0, "s": 10}, {}, {"o": 5}, {"o": 0, "s": 10}]
    ov = build_beat_overlay(STARTS, VALID, kps, None, fs=500)
    assert ov["beats"][0]["end"] == 600        # next start
    assert ov["beats"][1]["end"] == 1200       # next start
    assert ov["beats"][2]["end"] == 1900       # next start
    assert ov["beats"][3]["end"] is None       # last, no info


def test_no_beat_lengths_does_not_crash():
    # mirrors test_analysis_from_ctx_shape — ctx without beat templates
    ov = build_beat_overlay(STARTS, VALID, KPS, None, fs=500)
    assert len(ov["beats"]) == 4 and ov["fs"] == 500


def test_empty_inputs():
    ov = build_beat_overlay([], [], [], None, fs=500)
    assert ov == {"fs": 500, "beats": []}


def test_attach_beat_feats_valid_only_and_nan_clean():
    overlay = build_beat_overlay(STARTS, VALID, KPS, LENGTHS, fs=500)
    feats_list = [
        {"reflection_index": 0.85, "lasi": 2.1},
        {"reflection_index": float("nan"), "lasi": 2.5},
        {},
        {"reflection_index": 0.2, "lasi": float("inf")},
    ]
    out = attach_beat_feats(overlay, feats_list)
    assert "feats" in out["beats"][0]
    assert out["beats"][0]["feats"]["lasi"] == 2.1
    # NaN beat is valid here (mask says valid) → feats present, NaN→None
    assert out["beats"][1]["feats"]["reflection_index"] is None
    assert "feats" not in out["beats"][2]            # invalid beat
    assert out["beats"][3]["feats"]["lasi"] is None  # inf → None


def test_downsample_short_and_long():
    sig = np.linspace(0, 1, 100)
    assert downsample(sig, 200) == [float(x) for x in sig]  # passthrough
    out = downsample(sig, 10)
    assert len(out) == 10 and out[0] == 0.0 and out[-1] == 1.0


def test_waveform_payload_shape():
    sig = np.linspace(0, 1, 5000)
    beat_arrays = [np.zeros(n) for n in LENGTHS]  # lengths derive from templates
    payload = waveform_payload(sig, STARTS, VALID, KPS, beat_arrays,
                               fs=500, max_points=2000,
                               features_list=[{"ri": 0.8}] * 4)
    assert payload["fs"] == 500
    assert payload["n_samples"] == 5000
    assert payload["duration_sec"] == 10.0
    assert len(payload["points"]) == 2000
    assert len(payload["overlay"]["beats"]) == 4
    assert payload["overlay"]["beats"][0]["feats"] == {"ri": 0.8}


# ---------------------------------------------------------------------------
# enrich_overlay_feats (web analysis_service)
# ---------------------------------------------------------------------------

def _fake_report(abnormal: dict) -> dict:
    """stratified_percentiles with the given abnormal features + normal fill."""
    return {"stratified_percentiles": dict(abnormal),
            "baseline_source": "vital"}


def test_enrich_overlay_feats_unit(monkeypatch):
    from web.api.services import analysis_service as svc
    from ppg_agent.context import reset_context, get_context
    from ppg_agent.engine.percentile_engine import PercentileEngine

    calls = []

    def fake_query(self, feature, value, source, age=None, gender=None):
        calls.append((feature, value, source, age, gender))
        return {"level": "high", "percentile": "P85-P95"}

    monkeypatch.setattr(PercentileEngine, "query_stratified", fake_query)

    reset_context()
    ctx = get_context()
    ctx.age = 45.0
    ctx.segment_features_raw = {"gender": "F"}
    ctx.beat_records = [
        {"beat_idx": 0, "reflection_index": 0.85, "lasi": 2.0},
        {"beat_idx": 1, "reflection_index": float("nan"), "lasi": 2.1},
        {"beat_idx": 2, "reflection_index": 0.3, "lasi": 2.2},
    ]
    overlay = build_beat_overlay(STARTS[:3], VALID[:3], KPS[:3],
                                 LENGTHS[:3], fs=500)
    report = _fake_report({"reflection_index": {"level": "high"},
                           "lasi": {"level": "low"},
                           "t_sys_ms": {"level": "normal"}})  # normal → skipped
    out = svc.enrich_overlay_feats(overlay, report)
    # abnormal-only annotation, valid beats only, same query args as report
    assert "feats" in out["beats"][0]
    assert set(out["beats"][0]["feats"]) == {"reflection_index", "lasi"}
    assert out["beats"][0]["feats"]["reflection_index"] == {
        "v": 0.85, "level": "high", "percentile": "P85-P95"}
    assert calls[0] == ("reflection_index", 0.85, "vital", 45.0, "F")
    assert "feats" not in out["beats"][2]        # invalid beat
    assert out["beats"][1]["feats"] == {"lasi": {"v": 2.1, "level": "high",
                                                 "percentile": "P85-P95"}}  # NaN skipped


def test_enrich_overlay_feats_skips_missing_and_query_errors(monkeypatch):
    from web.api.services import analysis_service as svc
    from ppg_agent.context import reset_context, get_context
    from ppg_agent.engine.percentile_engine import PercentileEngine

    def fake_query(self, feature, value, source, age=None, gender=None):
        if feature == "no_baseline":
            return {"feature": feature, "error": "not in vital baseline"}
        return {"level": "high", "percentile": "P85-P95"}

    monkeypatch.setattr(PercentileEngine, "query_stratified", fake_query)
    reset_context()
    ctx = get_context()
    ctx.beat_records = [{"beat_idx": 0, "reflection_index": 0.85}]
    overlay = build_beat_overlay(STARTS[:1], VALID[:1], KPS[:1], LENGTHS[:1])
    report = _fake_report({"no_baseline": {"level": "high"}})
    out = svc.enrich_overlay_feats(overlay, report)
    assert "feats" not in out["beats"][0]      # query error → nothing attached
    # feature absent from beat record → skipped silently
    report2 = _fake_report({"reflection_index": {"level": "high"},
                            "absent_feat": {"level": "low"}})
    out2 = svc.enrich_overlay_feats(build_beat_overlay(STARTS[:1], VALID[:1],
                                                       KPS[:1], LENGTHS[:1]),
                                    report2)
    assert set(out2["beats"][0]["feats"]) == {"reflection_index"}


def test_enrich_overlay_feats_empty_abnormal_returns_unchanged():
    from web.api.services import analysis_service as svc
    overlay = build_beat_overlay(STARTS, VALID, KPS, LENGTHS)
    out = svc.enrich_overlay_feats(overlay, _fake_report({}))
    assert out is overlay and all("feats" not in b for b in out["beats"])


def test_enrich_overlay_feats_integration():
    """Real vital segment: per-beat levels match direct query_stratified."""
    from web.api.services import analysis_service as svc
    from ppg_agent.context import reset_context
    from ppg_agent.engine.percentile_engine import PercentileEngine

    reset_context()
    svc.load_signal("vital", "p000051", 1000)
    svc.normalize()
    svc.segment_beats_step()
    svc.extract_features_step("vital", "p000051", 1000)
    report = svc.run_phenotype("vital")
    overlay = svc.beat_overlay()
    svc.enrich_overlay_feats(overlay, report)

    assert all("end" in b for b in overlay["beats"])   # ctx.beats lengths present
    abnormal = {f for f, e in (report["stratified_percentiles"] or {}).items()
                if e.get("level") not in ("normal", "unknown", None)}
    engine = PercentileEngine()
    ctx_beats = 0
    for b in overlay["beats"]:
        if b["valid"]:
            ctx_beats += 1
        feats = b.get("feats") or {}
        assert set(feats) <= abnormal
        for f, info in feats.items():
            # per-beat annotation equals a direct query on the beat record
            v = info["v"]
            q = engine.query_stratified(f, v, report["baseline_source"],
                                        None, None)
            assert info["level"] == q["level"]
            assert info["percentile"] == q["percentile"]
    assert ctx_beats > 0


def test_pipeline_phenotype_overlay_enriched():
    """chat._pipeline(phenotype=True): overlay carries end + feats ⊆ abnormal."""
    from ppg_agent.context import fresh_async_task_context
    from web.api.routers import chat as chat_mod

    seg = {"source": "vital", "subject_id": "p000051", "seg_idx": 1000}
    out = fresh_async_task_context().run(
        lambda: chat_mod._pipeline(seg, phenotype=True))
    assert all("end" in b for b in out["overlay"]["beats"])
    abnormal = {f for f, e in (out["report"]["stratified_percentiles"] or {}).items()
                if e.get("level") not in ("normal", "unknown", None)}
    for b in out["overlay"]["beats"]:
        assert set(b.get("feats") or {}) <= abnormal
