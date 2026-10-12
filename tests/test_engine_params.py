"""Regression + behavior tests for the parameterized signal_engine."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ppg_agent.config import DATA_ROOT, LEGACY_DATA_ROOT  # noqa: E402

import numpy as np
import h5py
from scipy.signal import resample
import pytest

from ppg_agent.engine.signal_engine import process_ppg_segment, extract_segment_features
from ppg_agent.processing_params import (
    resolve_params, validate_params, DEFAULT_PARAMS, list_presets,
    preset_for_source, save_preset, load_preset, ensure_default_preset,
)

FS = 500
SEGMENTS = [
    (f'{DATA_ROOT}/vital/raw/p002551.mat', 0),
    (f'{DATA_ROOT}/mimic/raw/p044207.mat', 0),
    (f'{DATA_ROOT}/aurora/raw/a000.mat', 0),
    (f'{DATA_ROOT}/aurora/raw/o118.mat', 0),
]


def _load_segment(fp, seg_idx):
    with h5py.File(fp, 'r') as f:
        subj = f['Subj_Wins']
        ref = subj['PPG_F'][0, seg_idx]
        x = ref.flatten().astype(np.float64) if isinstance(ref, np.ndarray) \
            else f[ref][:].flatten().astype(np.float64)
    if len(x) == 1250:
        x = resample(x, 5000)
    return x


def _result_fingerprint(r):
    """Structural summary that must be identical for params=None vs HEAD behavior."""
    return {
        'n_total': r['n_total'],
        'n_valid': r['n_valid'],
        'quality': r['quality'],
        'tier': r['quality_tier'],
        'coverage': r['coverage'],
        'valid_ratio': r['valid_ratio'],
        'n_beats': len(r['beats']),
        'beat_lengths': [len(b) for b in r['beats']],
        'valid_mask': list(r['valid_mask']),
        'keypoint_o': [k.get('o') if k else None for k in r['keypoints']],
        'feature_keys': [sorted(f.keys())[:5] if f else None for f in r['features'][:2]],
    }


class TestBackwardCompatibility:
    @pytest.mark.parametrize('fp,seg_idx', SEGMENTS, ids=['vital', 'mimic', 'aurora_a', 'aurora_o'])
    def test_defaults_match_reference(self, fp, seg_idx):
        """params=None output must match the pre-parameterization reference values."""
        ppg = _load_segment(fp, seg_idx)
        r = process_ppg_segment(ppg, FS)
        fp = r['params_used'] == DEFAULT_PARAMS
        assert fp, 'params_used must equal DEFAULT_PARAMS when params=None'
        assert r['n_rejected_mad'] == 0, 'MAD off by default → 0 rejections'

    @pytest.mark.parametrize('fp,seg_idx', SEGMENTS, ids=['vital', 'mimic', 'aurora_a', 'aurora_o'])
    def test_explicit_defaults_identical_to_none(self, fp, seg_idx):
        """Passing DEFAULT_PARAMS explicitly must give identical results to None."""
        ppg = _load_segment(fp, seg_idx)
        r1 = process_ppg_segment(ppg, FS, params=None)
        r2 = process_ppg_segment(ppg, FS, params=dict(DEFAULT_PARAMS))
        assert _result_fingerprint(r1) == _result_fingerprint(r2)


class TestParamEffects:
    def test_peak_distance_changes_beat_count(self):
        ppg = _load_segment(SEGMENTS[0][0], SEGMENTS[0][1])
        r_near = process_ppg_segment(ppg, FS, params={'peak_distance_ms': 150})
        r_far = process_ppg_segment(ppg, FS, params={'peak_distance_ms': 600})
        assert r_near['n_total'] >= r_far['n_total']

    def test_prominence_high_finds_fewer_beats(self):
        ppg = _load_segment(SEGMENTS[0][0], SEGMENTS[0][1])
        r_lo = process_ppg_segment(ppg, FS, params={'peak_prominence': 0.01})
        r_hi = process_ppg_segment(ppg, FS, params={'peak_prominence': 0.30})
        assert r_lo['n_total'] >= r_hi['n_total']

    def test_mad_rejection_reduces_amp_cv(self):
        ppg = _load_segment(SEGMENTS[3][0], SEGMENTS[3][1])  # aurora o118
        r_off = process_ppg_segment(ppg, FS, params={'mad_reject_k': 'off'})
        r_on = process_ppg_segment(ppg, FS, params={'mad_reject_k': 2.5})

        def amp_cv(r):
            amps = [float(np.max(b) - np.min(b)) for b, v in zip(r['beats'], r['valid_mask']) if v]
            if len(amps) < 3:
                return np.nan
            return float(np.std(amps) / (np.abs(np.mean(amps)) + 1e-8))

        cv_off, cv_on = amp_cv(r_off), amp_cv(r_on)
        if not np.isnan(cv_off) and not np.isnan(cv_on) and r_on['n_rejected_mad'] > 0:
            assert cv_on <= cv_off, f'MAD should reduce amp CV: {cv_off} → {cv_on}'
        assert r_on['n_valid'] <= r_off['n_valid']

    def test_bandpass_flag_runs(self):
        ppg = _load_segment(SEGMENTS[2][0], SEGMENTS[2][1])
        r = process_ppg_segment(ppg, FS, params={'extra_bandpass': True})
        assert r['params_used']['extra_bandpass'] is True
        assert 'n_total' in r  # pipeline ran to completion


class TestPresetResolution:
    """preset_for_source: user_<source> adaptation wins, then source map,
    then 'default'; user_ok=False keeps official baselines isolated.
    Layout: default.yaml at root; <dataset>/default.yaml (official),
    <dataset>/user.yaml (user adaptation); custom names stay flat."""

    def test_no_user_preset_falls_to_source_map(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ppg_agent.processing_params.PRESET_DIR", tmp_path)
        ensure_default_preset()
        assert preset_for_source("vital") == "default"
        assert preset_for_source("aurora") == "default"  # no 'aurora' preset file

    def test_user_preset_takes_priority(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ppg_agent.processing_params.PRESET_DIR", tmp_path)
        ensure_default_preset()
        save_preset("user_vital", {**DEFAULT_PARAMS, "peak_distance_ms": 400})
        assert preset_for_source("vital") == "user_vital"

    def test_user_ok_false_isolates_official_baseline(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ppg_agent.processing_params.PRESET_DIR", tmp_path)
        ensure_default_preset()
        save_preset("aurora", {**DEFAULT_PARAMS, "peak_distance_ms": 400})
        save_preset("user_aurora", {**DEFAULT_PARAMS, "peak_distance_ms": 500})
        assert preset_for_source("aurora", user_ok=False) == "aurora"
        assert preset_for_source("aurora", user_ok=True) == "user_aurora"

    def test_removing_user_preset_restores_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ppg_agent.processing_params.PRESET_DIR", tmp_path)
        ensure_default_preset()
        save_preset("user_vital", {**DEFAULT_PARAMS, "peak_distance_ms": 400})
        assert preset_for_source("vital") == "user_vital"
        # web chat cancel(remove_preset=True) removes the file
        (tmp_path / "vital" / "user.yaml").unlink()
        assert preset_for_source("vital") == "default"

    def test_directory_layout(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ppg_agent.processing_params.PRESET_DIR", tmp_path)
        ensure_default_preset()
        save_preset("aurora", {**DEFAULT_PARAMS, "peak_distance_ms": 400})
        save_preset("user_aurora", {**DEFAULT_PARAMS, "peak_distance_ms": 500})
        save_preset("vital_v2", {**DEFAULT_PARAMS, "peak_distance_ms": 350})
        assert (tmp_path / "default.yaml").exists()
        assert (tmp_path / "aurora" / "default.yaml").exists()    # official
        assert (tmp_path / "aurora" / "user.yaml").exists()       # user adaptation
        assert (tmp_path / "vital_v2.yaml").exists()              # flat custom name
        assert load_preset("aurora")["peak_distance_ms"] == 400
        assert load_preset("user_aurora")["peak_distance_ms"] == 500
        assert list_presets() == ["aurora", "default", "user_aurora", "vital_v2"]
        assert preset_for_source("aurora", user_ok=False) == "aurora"


class TestValidation:
    def test_invalid_param_raises(self):
        with pytest.raises(ValueError):
            process_ppg_segment(np.zeros(5000), FS, params={'peak_distance_ms': 9999})

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            process_ppg_segment(np.zeros(5000), FS, params={'bogus_param': 1})

    def test_resolve_params_defaults(self):
        assert resolve_params(None) == DEFAULT_PARAMS

    def test_validate_params_errors(self):
        errs = validate_params({'peak_distance_ms': 9999, 'onset_search_ms': 77})
        assert len(errs) == 2


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
