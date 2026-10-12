"""40 维拍级形态学特征提取 —— 段级编排（分割 → 验证 → 重分割 → 逐拍特征）

本文件 2026-09-20 迁入本仓。代码为原项目自写，不派生自第三方；
第三方派生部分见 `third_party/vitalpy/`。
"""
# 从上游引擎裁取。
#
# ⚠ 兼容性补丁（必须在 import 第三方模块之前执行）：
#   scipy ≥1.14 删除了 integrate.simps / integrate.trapz，而
#   third_party/vitalpy/features/utils.py 用到 integrate.simps。
#   上游只补了 trapz；本仓补上 simps —— 读者用新版 scipy 时不会崩。
import scipy.integrate
import scipy.stats
import scipy.stats.stats
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
if not hasattr(scipy.integrate, 'trapz'):
    scipy.integrate.trapz = scipy.integrate.trapezoid
# third_party/vitalpy 用的是旧命名空间 scipy.stats.stats.zscore
# （scipy 1.9 起弃用、后续版本移除）。补个别名，避免读者在新版 scipy 上直接崩。
if not hasattr(scipy.stats.stats, 'zscore'):
    scipy.stats.stats.zscore = scipy.stats.zscore

import numpy as np
import pandas as pd

from third_party.vitalpy.preprocess.validate import validate_template

from .params import resolve_params
from .keypoints import extract_beat_keypoints_fixed
from .beat_features import extract_beat_features_fixed, validate_template_lenient

DEFAULT_FS = 500
DEFAULT_PEAK_THRESHOLD = 0.2
MIN_PULSE_CYCLE = 0.3   # 200 bpm
MAX_PULSE_CYCLE = 2.0   # 30 bpm

def process_ppg_segment(ppg: np.ndarray, fs: int = DEFAULT_FS,
                         merge_fragments: bool = True,
                         trim_to_onset: bool = True,
                         params: dict | None = None) -> dict:
    """Complete single-segment processing pipeline.

    Orchestrates the full flow in one call:
      0. Optional cascade bandpass (extra_bandpass switch)
      1. Initial segmentation by systolic peak midpoints
      2. Fragment merge (short beats grafted onto neighbors)
      3. Morphological validation
      4. Onset correction + beat trimming
      5. Clean feature extraction from corrected keypoints
      5b. Optional MAD-based amplitude outlier rejection (mad_reject_k switch)
      6. Quality score

    Args:
        ppg: PPG_F signal (already bandpass-filtered, 500 Hz).
        fs: Sampling rate.
        merge_fragments: Merge fragments < min_beat_len.
        trim_to_onset: Trim each beat to start at corrected onset 'o'.
        params: Processing parameter overrides (see `params.py`).
            None = defaults (identical to historical hardcoded behavior).

    Returns:
        dict with keys:
            n_total, n_valid: beat counts
            quality: float score
            quality_tier: 'good' / 'fair' / 'failed'
            coverage: fraction of signal covered by valid beats
            valid_ratio: n_valid / n_total
            beats: list of onset-trimmed beat templates
            beat_starts: absolute sample positions (onset-based)
            valid_mask: bool list per beat
            keypoints: list of dicts (on trimmed beats)
            features: list of dicts (clean feature set)
            onset_offsets: list of int (how many samples trimmed)
            n_rejected_mad: beats rejected by MAD outlier filter (0 when off)
            params_used: resolved parameter dict
    """
    from .params import resolve_params
    p = resolve_params(params)

    # ---------- Step 0: Optional cascade bandpass ----------
    if p['extra_bandpass'] and len(ppg) >= 50:
        try:
            from scipy.signal import butter, filtfilt
            nyq = fs / 2
            b, a = butter(4, [0.5 / nyq, 8.0 / nyq], btype='bandpass')
            ppg = filtfilt(b, a, ppg)
        except Exception:
            pass  # fall back to unbandpassed signal on failure

    total_samples = len(ppg)
    min_beat_len = p['min_beat_len']

    # ---------- Step 1: Initial segmentation ----------
    from scipy.signal import find_peaks as _find_peaks
    distance = int(fs * p['peak_distance_ms'] / 1000)
    prominence = p['peak_prominence'] * (np.max(ppg) - np.min(ppg))
    peaks, _ = _find_peaks(ppg, distance=distance, prominence=prominence)

    if len(peaks) < 2:
        return {'n_total': 0, 'n_valid': 0, 'quality': 0.0, 'quality_tier': 'failed',
                'beats': [], 'beat_starts': [], 'valid_mask': [],
                'keypoints': [], 'features': [], 'onset_offsets': [],
                'template_corr': [],
                'n_rejected_mad': 0, 'params_used': p}

    boundaries = [0]
    for i in range(len(peaks) - 1):
        boundaries.append((peaks[i] + peaks[i + 1]) // 2)
    boundaries.append(total_samples)

    templates, positions = [], []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s >= int(fs * p['min_segment_ms'] / 1000):
            templates.append(np.array(ppg[s:e]))
            positions.append(s)

    # ---------- Step 2: Fragment merge ----------
    # Fragments are beats < min_beat_len (300ms @ 500Hz).
    # They occur when: signal starts mid-cycle, noise creates extra peaks,
    # or the last beat in a 10s window is truncated.
    # Strategy: try appending each short fragment to the PREVIOUS beat.
    # If the combined beat passes lenient morphological validation, merge.
    # This preserves the complete systolic+diastolic phase.
    if merge_fragments and len(templates) > 1:
        merged_tmpl, merged_pos, merged_flags = [templates[0]], [positions[0]], [False]
        for i in range(1, len(templates)):
            tmpl = templates[i]
            if len(tmpl) < min_beat_len and merged_tmpl:
                candidate = np.concatenate([merged_tmpl[-1], tmpl])
                try:
                    if validate_template_lenient(candidate, fs):
                        merged_tmpl[-1] = candidate
                        merged_flags[-1] = True
                    elif len(merged_tmpl[-1]) < min_beat_len:
                        merged_tmpl[-1] = candidate
                        merged_flags[-1] = True
                    else:
                        merged_tmpl.append(tmpl)
                        merged_pos.append(positions[i])
                        merged_flags.append(False)
                except Exception:
                    merged_tmpl.append(tmpl)
                    merged_pos.append(positions[i])
                    merged_flags.append(False)
            else:
                merged_tmpl.append(tmpl)
                merged_pos.append(positions[i])
                merged_flags.append(False)
        # Filter to min length
        keep = [i for i, t in enumerate(merged_tmpl) if len(t) >= min_beat_len]
        templates = [merged_tmpl[i] for i in keep]
        positions = [merged_pos[i] for i in keep]
        was_merged = [merged_flags[i] for i in keep]
    else:
        templates = [t for t in templates if len(t) >= min_beat_len]
        was_merged = [False] * len(templates)

    # ---------- Step 3: Morphological validation ----------
    max_beat_sec = p['max_beat_len_ms'] / 1000
    use_lenient_all = (p['validation_mode'] == 'lenient')
    td_max = p.get('trough_diff_max', 0.2)

    def _validator_for(is_merged):
        # lenient path unchanged; strict path honours trough_diff_max
        if use_lenient_all or is_merged:
            return validate_template_lenient
        return lambda t, fs_: validate_template(t, fs_, trough_diff=td_max)

    valid_mask, good_tmpl, good_pos, good_merged = [], [], [], []
    for i, tmpl in enumerate(templates):
        if len(tmpl) / fs > max_beat_sec:  # too long = too slow HR, reject
            continue
        is_merged = i < len(was_merged) and was_merged[i]
        validator = _validator_for(is_merged)
        try:
            if validator(tmpl, fs):
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        except Exception:
            valid_mask.append(False)
        good_tmpl.append(tmpl)
        good_pos.append(positions[i])
        good_merged.append(is_merged)

    n_total = len(good_tmpl)
    if n_total == 0:
        return {'n_total': 0, 'n_valid': 0, 'quality': 0.0, 'quality_tier': 'failed',
                'beats': [], 'beat_starts': [], 'valid_mask': [],
                'keypoints': [], 'features': [], 'onset_offsets': [],
                'template_corr': [],
                'n_rejected_mad': 0, 'params_used': p}

    # ---------- Step 4: Onset-based re-segmentation ----------
    # Find corrected onset for each valid beat, then re-segment the full signal
    # at these onset positions. Each beat = [onset_i, onset_{i+1}] — zero gaps.
    onset_positions = []  # absolute positions in ppg of corrected onsets
    for i, (tmpl, pos, v) in enumerate(zip(good_tmpl, good_pos, valid_mask)):
        if not v:
            onset_positions.append(pos)  # use original boundary as fallback
            continue
        try:
            ok_v, kp_raw = extract_beat_keypoints_fixed(
                tmpl, fs,
                onset_search_ms=p['onset_search_ms'],
                onset_margin_ms=p['onset_margin_ms'],
                onset_rise_check_ms=p['onset_rise_check_ms'],
                diastolic_search_ms=p['diastolic_search_ms'])
            if ok_v and kp_raw:
                o_rel = kp_raw.get('o', 0)
                onset_positions.append(pos + o_rel)
            else:
                onset_positions.append(pos)
        except Exception:
            onset_positions.append(pos)

    # Build re-segmented beats: [onset_i, onset_{i+1}]
    re_beats, re_starts, re_valid, re_kps, re_offsets = [], [], [], [], []
    for i in range(len(onset_positions) - 1):
        s = onset_positions[i]
        e = onset_positions[i + 1]
        if e <= s or e - s < int(fs * p['resegment_min_ms'] / 1000):
            continue
        bt = np.array(ppg[s:e])

        # Validate length
        if len(bt) / fs > max_beat_sec:
            re_beats.append(bt)
            re_starts.append(s)
            re_valid.append(False)
            re_kps.append({})
            re_offsets.append(onset_positions[i] - good_pos[i] if i < len(good_pos) else 0)
            continue

        # Re-extract keypoints on the re-segmented beat
        is_merged = i < len(was_merged) and was_merged[i]
        validator = _validator_for(is_merged)
        try:
            shape_ok = validator(bt, fs)
        except Exception:
            shape_ok = False

        if not shape_ok:
            re_beats.append(bt)
            re_starts.append(s)
            re_valid.append(False)
            re_kps.append({})
            re_offsets.append(onset_positions[i] - good_pos[i] if i < len(good_pos) else 0)
            continue

        try:
            ok_v, kp_new = extract_beat_keypoints_fixed(
                bt, fs,
                onset_search_ms=p['onset_search_ms'],
                onset_margin_ms=p['onset_margin_ms'],
                onset_rise_check_ms=p['onset_rise_check_ms'],
                diastolic_search_ms=p['diastolic_search_ms'])
            if ok_v and kp_new:
                re_beats.append(bt)
                re_starts.append(s)
                re_valid.append(True)
                re_kps.append(kp_new)
            else:
                re_beats.append(bt)
                re_starts.append(s)
                re_valid.append(False)
                re_kps.append({})
        except Exception:
            re_beats.append(bt)
            re_starts.append(s)
            re_valid.append(False)
            re_kps.append({})

        offset_i = onset_positions[i] - good_pos[i] if i < len(good_pos) else 0
        re_offsets.append(offset_i)

    # ---------- Step 5: Feature extraction ----------
    features_list = []
    for i, (bt, kp, vv) in enumerate(zip(re_beats, re_kps, re_valid)):
        if vv and kp:
            try:
                rich = extract_beat_features_fixed(
                    bt, fs, groups='all',
                    diastolic_search_ms=p['diastolic_search_ms'],
                    notch_depth_threshold=p['notch_depth_threshold'])
                feats = rich.get('all_features_flat', {})
                # Beat metadata
                feats['beat_len_ms'] = round(len(bt) / fs * 1000, 1)
                feats['hr_bpm'] = round(60.0 / (len(bt) / fs), 1)
                feats['t_sys_ms'] = round((kp.get('s', 0) - kp.get('o', 0)) / fs * 1000, 1)
                feats['t_dia_ms'] = round((kp.get('v', len(bt)-1) - kp.get('s', 0)) / fs * 1000, 1)
                features_list.append(feats)
            except Exception:
                features_list.append({})
        else:
            features_list.append({})

    # ---------- Step 5b: MAD-based amplitude outlier rejection ----------
    n_rejected_mad = 0
    mad_k = p['mad_reject_k']
    if mad_k != 'off' and sum(re_valid) >= 6:
        amps = np.array([float(np.max(bt) - np.min(bt))
                         for bt, vv in zip(re_beats, re_valid) if vv])
        med = np.median(amps)
        mad = np.median(np.abs(amps - med))
        if mad > 0:
            keep = np.abs(amps - med) <= mad_k * mad
            vi = 0  # index into `keep`, aligned with valid-beat order
            for i in [i for i, vv in enumerate(re_valid) if vv]:
                if not keep[vi]:
                    re_valid[i] = False
                    re_kps[i] = {}
                    features_list[i] = {}
                    n_rejected_mad += 1
                vi += 1

    # ---------- Step 5c: Template-shape consistency check ----------
    # Beats admitted by a relaxed trough_diff_max but failing the strict 0.2
    # check must stay shape-consistent with the segment's median template
    # (built from strictly-validated beats). Guards against drift-admitted
    # artifacts; segments without >=2 strict beats keep their valid beats
    # as-is (no reference to test against).
    template_corr = [float('nan')] * len(re_beats)
    corr_min = p.get('template_corr_min', 0.0)
    if corr_min > 0:
        from scipy.signal import resample as _tmpl_resample
        strict_ok = []
        for i, (bt, vv) in enumerate(zip(re_beats, re_valid)):
            if not vv:
                continue
            try:
                if validate_template(np.asarray(bt, dtype=float), fs):
                    strict_ok.append(i)
            except Exception:
                pass
        if len(strict_ok) >= 2:
            L = 100
            arr = np.array([_tmpl_resample(np.asarray(re_beats[i], dtype=float), L)
                            for i in strict_ok])
            arr = (arr - arr.min(1, keepdims=True)) / (
                np.ptp(arr, axis=1, keepdims=True) + 1e-9)
            ref = np.median(arr, 0)
            ref = (ref - ref.min()) / (np.ptp(ref) + 1e-9)
            for i, (bt, vv) in enumerate(zip(re_beats, re_valid)):
                if not vv:
                    continue
                u = _tmpl_resample(np.asarray(bt, dtype=float), L)
                u = (u - u.min()) / (np.ptp(u) + 1e-9)
                try:
                    c = float(np.corrcoef(u, ref)[0, 1])
                except Exception:
                    c = float('nan')
                template_corr[i] = c
                try:
                    strict_pass = validate_template(np.asarray(bt, dtype=float), fs)
                except Exception:
                    strict_pass = False
                # corr gate only applies to beats the strict validator rejects
                if not strict_pass and not (c >= corr_min):
                    re_valid[i] = False
                    re_kps[i] = {}
                    features_list[i] = {}

    # ---------- Step 6: Quality score (computed on the kept set) ----------
    n_valid = sum(re_valid)
    tv = sum(len(t) for t, vv in zip(re_beats, re_valid) if vv)
    cov_valid = tv / total_samples if total_samples > 0 else 0
    vr = n_valid / n_total if n_total > 0 else 0
    quality = cov_valid * (0.4 + 0.6 * vr)
    tier = 'good' if cov_valid >= 0.60 else ('fair' if cov_valid >= 0.30 else 'failed')

    return {
        'n_total': n_total, 'n_valid': n_valid,
        'quality': round(quality, 3), 'quality_tier': tier,
        'coverage': round(cov_valid, 3), 'valid_ratio': round(vr, 3),
        'beats': re_beats,
        'beat_starts': re_starts,
        'template_corr': template_corr,
        'valid_mask': re_valid,
        'keypoints': re_kps,
        'features': features_list,
        'onset_offsets': re_offsets,
        'n_rejected_mad': n_rejected_mad,
        'params_used': p,
    }


def extract_segment_features(ppg: np.ndarray, abp: np.ndarray,
                              rpeaks: np.ndarray | None,
                              subject_id: str, seg_idx: int, dataset: str,
                              sbp_label: float = None, dbp_label: float = None,
                              fs: int = DEFAULT_FS,
                              params: dict | None = None) -> dict:
    """Complete segment feature extraction — segment-level + beat-level + PTT + HRV + ABP.

    Args:
        ppg: PPG_F signal at fs Hz.
        abp: ABP_Raw signal at fs Hz.
        rpeaks: ECG R-peak sample positions (original 125Hz samples), or None.
        subject_id: Subject identifier string (e.g. 'p000618').
        seg_idx: Segment index within subject.
        dataset: 'vital' or 'mimic'.
        sbp_label: Systolic BP label from data.
        dbp_label: Diastolic BP label from data.
        fs: Sampling rate.
        params: Processing parameter overrides (see `params.py`).

    Returns:
        {
            'segment': {segment-level dict: identifiers, aggregates, PTT, HRV, ABP, quality, cross-modal},
            'beats':   [{beat-level dict: subject_id, seg_idx, beat_idx, dataset, +44 features + ptt}, ...]
        }
    """
    import pandas as pd

    # ---------- Core PPG processing ----------
    ppg_result = process_ppg_segment(ppg, fs, merge_fragments=True, trim_to_onset=True,
                                     params=params)
    beats_list = ppg_result['beats']
    kps_list = ppg_result['keypoints']
    valid_mask = ppg_result['valid_mask']
    features_list = ppg_result['features']
    beat_starts = ppg_result['beat_starts']
    onset_offsets = ppg_result['onset_offsets']
    n_total = ppg_result['n_total']
    n_valid = ppg_result['n_valid']

    # ---------- Segment-level base ----------
    seg = {
        'subject_id': subject_id, 'seg_idx': seg_idx, 'dataset': dataset,
        'sbp_label': sbp_label, 'dbp_label': dbp_label,
        'extraction_status': 'success',
        'n_beats_ppg': n_total, 'n_beats_valid': n_valid,
    }

    # ---------- Segment-level feature aggregation ----------
    valid_feats = [f for f, v in zip(features_list, valid_mask) if v and f]
    if valid_feats:
        df = pd.DataFrame(valid_feats)
        for c in df.columns:
            v = df[c].dropna()
            if len(v) > 0:
                seg[f'{c}_mean'] = float(v.mean())
                seg[f'{c}_std'] = float(v.std()) if len(v) > 1 else 0.0
                seg[f'{c}_median'] = float(v.median())

    # ---------- ECG / HRV ----------
    eg = {}
    if rpeaks is not None and len(rpeaks) >= 2:
        # Resample R-peaks from 125Hz to fs
        rp = (rpeaks * (fs / 125)).astype(int)
        rp = rp[(rp > 0) & (rp < len(ppg))]
        rr = np.diff(rpeaks) / 125 * 1000  # RR intervals in ms
        eg['hrv_sdnn'] = round(float(np.std(rr)), 2)
        eg['hrv_rmssd'] = round(float(np.sqrt(np.mean(np.diff(rr)**2))), 2) if len(rr) > 2 else 0.0
        eg['hrv_pnn50'] = round(float(np.sum(np.abs(np.diff(rr)) > 50) / max(1, len(rr)) * 100), 1)
        eg['hrv_mean_rr'] = round(float(np.mean(rr)), 2)
        eg['hrv_std_rr'] = round(float(np.std(rr)), 2)
        eg['ecg_r_peak_count'] = len(rpeaks)
        hr_vals = 60000 / rr
        eg['ecg_hr_mean'] = round(float(np.mean(hr_vals)), 1)
        eg['ecg_hr_std'] = round(float(np.std(hr_vals)), 1)
    seg.update(eg)

    # ---------- ABP-PTT (ECG → ABP foot) ----------
    abp_500 = abp  # assume already at fs
    ptt_abp_vals = []
    if rpeaks is not None and len(rpeaks) >= 2:
        rp = (rpeaks * (fs / 125)).astype(int)
        rp = rp[(rp > 0) & (rp < len(abp_500))]
        for r in rp:
            end = min(r + int(fs * 0.5), len(abp_500))
            if end <= r + 5:
                continue
            ft = r + int(np.argmin(abp_500[r:end]))
            pt = (ft - r) / fs * 1000
            if 50 < pt < 500:
                ptt_abp_vals.append(pt)
        if ptt_abp_vals:
            seg['ptt_abp_mean_ms'] = round(float(np.mean(ptt_abp_vals)), 1)
            seg['ptt_abp_std_ms'] = round(float(np.std(ptt_abp_vals)), 1)
            seg['ptt_abp_min_ms'] = round(float(np.min(ptt_abp_vals)), 1)
            seg['ptt_abp_max_ms'] = round(float(np.max(ptt_abp_vals)), 1)
            seg['ptt_abp_count'] = len(ptt_abp_vals)

    # ---------- PPG-PTT (ECG → PPG onset) per-beat ----------
    ptt_ppg_vals = []
    if rpeaks is not None and len(rpeaks) >= 2:
        rp = (rpeaks * (fs / 125)).astype(int)
        rp = rp[(rp > 0) & (rp < len(ppg))]
        # For each R-peak, find the first valid beat whose onset is after it
        for r in rp:
            for bi in range(len(beat_starts)):
                if not valid_mask[bi]:
                    continue
                onset_sample = beat_starts[bi]
                if onset_sample > r:
                    ptt_ppg = (onset_sample - r) / fs * 1000
                    if 20 < ptt_ppg < 800:
                        ptt_ppg_vals.append(ptt_ppg)
                    break
        if ptt_ppg_vals:
            seg['ptt_ppg_mean_ms'] = round(float(np.mean(ptt_ppg_vals)), 1)
            seg['ptt_ppg_std_ms'] = round(float(np.std(ptt_ppg_vals)), 1)
            seg['ptt_ppg_min_ms'] = round(float(np.min(ptt_ppg_vals)), 1)
            seg['ptt_ppg_max_ms'] = round(float(np.max(ptt_ppg_vals)), 1)
            seg['ptt_ppg_count'] = len(ptt_ppg_vals)

    # PTT normalized by heart rate
    if seg.get('ptt_abp_mean_ms') and seg.get('hrv_mean_rr', 0) > 0:
        seg['ptt_abp_normalized'] = round(seg['ptt_abp_mean_ms'] / seg['hrv_mean_rr'], 4)
    if seg.get('ptt_ppg_mean_ms') and seg.get('hrv_mean_rr', 0) > 0:
        seg['ptt_ppg_normalized'] = round(seg['ptt_ppg_mean_ms'] / seg['hrv_mean_rr'], 4)

    # ---------- ABP stats ----------
    # Segment-level BP aggregates from per-beat window values (one SBP /
    # DBP per cardiac cycle).  Beat windows anchor the systolic peak, so
    # secondary bumps (dicrotic / reflected waves) are excluded — unlike
    # the old argrelextrema scan, which counted every local maximum and
    # systematically underestimated SBP.
    sbp_beats, dbp_beats, _mean_beats = _beat_window_bp(abp_500, beat_starts, fs)
    if len(sbp_beats) > 0:
        sv = sbp_beats[~np.isnan(sbp_beats)]
        dv = dbp_beats[~np.isnan(dbp_beats)]
        if len(sv) > 0:
            seg['abp_sbp_mean'] = round(float(np.mean(sv)), 1)
            seg['abp_sbp_std'] = round(float(np.std(sv)), 1)
        if len(dv) > 0:
            seg['abp_dbp_mean'] = round(float(np.mean(dv)), 1)
            seg['abp_dbp_std'] = round(float(np.std(dv)), 1)
        pp_mask = ~np.isnan(sbp_beats) & ~np.isnan(dbp_beats)
        if np.any(pp_mask):
            seg['abp_pp_mean'] = round(
                float(np.mean(sbp_beats[pp_mask] - dbp_beats[pp_mask])), 1)

    # ---------- Quality ----------
    total_samples = len(ppg)
    tv = sum(len(t) for t, vv in zip(beats_list, valid_mask) if vv)
    cov_valid = tv / total_samples
    vr = n_valid / n_total if n_total > 0 else 0
    score = cov_valid * (0.4 + 0.6 * vr)
    issues = []
    if cov_valid < 0.70: issues.append(f'low_coverage({int(cov_valid*100)}%)')
    if vr < 0.3 and n_total > 3: issues.append(f'low_valid_ratio({int(vr*100)}%)')
    if n_valid == 0 and n_total > 0: issues.append('no_valid_beats')
    tier = 'failed' if n_total == 0 else ('good' if cov_valid >= 0.60 else 'fair')
    seg['quality_score'] = round(score, 3)
    seg['quality_tier'] = tier
    seg['coverage'] = round(cov_valid, 3)
    seg['valid_ratio'] = round(vr, 3)
    seg['quality_flags'] = '; '.join(issues) if issues else 'ok'
    seg['quality_ok'] = (tier == 'good')

    # ---------- Cross-modal ----------
    seg['n_beats_ecg'] = eg.get('ecg_r_peak_count', 0)
    necg = seg['n_beats_ecg']
    if necg > 0 and n_total > 0:
        seg['beat_ratio_ppg_ecg'] = round(n_total / necg, 2)
    ptt = seg.get('ptt_abp_mean_ms')
    hr_v = seg.get('ecg_hr_mean')
    if ptt and hr_v:
        seg['ptt_hr_product'] = round(ptt * hr_v / 1000, 1)

    # ---------- Beat-level records ----------
    # Per-beat ABP BP shares the window logic with the segment-level
    # ABP stats above (single source of truth).
    beat_records = []
    for bi in range(len(beats_list)):
        rec = {
            'subject_id': subject_id, 'seg_idx': seg_idx, 'dataset': dataset,
            'beat_idx': bi,
        }
        if bi < len(features_list) and features_list[bi]:
            rec.update(features_list[bi])
        # Keypoint positions (sample offsets within beat)
        if bi < len(kps_list) and kps_list[bi]:
            for kp_name in ['o', 'a', 'md', 'b', 's', 'dn', 'ip', 'd', 'v']:
                val = kps_list[bi].get(kp_name)
                rec[f'kp_{kp_name}'] = int(val) if val is not None else None
        # Beat start position in signal
        if bi < len(beat_starts):
            rec['beat_start_sample'] = int(beat_starts[bi])
        # Template-shape consistency with segment median (only written when
        # the corr check is enabled; NaN = unchecked -> key omitted)
        tc = ppg_result.get('template_corr')
        if tc and bi < len(tc) and not np.isnan(tc[bi]):
            rec['template_corr'] = round(float(tc[bi]), 4)
        # Per-beat PTT
        if valid_mask[bi] and bi < len(beat_starts):
            onset_sample = beat_starts[bi]
            # Find the PTT for this beat (closest R-peak before onset)
            if rpeaks is not None and len(rpeaks) >= 2:
                rp = (rpeaks * (fs / 125)).astype(int)
                rp = rp[(rp > 0) & (rp < len(ppg))]
                # Find the R-peak just before this beat's onset
                candidates = rp[rp < onset_sample]
                if len(candidates) > 0:
                    closest_r = candidates[-1]
                    rec['ptt_ppg_ms'] = round((onset_sample - closest_r) / fs * 1000, 1)
                else:
                    rec['ptt_ppg_ms'] = None
            else:
                rec['ptt_ppg_ms'] = None
        else:
            rec['ptt_ppg_ms'] = None
        rec['abp_sbp'] = (float(sbp_beats[bi]) if bi < len(sbp_beats)
                          else float('nan'))
        rec['abp_dbp'] = (float(dbp_beats[bi]) if bi < len(dbp_beats)
                          else float('nan'))
        beat_records.append(rec)

    return {'segment': seg, 'beats': beat_records}


def _beat_window_bp(abp_500: np.ndarray, beat_starts, fs: int):
    """Per-beat ABP blood pressure via beat windows [onset_i, onset_{i+1}).

    Window max = SBP, min before the systolic peak = DBP (amplitude
    filters SBP 50-300 / DBP 20-200 mmHg).  Single source of truth for
    both beat-level records and segment-level ABP stats.

    Returns (sbp, dbp, mean) float arrays (NaN where no valid extremum).
    """
    n_sig = len(abp_500)
    n = len(beat_starts)
    sbp = np.full(n, np.nan)
    dbp = np.full(n, np.nan)
    mean = np.full(n, np.nan)
    if n_sig == 0 or not np.any(np.asarray(abp_500) > 0):
        return sbp, dbp, mean
    for i in range(n):
        w0 = max(0, min(int(beat_starts[i]), n_sig - 1))
        w1 = (int(beat_starts[i + 1]) if i + 1 < n else n_sig)
        w1 = max(w0 + 1, min(w1, n_sig))
        # A single cardiac cycle spans ~0.33-1.5s (40-180 bpm); longer
        # windows (trailing invalid beats) would cover multiple ABP
        # cycles — cap, then trim to the systolic peak below.
        w1 = min(w1, w0 + int(fs * 1.5))
        if w1 - w0 < int(fs * 0.04):  # need >=40ms of ABP
            continue
        win = abp_500[w0:w1]
        i_sys = int(np.argmax(win))
        if len(win) - i_sys > int(fs * 0.3):
            # multi-cycle window: keep only this beat's systole +
            # the following diastole
            win = win[:i_sys + int(fs * 0.3)]
            i_sys = int(np.argmax(win))
        sv = float(win.max())
        if 50 < sv < 300:
            sbp[i] = round(sv, 1)
        if i_sys > 0:
            dv = float(win[:i_sys].min())
            if 20 < dv < 200:
                dbp[i] = round(dv, 1)
        mean[i] = round(float(win.mean()), 1)
    return sbp, dbp, mean


__all__ = ['process_ppg_segment', 'extract_segment_features']
