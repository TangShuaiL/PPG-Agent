"""PPG Engine — 基于 VitalPy 底层函数的 numpy 友好处理原语。

设计原则：
- 输入输出使用 numpy ndarray（不要求 DataFrame 的 't' 列）
- 每个函数只做一件事，不耦合
- 参数可配置，不依赖修改 VitalPy 源码
- 分割算法复制自 VitalPy（因为原函数硬编码了 PPG_PEAK_DETECTION_THRESHOLD）

底层依赖：VitalPy 的 bandpass_filter、baseline_correction、extract_key_points、
FeatureExtractor 等。这些函数接受 pd.Series，我们在此做 ndarray↔Series 转换。
"""

import numpy as np
import pandas as pd

# ============================================================
# 兼容性补丁（必须在 import VitalPy 之前）
# ============================================================
import scipy.integrate
if not hasattr(scipy.integrate, 'trapz'):
    scipy.integrate.trapz = scipy.integrate.trapezoid

# VitalPy — local copy in engine/vital/
from ppg_agent.engine.vital.params import (
    MINIMUM_PULSE_CYCLE,
    POLY_COEFFS,
)
from ppg_agent.engine.vital.preprocess.validate import validate_template
from ppg_agent.engine.vital.features.utils import (
    extract_key_points,
    extract_key_temporal,
    extract_statistical,
    extract_temporal,
    extract_spectral,
    branch_width_features,
    extract_coeffs,
)

# ============================================================
# 默认参数
# ============================================================

# [0,1] 归一化数据的 beat 分割阈值
DEFAULT_PEAK_THRESHOLD = 0.2

# 默认采样率
DEFAULT_FS = 500

# 多项式拟合阶数
POLY_COEFFS = 15

# ============================================================
# 分割算法（复制自 VitalPy segment.py，参数化阈值）
# ============================================================

def _find_extremum(waveform: np.ndarray) -> list:
    """找到波形中的所有极值点（极大值 + 极小值）。

    复制自 VitalPy segment.py:find_extremum()

    Args:
        waveform: 一维信号数组

    Returns:
        list of (index, value) tuples，按索引排序
    """
    from scipy.signal import argrelmax, argrelmin
    extremum_idx = np.sort(np.unique(np.concatenate(
        (argrelmax(waveform)[0], argrelmin(waveform)[0])
    )))
    extremum_vals = waveform[extremum_idx]
    return list(zip(extremum_idx.tolist(), extremum_vals.tolist()))


def _segment_by_peaks(waveform: np.ndarray, fs: int = 500) -> list:
    """Beat segmentation using scipy.signal.find_peaks on the signal directly.

    Finds systolic peaks, then defines beat boundaries at the midpoints
    between adjacent peaks. This avoids splitting at dicrotic notches
    because peaks are detected on the systolic maximum, not on local minima.

    Returns list of beat templates.
    """
    from scipy.signal import find_peaks

    # Peak detection on the raw waveform
    distance = int(fs * 0.3)  # min 300ms between peaks (200 bpm max)
    prominence = 0.05 * (np.max(waveform) - np.min(waveform))
    peaks, props = find_peaks(waveform, distance=distance, prominence=prominence)

    if len(peaks) < 2:
        return []

    # Beat boundaries at midpoints between peaks
    boundaries = [0]
    for i in range(len(peaks) - 1):
        mid = (peaks[i] + peaks[i + 1]) // 2
        boundaries.append(mid)
    boundaries.append(len(waveform))

    templates = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start >= int(fs * 0.2):  # min 200ms
            templates.append(np.array(waveform[start:end]))

    return templates


def _segment_waveform(waveform: np.ndarray, peak_threshold: float = DEFAULT_PEAK_THRESHOLD) -> list:
    """从 PPG 波形中分割出 beat-to-beat 模板。

    基于自适应脉冲幅度阈值的相对极值分析。
    算法复制自 VitalPy segment.py:segmentation()，但将硬编码的
    PPG_PEAK_DETECTION_THRESHOLD 改为可配置参数。

    Args:
        waveform: 一维 PPG 信号（已滤波 + 基线校正）
        peak_threshold: 脉冲检测阈值系数 (0-1)，越小越敏感。
                        对 [0,1] 归一化数据推荐 0.2

    Returns:
        list of np.ndarray: 每个元素是一个 beat 模板
    """
    from scipy.signal import argrelmax, argrelmin

    max_value = np.median(waveform[argrelmax(waveform)])
    min_value = np.median(waveform[argrelmin(waveform)])
    threshold = (max_value - min_value) * peak_threshold

    template_list = []
    last_extreme_index = None
    last_extreme = None
    last_start_index = None

    for extreme_index, extreme in _find_extremum(waveform):
        if last_extreme is not None and extreme - last_extreme > threshold:
            if last_start_index is not None:
                single_waveform = waveform[last_start_index:last_extreme_index]
                template_list.append(single_waveform)
            last_start_index = last_extreme_index
        last_extreme_index = extreme_index
        last_extreme = extreme

    return template_list


# ============================================================
# 预处理原语
# ============================================================

def validate_template_lenient(template: np.ndarray, fs: float) -> bool:
    """Lenient validation: relaxed period check, skip trough-difference.

    Used for beats that received fragment merges (may have irregular
    morphology). Still enforces a reasonable upper bound to prevent
    over-merged beats.
    """
    period = len(template) / fs
    if period < MINIMUM_PULSE_CYCLE or period > 1.2:  # 50 bpm lower bound
        return False
    s_pos = float(np.argmax(template)) / float(len(template))
    if s_pos < 0.05 or s_pos >= 0.75:  # peak at edge or too late
        return False
    return True


def segment_beats(
    ppg: np.ndarray,
    fs: int = DEFAULT_FS,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    validate: bool = True,
    merge_fragments: bool = True,
    min_beat_len: int = 150,
) -> list:
    """Segment PPG signal into beat templates.

    Args:
        ppg: PPG signal (raw or filtered).
        fs: Sampling rate in Hz.
        peak_threshold: Detection sensitivity (0-1). Higher = fewer beats.
        validate: If True, keep only morphologically valid beats.
        merge_fragments: If True, merge short fragments into preceding beat
                         to preserve complete cardiac cycles (systolic + diastolic).
        min_beat_len: Minimum beat length in samples (default 150 = 0.3s @ 500Hz).

    Returns:
        list of np.ndarray: beat templates.
    """
    templates = _segment_by_peaks(ppg, fs)

    # Fallback: if peak-based segmentation produces too few beats,
    # use extremum-based (handles low-amplitude/irregular signals better)
    if len(templates) < 3:
        templates = _segment_waveform(ppg, peak_threshold=peak_threshold)

    # Smart merge: merge fragments, tracking which beats were merged
    was_merged = [False] * len(templates)  # track which beats got fragments merged in
    if merge_fragments and len(templates) > 1:
        merged = [templates[0]]
        merged_flag = [False]
        for tmpl in templates[1:]:
            if len(tmpl) < min_beat_len and merged:
                candidate = np.concatenate([merged[-1], tmpl])
                try:
                    if validate_template_lenient(candidate, fs):
                        merged[-1] = candidate
                        merged_flag[-1] = True  # mark as merged
                    elif len(merged[-1]) < min_beat_len:
                        merged[-1] = candidate
                        merged_flag[-1] = True
                    else:
                        merged.append(tmpl)
                        merged_flag.append(False)
                except Exception:
                    merged.append(tmpl)
                    merged_flag.append(False)
            else:
                merged.append(tmpl)
                merged_flag.append(False)
        templates = [t for t in merged if len(t) >= min_beat_len]
        was_merged = [merged_flag[i] for i, t in enumerate(merged) if len(t) >= min_beat_len]

    if not validate:
        templates = [t for t in templates if len(t) / fs <= 1.2]
        return templates

    # Quality filter: strict for original beats, lenient for merged beats
    good_templates = []
    for i, tmpl in enumerate(templates):
        try:
            # Hard cap: reject beats > 1.2s (50 bpm) regardless of validation
            if len(tmpl) / fs > 1.2:
                continue
            is_merged = i < len(was_merged) and was_merged[i]
            validator = validate_template_lenient if is_merged else validate_template
            if validator(tmpl, fs):
                good_templates.append(tmpl)
        except Exception:
            pass

    return good_templates


# ============================================================
# 单 beat 特征提取
# ============================================================

def extract_beat_keypoints(beat: np.ndarray, fs: int = DEFAULT_FS) -> tuple:
    """提取单个 beat 模板的关键点（fiducial points）。

    使用二阶导数分析检测 PPG 波形的 9 个关键点：
    o(起始), a, md(最大斜率), b, s(收缩峰), dn(重搏切迹), ip(拐点), d(舒张峰), v(结束)

    Args:
        beat: 单个 beat 模板
        fs: 采样率

    Returns:
        (valid, key_points): valid 表示检测是否可靠，key_points 是 {name: index} 字典
    """
    template = pd.Series(beat)
    return extract_key_points(template, fs)


def _find_true_onset(beat: np.ndarray, md_pos: int, fs: int = DEFAULT_FS,
                     onset_search_ms: int = 120, onset_margin_ms: int = 5,
                     onset_rise_check_ms: int = 50) -> int:
    """Find the true foot of the pulse wave (onset 'o').

    The beat boundary (position 0) is a midpoint between systolic peaks, not
    necessarily the true foot. The true foot is the lowest point right before
    the systolic upstroke begins — a local minimum immediately preceding 'md'
    (the max-derivative point).

    Key insight: the foot is ALWAYS close to 'md' (typically within 60-100ms).
    Searching too far back catches false valleys in the diastolic decay tail,
    especially in vital data where beats are long and the diastolic tail can
    have multiple small fluctuations.

    Args:
        beat: Beat template.
        md_pos: Position of max derivative (systolic upstroke).
        fs: Sampling rate.
        onset_search_ms: Search window before max-derivative (ms).
        onset_margin_ms: Margin before max-derivative excluded from search (ms).
        onset_rise_check_ms: Rise verification window after onset (ms).

    Returns:
        Sample index of the true onset within the beat, or 0 if not found.
    """
    n = len(beat)
    if md_pos <= 5 or md_pos >= n:
        return 0

    # --- Phase 1: argmin in a narrow window right before md ---
    # The foot is the lowest point in the ~onset_search_ms before the
    # max-derivative. It is enough to span the foot-to-md gap,
    # while narrow enough to avoid false minima from the diastolic tail.
    window_ms = onset_search_ms
    margin_ms = onset_margin_ms  # foot cannot be AT md — back off at least margin
    search_start = max(0, md_pos - int(fs * window_ms / 1000))
    search_end = max(search_start + 3, md_pos - int(fs * margin_ms / 1000))

    if search_end <= search_start:
        return 0

    region = beat[search_start:search_end]
    foot_rel = int(np.argmin(region))
    foot = search_start + foot_rel

    # --- Phase 2: Verify — signal should RISE after the foot ---
    # Check that the median derivative in the rise-check window after the
    # foot is positive. If not, the argmin found a false dip; widen the search.
    check_len = int(fs * onset_rise_check_ms / 1000)
    check_start = foot
    check_end = min(foot + check_len, n - 2)
    if check_end > check_start + 2:
        post_deriv = np.diff(beat[check_start:check_end])
        if np.median(post_deriv) <= 0:
            # Foot failed the rise check — it's not a true upstroke start.
            # Try phase 3 fallback (derivative crossing, closer to md).
            pass
        else:
            return foot

    # --- Phase 3: Derivative zero-crossing in tight window ---
    # As a fallback, find the LAST valley (neg→pos crossing) in the
    # same window. This catches beats where the foot isn't the
    # absolute minimum but the last local minimum before the upstroke.
    d1 = np.diff(region)
    for i in range(len(d1) - 1, 0, -1):
        if d1[i - 1] < 0 and d1[i] > 0:
            foot = search_start + i
            # Verify rise again
            c2s = foot
            c2e = min(foot + check_len, n - 2)
            if c2e > c2s + 2:
                post_d = np.diff(beat[c2s:c2e])
                if np.median(post_d) > 0:
                    return foot
            break

    # --- Phase 4: Absolute minimum in [0, md_pos] as last resort ---
    # Only used if all else fails — widens all the way back to beat start.
    if md_pos > 0:
        foot = int(np.argmin(beat[:md_pos]))
        return foot

    return 0


def extract_beat_keypoints_fixed(beat: np.ndarray, fs: int = DEFAULT_FS,
                                 onset_search_ms: int = 120, onset_margin_ms: int = 5,
                                 onset_rise_check_ms: int = 50,
                                 diastolic_search_ms: int = 200) -> tuple:
    """Extract keypoints with diastolic correction.

    VitalPy's extract_key_points often fails to detect dn/ip/d correctly on
    clinical PPG data, setting them to len(waveform)-3. This wrapper detects
    that case and applies local-minima-based correction.

    B-class parameters (onset/diastolic search) are threaded through so the
    processing pipeline can tune feature-point detection per dataset.

    Returns:
        (valid, key_points): Always all 9 points. valid=False only if the beat
        is truly unusable (systolic peak at edge, etc.).
    """
    template = pd.Series(beat)
    n = len(beat)

    try:
        valid, kp = extract_key_points(template, fs)
    except Exception:
        valid, kp = False, {}

    # Ensure all 9 keys exist with at least fallback values
    s = kp.get('s') if kp.get('s') is not None and 0 < kp.get('s', 0) < n else int(np.argmax(beat))
    defaults = {
        'o': 0,
        'v': n - 1,
        's': s,
        'a': kp.get('a') if kp.get('a') is not None and 0 < kp.get('a', 0) < s else max(1, s // 2),
        'md': kp.get('md') if kp.get('md') is not None and 0 < kp.get('md', 0) < s else max(2, int(s * 0.6)),
        'b': kp.get('b') if kp.get('b') is not None and 0 < kp.get('b', 0) < s else max(3, int(s * 0.8)),
    }
    # Ensure a < md < b < s ordering
    defaults['a'] = min(defaults['a'], defaults['md'] - 1) if defaults['md'] > 2 else 1
    defaults['md'] = min(max(defaults['a'] + 1, defaults['md']), defaults['b'] - 1) if defaults['b'] > defaults['a'] + 2 else defaults['a'] + 2
    defaults['b'] = min(max(defaults['md'] + 1, defaults['b']), s - 1) if s > defaults['md'] + 2 else defaults['md'] + 2

    kp = {**defaults, **kp}  # VitalPy values take precedence for systolic points

    # Enforce strict ordering: o < a < md < b < s < dn < ip < d < v
    # (re-fix any overlaps that VitalPy or defaults introduced)
    def _force_order(kp, n):
        o = kp.get('o', 0)
        v = kp.get('v', n - 1)
        s = kp.get('s', n // 2)
        # Systolic points: a, md, b must be between o and s
        a = kp.get('a', 1)
        md = kp.get('md', 2)
        b = kp.get('b', 3)
        a = max(o + 1, min(a, s - 3))
        md = max(a + 1, min(md, s - 2))
        b = max(md + 1, min(b, s - 1))
        kp['a'] = a
        kp['md'] = md
        kp['b'] = b
        # Diastolic points handled by din/dip/d phase below
        return kp
    kp = _force_order(kp, n)

    # Detect default/incorrect diastolic keypoints
    v = kp.get('v', n - 1)
    dn_def = v - 3
    dn_bad = (kp.get('dn') in (None, dn_def, n-1, n-2) or
              kp.get('ip') in (None, dn_def, n-1, n-2) or
              kp.get('d') in (None, dn_def, n-1, n-2) or
              kp.get('dn') == kp.get('ip') == kp.get('d'))

    if not valid or dn_bad:
        fixed = fix_diastolic_keypoints(beat, fs, diastolic_search_ms=diastolic_search_ms)
        if fixed:
            kp.update(fixed)
            # Mark valid=True if systolic is reasonable
            if s > 10 and s < n - 10:
                valid = True

    # Final sanity: ensure dn < ip < d < v
    dn = kp.get('dn', 0)
    ip = kp.get('ip', 0)
    d_pos = kp.get('d', 0)
    if dn is None: dn = int(n * 0.4)
    if ip is None: ip = int(n * 0.5)
    if d_pos is None: d_pos = int(n * 0.6)
    if ip <= dn: ip = min(dn + 3, n - 5)
    if d_pos <= ip: d_pos = min(ip + 3, n - 2)
    kp['dn'] = min(max(dn, 0), n - 1)
    kp['ip'] = min(max(ip, 0), n - 1)
    kp['d'] = min(max(d_pos, 0), n - 1)

    # --- True onset detection: find the actual foot of the pulse ---
    # The beat boundary (position 0) is a midpoint between systolic peaks.
    # The true foot is the last valley before the systolic upstroke.
    # This is especially important for MIMIC data where the boundary may
    # fall well before the foot, causing a "dip then rise" pattern.
    md_curr = kp.get('md', 0)
    if md_curr > 0 and md_curr < n:
        true_o = _find_true_onset(beat, md_curr, fs,
                                  onset_search_ms=onset_search_ms,
                                  onset_margin_ms=onset_margin_ms,
                                  onset_rise_check_ms=onset_rise_check_ms)
        if true_o > 0:
            kp['o'] = true_o
            # Re-apply systolic ordering with corrected onset
            # If a/md/b are before new onset, clamp them after it
            kp = _force_order(kp, n)

    return valid, kp


def extract_beat_features_clean(beat: np.ndarray, keypoints: dict,
                                 fs: int = DEFAULT_FS) -> dict:
    """Extract a compact, non-redundant feature set directly from keypoints.

    Only computes features that are independent fundamental measurements,
    avoiding composite/derived features that can be recomputed downstream.
    All features are computed from the 9 fiducial points + beat signal.

    Args:
        beat: Beat template (should be onset-trimmed: position 0 = true foot).
        keypoints: 9 fiducial points {o,a,md,b,s,dn,ip,d,v} on this beat.
        fs: Sampling rate.

    Returns:
        dict of ~15 physiologically meaningful scalar features.
    """
    n = len(beat)
    kp = keypoints
    feats = {}

    # --- Beat-level metadata ---
    feats['beat_len_ms'] = round(n / fs * 1000, 1)
    feats['hr_bpm'] = round(60.0 / (n / fs), 1)

    # Keypoint positions
    o, s, dn, ip, d_pos, v_pos = [kp.get(k, 0) or 0 for k in ['o', 's', 'dn', 'ip', 'd', 'v']]
    if v_pos <= 0:
        v_pos = n - 1

    # --- Amplitude features ---
    amp_s = float(beat[s]) if 0 <= s < n else 0.0
    amp_o = float(beat[o]) if 0 <= o < n else 0.0
    amp_dn = float(beat[dn]) if 0 <= dn < n else 0.0
    amp_ip = float(beat[ip]) if 0 <= ip < n else 0.0
    amp_d = float(beat[d_pos]) if 0 <= d_pos < n else 0.0

    # Pulse amplitude (AC component)
    feats['pulse_amp'] = round(amp_s - amp_o, 6)

    # Intensity ratios (key vascular health indicators)
    feats['ri'] = round(amp_ip / amp_s, 6) if amp_s != 0 else None          # Reflection Index
    feats['ir_s_d'] = round(amp_s / amp_d, 4) if amp_d != 0 else None       # Systolic/Diastolic ratio
    feats['ni_dn'] = round(amp_dn / amp_s, 6) if amp_s != 0 else None       # Normalized dicrotic notch
    feats['ni_ip'] = round(amp_ip / amp_s, 6) if amp_s != 0 else None       # Normalized inflection point

    # --- Timing features (ms) ---
    feats['t_sys'] = round((s - o) / fs * 1000, 1)           # Systolic duration
    feats['t_dia'] = round((v_pos - s) / fs * 1000, 1)       # Diastolic duration
    feats['t_dn'] = round((dn - s) / fs * 1000, 1) if dn > s else None     # Time to dicrotic notch
    feats['t_dn_d'] = round((d_pos - dn) / fs * 1000, 1) if d_pos > dn > 0 else None  # Notch→diastolic peak
    feats['t_ratio_sd'] = round((s - o) / max(1, v_pos - s), 4)  # Systolic/Diastolic time ratio

    # --- Slope features ---
    if s > o and amp_s != amp_o:
        feats['upstroke_slope'] = round((amp_s - amp_o) / ((s - o) / fs), 4)  # Raw upstroke slope
    else:
        feats['upstroke_slope'] = None

    # LASI (Large Artery Stiffness Index): 1 / Δt from systolic peak to inflection point
    if ip > s:
        feats['lasi'] = round(1.0 / ((ip - s) / fs), 2)
    else:
        feats['lasi'] = None

    return feats


def extract_beat_features(beat: np.ndarray, fs: int = DEFAULT_FS) -> dict:
    """提取单个 beat 的全部特征。

    包括：统计特征、时域特征、频域特征、模板关键点时域特征、
    带宽特征、多项式拟合系数。

    Args:
        beat: 单个 beat 模板
        fs: 采样率

    Returns:
        特征字典（~380+ keys）
    """
    template = pd.Series(beat)

    # 1. 提取关键点
    valid, key_points = extract_key_points(template, fs)

    # 2. 使用 FeatureExtractor 提取全部特征
    fe = FeatureExtractor(template, fs, key_points)
    all_features = fe.extract_features()

    return all_features


def extract_beat_features_compact(beat: np.ndarray, fs: int = DEFAULT_FS) -> dict:
    """提取单个 beat 的紧凑特征集。

    与 extract_beat_features 不同，此函数返回一组精选的、LLM 友好的标量特征。
    适合直接作为 Agent 工具的输出。

    Args:
        beat: 单个 beat 模板
        fs: 采样率

    Returns:
        紧凑特征字典（~25 keys），所有值都是标量
    """
    template = pd.Series(beat)

    # 关键点
    valid, key_points = extract_key_points(template, fs)

    # 统计特征
    stat = extract_statistical(template)

    # 时域特征
    temp = extract_temporal(template, fs)

    # 频域特征
    try:
        spec = extract_spectral(template, fs)
    except Exception:
        spec = {}

    # 关键点时域特征（RI, LASI, mNPV 等）
    kt_feat = {}
    try:
        kt_feat = extract_key_temporal(template, fs, key_points, norm_method='z_score')
    except Exception:
        pass

    # 心率
    heart_rate = None
    if spec.get('fundamental_frequency'):
        heart_rate = round(spec['fundamental_frequency'] * 60, 1)
    if not heart_rate:
        # 从脉搏宽度估算
        period = len(beat) / fs
        if period > 0:
            heart_rate = round(60.0 / period, 1)

    return {
        # 基础
        "valid": valid,
        "period_seconds": round(len(beat) / fs, 3),
        "heart_rate_bpm": heart_rate,
        # 模板关键特征
        "ri": kt_feat.get('RI'),
        "lasi": kt_feat.get('LASI'),
        "mnpv": kt_feat.get('mNPV'),
        "ppgk": kt_feat.get('PPGK'),
        # 统计特征
        "skewness": stat.get('s'),
        "kurtosis": stat.get('k'),
        "rms": stat.get('rms_rms') or stat.get('rms'),
        "perfusion": stat.get('p') or stat.get('p_p'),
        "mav": stat.get('mav'),
        # 时域特征
        "corr": temp.get('corr'),
        "mean_abs_diff": temp.get('mean_abs_diff'),
        # 频域特征
        "fundamental_frequency": spec.get('fundamental_frequency'),
        "spectral_entropy": spec.get('spectral_entropy'),
        "wavelet_entropy": spec.get('wavelet_entropy'),
        "human_range_energy": spec.get('human_range_energy'),
        "fsqi": spec.get('fsqi'),
        # 关键点索引
        "keypoint_s": key_points.get('s'),       # 收缩峰
        "keypoint_dn": key_points.get('dn'),     # 重搏切迹
        "keypoint_d": key_points.get('d'),       # 舒张峰
    }


# ============================================================
# Rich 特征集（~50 维，按生理含义分组）
# ============================================================

# 特征分组定义：{group_name: {output_key: vitalpy_key}}
# 用于 extract_beat_features_rich() 和 analyze_beat 的 feature_group 参数
FEATURE_GROUPS = {
    "arterial_stiffness": {
        "label": "动脉硬度",
        "features": {
            "lasi":              "LASI",
            "slope_systolic":    "slp_s",
            "slope_systolic_n":  "slp_s_n",
            "slope_md":          "slp_md",
            "slope_md_n":        "slp_md_n",
            "pulse_width_50":    "bw50",
            "pulse_width_90":    "bw90",
            "bwr_50":            "bwr50",
        },
        "description": "大动脉弹性、血管硬度、收缩期动力学",
    },
    "wave_reflection": {
        "label": "波反射",
        "features": {
            "reflection_index":  "RI",
            "ni_dn":             "ni_dn",
            "ni_ip":             "ni_ip",
            "t_dn":              "t_dn",
            "nt_dn":             "nt_dn",
        },
        "description": "外周波反射、重搏切迹特征、血管树状态",
    },
    "pulse_morphology": {
        "label": "脉宽形态",
        "features": {
            "systolic_width_50":  "sbw50",
            "systolic_width_75":  "sbw75",
            "diastolic_width_50": "dbw50",
            "diastolic_width_75": "dbw75",
        },
        "description": "收缩期/舒张期各高度脉宽，与顺应性和外周阻力相关",
    },
    "perfusion": {
        "label": "灌注",
        "features": {
            "perfusion_index":       "p",
            "normalized_pulse_vol":  "mNPV",
            "ppg_characteristic":    "PPGK",
            "mean_intensity":        "Im",
            "pulse_amp":             "pulse_amp",  # AC component, computed below
        },
        "description": "外周血流灌注水平、脉容积",
    },
    "statistical_shape": {
        "label": "统计形态",
        "features": {
            "skewness":   "s",
            "kurtosis":   "k",
            "rms":        "rms",
            "crest_factor": "cf",
            "iqr":        "irq",
        },
        "description": "波形整体形态的统计描述",
    },
    "spectral": {
        "label": "频域特征",
        "features": {
            "freq_dominant_1":   "f1",
            "freq_dominant_2":   "f2",
            "freq_dominant_3":   "f3",
            "spectral_entropy":  "spectral_entropy",
            "wavelet_entropy":   "wavelet_entropy",
            "spectral_centroid": "spectral_centroid",
            "fsqi":              "fsqi",
        },
        "description": "频谱分布、信号复杂度、频域质量指标",
    },
}

# 预计算所有 VitalPy key → output_key 的反向映射
_FEATURE_KEY_TO_GROUP = {}
for _group_name, _group_info in FEATURE_GROUPS.items():
    for _output_key, _vitalpy_key in _group_info["features"].items():
        _FEATURE_KEY_TO_GROUP[_vitalpy_key] = (_group_name, _output_key)


def extract_beat_features_rich(
    beat: np.ndarray,
    fs: int = DEFAULT_FS,
    groups: list | str = "all",
) -> dict:
    """提取单个 beat 的全维度 Rich 特征集。

    从 VitalPy 的 ~385 维特征中精选 ~50 个最具生理意义
    的特征，按 7 个生理类别分组返回。

    Args:
        beat: 单个 beat 模板
        fs: 采样率
        groups: 要提取的特征组。
                "all" — 全部 7 组
                "vascular" — arterial_stiffness + wave_reflection + pulse_morphology
                "basic" — 基础组（perfusion + statistical_shape）
                或 list of group names

    Returns:
        dict: {
            "beat_info": {周期, 心率},
            "groups": {
                "arterial_stiffness": {"label": "...", "features": {...}},
                ...
            },
            "all_features_flat": {...}  # 扁平化版本，方便 Agent 直接引用
        }
    """
    import pandas as pd

    template = pd.Series(beat)

    # 解析 groups 参数
    if groups == "all":
        active_groups = list(FEATURE_GROUPS.keys())
    elif groups == "vascular":
        active_groups = ["arterial_stiffness", "wave_reflection", "pulse_morphology"]
    elif groups == "basic":
        active_groups = ["perfusion", "statistical_shape"]
    elif isinstance(groups, list):
        active_groups = [g for g in groups if g in FEATURE_GROUPS]
    else:
        active_groups = list(FEATURE_GROUPS.keys())

    # 提取关键点
    valid, key_points = extract_key_points(template, fs)

    # Only call VitalPy functions whose output is actually used by FEATURE_GROUPS.
    # extract_temporal (KDE entropy, 74% of time) and extract_coeffs are NOT used.
    all_features = {}
    try:
        all_features.update(extract_key_temporal(template, fs, key_points, norm_method='z_score'))
    except Exception:
        pass
    try:
        all_features.update(branch_width_features(template, fs))
    except Exception:
        pass
    try:
        all_features.update(extract_statistical(template))
    except Exception:
        pass
    try:
        all_features.update(extract_spectral(template, fs))
    except Exception:
        pass

    # 心率
    period = len(beat) / fs
    heart_rate = round(60.0 / period, 1) if period > 0 else None

    # 手动补充：pulse_amp = AC component (systolic peak - beat minimum)
    s_idx = int(np.argmax(beat))
    pulse_amp_val = float(beat[s_idx] - np.min(beat))
    all_features['pulse_amp'] = pulse_amp_val

    result = {
        "beat_info": {
            "period_seconds": round(period, 3),
            "heart_rate_bpm": heart_rate,
            "sample_count": len(beat),
            "keypoints_valid": valid,
        },
        "groups": {},
        "all_features_flat": {},
    }

    for group_name in active_groups:
        group_info = FEATURE_GROUPS[group_name]
        group_features = {}

        for output_key, vitalpy_key in group_info["features"].items():
            value = all_features.get(vitalpy_key)
            if value is not None and isinstance(value, (int, float, np.floating)):
                value = float(value)
            group_features[output_key] = value
            result["all_features_flat"][output_key] = value

        result["groups"][group_name] = {
            "label": group_info["label"],
            "description": group_info["description"],
            "features": group_features,
        }

    return result


# ============================================================
# Diastolic keypoint fix — VitalPy fails on PulseDB data
# ============================================================

def fix_diastolic_keypoints(beat: np.ndarray, fs: float = 500.0,
                            diastolic_search_ms: int = 200) -> dict:
    """Find dn/ip/d using 2nd-derivative zero-crossing (inflection point).

    Key insight: use TIME-DISTANCE from systolic peak, not just amplitude.
    - Crossings within 30ms of s with >90% amplitude: systolic decay artifact (skip)
    - First crossing 30ms..diastolic_search_ms from s: accept as dicrotic notch
    - Works for both deep notches (p000451) and subtle plateaus (p000001)

    Physiology: the dicrotic notch occurs 80-200ms after the systolic peak.
    The 2nd-derivative crossing immediately after s (<30ms) is just the
    end of the rapid systolic upstroke, not the notch.

    Detection:
      1. dn: first 2nd-derivative zero-crossing (neg→pos) at least 30ms after s
      2. d:  maximum amplitude between dn+60ms and dn+350ms
      3. ip: maximum of 1st derivative between dn and d

    Returns empty dict if beat is too short or systolic peak at edge.
    """
    n = len(beat)
    s = int(np.argmax(beat))
    s_amp = beat[s]
    if s_amp <= 0 or s >= n - 30 or s < 5:
        return {}

    d2 = np.gradient(np.gradient(beat))

    # Skip crossings too close to s (systolic decay artifact)
    near_margin = int(fs * 0.03)   # 30ms — crossings before this are NOT the notch
    far_limit = int(fs * diastolic_search_ms / 1000)  # notch should be within this range
    search_start = s + near_margin
    search_end = min(n - 10, s + far_limit)

    # ---- 1. dn: first STABLE zero-crossing (not just hovering near zero) ----
    dn = None
    dn_conf = 'none'

    # A stable crossing means d2 is clearly negative BEFORE and clearly positive AFTER.
    # This filters out "hovering near zero" noise where d2 oscillates around 0
    # during the smooth systolic decay.
    def _is_stable_crossing(d2, idx, window=3):
        """Check if d2 crosses from clearly negative to clearly positive."""
        if idx < window or idx + window >= len(d2):
            return False
        # Before: all negative with at least one significantly negative
        before = d2[idx - window:idx + 1]
        after = d2[idx + 1:idx + 1 + window]
        before_neg = np.all(before < 0) and np.min(before) < -0.001 * np.std(d2)
        after_pos = np.all(after > 0) and np.max(after) > 0.001 * np.std(d2)
        return before_neg and after_pos

    # Pass 1: stable crossing in 30-200ms window
    for i in range(search_start, search_end):
        if d2[i] < 0 and d2[i + 1] > 0 and _is_stable_crossing(d2, i, window=4):
            dn = i
            dn_conf = 'd2'
            break

    # Pass 2: any crossing in 30-200ms (less strict)
    if dn is None:
        for i in range(search_start, search_end):
            if d2[i] < 0 and d2[i + 1] > 0:
                dn = i
                dn_conf = 'd2_weak'
                break

    # Pass 3: extend search to end of beat
    if dn is None:
        for i in range(search_start, n - 10):
            if d2[i] < 0 and d2[i + 1] > 0:
                dn = i
                dn_conf = 'd2_late'
                break

    # Pass 3: fallback — argmin of d2 (point of maximum concavity change)
    if dn is None and search_end > search_start + 3:
        dn = search_start + int(np.argmin(d2[search_start:search_end]))
        dn_conf = 'd2min'

    if dn is None:
        dn = s + int((n - s) * 0.20)
        dn_conf = 'prop'

    # ---- 2. d: diastolic peak / plateau end ----
    # Two cases:
    #   a) Real diastolic wave: local max in dn+30ms..dn+200ms → use peak
    #   b) Plateau with no wave: signal flattens after dn → use flattest point
    from scipy.signal import argrelextrema
    d1 = np.gradient(beat)
    d_start = dn + int(fs * 0.03)
    d_end = min(n - 3, dn + int(fs * 0.20))

    d = None
    if d_end > d_start + 5:
        # Try finding a local max (diastolic wave)
        for order in [5, 3, 2]:
            d_candidates = argrelextrema(beat[d_start:d_end], np.greater, order=order)[0]
            if len(d_candidates) > 0:
                d = d_start + int(d_candidates[np.argmax(beat[d_start:d_end][d_candidates])])
                break

        if d is None:
            # No diastolic wave — find the plateau: where 1st derivative is
            # closest to zero after dn (the "flattest" part of the descent)
            # Weight toward later positions (plateau is usually not right at dn)
            flat_region = d1[d_start:d_end]
            # Find minimum absolute derivative (flattest point)
            flat_idx = int(np.argmin(np.abs(flat_region)))
            d = d_start + flat_idx

            # Ensure we're not right at dn edge
            if d - dn < int(fs * 0.04):
                d = d_start + int(np.argmin(np.abs(d1[d_start+int(fs*0.02):d_end]))) + int(fs*0.02)
    else:
        d = min(dn + int(fs * 0.10), n - 2)

    # ---- 3. ip: max of 1st derivative between dn and d ----
    ip_start = dn + int(fs * 0.01)
    ip_end = max(d - 2, ip_start + 3)
    if ip_end > ip_start + 3:
        ip_candidates = argrelextrema(d1[ip_start:ip_end], np.greater, order=3)[0]
        if len(ip_candidates) > 0:
            ip = ip_start + int(ip_candidates[np.argmax(d1[ip_start:ip_end][ip_candidates])])
        else:
            # If no clear 1st-deriv peak, ip is at max of d1 in range
            ip = ip_start + int(np.argmax(d1[ip_start:ip_end]))
    else:
        ip = int((dn + d) / 2)

    # ---- 4. Enforce ordering: s < dn < ip < d < v ----
    dn = max(s + near_margin, min(dn, ip - 2, n - 10))
    ip = max(dn + 2, min(ip, d - 2))
    d = min(max(ip + 2, d), n - 2)

    return {
        'dn': int(dn),
        'ip': int(ip),
        'd': int(d),
    }


def extract_beat_features_fixed(beat: np.ndarray, fs: int = DEFAULT_FS,
                                 groups: str = "all",
                                 diastolic_search_ms: int = 200,
                                 notch_depth_threshold: float = 0.90) -> dict:
    """Extract rich features with CORRECTED diastolic keypoints.

    Wraps extract_beat_features_rich but fixes dn/ip/d before computing
    keypoint-dependent features (RI, LASI, ratio_sys_dia, etc.).
    """
    result = extract_beat_features_rich(beat, fs, groups=groups)
    feats = result.get("all_features_flat", {})
    if not feats:
        return result

    fixed = fix_diastolic_keypoints(beat, fs, diastolic_search_ms=diastolic_search_ms)
    if not fixed:
        return result

    s = int(np.argmax(beat))
    dn = fixed['dn']
    ip = fixed['ip']
    d = fixed['d']
    v = len(beat) - 1

    # Recompute keypoint-dependent features
    # Only compute these if the dicrotic notch amplitude is substantially below
    # systolic peak (indicating a real notch, not just a slight slope change)
    dn_depth = float(beat[dn] / beat[s]) if beat[s] != 0 else 1.0
    # For subtle/shoulder-type notches, amplitude 75-85% is normal
    # Only reject if notch is essentially at systolic level (no real decay)
    is_real_notch = (dn_depth < notch_depth_threshold)

    if is_real_notch:
        # Overwrite VitalPy values with corrected-keypoint versions
        feats['reflection_index'] = round(float(beat[ip] / beat[s]), 6)
        t_s = s / fs
        t_ip = ip / fs
        feats['lasi'] = round(1.0 / (t_ip - t_s), 4) if t_ip > t_s else 0.0
        feats['ni_dn'] = round(float(beat[dn] / beat[s]), 6)
        feats['ni_ip'] = round(float(beat[ip] / beat[s]), 6)
        feats['t_dn'] = round(float((dn - s) / fs), 6)
        feats['nt_dn'] = round(float((dn - s) / (v - s)), 6) if v > s else 0.0
    else:
        feats['reflection_index'] = None
        feats['lasi'] = None
        feats['ni_dn'] = None
        feats['ni_ip'] = None
        feats['t_dn'] = None
        feats['nt_dn'] = None

    # Intensity ratio s/d (keep as is, not in FEATURE_GROUPS but useful)
    feats['intensity_ratio_s_d'] = round(float(beat[s] / beat[d]), 6) if beat[d] != 0 else 0.0

    # Slope features need derivative, keep VitalPy's values for now
    # (they use normalized templates, so keypoint shift matters less)

    result["all_features_flat"] = feats
    return result


# ============================================================
# 统一流水线：segment → merge → validate → onset-trim → extract
# ============================================================

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
        params: Processing parameter overrides (see ppg_agent.processing_params).
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
    from ppg_agent.processing_params import resolve_params
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


# ============================================================
# 完整 segment 级特征提取（beat 级 + segment 聚合 + PTT + HRV + ABP）
# ============================================================

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
        params: Processing parameter overrides (see ppg_agent.processing_params).

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


def extract_keypoints_from_segment(ppg: np.ndarray, fs: int = DEFAULT_FS,
                                   params: dict | None = None) -> dict:
    """Lightweight: extract only keypoints + beat positions, no feature computation.

    ~10x faster than extract_segment_features. Useful for adding keypoint
    columns to existing parquet without re-running the full pipeline.

    Returns:
        {'beats': [{beat_idx, beat_start_sample, kp_o..kp_v}, ...]}
    """
    ppg_result = process_ppg_segment(ppg, fs, merge_fragments=True, trim_to_onset=True,
                                     params=params)
    kps_list = ppg_result['keypoints']
    starts = ppg_result['beat_starts']

    beat_records = []
    for bi in range(len(starts)):
        rec = {'beat_idx': bi, 'beat_start_sample': int(starts[bi])}
        if bi < len(kps_list) and kps_list[bi]:
            for kp_name in ['o', 'a', 'md', 'b', 's', 'dn', 'ip', 'd', 'v']:
                val = kps_list[bi].get(kp_name)
                rec[f'kp_{kp_name}'] = int(val) if val is not None else None
        beat_records.append(rec)
    return {'beats': beat_records}
