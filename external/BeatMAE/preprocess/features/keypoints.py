"""40 维拍级形态学特征提取 —— 拍级关键点（含 onset 修正与舒张点修复）

本文件 2026-09-20 迁入本仓。代码为原项目自写，不派生自第三方；
第三方派生部分见 `third_party/vitalpy/`。
"""
# 从上游引擎裁取。函数内依赖（scipy 等）保持原样，含各自的函数级 import。
import numpy as np
import pandas as pd          # extract_beat_keypoints_fixed 用 pd.Series(beat)

from third_party.vitalpy.features.utils import extract_key_points

DEFAULT_FS = 500   # 上游 ppg_engine.py 顶层常量

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


__all__ = ['extract_beat_keypoints_fixed', 'fix_diastolic_keypoints']
