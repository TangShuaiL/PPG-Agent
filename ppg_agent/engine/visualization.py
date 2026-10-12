"""Shared visualization engine — single-axis PPG plots with beat annotations.

All agents use these functions.  Images saved under reports/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ppg_agent.config import REPORTS_ROOT

matplotlib.use("Agg")

FIGURES_DIR = Path(REPORTS_ROOT) / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLOR_RAW = "#7f8c8d"
COLOR_PROCESSED = "#2c3e50"
BG_VALID = "#d5f5e3"        # light green
BG_REJECTED = "#fadbd8"      # light red
COLOR_KEYPOINT = "#e74c3c"
COLOR_KEYPOINT_S = "#2980b9"  # systolic peak in blue

KEYPOINT_DISPLAY = {
    "o":  ("onset",       "D"),
    "a":  ("a-wave",      "s"),
    "md": ("max deriv.",  "p"),
    "b":  ("b-point",     "^"),
    "s":  ("sys. peak",   "o"),
    "dn": ("dicrotic n.", "v"),
    "ip": ("infl. pt.",   "<"),
    "d":  ("dia. peak",   "o"),
    "v":  ("end",         "D"),
}


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def plot_signal_with_beats(
    raw: np.ndarray,
    processed: np.ndarray | None,
    beats: list[np.ndarray],
    beat_starts: list[int],
    valid_mask: list[bool],
    fs: int,
    keypoints_per_beat: list[dict] | None = None,
    title: str = "PPG Signal with Beat Annotations",
    filename: str | None = None,
    show_keypoints: bool = True,
    show_beat_details: bool = False,
) -> dict:
    """PPG signal with green/red beat backgrounds and fiducial keypoint annotations.

    Each beat span gets a green (valid) or red (rejected) background.
    When show_keypoints=True, all 9 fiducial points (o,a,md,b,s,dn,ip,d,v)
    are marked directly on the main waveform with labels.

    Args:
        raw:              Raw PPG signal.
        processed:        Preprocessed signal (overlaid if provided).
        beats:            Beat templates (must match beat_starts length).
        beat_starts:      Start sample index per beat.
        valid_mask:       True = valid, False = rejected.
        fs:               Sampling rate in Hz.
        keypoints_per_beat: Optional [{name: sample_offset}, ...] per beat.
        title:            Plot title.
        filename:         Output filename stem (without .png).
        show_keypoints:   Mark fiducial points on the waveform.
    """
    n_beats = len(beats)
    main_signal = processed if processed is not None else raw
    t = np.arange(len(main_signal)) / fs

    # Layout: main + up to 6 beat detail rows
    has_details = show_beat_details and keypoints_per_beat and n_beats > 1
    detail_rows = min(n_beats, 6) if has_details else 0
    fig_height = 5 + 2.5 * detail_rows
    fig, ax_main = plt.subplots(figsize=(18, fig_height))

    # Create detail axes
    detail_axes = []
    if has_details:
        gs = fig.add_gridspec(1 + detail_rows, 1, height_ratios=[5] + [2.5]*detail_rows, hspace=0.35)
        ax_main = fig.add_subplot(gs[0])
        for i in range(detail_rows):
            detail_axes.append(fig.add_subplot(gs[1 + i]))
    else:
        gs = None
        ax_main = fig.add_subplot(111)

    # ---- Main axis: beat backgrounds ----
    y_min, y_max = float(np.min(main_signal)), float(np.max(main_signal))
    y_pad = (y_max - y_min) * 0.1

    for i, (start, valid) in enumerate(zip(beat_starts, valid_mask)):
        t_start = start / fs
        t_end = t_start + len(beats[i]) / fs
        color = BG_VALID if valid else BG_REJECTED
        ax_main.axvspan(t_start, t_end, facecolor=color, alpha=0.5, zorder=0)

    # ---- Beat separator lines ----
    for i in range(len(beat_starts)):
        # Left boundary of each beat
        t_boundary = beat_starts[i] / fs
        ax_main.axvline(x=t_boundary, color='#555555', linewidth=0.6,
                       linestyle='--', alpha=0.5, zorder=3)
    # Right boundary of last beat
    if beat_starts:
        t_end = (beat_starts[-1] + len(beats[-1])) / fs
        ax_main.axvline(x=t_end, color='#555555', linewidth=0.6,
                       linestyle='--', alpha=0.5, zorder=3)

    if processed is not None:
        ax_main.plot(np.arange(len(raw))/fs, raw, color=COLOR_RAW, linewidth=0.6, alpha=0.7, label="Raw", zorder=1)
    ax_main.plot(t, main_signal, color=COLOR_PROCESSED, linewidth=1.2,
                 label="Preprocessed" if processed is not None else "Signal", zorder=2)

    if show_keypoints and keypoints_per_beat:
        _mark_keypoints(ax_main, main_signal, beats, beat_starts, valid_mask,
                        keypoints_per_beat, fs, y_min, y_max)

    ax_main.set_xlim(0, t[-1])
    ax_main.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_main.set_title(title, fontsize=14, fontweight="bold")
    ax_main.set_xlabel("Time (s)", fontsize=11)
    ax_main.set_ylabel("Amplitude", fontsize=11)
    # Keypoint labels: horizontal strip BELOW the axis (never covers the
    # waveform); the in-axis legend keeps only the signal lines.
    if show_keypoints and not has_details:
        _draw_keypoint_label_strip(fig, y_bottom=0.06)
    if ax_main.get_legend_handles_labels()[0]:
        ax_main.legend(loc="upper right", fontsize=9)

    for i, start in enumerate(beat_starts):
        t_mid = (start + len(beats[i]) / 2) / fs
        ax_main.annotate(f"B{i}", (t_mid, y_max + y_pad * 0.3), fontsize=7,
                         ha="center", va="bottom", color="#555555")

    # ---- Beat detail subplots ----
    if has_details:
        # Sort: show rejected beats first (red), then validated (green) up to 6 total
        indices = list(range(n_beats))
        rejected = [i for i in indices if not valid_mask[i]]
        validated = [i for i in indices if valid_mask[i]]
        show_indices = (rejected + validated)[:6]

        for row, beat_i in enumerate(show_indices):
            ax = detail_axes[row]
            bt = beats[beat_i]
            bt_t = np.arange(len(bt)) / fs
            is_val = valid_mask[beat_i]
            bg = BG_VALID if is_val else BG_REJECTED
            label = f"Beat {beat_i} {'✓' if is_val else '✗'}"
            ax.fill_between(bt_t, bt, alpha=0.15, color=bg)
            ax.plot(bt_t, bt, color=COLOR_PROCESSED, linewidth=1.0, label=label)

            # Annotate all keypoints on this beat
            kp = keypoints_per_beat[beat_i] if beat_i < len(keypoints_per_beat) else {}
            for name, (disp_label, marker) in KEYPOINT_DISPLAY.items():
                idx = kp.get(name)
                if idx is None or idx < 0 or idx >= len(bt):
                    continue
                color_kp = COLOR_KEYPOINT_S if name == 's' else COLOR_KEYPOINT
                ax.plot(bt_t[idx], bt[idx], marker=marker, color=color_kp,
                        markersize=5, markeredgewidth=1, markerfacecolor='white', zorder=5)

            ax.set_ylabel(f'B{beat_i}', fontsize=8)
            ax.legend(fontsize=7, loc='upper right')
            if row == len(show_indices) - 1:
                ax.set_xlabel("Time (s)", fontsize=9)

    fname = filename or "ppg_beats"
    path = FIGURES_DIR / f"{fname}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_valid = sum(valid_mask)
    return {
        "status": "success",
        "path": str(path),
        "filename": path.name,
        "n_beats_total": n_beats,
        "n_beats_valid": n_valid,
        "n_beats_rejected": n_beats - n_valid,
        "dimensions": f"{int(fig.get_figwidth())}x{int(fig.get_figheight())} inches",
    }


KP_CATEGORY = {
    'o': ('#27ae60', 'o onset'), 'a': ('#8e44ad', 'a a-wave'),
    'md': ('#8e44ad', 'md pre-peak'), 'b': ('#8e44ad', 'b slope'),
    's': ('#2980b9', 's systolic peak'), 'dn': ('#e67e22', 'dn notch'),
    'ip': ('#e74c3c', 'ip diastolic'), 'd': ('#e74c3c', 'd end-dia'),
    'v': ('#27ae60', 'v end'),
}
KP_ORDER = ['o', 'a', 'md', 'b', 's', 'dn', 'ip', 'd', 'v']


def _draw_keypoint_label_strip(fig: plt.Figure, y_bottom: float = 0.06) -> None:
    """Horizontal keypoint legend below the axis — never covers the waveform.

    One colored dot + name per fiducial point, laid out left-to-right in
    signal order across the bottom of the figure.
    """
    fig.subplots_adjust(bottom=0.20)
    n = len(KP_ORDER)
    for i, name in enumerate(KP_ORDER):
        color, label = KP_CATEGORY[name]
        x_pos = 0.03 + i * (0.94 / n)
        fig.text(x_pos, y_bottom, '●', color=color, fontsize=11,
                 ha='center', va='center', transform=fig.transFigure)
        fig.text(x_pos + 0.012, y_bottom, label, fontsize=8.5, ha='left',
                 va='center', color='#333333', transform=fig.transFigure)


def _mark_keypoints(
    ax: plt.Axes,
    signal: np.ndarray,
    beats: list[np.ndarray],
    beat_starts: list[int],
    valid_mask: list[bool],
    keypoints_per_beat: list[dict],
    fs: int,
    y_min: float,
    y_max: float,
) -> None:
    """Mark all 9 fiducial points on the main signal axis with labels.

    Only marks keypoints on validated beats.
    Alternates label offset direction to avoid overlap.
    """
    y_span = y_max - y_min

    for beat_i, (start, kp, valid) in enumerate(zip(beat_starts, keypoints_per_beat, valid_mask)):
        if not valid or not kp:
            continue

        _draw_kp_markers(ax, signal, beats[beat_i], start, kp, fs)


def _draw_kp_markers(
    ax: plt.Axes,
    signal: np.ndarray,
    beat: np.ndarray,
    start: int,
    kp: dict,
    fs: int,
) -> None:
    """Draw keypoint markers with distinct colors per category."""

    # Bold distinct colors
    SYS_COLOR = "#2980b9"     # blue — systolic peak (s)
    DIA_COLOR = "#e74c3c"     # red — diastolic (ip, d)
    NOTCH_COLOR = "#e67e22"   # orange — dicrotic notch (dn)
    ONSET_COLOR = "#27ae60"   # green — onset/end (o, v)
    SLOPE_COLOR = "#8e44ad"   # purple — derivative/a-wave (a, md, b)

    category_colors = {
        's': SYS_COLOR, 'dn': NOTCH_COLOR,
        'ip': DIA_COLOR, 'd': DIA_COLOR,
        'o': ONSET_COLOR, 'v': ONSET_COLOR,
        'a': SLOPE_COLOR, 'md': SLOPE_COLOR, 'b': SLOPE_COLOR,
    }

    # Track position → count for overlapping offsets
    pos_count: dict[int, int] = {}
    for name in ['o', 'a', 'md', 'b', 's', 'dn', 'ip', 'd', 'v']:
        idx = kp.get(name)
        if idx is None or idx < 0 or idx >= len(beat):
            continue
        global_idx = start + idx
        if global_idx >= len(signal):
            continue
        color = category_colors.get(name, "#95a5a6")

        # Offset overlapping points vertically so all are visible
        n = pos_count.get(global_idx, 0)
        pos_count[global_idx] = n + 1
        y_val = signal[global_idx]
        y_range = float(np.max(signal) - np.min(signal))
        y_offset = n * y_range * 0.015  # small vertical stagger

        ax.plot(global_idx / fs, y_val + y_offset, 'o',
                color=color, markersize=6, markeredgewidth=0.5,
                markeredgecolor='white',
                markerfacecolor=color, zorder=6)


# ---------------------------------------------------------------------------
# Feature extraction overview (adaptation session)
# ---------------------------------------------------------------------------

# Keypoint category colors — matches the analysis-script style
KEYPOINT_COLORS = {
    "o": "#27ae60", "v": "#27ae60",        # onset / end — green
    "s": "#2980b9",                         # systolic peak — blue
    "dn": "#e67e22",                        # dicrotic notch — orange
    "ip": "#e74c3c", "d": "#e74c3c",        # diastolic — red
    "a": "#8e44ad", "md": "#8e44ad", "b": "#8e44ad",  # slope — purple
}


def plot_feature_extraction_overview(
    beats: list[np.ndarray],
    beat_starts: list[int],
    valid_mask: list[bool],
    keypoints: list[dict],
    fs: int,
    title: str = "Feature Extraction Overview",
    filename: str | None = None,
    mad_cutoffs: tuple[float, float] | None = None,
) -> dict:
    """Three-panel feature-extraction overview for the adaptation session.

    Row 1: zoomed grid of up to 6 beats with the 9 fiducial points marked
           (rejected beats first — same ordering rule as show_beat_details).
    Row 2: per-beat scatter of pulse amplitude and rise time, valid/rejected
           colored, with optional MAD cutoff lines.
    Row 3: mean valid-beat template ± 1 std with keypoint positions.

    Args:
        beats: Beat templates.
        beat_starts: Absolute sample positions.
        valid_mask: Per-beat validity.
        keypoints: Per-beat keypoint dicts ({} for invalid).
        fs: Sampling rate.
        title: Figure title.
        filename: Output filename stem.
        mad_cutoffs: Optional (lo, hi) amplitude cutoffs for the scatter panel.

    Returns:
        {status, path, filename, n_valid, n_rejected}
    """
    n_valid = sum(valid_mask)
    n_rejected = len(valid_mask) - n_valid
    ordered = sorted(range(len(beats)),
                     key=lambda i: (valid_mask[i], i))  # rejected first

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 1.0], hspace=0.38)

    # ---- Row 1: beat grid ----
    ax_grid = fig.add_subplot(gs[0])
    n_show = min(6, len(beats))
    for slot, bi in enumerate(ordered[:n_show]):
        bt = beats[bi]
        kp = keypoints[bi] if bi < len(keypoints) else {}
        t = (np.arange(len(bt)) + beat_starts[bi]) / fs
        y_off = slot * 1.6 * (float(np.max(bt) - np.min(bt)) + 1e-9)
        ax_grid.plot(t, bt + y_off, color=COLOR_PROCESSED, lw=0.9, zorder=2)
        bg = BG_VALID if valid_mask[bi] else BG_REJECTED
        ax_grid.axvspan(t[0], t[-1], ymin=0.06 + slot * 0.15, ymax=0.20 + slot * 0.15,
                        facecolor=bg, alpha=0.5, zorder=0)
        ax_grid.text(t[0] - 0.12, y_off + (np.max(bt) + np.min(bt)) / 2,
                     f"B{bi}", fontsize=7, va="center", ha="right")
        for name, idx in (kp or {}).items():
            if idx is None or idx < 0 or idx >= len(bt):
                continue
            marker = KEYPOINT_DISPLAY[name][1]
            color = KEYPOINT_COLORS.get(name, COLOR_KEYPOINT)
            ax_grid.plot(t[idx], bt[idx] + y_off, marker=marker, color=color,
                         markersize=5, markeredgewidth=0.3, zorder=5)
    ax_grid.set_title("Beat keypoints (rejected first)", fontsize=10, fontweight="bold")
    ax_grid.set_ylabel("Amplitude (offset per beat)")
    ax_grid.set_yticks([])
    ax_grid.grid(True, alpha=0.2)

    # ---- Row 2: per-beat scatter ----
    ax_sc = fig.add_subplot(gs[1])
    amps, rises = [], []
    for i, (bt, kp) in enumerate(zip(beats, keypoints)):
        y = bt.astype(np.float64)
        amps.append(float(np.max(y) - np.min(y)))
        s_idx, o_idx = (kp or {}).get('s'), (kp or {}).get('o')
        if s_idx is not None and o_idx is not None and 0 <= s_idx < len(y) and 0 <= o_idx < len(y):
            rises.append((s_idx - o_idx) / fs * 1000)
        else:
            rises.append(np.nan)
    vc = np.array(valid_mask)
    rc = ~vc
    ax_sc.scatter(np.arange(len(beats))[vc], np.array(amps)[vc], c=BG_VALID,
                  edgecolors="#27ae60", label="valid", s=40, zorder=2)
    ax_sc.scatter(np.arange(len(beats))[rc], np.array(amps)[rc], c=BG_REJECTED,
                  edgecolors="#e74c3c", label="rejected", s=40, zorder=2)
    ax_sc.set_xlabel("Beat index")
    ax_sc.set_ylabel("Pulse amplitude", color="#2c3e50")
    ax_sc.set_ylim(0, max(amps) * 1.2 if amps else 1)
    ax_sc.set_title("Per-beat amplitude & rise time", fontsize=10, fontweight="bold")
    if mad_cutoffs is not None:
        lo, hi = mad_cutoffs
        ax_sc.axhline(hi, color="#e67e22", ls="--", lw=1.2, label=f"MAD cutoff {hi:.3f}")
        ax_sc.axhline(lo, color="#e67e22", ls="--", lw=1.2, alpha=0.6)
    ax_r = ax_sc.twinx()
    ax_r.scatter(np.arange(len(beats))[vc], np.array(rises)[vc], marker="x",
                 color="#2980b9", s=35, alpha=0.8, label="rise time")
    ax_r.set_ylabel("Rise time (ms)", color="#2980b9")
    ax_sc.grid(True, alpha=0.2)

    # ---- Row 3: mean template ----
    ax_t = fig.add_subplot(gs[2])
    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    valid_beats = [beats[i] for i in valid_idx]
    if valid_beats:
        L = int(np.median([len(b) for b in valid_beats]))
        resampled = []
        for b in valid_beats:
            resampled.append(np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(b)), b))
        M = np.vstack(resampled)
        mean_t = M.mean(axis=0)
        std_t = M.std(axis=0)
        t_axis = np.arange(L) / fs
        ax_t.plot(t_axis, mean_t, color=COLOR_PROCESSED, lw=1.5, label="mean")
        ax_t.fill_between(t_axis, mean_t - std_t, mean_t + std_t, alpha=0.25,
                          color=COLOR_PROCESSED, label="±1 std")
        # Keypoint positions from a representative valid beat (closest length)
        rep_pos = min(range(len(valid_beats)), key=lambda j: abs(len(valid_beats[j]) - L))
        rep = valid_beats[rep_pos]
        rep_kp = keypoints[valid_idx[rep_pos]] if rep_pos < len(keypoints) else {}
        for name, idx in (rep_kp or {}).items():
            if idx is None or idx < 0:
                continue
            x = idx / len(rep) * len(t_axis)
            if x >= len(t_axis):
                continue
            ax_t.plot(t_axis[int(x)], mean_t[int(x)],
                      marker=KEYPOINT_DISPLAY[name][1],
                      color=KEYPOINT_COLORS.get(name, COLOR_KEYPOINT),
                      markersize=7, markeredgewidth=0.3, zorder=5)
    ax_t.set_title(f"Mean valid-beat template ± 1 std (n={len(valid_beats)})",
                   fontsize=10, fontweight="bold")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Amplitude")
    ax_t.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fname = filename or "feature_extraction_overview"
    path = FIGURES_DIR / f"{fname}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"status": "success", "path": str(path), "filename": path.name,
            "n_valid": n_valid, "n_rejected": n_rejected}
