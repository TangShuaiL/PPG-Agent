"""Single source of truth for processing parameters, preset I/O, and validation.

Replaces the tuner's PARAM_SPEC and the engine's hardcoded constants.
All 14 parameters: 7 A-class (beat detection, free ranges), 5 B-class
(feature-point detection, predefined candidates), 2 switches.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ppg_agent.config import MODELS_ROOT

PRESET_DIR = MODELS_ROOT / "param_presets"
LOCKED_PRESETS = {"default"}


@dataclass(frozen=True)
class ParamSpec:
    key: str
    default: Any
    kind: str                    # "range" | "choice" | "switch"
    cls: str = "A"               # A=beat detection, B=feature-point, S=switch
    min: float | None = None
    max: float | None = None
    candidates: list | None = None
    desc: str = ""

    def validate(self, value: Any) -> str | None:
        """Return a human-readable error string, or None if valid."""
        if self.kind == "range":
            try:
                v = float(value)
            except (TypeError, ValueError):
                return f"{self.key}: {value!r} is not numeric"
            if not (self.min <= v <= self.max):
                return f"{self.key}: {value} out of range [{self.min}, {self.max}]"
        else:  # choice / switch
            if value not in (self.candidates or []):
                return f"{self.key}: {value!r} not in {self.candidates}"
        return None


PARAM_SPEC: dict[str, ParamSpec] = {
    # ── A-class: beat detection (free ranges) ──
    "peak_distance_ms": ParamSpec("peak_distance_ms", 300, "range", "A", 150, 600,
                                  desc="Minimum distance between systolic peaks (ms)"),
    "peak_prominence": ParamSpec("peak_prominence", 0.05, "range", "A", 0.01, 0.30,
                                 desc="Peak prominence as fraction of signal range"),
    "min_segment_ms": ParamSpec("min_segment_ms", 200, "range", "A", 100, 400,
                                desc="Minimum initial segment length (ms)"),
    "min_beat_len": ParamSpec("min_beat_len", 150, "range", "A", 80, 300,
                              desc="Minimum beat length in samples @500Hz"),
    "max_beat_len_ms": ParamSpec("max_beat_len_ms", 1200, "range", "A", 600, 2000,
                                 desc="Maximum beat length (ms)"),
    "validation_mode": ParamSpec("validation_mode", "strict", "choice", "A",
                                 candidates=["strict", "lenient"],
                                 desc="Template validation strictness"),
    "trough_diff_max": ParamSpec("trough_diff_max", 0.2, "range", "A", 0.1, 0.6,
                                 desc="Max start/end trough difference as fraction of "
                                      "beat amplitude (0.2 = clinical strict default; "
                                      "free-living wrist data typically 0.4)"),
    "template_corr_min": ParamSpec("template_corr_min", 0.0, "range", "S", 0.0, 1.0,
                                   desc="Min Pearson corr with the segment median "
                                        "template for valid beats that fail the strict "
                                        "(0.2) trough check; 0 disables the check"),
    "resegment_min_ms": ParamSpec("resegment_min_ms", 150, "range", "A", 80, 300,
                                  desc="Minimum re-segmented beat length (ms)"),
    # ── B-class: feature-point detection (predefined candidates) ──
    "onset_search_ms": ParamSpec("onset_search_ms", 120, "choice", "B",
                                 candidates=[80, 120, 160, 200],
                                 desc="Onset search window before max-derivative (ms)"),
    "onset_margin_ms": ParamSpec("onset_margin_ms", 5, "choice", "B",
                                 candidates=[5, 10],
                                 desc="Margin before max-derivative excluded from search (ms)"),
    "onset_rise_check_ms": ParamSpec("onset_rise_check_ms", 50, "choice", "B",
                                     candidates=[30, 50, 80],
                                     desc="Rise verification window after onset (ms)"),
    "diastolic_search_ms": ParamSpec("diastolic_search_ms", 200, "choice", "B",
                                     candidates=[150, 200, 250],
                                     desc="Diastolic keypoint search limit (ms)"),
    "notch_depth_threshold": ParamSpec("notch_depth_threshold", 0.90, "choice", "B",
                                       candidates=[0.85, 0.90, 0.95],
                                       desc="Dicrotic notch amplitude threshold"),
    # ── S-class: pipeline switches ──
    "extra_bandpass": ParamSpec("extra_bandpass", False, "switch", "S",
                                candidates=[False, True],
                                desc="Cascade 0.5-8Hz Butterworth bandpass before segmentation"),
    "mad_reject_k": ParamSpec("mad_reject_k", "off", "switch", "S",
                              candidates=["off", 2.0, 2.5, 3.0],
                              desc="MAD-based amplitude outlier beat rejection (k, or off)"),
}

DEFAULT_PARAMS: dict[str, Any] = {k: s.default for k, s in PARAM_SPEC.items()}

# source → preset name resolution (missing presets fall back to "default")
SOURCE_PRESET_MAP = {
    "vital": "default",
    "mimic": "default",
    "aurora": "aurora",
    "aurorabp": "aurora",
    "anker": "anker",
}


def validate_params(params: dict, *, require_all: bool = False) -> list[str]:
    """Validate a params dict against PARAM_SPEC. Returns list of error strings."""
    errors = []
    unknown = set(params) - set(PARAM_SPEC)
    if unknown:
        errors.append(f"unknown parameter(s): {sorted(unknown)}")
    for key, spec in PARAM_SPEC.items():
        if key not in params:
            if require_all:
                errors.append(f"missing parameter: {key}")
            continue
        err = spec.validate(params[key])
        if err:
            errors.append(err)
    return errors


def resolve_params(params: dict | None) -> dict:
    """Merge params over defaults and validate. Raises ValueError on invalid input."""
    resolved = {**DEFAULT_PARAMS, **(params or {})}
    errors = validate_params(resolved)
    if errors:
        raise ValueError("invalid processing params: " + "; ".join(errors))
    return resolved


def _preset_path(name: str) -> Path:
    """Map a preset name to its file, organized per dataset:

        default            → param_presets/default.yaml          (global fallback)
        user_<dataset>     → param_presets/<dataset>/user.yaml   (user adaptation)
        <dataset>          → param_presets/<dataset>/default.yaml (official default)
        <custom>           → param_presets/<custom>.yaml          (flat, user-named)

    "Dataset" here means any source-mapped official preset name — the flat
    layout is kept for arbitrary names so historical presets keep working.
    """
    name = name.lower()
    if name == "default":
        return PRESET_DIR / "default.yaml"
    if name.startswith("user_"):
        return PRESET_DIR / name[5:] / "user.yaml"
    if name in SOURCE_PRESET_MAP.values():
        return PRESET_DIR / name / "default.yaml"
    return PRESET_DIR / f"{name}.yaml"


def load_preset(name: str) -> dict:
    """Load a preset, merging over defaults (old 7-key presets remain valid)."""
    fp = _preset_path(name)
    if not fp.exists():
        raise FileNotFoundError(f"preset not found: {fp}")
    with open(fp, "r") as f:
        data = yaml.safe_load(f) or {}
    return resolve_params(data)


def save_preset(name: str, params: dict) -> dict:
    """Write all 14 keys to a preset file. Refuses to overwrite 'default'."""
    if name in LOCKED_PRESETS:
        raise ValueError(f"cannot overwrite locked preset: {name}")
    resolved = resolve_params(params)
    fp = _preset_path(name)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as f:
        yaml.dump(dict(sorted(resolved.items())), f, default_flow_style=False, allow_unicode=True)
    return {"preset": name, "path": str(fp), "params": resolved}


def list_presets() -> list[str]:
    """All available presets as names (inverse of _preset_path):
    root default.yaml → 'default'; <dir>/default.yaml → '<dir>';
    <dir>/user.yaml → 'user_<dir>'; flat *.yaml → their stem."""
    if not PRESET_DIR.exists():
        return []
    out: list[str] = []
    if (PRESET_DIR / "default.yaml").exists():
        out.append("default")
    for d in sorted(PRESET_DIR.iterdir()):
        if d.is_dir():
            if (d / "default.yaml").exists():
                out.append(d.name)
            if (d / "user.yaml").exists():
                out.append(f"user_{d.name}")
        elif d.suffix == ".yaml" and d.stem != "default":
            out.append(d.stem)
    return sorted(set(out))


def preset_for_source(source: str, user_ok: bool = True) -> str:
    """Resolve the preset name for a dataset source; falls back to 'default'.

    Priority: user adaptation (``user_<source>``, saved by the web chat /
    Agent 1 adapt mode) → source-mapped official preset → 'default'.
    ``user_ok=False`` skips the user preset (scientific re-runs of official
    baselines must never be silently affected by a user's tuning).
    """
    if user_ok:
        user = f"user_{source}"
        if user in list_presets():
            return user
    name = SOURCE_PRESET_MAP.get(source, "default")
    return name if name in list_presets() else "default"


def ensure_default_preset() -> str:
    """Create/refresh default.yaml with all 14 defaults. Returns its path."""
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    fp = _preset_path("default")
    with open(fp, "w") as f:
        yaml.dump(dict(sorted(DEFAULT_PARAMS.items())), f,
                  default_flow_style=False, allow_unicode=True)
    return str(fp)
