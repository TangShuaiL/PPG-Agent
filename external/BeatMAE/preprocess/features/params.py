"""40 维拍级形态学特征提取 —— 处理参数（14 键，硬编码，零文件 I/O）

本文件 2026-09-20 迁入本仓。代码为原项目自写，不派生自第三方；
第三方派生部分见 `third_party/vitalpy/`。
"""
# 从上游参数模块裁取：只保留参数定义与校验/解析。
# 上游的 preset YAML 读写（load_preset / preset_for_source / list_presets …）不在
# 40 维闭包内，未搬运 —— 默认参数全部硬编码在 PARAM_SPEC 里，解析不读任何文件。
from dataclasses import dataclass
from typing import Any

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


__all__ = ['PARAM_SPEC', 'DEFAULT_PARAMS', 'validate_params', 'resolve_params']
