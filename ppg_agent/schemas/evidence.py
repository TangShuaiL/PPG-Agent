"""Evidence schemas — Agent 2 (Cardiovascular Analysis) output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .quality import QualityReport
from .signal import SignalRef


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PercentileLevel(str, Enum):
    """Feature value classification against cohort reference distribution."""

    ABOVE_P95 = "above_p95"
    ABOVE_P85 = "above_p85"
    ABOVE_P75 = "above_p75"
    WITHIN_IQR = "within_iqr"
    BELOW_P25 = "below_p25"
    BELOW_P20 = "below_p20"
    BELOW_P05 = "below_p05"
    UNKNOWN = "unknown"


class EvidenceStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"


class EvidenceDirection(str, Enum):
    """Directional labels for physiological interpretation.

    These replace the free-form strings in the old report_schema.py.
    """

    # --- PPG morphology ---
    WAVE_REFLECTION_INCREASED = "wave_reflection_increased"
    WAVE_REFLECTION_DECREASED = "wave_reflection_decreased"
    STIFFNESS_PROXY_INCREASED = "stiffness_proxy_increased"
    STIFFNESS_PROXY_DECREASED = "stiffness_proxy_decreased"

    # --- PTT ---
    SHORT_TRANSIT_TIME = "short_transit_time"
    LONG_TRANSIT_TIME = "long_transit_time"

    # --- Perfusion / sensor ---
    LOW_PERFUSION = "low_perfusion"
    SENSOR_COUPLING_ISSUE = "sensor_coupling_issue"
    SIGNAL_QUALITY_ISSUE = "signal_quality_issue"

    # --- BP consistency ---
    CONSISTENT_WITH_HIGH_SBP = "consistent_with_high_sbp"
    CONSISTENT_WITH_NORMAL_SBP = "consistent_with_normal_sbp"
    NOT_ALIGNED_WITH_REFERENCE_SBP = "not_aligned_with_reference_sbp"

    # --- Generic ---
    NOT_SPECIFIC = "not_specific"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"


# ---------------------------------------------------------------------------
# Reference context
# ---------------------------------------------------------------------------

class ReferenceContext(BaseModel):
    """Cohort reference statistics for one feature."""

    n: int | None = None
    mean: float | None = None
    std: float | None = None
    p05: float | None = None
    p20: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p85: float | None = None
    p95: float | None = None


# ---------------------------------------------------------------------------
# Evidence items
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """A single piece of supporting or contradicting evidence."""

    feature: str = Field(..., description="Feature name, e.g. 'reflection_index_mean'.")
    value: float | None = Field(default=None, description="Measured value.")
    percentile_level: PercentileLevel = Field(
        default=PercentileLevel.UNKNOWN,
        description="Where the value falls in the cohort distribution.",
    )
    direction: EvidenceDirection = Field(
        ...,
        description="Interpretation direction.",
    )
    strength: EvidenceStrength = Field(
        default=EvidenceStrength.MODERATE,
        description="Evidence strength.",
    )
    rationale: str = Field(
        default="",
        description="Physiological rationale from feature rules.",
    )
    reference: ReferenceContext | None = Field(
        default=None,
        description="Cohort statistics used for percentile classification.",
    )


# ---------------------------------------------------------------------------
# Condition / pattern statistics
# ---------------------------------------------------------------------------

class ConditionStats(BaseModel):
    """BP distribution under a predefined feature pattern."""

    pattern: str = Field(..., description="Pattern name, e.g. 'ri_high'.")
    condition_desc: str = Field(default="", description="Human-readable condition.")
    n: int = Field(default=0, ge=0)
    sbp_median: float | None = None
    sbp_mean: float | None = None
    sbp_gt_130_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    sbp_gt_140_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    dbp_median: float | None = None
    dbp_gt_90_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class StatisticalContext(BaseModel):
    """Statistical background for a cardiovascular evidence report."""

    matched_pattern: str | None = Field(
        default=None, description="The feature pattern that this segment matched."
    )
    condition_stats: ConditionStats | None = Field(
        default=None, description="BP distribution under the matched pattern."
    )
    feature_percentiles: dict[str, PercentileLevel] = Field(
        default_factory=dict,
        description="Per-feature percentile classification.",
    )
    reference_labels: dict[str, float | None] = Field(
        default_factory=dict,
        description="Reference BP labels from ABP measurement, e.g. {'sbp_label': 142.3}.",
    )


# ---------------------------------------------------------------------------
# Evidence report (Agent 2 output)
# ---------------------------------------------------------------------------

class EvidenceReport(BaseModel):
    """Cardiovascular analysis report — PRIMARY output of Agent 2.

    This is the structured result consumed by the Supervisor for final synthesis.
    """

    signal: SignalRef
    quality: QualityReport

    confidence: Confidence = Field(
        default=Confidence.LOW,
        description="Overall confidence of the cardiovascular analysis.",
    )

    supporting: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items that support the directional hypothesis.",
    )
    contradicting: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items that contradict or are non-specific.",
    )

    statistical: StatisticalContext = Field(
        default_factory=StatisticalContext,
        description="Cohort-level statistical context.",
    )

    direct_answer: str = Field(
        default="",
        description="One-sentence summary of the cardiovascular analysis.",
    )
    uncertainty: list[str] = Field(
        default_factory=list,
        description="Uncertainty caveats.",
    )
    safety_note: str = Field(
        default="This is directional evidence from PPG-derived features, not a medical diagnosis.",
    )
