"""Quality assessment schemas — Agent 1 output."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from .signal import SignalRef


# ---------------------------------------------------------------------------
# Quality flags (enum-constrained)
# ---------------------------------------------------------------------------

class QualityFlag(str, Enum):
    """Enumeration of possible quality issues."""

    TIER_NOT_GOOD = "tier_not_good"
    SCORE_LOW = "score_low"
    COVERAGE_LOW = "coverage_low"
    VALID_BEAT_RATIO_LOW = "valid_beat_ratio_low"
    PPG_OVERSEGMENT = "ppg_oversegment"
    PPG_UNDERSEGMENT = "ppg_undersegment"
    FEW_BEATS = "few_beats"
    NO_VALID_BEATS = "no_valid_beats"
    EXTRACTION_FAILED = "extraction_failed"


# Human-readable descriptions for each flag
QUALITY_FLAG_DESCRIPTIONS: dict[QualityFlag, str] = {
    QualityFlag.TIER_NOT_GOOD: "Quality tier is not 'good'.",
    QualityFlag.SCORE_LOW: "Quality score below 0.75.",
    QualityFlag.COVERAGE_LOW: "Signal coverage below 75 %.",
    QualityFlag.VALID_BEAT_RATIO_LOW: "Valid beat ratio below 80 %.",
    QualityFlag.PPG_OVERSEGMENT: "PPG beat count is significantly higher than ECG beat count (possible over-segmentation).",
    QualityFlag.PPG_UNDERSEGMENT: "PPG beat count is significantly lower than ECG beat count (possible under-segmentation).",
    QualityFlag.FEW_BEATS: "Both PPG and ECG detected fewer than 3 beats.",
    QualityFlag.NO_VALID_BEATS: "No beats passed morphological validation.",
    QualityFlag.EXTRACTION_FAILED: "Feature extraction raised an exception.",
}


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

class QualityReport(BaseModel):
    """Signal quality evaluation produced by Agent 1.

    This is consumed by Agent 2 to adjust confidence levels and by the
    Supervisor to decide whether the segment is trustworthy.
    """

    signal: SignalRef

    quality_tier: Literal["good", "fair", "poor", "failed"] = Field(
        ...,
        description="Overall quality category.",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Composite quality score (0 = worst, 1 = best).",
    )
    coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of the 10-second window covered by beat templates (all beats).",
    )
    valid_beat_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of beats that passed morphological validation.",
    )

    flags: list[QualityFlag] = Field(
        default_factory=list,
        description="Triggered quality flags.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Human-readable issue descriptions.",
    )

    @property
    def is_reliable(self) -> bool:
        """True if no quality flags were triggered."""
        return len(self.flags) == 0
