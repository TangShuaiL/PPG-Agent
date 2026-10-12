"""Feature schemas — beat-level and segment-level feature containers.

These are the PRIMARY data bridge from Agent 1 (Signal Processor) to
Agent 2 (Cardiovascular Analysis) and Agent 3 (ML Models).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .signal import SignalRef


# ---------------------------------------------------------------------------
# Beat-level
# ---------------------------------------------------------------------------

class BeatRecord(BaseModel):
    """Morphology and quality features for a single heartbeat."""

    beat_idx: int = Field(..., ge=0)

    # -- Validation ---------------------------------------------------------
    is_validated: bool = Field(
        default=False,
        description="True if the beat passed VitalPy morphological validation.",
    )
    start_sample: int | None = Field(
        default=None, ge=0, description="Start sample index in the raw signal."
    )
    end_sample: int | None = Field(
        default=None, ge=0, description="End sample index in the raw signal."
    )
    period_seconds: float | None = Field(
        default=None, gt=0, description="Beat-to-beat interval in seconds."
    )

    # -- Morphology ---------------------------------------------------------
    reflection_index: float | None = Field(default=None, description="Reflection Index.")
    lasi: float | None = Field(default=None, description="Large Artery Stiffness Index.")
    pulse_width_50: float | None = Field(default=None, description="Pulse width at 50 % amplitude.")
    perfusion_index: float | None = Field(default=None, description="Perfusion Index.")
    pulse_width_90: float | None = Field(default=None, description="Pulse width at 90 % amplitude.")

    # -- Statistical --------------------------------------------------------
    skewness: float | None = Field(default=None)
    kurtosis: float | None = Field(default=None)

    # -- Spectral -----------------------------------------------------------
    spectral_entropy: float | None = Field(default=None)
    wavelet_entropy: float | None = Field(default=None)

    # -- Quality proxy ------------------------------------------------------
    signal_quality_fsqi: float | None = Field(
        default=None, description="Feature-based Signal Quality Index."
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class AggStats(BaseModel):
    """Per-feature aggregation statistics across beats."""

    mean: float | None = None
    std: float | None = None
    median: float | None = None
    min: float | None = None
    max: float | None = None
    count: int | None = None


# ---------------------------------------------------------------------------
# Segment-level (the key bridge object)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "reflection_index",
    "lasi",
    "pulse_width_50",
    "pulse_width_90",
    "perfusion_index",
    "skewness",
    "kurtosis",
    "spectral_entropy",
    "wavelet_entropy",
    "signal_quality_fsqi",
    "period_seconds",
]


class SegmentFeatures(BaseModel):
    """Aggregated features for one 10-second segment.

    This is the PRIMARY output of Agent 1 and the PRIMARY input to Agent 2 / Agent 3.
    """

    signal: SignalRef
    n_beats_total: int = Field(..., ge=0, description="Total number of beats segmented.")
    n_beats_valid: int = Field(..., ge=0, description="Number of beats that passed validation.")

    beats: list[BeatRecord] = Field(
        default_factory=list, description="Per-beat feature records."
    )

    aggregation: dict[str, AggStats] = Field(
        default_factory=dict,
        description="Per-feature aggregation (mean/std/median/min/max) across all validated beats.",
    )

    # Convenience accessors for the most-used segment-level means
    @property
    def ri(self) -> float | None:
        return self._agg_value("reflection_index", "mean")

    @property
    def lasi(self) -> float | None:
        return self._agg_value("lasi", "mean")

    @property
    def perfusion(self) -> float | None:
        return self._agg_value("perfusion_index", "mean")

    @property
    def pulse_width_50(self) -> float | None:
        return self._agg_value("pulse_width_50", "mean")

    @property
    def skewness(self) -> float | None:
        return self._agg_value("skewness", "mean")

    @property
    def spectral_entropy(self) -> float | None:
        return self._agg_value("spectral_entropy", "mean")

    @property
    def fsqi(self) -> float | None:
        return self._agg_value("signal_quality_fsqi", "mean")

    def _agg_value(self, feature: str, stat: str) -> float | None:
        agg = self.aggregation.get(feature)
        return getattr(agg, stat, None) if agg else None
