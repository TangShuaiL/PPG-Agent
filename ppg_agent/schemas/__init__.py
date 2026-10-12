"""PPG Agent schemas — Pydantic data contracts."""

from .evidence import (
    ConditionStats, Confidence, EvidenceDirection, EvidenceItem,
    EvidenceReport, EvidenceStrength, PercentileLevel,
    ReferenceContext, StatisticalContext,
)
from .features import AggStats, BeatRecord, SegmentFeatures
from .quality import QUALITY_FLAG_DESCRIPTIONS, QualityFlag, QualityReport
from .signal import SignalRef

__all__ = [
    "SignalRef",
    "BeatRecord", "AggStats", "SegmentFeatures",
    "QualityFlag", "QUALITY_FLAG_DESCRIPTIONS", "QualityReport",
    "PercentileLevel", "EvidenceStrength", "EvidenceDirection", "Confidence",
    "ReferenceContext", "EvidenceItem", "ConditionStats",
    "StatisticalContext", "EvidenceReport",
]
