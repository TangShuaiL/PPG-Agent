"""Signal reference schema — uniquely identifies one PPG segment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SignalRef(BaseModel):
    """Lightweight identifier for a single PPG segment.

    This is the entry point of the entire pipeline. Passed from user to
    Supervisor, then forwarded to Agent 1 (Signal Processor).
    """

    source: Literal["vital", "mimic", "aurora"] = Field(
        ...,
        description="Dataset source: 'vital' for VitalDB, 'mimic' for "
                    "MIMIC-III, 'aurora' for AuroraBP (wearable).",
    )
    subject_id: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="Subject identifier, e.g. 'p000160'.",
    )
    seg_idx: int = Field(
        ...,
        ge=0,
        description="Zero-based segment index within the subject.",
    )
    fs: int = Field(
        default=125,
        ge=1,
        description="Sampling rate in Hz.",
    )
    duration_sec: float = Field(
        default=10.0,
        gt=0,
        description="Segment duration in seconds.",
    )
