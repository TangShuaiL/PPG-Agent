"""PPGContext — shared runtime state bridging Agent 1 → Agent 2 → Agent 3.

Design principles:
    - Raw signal data (numpy arrays) stays here, NEVER reaches the LLM.
    - ctx.ppg_raw is READ-ONLY — it is the untouched original signal.
    - ctx.ppg_processed holds the filtered+corrected signal for downstream use.
    - Structured outputs use Pydantic schemas from ppg_agent.schemas.
    - Tools access this context via get_context() / reset_context().

Data flow (Agent 1):
    load_raw_signal    → ctx.ppg_raw         (original, never mutated)
    preprocess_signal  → ctx.ppg_processed    (bandpass + baseline corrected)
    segment_beats      → ctx.beats            (from ctx.ppg_processed)
    extract_features   → ctx.segment_features (from ctx.beats)
    assess_quality     → ctx.quality_report   (from ctx.beats + ctx.ppg_processed)
"""

from __future__ import annotations

import asyncio
import contextvars
import sys

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ppg_agent.schemas.evidence import EvidenceReport
from ppg_agent.schemas.features import SegmentFeatures
from ppg_agent.schemas.quality import QualityReport
from ppg_agent.schemas.signal import SignalRef


@dataclass
class PPGContext:
    """Runtime context holding raw/processed signals and structured results.

    This object is intentionally opaque to the LLM — tools read/write it
    directly in Python, and only structured (schema-validated) results
    are returned to the Agent layer.
    """

    # ===== PPG signal (ready for downstream use) =====
    ppg_signal: np.ndarray | None = None     # PPG_F from PulseDB, or raw + bandpass for others
    has_filtered: bool = False               # True if using official PPG_F

    # ===== Reference signals (optional) =====
    abp_signal: np.ndarray | None = None
    ecg: np.ndarray | None = None
    rpeaks_raw: np.ndarray | None = None   # ECG R-peak sample indices (125 Hz)
    age: float | None = None
    sbp_label: float | None = None
    dbp_label: float | None = None

    # ===== Metadata =====
    fs: int = 125
    signal_ref: SignalRef | None = None

    # ===== Agent 1 outputs (bridge to Agent 2 / Agent 3) =====
    all_beats: list[np.ndarray] = field(default_factory=list)   # all templates (incl. rejected)
    beats: list[np.ndarray] = field(default_factory=list)       # onset-corrected
    beat_starts: list[int] = field(default_factory=list)        # onset-based sample indices
    beat_valid_mask: list[bool] = field(default_factory=list)   # True=validated
    beat_keypoints: list[dict] = field(default_factory=list)    # 9 fiducial points per beat
    beat_records: list[dict] = field(default_factory=list)      # full beat records with identifiers
    num_beats: int = 0
    n_valid_beats: int = 0
    ppg_quality: float = 0.0
    ppg_quality_tier: str = ""

    segment_features: SegmentFeatures | None = None
    segment_features_raw: dict | None = None   # raw dict from extract_segment_features
    quality_report: QualityReport | None = None

    # ===== Agent 2 outputs (bridge to Supervisor) =====
    evidence_report: EvidenceReport | None = None
    # Agent 2 handoff: subject identity installed by the assess node so the
    # longitudinal follow-up tools need no dataset/subject arguments.
    agent2_identity: dict | None = None

    # ===== Dataset adaptation (Agent 1 adapt mode) =====
    # AdaptationSession handle, created lazily by the adapt_pipeline tool;
    # typed Any to avoid a circular import with ppg_agent.agents.adaptation_session.
    adapt_session: Any = None
    # Routing-mode memory for multi-turn conversations: 'adapt' | 'process' | None.
    # Keeps letter choices ("选B", "finalize") inside the ongoing mode instead
    # of re-classifying them against the entry keywords.
    session_mode: str | None = None

    # ===== Legacy (write-only diagnostic log) =====
    analysis_log: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Pipeline stage helpers
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True after load_raw_signal."""
        return self.ppg_signal is not None

    @property
    def is_segmented(self) -> bool:
        """True after segment_beats."""
        return len(self.beats) > 0

    @property
    def is_features_extracted(self) -> bool:
        """True after extract_all_features."""
        return self.segment_features is not None

    @property
    def is_quality_assessed(self) -> bool:
        """True after assess_quality."""
        return self.quality_report is not None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_log(self, entry: str) -> None:
        self.analysis_log.append(entry)

    def beat_summary(self) -> dict[str, Any]:
        """Agent 1 → Agent 2 handoff: structured beat-level summary of the
        CURRENT segment, computed from the live extraction (beat_records).

        Agent 2's engine reads the same segment from the parquet store; this
        summary lets it cross-check counts and gives beat-level statistics
        without re-reading the store.  Extraction logic is untouched.
        """
        recs = self.beat_records or []
        sbp = [r.get("abp_sbp") for r in recs
               if isinstance(r.get("abp_sbp"), (int, float))
               and np.isfinite(r["abp_sbp"])]
        dbp = [r.get("abp_dbp") for r in recs
               if isinstance(r.get("abp_dbp"), (int, float))
               and np.isfinite(r["abp_dbp"])]
        n = len(recs) or self.num_beats
        out: dict[str, Any] = {
            "n_beats": n,
            "n_valid": self.n_valid_beats,
            "valid_ratio": (round(self.n_valid_beats / n, 3) if n else None),
            "quality": self.ppg_quality,
            "quality_tier": self.ppg_quality_tier,
            "has_beat_bp": bool(sbp),
        }
        if sbp:
            out["beat_sbp_median"] = round(float(np.median(sbp)), 1)
            out["beat_dbp_median"] = round(float(np.median(dbp)), 1) if dbp else None
        return out

    def stage_summary(self) -> dict[str, bool | int | str | None]:
        """Return which pipeline stages are complete (for skip logic)."""
        return {
            "loaded": self.is_loaded,
            "has_filtered": self.has_filtered,
            "segmented": self.is_segmented,
            "features_extracted": self.is_features_extracted,
            "quality_assessed": self.is_quality_assessed,
            "num_beats": self.num_beats,
            "quality_tier": self.quality_report.quality_tier if self.quality_report else None,
        }

    def to_summary(self) -> dict[str, Any]:
        return {
            "signal_ref": self.signal_ref.model_dump() if self.signal_ref else None,
            "fs": self.fs,
            "has_filtered": self.has_filtered,
            "stage": {
                "loaded": self.is_loaded,
                "segmented": self.is_segmented,
                "features_extracted": self.is_features_extracted,
                "quality_assessed": self.is_quality_assessed,
            },
            "n_samples": len(self.ppg_signal) if self.ppg_signal is not None else 0,
            "num_beats": self.num_beats,
            "quality_tier": self.quality_report.quality_tier if self.quality_report else None,
            "has_evidence": self.evidence_report is not None,
            "ml_predictions": len(getattr(self, "ml_predictions", [])),
        }


# ------------------------------------------------------------------
# Request-scoped context (contextvars)
#
# Each asyncio task / thread gets its own PPGContext, so concurrent
# web requests never share signal state.  get_context()/reset_context()
# keep their original semantics for single-process scripts: lazy create
# on first access, replace with a fresh context on reset.
#
# NOTE: when dispatching work to a ThreadPoolExecutor, propagate the
# current context explicitly, e.g. contextvars.copy_context().run(fn).
# ------------------------------------------------------------------

_ctx_var: contextvars.ContextVar[PPGContext | None] = contextvars.ContextVar(
    "ppg_context", default=None
)


def get_context() -> PPGContext:
    ctx = _ctx_var.get()
    if ctx is None:
        ctx = PPGContext()
        _ctx_var.set(ctx)
    return ctx


def reset_context() -> PPGContext:
    """Replace the current scope's context with a fresh one."""
    ctx = PPGContext()
    _ctx_var.set(ctx)
    return ctx


def set_context(ctx: PPGContext) -> contextvars.Token:
    """Install an explicit context (used by the web layer per request)."""
    return _ctx_var.set(ctx)


def fresh_async_task_context() -> contextvars.Context:
    """Return a context for ``asyncio.create_task(coro, context=...)``.

    Combines the two required isolation steps: copies the current contextvar
    bindings AND installs a fresh PPGContext inside the copy.  Usage:

        task = asyncio.create_task(
            handle_request(...), context=fresh_async_task_context()
        )

    NOTE: copy_context() alone is NOT sufficient — it copies var bindings, so
    an already-created PPGContext object would still be shared across tasks.
    """
    copied = contextvars.copy_context()
    copied.run(reset_context)
    return copied


def spawn_isolated_task(coro):
    """Web 层任务隔离的唯一入口：create_task + fresh_async_task_context。

    等价于 ``asyncio.create_task(coro, context=fresh_async_task_context())``，
    但同时兼容 3.10（torch_env 运行时，create_task 无 context= 形参）与
    3.11+：3.10 路径在 fresh 上下文内调用 create_task，Task 按创建时的
    当前上下文（PEP 567）继承同一份隔离语义。用法：

        task = spawn_isolated_task(handle_request(...))
    """
    ctx = fresh_async_task_context()
    if sys.version_info >= (3, 11):
        return asyncio.create_task(coro, context=ctx)
    return ctx.run(asyncio.create_task, coro)
