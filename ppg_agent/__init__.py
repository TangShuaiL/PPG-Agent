"""PPG Agent — multi-agent framework for PPG cardiovascular phenotyping."""

from __future__ import annotations

from . import schemas
from .context import (
    PPGContext, fresh_async_task_context, get_context, reset_context, set_context,
)
from .llm_config import get_llm, test_llm_connection
from .llm_service import AgentService, get_llm as get_service_llm
from .knowledge import FeatureRule, RuleEngine
from .data_loader import get_sample_signal, load_pulse_db_signal
from .engine.signal_engine import (
    DEFAULT_PEAK_THRESHOLD, FEATURE_GROUPS,
    extract_beat_features_compact, extract_beat_features_rich,
    extract_segment_features, process_ppg_segment, segment_beats,
)

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "create_signal_processor_agent": ("ppg_agent.agents.signal_processor_agent", "create_signal_processor_agent"),
    "create_cardiovascular_agent": ("ppg_agent.agents.cardiovascular_agent", "create_cardiovascular_agent"),
    "create_ml_prediction_agent": ("ppg_agent.agents.ml_prediction_agent", "create_ml_prediction_agent"),
    "create_adaptation_session": ("ppg_agent.agents.adaptation_session", "create_adaptation_session"),
    "AgentService": ("ppg_agent.llm_service", "AgentService"),
}

def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name][0])
        return getattr(mod, _LAZY_IMPORTS[name][1])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "schemas", "RuleEngine", "FeatureRule",
    "PPGContext", "get_context", "reset_context", "set_context",
    "fresh_async_task_context",
    "get_llm", "test_llm_connection", "AgentService", "get_service_llm",
    "create_signal_processor_agent", "create_cardiovascular_agent",
    "create_ml_prediction_agent",
    "create_adaptation_session",
    "load_pulse_db_signal", "get_sample_signal",
    "DEFAULT_PEAK_THRESHOLD", "FEATURE_GROUPS",
    "segment_beats", "extract_beat_features_rich", "extract_beat_features_compact",
    "process_ppg_segment", "extract_segment_features",
]
