"""Physiological interpretation rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ppg_agent.config import KNOWLEDGE_ROOT


@dataclass
class WaveformKeypoint:
    """Interpretation vocabulary for one PPG fiducial keypoint (L2/L3 chain)."""

    name: str
    label: str = ""
    physiology: str = ""
    related_features: list[str] = field(default_factory=list)
    up: dict[str, str] = field(default_factory=dict)  # {sbp|dbp|age: phrase}


@dataclass
class FeatureRule:
    """Directional interpretation rule for one feature."""

    feature: str
    label: str = ""
    short_name: str = ""
    high_direction: str = ""
    low_direction: str = ""
    supports_elevated_sbp_when: str = "none"  # "high" | "low" | "none"
    rationale: str = ""


class RuleEngine:
    """Load and query physiological feature rules."""

    def __init__(self, knowledge_root: str | Path | None = None):
        self.root = Path(knowledge_root).expanduser() if knowledge_root else KNOWLEDGE_ROOT
        self._rules = self._load("feature_rules.json")

    def get(self, feature: str) -> FeatureRule:
        data = self._rules.get("features", {}).get(feature, {})
        return FeatureRule(
            feature=feature,
            label=data.get("label", feature),
            short_name=data.get("short_name", feature),
            high_direction=data.get("high_direction", ""),
            low_direction=data.get("low_direction", ""),
            supports_elevated_sbp_when=data.get("supports_elevated_sbp_when", "none"),
            rationale=data.get("rationale", ""),
        )

    def waveform_keypoint(self, name: str) -> WaveformKeypoint:
        """Return the L2/L3 keypoint interpretation entry for one fiducial
        point (o/a/md/b/s/dn/ip/d/v). Missing entries degrade to a labeled
        stub so the narrative never crashes on an unexpected keypoint."""
        data = self._rules.get("waveform_keypoints", {}).get(name, {})
        return WaveformKeypoint(
            name=name,
            label=data.get("label", name),
            physiology=data.get("physiology", ""),
            related_features=list(data.get("related_features", [])),
            up=dict(data.get("up", {})),
        )

    @property
    def keypoint_note(self) -> str:
        return self._rules.get("waveform_keypoints", {}).get("note", "")

    @property
    def safety_note(self) -> str:
        return self._rules.get("diagnostic_safety_note", "")

    @property
    def alternative_explanations(self) -> list[str]:
        return self._rules.get("default_alternative_explanations", [])

    def _load(self, filename: str) -> dict[str, Any]:
        path = self.root / filename
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
