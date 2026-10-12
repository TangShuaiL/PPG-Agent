"""Physiological interpretability knowledge (feature_interpretation tool).

Loads ``knowledge/feature_interpretation.json`` (built by
``scripts/build_feature_interpretation_kb.py`` from the train-split,
FDR-controlled association tables) and answers "what does this feature
mean" questions with three evidence tiers:

  * mechanism        — physiological rationale + high/low direction
                       (feature_rules.json)
  * age_patterns     — sex-adjusted partial rho(feature, age) per dataset
  * population_patterns — labeled-group descriptions from rank-ANCOVA
                       (age+sex adjusted, global BH FDR < 0.05)
  * bp_patterns      — weak population rho of per-patient feature medians
                       vs cuff SBP/DBP (age+sex rank partial, per-source
                       BH-FDR; aurora a/o never pooled)

DISCIPLINE: population patterns are FORWARD descriptions of labeled
groups ("the hypertension-labeled group shows higher values of this
feature") — the loader repeats the caveat in every payload so the LLM
never reverses them into a condition or symptom claim for the
individual.  The tool needs NO blood-pressure labels (pure PPG), which
is the deployment shape on real-world wearable data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ppg_agent.config import KNOWLEDGE_ROOT

_KB_FILE = "feature_interpretation.json"
_MAX_FEATURES = 8


class FeatureInterpretation:
    """Read-only view over the interpretability knowledge base."""

    def __init__(self, knowledge_root: str | Path | None = None):
        root = (Path(knowledge_root).expanduser() if knowledge_root
                else KNOWLEDGE_ROOT)
        path = root / _KB_FILE
        self._kb: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists()
            else {})
        self._feats: dict[str, dict] = self._kb.get("features", {})

    @property
    def available(self) -> bool:
        return bool(self._feats)

    def entry(self, feature: str) -> dict | None:
        """Raw KB entry for ``feature`` (bare name; _mean tolerated)."""
        return self._feats.get(str(feature).removesuffix("_mean"))

    def interpret(self, features: list[str] | None = None,
                  representatives: list[str] | None = None,
                  fc: Any = None) -> dict:
        """Interpretation payload for ``features``.

        ``features`` omitted → fall back to joint-cluster
        ``representatives`` (fold-aware overview), else the first KB
        features.  When ``fc`` (FeatureClusters) is available, family
        members carry their representative so the narrative stays
        fold-aware.
        """
        if not self._feats:
            return {"available": False,
                    "reason": "feature_interpretation KB missing"}
        if features:
            names = [str(f).removesuffix("_mean") for f in features]
        elif representatives:
            names = list(representatives)
        else:
            names = list(self._feats)
        names = list(dict.fromkeys(names))[:_MAX_FEATURES]
        out_feats: dict[str, dict] = {}
        missing: list[str] = []
        for n in names:
            e = self._feats.get(n)
            if e is None:
                missing.append(n)
                continue
            entry = {
                "mechanism": e.get("mechanism", {}),
                "age_patterns": e.get("age_patterns", []),
                "population_patterns": e.get("population_patterns", []),
                "bp_patterns": e.get("bp_patterns", []),
            }
            if fc is not None and getattr(fc, "available", False):
                cl = fc.cluster_of(n)
                if cl:
                    entry["cluster_representative"] = cl["representative"]
                    if cl["representative"] != n:
                        entry["fold_note"] = (
                            f"family member — family-level reading, "
                            f"representative {cl['representative']}")
            out_feats[n] = entry
        return {
            "available": True,
            "discipline": self._kb.get("discipline", ""),
            "features": out_feats,
            "missing": missing,
            "caveat": "population patterns are FORWARD descriptions of "
                      "labeled groups (age+sex adjusted, FDR<0.05, small "
                      "effect sizes |d|~0.1-0.5); never reverse them into "
                      "a condition or symptom claim for this subject. "
                      "bp_patterns are weak population correlations "
                      "(|rho| mostly < 0.3) — never use them to estimate "
                      "this subject's blood pressure",
        }
