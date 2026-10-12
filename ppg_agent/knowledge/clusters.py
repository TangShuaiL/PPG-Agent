"""Joint feature-redundancy knowledge (step-28/29 analysis).

Loads ``knowledge/feature_clusters.json`` and provides deterministic
narrative-folding annotations for Agent 2's two consumers:

  * compare_segment (longitudinal): joint clusters only — a flagged
    change is annotated with its cluster and whether same-cluster
    co-members were flagged with the direction-consistent sign
    (``cluster_consistent`` / ``isolated`` / ``direction_divergent``);
  * population_position (population): joint clusters PLUS between_only
    pairs (safe only here — within-person consistency < 0.8).

Folding is a NARRATIVE layer on top of the engine: all features are
still computed and reported in the raw payloads; the annotations only
guide the Supervisor's wording.  The decision engine is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ppg_agent.config import KNOWLEDGE_ROOT

_FEATURES_FILE = "feature_clusters.json"


class FeatureClusters:
    """Read-only view over the joint redundancy knowledge base."""

    def __init__(self, knowledge_root: str | Path | None = None):
        root = (Path(knowledge_root).expanduser() if knowledge_root
                else KNOWLEDGE_ROOT)
        path = root / _FEATURES_FILE
        self._kb: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists()
            else {})
        self._clusters: list[dict] = self._kb.get("joint_clusters", [])
        self._by_feature: dict[str, dict] = {}
        for cl in self._clusters:
            for f in cl["members"]:
                self._by_feature[f] = cl
        # between-only pair index (population layer only)
        self._between_pairs: dict[frozenset, dict] = {
            frozenset((p["a"], p["b"])): p
            for p in self._kb.get("between_only_pairs", [])}

    @property
    def available(self) -> bool:
        return bool(self._kb)

    def cluster_of(self, feature: str) -> dict | None:
        """Joint cluster containing ``feature`` (None → singleton)."""
        return self._by_feature.get(feature)

    def representative(self, feature: str) -> str:
        cl = self._by_feature.get(feature)
        return cl["representative"] if cl else feature

    def is_representative(self, feature: str) -> bool:
        cl = self._by_feature.get(feature)
        return bool(cl) and cl["representative"] == feature

    def direction_vs_rep(self, feature: str) -> str:
        """'same'/'inverse' relative to the cluster representative."""
        cl = self._by_feature.get(feature)
        if not cl or cl["representative"] == feature:
            return "same"
        return (cl["members_info"].get(feature) or {}).get(
            "direction_vs_rep", "same")

    def representatives(self) -> list[str]:
        """Cluster representatives in KB order (fold-aware overview)."""
        return [cl["representative"] for cl in self._clusters]


def annotate_changes(report: dict, fc: FeatureClusters | None = None) -> dict:
    """Attach cluster folding annotations to a compare_segment report.

    Adds to each real_change feature entry:
      cluster_representative / is_representative / cluster_consistency
      where consistency is:
        cluster_consistent  — >=1 same-cluster co-member also real_change
                              with the direction-consistent sign
        direction_divergent — a co-member changed with the opposite
                              (direction-adjusted) sign — genuinely
                              independent movement, keep both visible
        isolated            — no co-member flagged
    Singletons get cluster_consistency="singleton".  Also adds a top-level
    "cluster_fold" summary: per joint cluster, representative + the
    (direction-adjusted) count of flagged members.  The engine payload is
    not altered otherwise.
    """
    fc = fc or FeatureClusters()
    if not fc.available:
        return report
    feats = report.get("features") or {}
    flagged = {name.removesuffix("_mean"): name
               for name, f in feats.items()
               if f.get("verdict_long") == "real_change"}
    fold: dict[int, dict] = {}
    for bare, name in flagged.items():
        cl = fc.cluster_of(bare)
        entry = feats[name]
        if cl is None:
            entry["cluster_consistency"] = "singleton"
            continue
        rep = cl["representative"]
        entry["cluster_representative"] = rep
        entry["is_representative"] = rep == bare
        # direction-adjusted sign: +1 = "moves with the representative"
        own = (1 if (feats[name].get("delta_vs_long") or 0) > 0 else -1)
        own *= (1 if fc.direction_vs_rep(bare) == "same" else -1)
        agree, diverge = 0, 0
        for co in cl["members"]:
            if co == bare or co not in flagged:
                continue
            co_entry = feats[flagged[co]]
            co_sign = (1 if (co_entry.get("delta_vs_long") or 0) > 0 else -1)
            co_sign *= (1 if fc.direction_vs_rep(co) == "same" else -1)
            agree, diverge = (agree + 1, diverge) if co_sign == own \
                else (agree, diverge + 1)
        entry["cluster_consistency"] = (
            "direction_divergent" if diverge else
            "cluster_consistent" if agree else "isolated")
        f = fold.setdefault(cl["id"], {
            "representative": rep, "flagged": [], "consistent": True})
        f["flagged"].append(bare)
        if entry["cluster_consistency"] == "direction_divergent":
            f["consistent"] = False
    if fold:
        report = {**report, "cluster_fold": fold}
    return report


def annotate_positions(out: dict, fc: FeatureClusters | None = None) -> dict:
    """Attach cluster folding annotations to a population_position result.

    Each entry in "positions"/"recent_positions" gains
    cluster_representative / is_representative; a top-level
    "position_fold" lists, per joint cluster with >=2 positioned members,
    the representative and the members whose population band matches the
    representative's (safe to state as one family-level position).
    Between-only pairs are exposed under "between_only_note" as candidate
    same-family phrasings that ONLY apply at population level.
    """
    fc = fc or FeatureClusters()
    if not fc.available:
        return out
    pos = out.get("positions") or {}
    for f, q in pos.items():
        rep = fc.representative(f)
        q["cluster_representative"] = rep
        q["is_representative"] = fc.is_representative(f)
        for rf, rq in (out.get("recent_positions") or {}).items():
            if rf == f:
                rq["cluster_representative"] = rep
                rq["is_representative"] = fc.is_representative(f)
    fold = []
    seen: set[int] = set()
    for f, q in pos.items():
        cl = fc.cluster_of(f)
        if cl is None or cl["id"] in seen:
            continue
        seen.add(cl["id"])
        members_in = [m for m in cl["members"] if m in pos]
        if len(members_in) < 2:
            continue
        rep = cl["representative"]
        rep_band = pos[rep]["percentile"] if rep in pos else None
        matched = [m for m in members_in
                   if rep_band and pos[m]["percentile"] == rep_band]
        fold.append({"cluster": cl["id"], "representative": rep,
                     "members_positioned": members_in,
                     "band_match": matched})
    if fold:
        out = {**out, "position_fold": fold}
    bo = [p for p in fc._between_pairs.values()
          if p["a"] in pos and p["b"] in pos]
    if bo:
        out = {**out, "between_only_note": {
            "scope": "population narrative only (within-person "
                     "consistency < 0.8)",
            "pairs": bo}}
    return out
