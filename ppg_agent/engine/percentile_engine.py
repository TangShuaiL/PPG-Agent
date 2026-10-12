"""Percentile lookup engine for Agent 2 cardiovascular phenotyping.

Supports:
  - Multi-source baselines (VITAL, MIMIC)
  - Age/gender stratified queries
  - Disease-conditioned baselines (MIMIC only)
  - Cross-population comparison
  - Similar patient retrieval from SQLite DB
  - Vascular Age Index computation
"""
from __future__ import annotations

import json, sqlite3
from pathlib import Path
from typing import Any

import numpy as np, pandas as pd

FEATURE_RELIABILITY: dict[str, float] = {
    "t_sys_ms": 0.46, "systolic_width_50": 0.45, "systolic_width_75": 0.44,
    "pulse_width_90": 0.41, "pulse_width_50": 0.40, "diastolic_width_75": 0.36,
    "diastolic_width_50": 0.36, "ni_dn": 0.28, "reflection_index": 0.26,
    "t_dn": 0.26, "iqr": 0.22, "ni_ip": 0.20, "rms": 0.19, "t_dia_ms": 0.19,
    "bwr_50": 0.18, "beat_len_ms": 0.16, "pulse_amp": 0.09, "lasi": 0.07,
    "slope_systolic_n": 0.06, "slope_md_n": 0.05, "skewness": 0.05,
    "crest_factor": 0.04, "intensity_ratio_s_d": 0.04, "freq_dominant_3": 0.04,
    "freq_dominant_2": 0.04, "slope_md": 0.03, "slope_systolic": 0.03,
    "spectral_entropy": 0.03, "perfusion_index": 0.02, "normalized_pulse_vol": 0.02,
    "nt_dn": 0.02, "freq_dominant_1": 0.02, "kurtosis": 0.02, "wavelet_entropy": 0.01,
    "mean_intensity": 0.01, "ppg_characteristic": 0.01, "spectral_centroid": 0.01, "fsqi": 0.00,
}

VAI_FEATURES = [
    "t_sys_ms", "systolic_width_50", "systolic_width_75", "pulse_width_50",
    "pulse_width_90", "diastolic_width_50", "diastolic_width_75",
    "reflection_index", "ni_dn", "ni_ip", "t_dn", "t_dia_ms", "iqr", "rms",
]

AGE_GROUPS = ["0-20","20-30","30-40","40-50","50-60","60-70","70-80","80+"]

# VAI feature-selection threshold for aurora: only features whose |ρ(age)| on
# the scenario's train split exceeds this enter the VAI (step-06 methodology).
VAI_SELECTION_RHO = 0.15

# Disease-condition baseline feature list (only features with ρ>0.1)
CONDITION_FEATURES = ["reflection_index","lasi","t_sys_ms","systolic_width_50",
                       "pulse_width_50","diastolic_width_50","pulse_amp","ni_dn"]

# ── Agent 2 knowledge-layer constants (analysis steps 16-27) ────────────────
# Short-window discipline threshold: below this many valid beats the segment
# is treated as a short window — between-patient directional language is not
# transferable to within-patient / short-window state interpretation
# (knowledge/scale_direction_warnings.json meta.rule; window validity from
# analysis step 27 saturates at N=20 beats).
SHORT_WINDOW_BEATS = 20

# Subtype assignment (analysis step 25): k=2 patient clusters per source,
# z=(x−mean)/std centers.  Fewer usable features than this → no assignment.
SUBTYPE_MIN_FEATURES = 15
SUBTYPE_TOP_CONTRIBUTORS = 5

# Disease-association evidence (analysis step 24): per out-of-range feature,
# top-k FDR-significant population associations, capped overall.
EVIDENCE_TOP_K = 2
EVIDENCE_MAX_FEATURES = 6
EVIDENCE_MIN_RELIABILITY = 0.15

# Disease unit (feature_disease_knowledge.json keys) → human display names.
# Fallback strips the source prefix and replaces "_" with " ".
DISEASE_DISPLAY: dict[str, str] = {
    "mimic_hypertension_essential": "hypertension",
    "mimic_heart_failure": "heart failure",
    "mimic_acute_kidney_failure": "acute kidney failure",
    "mimic_diabetes_type2": "type 2 diabetes",
    "mimic_sepsis": "sepsis",
    "mimic_coronary_artery_disease": "coronary artery disease",
    "mimic_myocardial_infarction": "myocardial infarction",
    "mimic_hypertensive_ckd": "hypertensive CKD",
    "mimic_atrial_fibrillation": "atrial fibrillation",
    "mimic_hyperlipidemia": "hyperlipidemia",
    "mimic_ckd": "chronic kidney disease",
    "mimic_copd": "COPD",
    "mimic_stroke": "stroke",
    "mimic_obesity": "obesity",
    "mimic_diabetes_complicated": "diabetes (complicated)",
    "vital_preop_htn": "preoperative hypertension",
    "vital_preop_dm": "preoperative diabetes",
    # TODO: confirm the VitalDB label meaning of "emop" (emergency operation?)
    # against the label source before the display name is user-facing.
    "vital_emop": "emop",
    "aurora_high_blood_pressure": "high blood pressure",
    "aurora_diabetes": "diabetes",
}


def _disease_display_name(unit: str) -> str:
    """Disease unit → display name; fallback strips the source prefix."""
    if unit in DISEASE_DISPLAY:
        return DISEASE_DISPLAY[unit]
    for prefix in ("mimic_", "vital_", "aurora_"):
        if unit.startswith(prefix):
            return unit[len(prefix):].replace("_", " ")
    return unit.replace("_", " ")


class PercentileEngine:
    def __init__(self, baseline_dir: str | None = None, db_path: str | None = None):
        if baseline_dir is None:
            baseline_dir = str(Path(__file__).parent.parent.parent
                / "analysis/05_reference_baselines/output/tables")
        if db_path is None:
            from ppg_agent.config import PHENOTYPE_DB
            db_path = str(PHENOTYPE_DB)
        self.baseline_dir = Path(baseline_dir)
        self.db_path = Path(db_path)
        self._baselines: dict[str, pd.DataFrame] = {}
        self._cache: dict[str, dict[str, dict]] = {}
        self._load_all()

    def _load_all(self):
        # Prefer database over CSV files
        if self.db_path.exists():
            try:
                conn = sqlite3.connect(str(self.db_path))
                df = pd.read_sql("SELECT * FROM baselines", conn)
                conn.close()
                if len(df) > 0:
                    for source in df["dataset"].unique():
                        sub = df[df["dataset"] == source]
                        self._baselines[source] = sub
                        gl = sub[(sub["age_group"]=="all") & (sub["gender"]=="all")]
                        self._cache[source] = gl.set_index("feature").to_dict("index")
                    return
            except Exception:
                pass  # fallback to CSV

        for path in sorted(self.baseline_dir.glob("*_percentile_baselines.csv")):
            source = path.stem.replace("_percentile_baselines", "")
            df = pd.read_csv(path)
            self._baselines[source] = df
            gl = df[(df["age_group"] == "all") & (df["gender"] == "all")]
            self._cache[source] = gl.set_index("feature").to_dict("index")

    @property
    def available_sources(self) -> list[str]:
        return list(self._baselines.keys())

    # ------------------------------------------------------------------
    # Source / knowledge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_source(source: str, scenario: str | None = None) -> str:
        """Map an aurora request to its scenario-specific baseline.

        The aurora a/o scenarios are calibrated differently (~30-40pp offset);
        their baselines are hard-separated (``aurora_a`` / ``aurora_o``) and
        must never be pooled.  ``scenario`` is 'a'/'o' (from the subject id's
        first char or user selection).  Non-aurora sources pass through.
        """
        if source not in ("aurora", "aurorabp"):
            return source
        sc = (scenario or "").strip().lower()
        if sc in ("a", "auscultatory", "aus"):
            return "aurora_a"
        if sc in ("o", "oscillometric", "osc"):
            return "aurora_o"
        return source  # unresolved → caller decides (engine will error cleanly)

    def _reliability(self, source: str) -> dict[str, float]:
        """Per-source split-half reliability weights (data-driven, step 06).

        Loaded once from ``knowledge/feature_reliability.json``; a missing
        source or feature falls back to the hardcoded FEATURE_RELIABILITY in
        ``_weight()``.
        """
        if not hasattr(self, "_reliability_map"):
            self._reliability_map = {}
            path = Path(__file__).parent.parent.parent / "knowledge" / "feature_reliability.json"
            try:
                self._reliability_map = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 — optional knowledge
                pass
        return self._reliability_map.get(source, {})

    def _weight(self, source: str, feature: str) -> float:
        """Reliability weight for (source, feature); falls back to hardcode."""
        return self._reliability(source).get(feature,
                                             FEATURE_RELIABILITY.get(feature, 0))

    def _icc(self, source: str, feature: str) -> float | None:
        """Single-segment ICC(1,1) for (source, feature) — beat-level step 02.

        Henderson type-3 variance decomposition (knowledge/feature_icc.json).
        None when the feature has no per-beat decomposition (e.g. spectral
        features — a single beat has no spectrum).
        """
        if not hasattr(self, "_icc_map"):
            self._icc_map = {}
            path = Path(__file__).parent.parent.parent / "knowledge" / "feature_icc.json"
            try:
                self._icc_map = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 — optional knowledge
                pass
        v = self._icc_map.get(source, {}).get(feature)
        return v if v is not None and not pd.isna(v) else None

    def _icc_2k(self, source: str, feature: str, k: int) -> float | None:
        """ICC(2,k) — reliability of the mean of k segments, from ICC(1,1).

        icc2_k = icc1·k / (icc1·k + (1 − icc1)): the within-person noise
        (σ²s + σ²w) is averaged over k independent segments.  k=1 → icc1.
        """
        icc1 = self._icc(source, feature)
        if icc1 is None or k < 1:
            return icc1 if icc1 is not None else None
        return round(icc1 * k / (icc1 * k + (1 - icc1)), 4)

    def _cv_reference(self, source: str, feature: str) -> dict | None:
        """Per-source p50/p90 of segment-level beat-to-beat CV (step 08).

        Used to flag "elevated beat-to-beat variability" — a segment whose
        CV exceeds the population p90 deviates from the reference stability.
        """
        if not hasattr(self, "_cv_ref"):
            self._cv_ref = {}
            path = (Path(__file__).parent.parent.parent / "knowledge"
                    / "beat_cv_reference.json")
            try:
                self._cv_ref = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 — optional knowledge
                pass
        return self._cv_ref.get(source, {}).get(feature)

    def _sign_alignment(self, source: str) -> dict[str, int]:
        """Feature→direction map for VAI on aurora (observed ρ(age) sign).

        Generated by step 05; vital/mimic have no map (their VAI path keeps
        sign=1).  Features mapped to 0 are skipped by compute_vai.
        """
        if not hasattr(self, "_sign_maps"):
            self._sign_maps: dict[str, dict[str, int]] = {}
        if not hasattr(self, "_sign_rhos"):
            self._sign_rhos: dict[str, dict[str, float]] = {}
        if source not in self._sign_maps:
            path = (Path(__file__).parent.parent.parent / "knowledge"
                    / f"feature_sign_alignment_{source}.json")
            try:
                data = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 — optional knowledge
                data = {}
            self._sign_maps[source] = data.get("features", {})
            self._sign_rhos[source] = data.get("rho", {})
        return self._sign_maps[source]

    def _vai_candidates(self, source: str) -> list[str]:
        """VAI feature set for a source.

        aurora (has a sign map): the map's features with sign≠0 and
        |ρ(age)|>VAI_SELECTION_RHO — data-driven selection (step-06
        methodology, |ρ|>0.15 recovers the age signal; the fixed 14-feature
        set mixes in noisy features on aurora).  vital/mimic: VAI_FEATURES
        (unchanged path).
        """
        sign_map = self._sign_alignment(source)
        if not sign_map:
            return VAI_FEATURES
        rho_tab = getattr(self, "_sign_rhos", {}).get(source, {})
        return sorted(fc for fc, s in sign_map.items() if s != 0
                      and abs(rho_tab.get(fc, 0)) > VAI_SELECTION_RHO)

    def _rule_for(self, feature: str) -> dict | None:
        """Physiological rule text for one feature (knowledge/rules.json)."""
        try:
            from ppg_agent.knowledge.rules import RuleEngine

            rule = RuleEngine().get(feature + "_mean")
        except Exception:  # noqa: BLE001 — rules are optional
            return None
        bits: dict = {}
        if rule.high_direction:
            bits["high"] = rule.high_direction
        if rule.low_direction:
            bits["low"] = rule.low_direction
        if rule.rationale:
            bits["rationale"] = rule.rationale
        if rule.label:
            bits["label"] = rule.label
        return bits or None

    def _get_baseline(self, source: str, feature: str,
                      age_group: str = "all", gender: str = "all") -> dict | None:
        """Percentile row for (source, feature) with a 4-step fallback ladder.

        (ag,g) → (ag,'all') → ('all',g) → ('all','all').  aurora baselines skip
        cells with n<15 (tail age bins), so a missing cell falls back to a
        broader reference instead of returning None.  vital/mimic cells are
        dense and hit the exact match first, so their behavior is unchanged.
        """
        if source not in self._baselines: return None
        df = self._baselines[source]
        for ag, g in dict.fromkeys(((age_group, gender), (age_group, "all"),
                                    ("all", gender), ("all", "all"))):
            rows = df[(df["feature"]==feature) & (df["age_group"]==ag) & (df["gender"]==g)]
            if len(rows):
                return rows.iloc[0].to_dict()
        return None

    def _age_group(self, age: float | None) -> str:
        if age is None:
            return "all"
        for ag in AGE_GROUPS:
            if ag.endswith("+"):
                lo = float(ag[:-1])
                if age >= lo:
                    return ag
                continue
            lo, hi = ag.split("-")
            if float(lo) <= age < float(hi):
                return ag
        return "all"

    # ──────────────────────────────────────────────────────────────
    # Agent 2 knowledge layer (analysis steps 16-27)
    #   All loaders are lazy + defensive: a missing file or a malformed
    #   entry degrades to empty/None — the engine never raises on optional
    #   knowledge.  Tier-1 sources are the deployed knowledge/ JSONs; the
    #   enrichment / window / MDC tables fall back to the raw analysis CSV
    #   (same pattern as _load_all reading analysis tables).
    # ──────────────────────────────────────────────────────────────

    def _analysis_tables(self) -> Path:
        return (Path(__file__).parent.parent.parent
                / "analysis/09_multidimensional_association/output/tables")

    def _disease_knowledge(self) -> dict:
        """feature_disease_knowledge.json — {source: {unit: {feature: stats}}}."""
        if not hasattr(self, "_disease_map"):
            self._disease_map = {}
            path = (Path(__file__).parent.parent.parent / "knowledge"
                    / "feature_disease_knowledge.json")
            try:
                self._disease_map = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 — optional knowledge
                pass
        return self._disease_map

    def _cluster_centers(self, source: str) -> dict | None:
        """Patient-cluster z centers for a source (analysis step 25).

        {source, k, features[], means[], stds[], centers_z[k][31]} —
        z-scores are (x − mean)/std with ddof=0 (fixed in the JSON).
        """
        if not hasattr(self, "_cluster_center_maps"):
            self._cluster_center_maps: dict[str, dict] = {}
        if source not in self._cluster_center_maps:
            path = (Path(__file__).parent.parent.parent / "knowledge"
                    / f"patient_cluster_centers_{source}.json")
            data = None
            try:
                data = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 — optional knowledge
                data = None
            self._cluster_center_maps[source] = data or {}
        return self._cluster_center_maps[source] or None

    def _enrichment(self) -> dict:
        """Per-source cluster disease enrichment (patient_cluster_disease.csv).

        Tier 1: raw analysis CSV.  Tier 2: knowledge/patient_cluster_enrichment.json
        (loaded by _enrichment_json when present, flipping priority).
        """
        if not hasattr(self, "_enrich_map"):
            self._enrich_map = {}
            data = self._enrichment_json() if self._enrichment_json_available() else None
            if data is not None:
                self._enrich_map = data
                return self._enrich_map
            path = self._analysis_tables() / "patient_cluster_disease.csv"
            try:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    self._enrich_map.setdefault(row["source"], {}).setdefault(
                        int(row["cluster"]), []).append(row.to_dict())
            except Exception:  # noqa: BLE001 — optional knowledge
                pass
        return self._enrich_map

    def _enrichment_json_available(self) -> bool:
        return True  # tier-2 deployment: knowledge JSON takes priority

    def _enrichment_json(self) -> dict | None:
        """Deployed enrichment — cluster keys int-ified (JSON keys are
        strings; the CSV path yields ints, and assign_subtype indexes by int)."""
        try:
            path = (Path(__file__).parent.parent.parent / "knowledge"
                    / "patient_cluster_enrichment.json")
            data = json.loads(path.read_text()).get("sources", {})
            return {source: {int(c): rows for c, rows in clusters.items()}
                    for source, clusters in data.items()}
        except Exception:  # noqa: BLE001 — optional knowledge
            return None

    def _window_spec(self) -> dict:
        """Window-aggregation spec (analysis step 27) — N saturation, median.

        Tier 1: raw analysis CSV.  Tier 2: knowledge/window_aggregation.json.
        """
        if not hasattr(self, "_window_map"):
            self._window_map = {}
            data = self._window_json() if self._window_json_available() else None
            if data is not None:
                self._window_map = data
                return self._window_map
            path = self._analysis_tables() / "window_aggregation.csv"
            try:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    self._window_map.setdefault(row["source"], {}).setdefault(
                        int(row["N"]), {})[row["agg"]] = {
                        k: row[k] for k in ("rho_bp_top", "bias_p90", "mad_median")}
            except Exception:  # noqa: BLE001 — optional knowledge
                pass
        return self._window_map

    def _window_json_available(self) -> bool:
        return True  # tier-2 deployment: knowledge JSON takes priority

    def _window_json(self) -> dict | None:
        """Deployed window spec — converted to the {source: {N: {agg: metrics}}}
        shape the CSV path produces, so _recommended_window is format-agnostic."""
        try:
            path = (Path(__file__).parent.parent.parent / "knowledge"
                    / "window_aggregation.json")
            data = json.loads(path.read_text()).get("sources", {})
            out: dict = {}
            for source, s in data.items():
                out[source] = {}
                for n_str, metrics in (s.get("rows") or {}).items():
                    out[source][int(n_str)] = {"median": metrics}
            return out
        except Exception:  # noqa: BLE001 — optional knowledge
            return None

    def _scale_warnings(self) -> dict:
        """scale_direction_warnings.json — {meta: {rule}, pairs: {source: {SBP/DBP: [...]}}}."""
        if not hasattr(self, "_scale_warn"):
            self._scale_warn = {}
            path = (Path(__file__).parent.parent.parent / "knowledge"
                    / "scale_direction_warnings.json")
            try:
                self._scale_warn = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 — optional knowledge
                pass
        return self._scale_warn

    def _mdc(self, source: str, feature: str) -> dict | None:
        """Minimal detectable change for (source, feature) — tier 2 only.

        MDC = 1.96 × sd_diff (Bland-Altman of two-day retest, analysis step 16).
        aurora_a/aurora_o share the retest reference (no scenario column) →
        both map to the "aurora" key.  None when the reference is missing.
        """
        if not hasattr(self, "_mdc_map"):
            self._mdc_map = {}
            path = (Path(__file__).parent.parent.parent / "knowledge"
                    / "mdc_reference.json")
            try:
                data = json.loads(path.read_text())
                self._mdc_map = data.get("features", {})
            except Exception:  # noqa: BLE001 — optional knowledge
                pass
        if self._mdc_map is None:
            return None
        key = "aurora" if source in ("aurora_a", "aurora_o") else source
        return self._mdc_map.get(feature) or None

    # ── public capability methods ──────────────────────────────────

    def disease_associations(self, stratified: dict, features: dict,
                             source: str, top_k: int = EVIDENCE_TOP_K,
                             max_features: int = EVIDENCE_MAX_FEATURES,
                             min_reliability: float = EVIDENCE_MIN_RELIABILITY) -> dict:
        """Population disease-association evidence for out-of-range features.

        For every feature whose stratified level is not "normal" and whose
        reliability ≥ min_reliability, take the FDR-significant (sig=True)
        disease associations from the knowledge layer (step 24), sorted by
        |ρ|, capped at top_k per feature and max_features overall.

        Returns {feature: [{disease, display, rho, rho_raw, direction, q_fdr,
        n_pos, n_neg}]} — empty when no knowledge or nothing out-of-range.
        """
        knowledge = self._disease_knowledge().get(source, {})
        if not knowledge:
            return {}
        out = {}
        for fname, entry in stratified.items():
            if entry.get("level") in (None, "normal"):
                continue
            if (entry.get("reliability") or 0) < min_reliability:
                continue
            if fname not in features:
                continue
            hits = []
            for unit, feat_map in knowledge.items():
                if unit in ("age_association", "sex_association"):
                    continue
                row = feat_map.get(fname)
                if row and row.get("sig"):
                    hits.append({
                        "disease": unit,
                        "display": _disease_display_name(unit),
                        "rho": row.get("rho"),
                        "rho_raw": row.get("rho_raw"),
                        "direction": row.get("direction"),
                        "q_fdr": row.get("q_fdr"),
                        "n_pos": row.get("n_pos"),
                        "n_neg": row.get("n_neg"),
                    })
            hits.sort(key=lambda h: abs(h["rho"] or 0), reverse=True)
            if hits:
                out[fname] = hits[:top_k]
                if len(out) >= max_features:
                    break
        return out

    def assign_subtype(self, features: dict, source: str) -> dict:
        """Assign the patient to the nearest reference-population subtype.

        z_i = (x_i − mean_i)/std_i (ddof=0, as fixed by analysis step 25);
        distance to each cluster center is the RMS z-distance over the
        features both present in the request and the center file — RMS is
        normalized for missing features, so partial feature sets stay
        comparable.  Enrichment (step 25 tables) is FDR-filtered, OR-sorted,
        capped at 3.  No hardcoded subtype labels — the z centers speak.
        """
        centers = self._cluster_centers(source)
        if not centers or not centers.get("centers_z"):
            return {"available": False, "reason": "no subtype reference for source"}
        feat_list = centers.get("features") or []
        stds, means = centers.get("stds") or [], centers.get("means") or []
        z = []
        used = []
        for i, fname in enumerate(feat_list):
            if fname not in features:
                continue
            v = features[fname]
            if not isinstance(v, (int, float)):
                continue
            s = stds[i] if i < len(stds) else None
            m = means[i] if i < len(means) else None
            if not s or float(s) == 0 or m is None:
                continue
            z.append((float(v) - float(m)) / float(s))
            used.append(fname)
        if len(z) < SUBTYPE_MIN_FEATURES:
            return {"available": False, "reason": "insufficient features",
                    "n_features_used": len(z)}
        c_z = np.array(centers["centers_z"])
        # indices of used features within the full feature list
        idx = [feat_list.index(f) for f in used]
        z_arr = np.array(z)
        dists = np.sqrt(np.mean((z_arr[:, None] - c_z[:, idx].T) ** 2, axis=0))
        cluster = int(np.argmin(dists))
        # top contributing features: |z − center| largest
        contrib = []
        for i, fname in enumerate(used):
            contrib.append({"feature": fname, "z": round(float(z_arr[i]), 3),
                            "center": round(float(c_z[cluster, idx[i]]), 3)})
        contrib.sort(key=lambda c: abs(c["z"] - c["center"]), reverse=True)
        result = {
            "available": True,
            "source": source,
            "cluster": cluster,
            "distance": round(float(dists[cluster]), 3),
            "n_features_used": len(z),
            "features_contributing": contrib[:SUBTYPE_TOP_CONTRIBUTORS],
            "disease_enrichment": self._subtype_enrichment(source, cluster),
        }
        return result

    def _subtype_enrichment(self, source: str, cluster: int) -> list[dict]:
        """FDR-significant enrichment rows for (source, cluster), OR-desc, ≤3."""
        rows = self._enrichment().get(source, {}).get(cluster, [])
        rows = [r for r in rows if r.get("q_fdr") is not None and r["q_fdr"] < 0.05]
        rows.sort(key=lambda r: r.get("odds_ratio") or 0, reverse=True)
        out = []
        for r in rows[:3]:
            out.append({
                "disease": r["disease"],
                "display": _disease_display_name(r["disease"]),
                "odds_ratio": round(float(r["odds_ratio"]), 3),
                "prev_cluster": round(float(r["prev_cluster"]), 3),
                "prev_rest": round(float(r["prev_rest"]), 3),
                "q_fdr": round(float(r["q_fdr"]), 4),
            })
        return out

    def window_context(self, source: str, n_beats_valid: int | None = None,
                       window_n_segments: int | None = None) -> dict:
        """Short-window context: beats axis (N<20) and/or segments axis.

        Window validity (feature↔BP rho) saturates at N=20 beats (analysis
        step 27, median aggregation).  Below it the segment is a short window:
        interpretation must stay descriptive — between-patient direction is
        not transferable (see scale_direction_warnings meta.rule).
        """
        rec = self._recommended_window(source)
        n_rec = rec.get("recommended_N") if rec else SHORT_WINDOW_BEATS
        is_short, reason = False, None
        if n_beats_valid is not None and n_beats_valid < SHORT_WINDOW_BEATS:
            is_short, reason = True, "beats"
        elif window_n_segments is not None and window_n_segments < (n_rec or SHORT_WINDOW_BEATS):
            is_short, reason = True, "segments"
        return {
            "n_beats_valid": n_beats_valid,
            "is_short_window": is_short,
            "short_window_reason": reason,
            "recommended_N": n_rec,
            "recommended_agg": rec.get("recommended_agg") if rec else "median",
            "note": ("Median pooling of ~20 segments gives a large, "
                     "held-out-validated stability gain (within-subject CV "
                     "drops ~76-89%). The step-27 rho(bp) saturation at "
                     "N=20 did NOT replicate on vital/mimic held-out test — "
                     "treat recommended_N as indicative.")
            if rec else "No window reference for this source.",
        }

    def _recommended_window(self, source: str) -> dict | None:
        """Median-row recommendation for a source from the step-27 spec."""
        spec = self._window_spec().get(source)
        if not spec:
            return None
        # validity-saturation: smallest N ≥ 20 whose rho_bp_top is within
        # 0.01 of the N=50 value; default 20.
        median_rows = {n: s.get("median") for n, s in spec.items() if "median" in s}
        if not median_rows:
            return None
        n50 = median_rows.get(50) or max(median_rows)
        ref = n50.get("rho_bp_top") or 0
        rec_n = 20
        for n in sorted(median_rows):
            if n >= 20 and abs((median_rows[n].get("rho_bp_top") or 0) - ref) < 0.01:
                rec_n = n
                break
        return {"recommended_N": rec_n, "recommended_agg": "median"}

    def scale_warning_features(self, source: str) -> list[str]:
        """Features whose within-patient sign flips vs the between-patient
        dictionary for this source (SBP ∪ DBP from step 26b).  Empty when the
        source has no flips or no knowledge file.
        """
        pairs = self._scale_warnings().get("pairs", {}).get(source)
        if not pairs:
            return []
        seen, out = set(), []
        for bp in ("SBP", "DBP"):
            for p in pairs.get(bp, []):
                f = p.get("feature")
                if f and f not in seen:
                    seen.add(f)
                    out.append(f)
        return out

    def assess_change(self, features_curr: dict, features_prev: dict,
                      source: str) -> dict:
        """Minimal-detectable-change comparison (tier 2; step 16 MDC).

        For features present in both measurements with an MDC reference:
        |Δ| > MDC → "exceeds_measurement_error".  MDC = 1.96 × sd_diff of the
        two-day retest Bland-Altman.  No MDC reference → unavailable, and the
        LLM must NOT make change claims (prompt rule).
        """
        if not features_prev:
            return {"available": False, "reason": "no previous measurement provided"}
        common = [f for f in features_curr if f in features_prev
                  and isinstance(features_curr[f], (int, float))
                  and isinstance(features_prev[f], (int, float))]
        feats = {}
        found = False
        for f in common:
            ref = self._mdc(source, f)
            if not ref:
                continue
            found = True
            delta = float(features_curr[f]) - float(features_prev[f])
            mdc = ref.get("mdc")
            if not mdc:
                continue
            feats[f] = {
                "delta": round(delta, 3),
                "mdc": round(float(mdc), 3),
                "verdict": ("exceeds_measurement_error"
                            if abs(delta) > mdc else "within_measurement_error"),
                "icc": ref.get("icc"),
                "protocol": ref.get("protocol"),
                "n": ref.get("n"),
                "bias": ref.get("bias"),
            }
        if not found:
            return {"available": False, "reason": "no MDC reference for source"}
        return {"available": True, "features": feats}

    # ══════════════════════════════════════════════════════════════
    # Tier 1: Basic query
    # ══════════════════════════════════════════════════════════════

    def query(self, feature: str, value: float, source: str = "vital",
              age_group: str = "all", gender: str = "all") -> dict[str, Any]:
        bl = self._get_baseline(source, feature, age_group, gender)
        if bl is None:
            return {"feature": feature, "error": f"not in {source} baseline"}
        p75, p85, p95 = bl.get("p75",0), bl.get("p85",0), bl.get("p95",0)
        p25, p05, p50 = bl.get("p25",0), bl.get("p05",0), bl.get("p50",0)
        if pd.isna(value):
            return {"feature": feature, "percentile": "unknown", "level": "unknown",
                    "reliability": self._weight(source, feature), "source": source}
        if value > p95:       pct, lvl = "P95+", "very_high"
        elif value > p85:     pct, lvl = "P85-P95", "high"
        elif value > p75:     pct, lvl = "P75-P85", "mildly_high"
        elif value < p05:     pct, lvl = "P5-", "very_low"
        elif value < p25:     pct, lvl = "P5-P25", "low"
        else:                 pct, lvl = "P25-P75", "normal"
        return {"feature": feature, "value": round(value,6), "percentile": pct,
                "level": lvl, "reliability": self._weight(source, feature),
                "source": source, "age_group": age_group, "gender": gender,
                "p50": p50, "p75": p75, "p95": p95, "p25": p25, "p05": p05,
                "n_baseline": bl.get("n",0)}

    def query_stratified(self, feature: str, value: float, source: str = "vital",
                         age: float | None = None, gender: str | None = None) -> dict[str, Any]:
        """Query with automatic age/gender stratification."""
        ag = self._age_group(age) if age else "all"
        g = gender if gender else "all"
        stratified = self.query(feature, value, source, ag, g)
        global_q = self.query(feature, value, source, "all", "all")
        # If stratified returned fallback (n_baseline=0), add note
        if stratified.get("n_baseline", 0) == 0:
            stratified["note"] = "fallback to global baseline (insufficient samples in your age/gender group)"
        stratified["global_percentile"] = global_q.get("percentile")
        return stratified

    def query_compare(self, feature: str, value: float,
                      age: float | None = None, gender: str | None = None) -> dict[str, Any]:
        """Cross-population comparison across all available baselines."""
        results = {}
        for src in self.available_sources:
            results[src] = self.query_stratified(feature, value, src, age, gender)
        return {"feature": feature, "value": value, "results": results,
                "interpretation": self._compare_interpretation(results)}

    def _compare_interpretation(self, results: dict) -> str:
        pcts = {src: r.get("percentile","?") for src, r in results.items()}
        values = set(pcts.values())
        if len(values) == 1:
            return f"Consistent across populations: {pcts}"
        else:
            return f"Varies across populations: {pcts}. Consider population-specific context."

    # ══════════════════════════════════════════════════════════════
    # Tier 2: Disease-condition baselines (MIMIC only)
    # ══════════════════════════════════════════════════════════════

    def query_disease_conditioned(self, feature: str, value: float,
                                   disease: str, source: str = "mimic") -> dict[str, Any]:
        """Query percentile within a disease-specific subpopulation.

        Uses pre-computed disease_percentiles table for fast lookup (< 1ms).
        Only available for MIMIC (has disease labels).
        """
        if not self.db_path.exists():
            return {"error": "Database not available."}

        global_q = self.query(feature, value, source)
        conn = sqlite3.connect(str(self.db_path))

        try:
            # Fast lookup from pre-computed percentile table
            row = conn.execute("""
                SELECT n, p25, p50, p75, mean_val, std_val
                FROM disease_percentiles
                WHERE disease=? AND feature=? AND has_disease=1
            """, (disease, feature)).fetchone()

            if not row:
                return {"feature": feature, "error": f"No data for {disease}/{feature}"}

            n, p25, p50, p75, mean_val, std_val = row
            if pd.isna(value):
                return {"feature": feature, "disease": disease, "percentile": "unknown",
                        "global_percentile": global_q.get("percentile")}

            # Classify within disease population
            if value > p75:        d_pct, d_lvl = "P75+", "high_for_condition"
            elif value < p25:      d_pct, d_lvl = "P25-", "low_for_condition"
            else:                  d_pct, d_lvl = "P25-P75", "normal_for_condition"

            # Also get disease-negative stats for comparison
            row_neg = conn.execute("""
                SELECT n, p50, mean_val FROM disease_percentiles
                WHERE disease=? AND feature=? AND has_disease=0
            """, (disease, feature)).fetchone()

            neg_n = row_neg[0] if row_neg else 0
            neg_p50 = row_neg[1] if row_neg else None

            interpretation = self._disease_interpretation(
                feature, value, d_pct, global_q.get("percentile","?"),
                disease, p50, neg_p50)

            return {
                "feature": feature, "value": round(value, 6),
                "disease": disease,
                "disease_percentile": d_pct, "disease_level": d_lvl,
                "global_percentile": global_q.get("percentile"),
                "disease_p50": p50, "disease_p25": p25, "disease_p75": p75,
                "disease_n": n,
                "non_disease_p50": neg_p50, "non_disease_n": neg_n,
                "interpretation": interpretation,
            }
        finally:
            conn.close()

    def _disease_interpretation(self, feature, value, d_pct, g_pct, disease, d_p50, nd_p50):
        """Generate interpretation comparing disease vs general population percentile."""
        v_str = f"{value:.3f}"; d50_str = f"{float(d_p50):.3f}" if d_p50 else "N/A"
        nd50_str = f"{float(nd_p50):.3f}" if nd_p50 else "N/A"
        g_str = str(g_pct)

        if d_pct == "P25-P75" and ("P75" in g_str or "P85" in g_str or "P95" in g_str):
            return (f"Your {feature}={v_str} is normal within the {disease} population "
                    f"(P50={d50_str}), but elevated in the general population. "
                    f"This pattern is consistent with {disease}-related cardiovascular changes.")
        elif "P75" in str(d_pct) and "P75" not in g_str:
            return (f"Your {feature}={v_str} is elevated even within the {disease} population "
                    f"(disease P50={d50_str}). This may indicate additional risk beyond {disease}.")
        else:
            return (f"Your {feature}={v_str}. Disease population P50={d50_str}, "
                    f"non-disease P50={nd50_str}.")

    # ══════════════════════════════════════════════════════════════
    # Tier 3: Similar patient retrieval
    # ══════════════════════════════════════════════════════════════

    def find_similar_patients(self, features: dict[str, float],
                               source: str = "mimic", top_k: int = 5) -> dict[str, Any]:
        """Find K most similar patients in the database by feature distance."""
        if not self.db_path.exists():
            return {"error": "Database not available."}

        conn = sqlite3.connect(str(self.db_path))
        try:
            # Build query: compute Euclidean distance for each patient in DB
            feat_cols = [f"{fc}_mean" for fc in VAI_FEATURES if fc in features]
            dist_terms = []
            notnull_terms = []
            for fc in VAI_FEATURES:
                if fc in features:
                    col = fc + "_mean"
                    dist_terms.append(f"(sf.{col} - {features[fc]})*(sf.{col} - {features[fc]})")
                    notnull_terms.append(f"sf.{col} IS NOT NULL")

            if not dist_terms:
                return {"error": "No matching features in database"}
            dist_expr = " + ".join(dist_terms)

            query = f"""
                SELECT s.subject_id, s.source, SQRT({dist_expr}) as distance
                FROM segment_features sf
                JOIN segments s ON sf.seg_id = s.id
                WHERE s.source = '{source}' AND {' AND '.join(notnull_terms)}
                ORDER BY distance ASC
                LIMIT {top_k}
            """
            rows = conn.execute(query).fetchall()
            return {
                "similar_patients": [
                    {"subject_id": r[0], "source": r[1], "distance": round(r[2], 4)}
                    for r in rows
                ],
                "interpretation": f"Found {len(rows)} most similar patients in {source} database."
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════
    # Tier 4: VAI computation
    # ══════════════════════════════════════════════════════════════

    def compute_vai(self, features: dict[str, float], source: str = "vital") -> dict[str, Any]:
        if source not in self._cache:
            return {"error": f"baseline '{source}' not loaded"}
        baseline = self._cache[source]
        sign_map = self._sign_alignment(source)  # aurora only; others sign=1
        z_sum, count, contributions = 0.0, 0, []
        for fc in self._vai_candidates(source):
            if fc not in features or fc not in baseline: continue
            val = features[fc]
            if pd.isna(val): continue
            sign = sign_map.get(fc, 1)
            if sign == 0: continue  # no reliable direction → skip feature
            bl = baseline[fc]; p50 = bl.get("p50",0)
            std_est = (bl.get("p75",0) - bl.get("p25",0)) / 1.35
            if std_est == 0: continue
            z = sign * (val - p50) / std_est; z_sum += z; count += 1
            contributions.append({"feature": fc, "z_score": round(float(z),3)})
        vai = z_sum / max(count, 1)
        contributions.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return {"vai": round(float(vai),3), "source": source,
                "n_features_used": count, "top_contributors": contributions[:5]}

    def assess_vascular_age(self, features: dict[str, float],
                             actual_age: float | None = None,
                             source: str = "vital") -> dict[str, Any]:
        vai_r = self.compute_vai(features, source)
        feature_details = []
        for c in vai_r.get("top_contributors", [])[:5]:
            q = self.query(c["feature"], features.get(c["feature"], float("nan")), source)
            feature_details.append({"feature": c["feature"], "value": q.get("value"),
                                     "percentile": q.get("percentile"), "level": q.get("level"),
                                     "reliability": q.get("reliability"), "z_score": c["z_score"]})
        assessment = {"vai": vai_r["vai"], "source": source,
                       "n_features": vai_r["n_features_used"],
                       "feature_details": feature_details, "actual_age": actual_age}
        if actual_age is not None:
            # Use age-stratified VAI from database if available
            ag = self._age_group(actual_age)
            age_strat_ref = None
            if self.db_path.exists():
                try:
                    conn = sqlite3.connect(str(self.db_path))
                    row = conn.execute("""SELECT vai_mean, vai_std, vai_p25, vai_p50, vai_p75, n
                        FROM vai_age_stratified WHERE source=? AND age_group=? AND gender='all'""",
                        (source, ag)).fetchone()
                    conn.close()
                    if row:
                        age_strat_ref = {"mean": row[0], "std": row[1], "p25": row[2],
                                          "p50": row[3], "p75": row[4], "n": row[5]}
                except: pass

            if age_strat_ref:
                delta = vai_r["vai"] - age_strat_ref["p50"]
                z_score = delta / max(age_strat_ref["std"], 0.01)
                vs = ("older" if delta>0.5 else ("younger" if delta<-0.5 else "matches"))
                assessment["vascular_age_vs_actual"] = vs
                assessment["age_stratified"] = age_strat_ref
                assessment["interpretation"] = (
                    f"VAI={vai_r['vai']:+.2f}. Age group {ag} reference: "
                    f"mean={age_strat_ref['mean']:+.2f}±{age_strat_ref['std']:.2f} (n={age_strat_ref['n']}). "
                    f"Vascular age {'older than' if delta>0.5 else ('younger than' if delta<-0.5 else 'consistent with')} "
                    f"age-matched peers (z={z_score:+.1f}σ)."
                )
            else:
                # Fallback to fixed reference
                expected = -1.0 if actual_age<40 else (0.0 if actual_age<60 else (0.3 if actual_age<70 else 0.7))
                delta = vai_r["vai"] - expected
                assessment["vascular_age_vs_actual"] = "older" if delta>0.5 else ("younger" if delta<-0.5 else "matches")
                assessment["interpretation"] = (
                    f"VAI={vai_r['vai']:+.2f}, expected ~{expected:+.1f} for age {actual_age:.0f}. "
                    f"Vascular age {'older than' if delta>0.5 else ('younger than' if delta<-0.5 else 'consistent with')} chronological age."
                )
        else:
            assessment["vascular_age_vs_actual"] = "unknown"
            assessment["interpretation"] = f"VAI={vai_r['vai']:+.2f}. No age reference."
        return assessment

    # ══════════════════════════════════════════════════════════════
    # Full report
    # ══════════════════════════════════════════════════════════════

    def generate_report(self, features: dict[str, float],
                         actual_age: float | None = None,
                         gender: str | None = None,
                         n_segments: int = 1,
                         source: str = "vital",
                         include_disease_condition: str | None = None,
                         include_similar_patients: bool = False,
                         beat_cv: dict[str, float] | None = None,
                         n_beats_valid: int | None = None,
                         window_n_segments: int | None = None,
                         previous_features: dict | None = None) -> dict[str, Any]:
        """Full phenotype report with optional enhanced features.

        Args:
            include_disease_condition: Disease name for condition-specific baseline (MIMIC only).
            include_similar_patients:  Whether to find similar patients in the DB.
            beat_cv: Per-feature beat-to-beat variability of the assessed
                segment(s) — CV = beat std / |beat mean| (ddof=1), same
                definition as the reference table.  Used for the
                ``beat_consistency`` output and confidence fusion.
            n_beats_valid: Valid-beat count of the assessed segment — drives
                the short-window discipline flag (window_context).
            window_n_segments: Optional segment-count axis for the window
                spec (multi-segment flows; None = single-segment).
            previous_features: Optional earlier measurement of the same
                subject for minimal-detectable-change comparison
                (change_reference; tier-2 MDC knowledge).

        Returns:
            Report dict; ``beat_consistency`` maps each analyzed feature to
            {icc1_single, icc2_k5, icc2_k10, cv, cv_p90, cv_level} — the
            ICC-driven basis for "how many segments are enough" (5 → ~0.85,
            10 → ~0.92 on the most reliable features).  New Agent-2 fields:
            disease_associations / phenotype_subtype / window_context /
            scale_warnings / change_reference — all optional knowledge,
            absent or available=False when the reference is missing.
        """
        va = self.assess_vascular_age(features, actual_age, source)
        rel_map = self._reliability(source)
        age_f = sorted(((k, rel_map.get(k, FEATURE_RELIABILITY.get(k, 0)))
                        for k in set(rel_map) | set(FEATURE_RELIABILITY)),
                       key=lambda x: x[1], reverse=True)[:10]

        # Analysis set = top-10 reliability ∪ VAI top contributors: the
        # features driving the assessment must be interpreted even when they
        # fall outside the reliability top-10 (e.g. reflection_index), else
        # the report misses the strongest signal.
        key_set = {fname for fname, _ in age_f}
        for c in va.get("feature_details", []):
            fc = c.get("feature")
            if fc in features:
                key_set.add(fc)
        key_set = {f for f in key_set if f in features}

        # Stratified percentile (age/gender matched)
        stratified = {}
        for fname in key_set:
            rel = rel_map.get(fname, FEATURE_RELIABILITY.get(fname, 0))
            q = self.query_stratified(fname, features[fname], source, actual_age, gender)
            stratified[fname] = {"percentile": q.get("percentile"),
                                  "global": q.get("global_percentile"),
                                  "level": q.get("level"), "reliability": rel}

        # Disease condition
        disease_info = None
        if include_disease_condition and source == "mimic":
            disease_info = {}
            for fc in CONDITION_FEATURES:
                if fc in features:
                    disease_info[fc] = self.query_disease_conditioned(
                        fc, features[fc], include_disease_condition)

        # Similar patients
        similar = None
        if include_similar_patients:
            similar = self.find_similar_patients(features, source)

        # ── Beat-level consistency (ICC + beat-to-beat CV) ──
        beat_consistency = {}
        for fname in stratified:
            icc1 = self._icc(source, fname)
            if icc1 is None:
                continue
            ref = self._cv_reference(source, fname) or {}
            cv = (beat_cv or {}).get(fname)
            cv_level = "unknown"
            if cv is not None and ref.get("p90"):
                cv_level = "elevated" if cv > ref["p90"] else "normal"
            beat_consistency[fname] = {
                "icc1_single": icc1,
                "icc2_k5": self._icc_2k(source, fname, 5),
                "icc2_k10": self._icc_2k(source, fname, 10),
                "cv": None if cv is None else round(float(cv), 4),
                "cv_p90": ref.get("p90"),
                "cv_level": cv_level,
            }

        # ── Confidence: ICC(2, n_segments) across the analyzed features ──
        icc2_ks = [self._icc_2k(source, f, max(1, n_segments))
                   for f in stratified]
        icc2_ks = [x for x in icc2_ks if x is not None]
        if icc2_ks:
            mean_icc2k = float(np.mean(icc2_ks))
            if mean_icc2k >= 0.80:
                conf = "high"
            elif mean_icc2k >= 0.55:
                conf = "medium"
            else:
                conf = "low"
            conf_note = (f"ICC(2,{n_segments}) averaged {mean_icc2k:.2f} over "
                         f"{len(icc2_ks)} features"
                         + ("; some features above the p90 beat-to-beat "
                            "variability reference — treat those as unstable."
                            if any(b.get("cv_level") == "elevated"
                                   for b in beat_consistency.values()) else ""))
        else:
            conf = "medium" if n_segments >= 5 else "low"
            conf_note = ("≥5 segments, noise reduced." if n_segments >= 5 else
                         "Single segment, within-person noise ~60% of "
                         "between-person variation.")

        # Physiological rules for out-of-range features (directional wording;
        # the synthesis phase cites these, never diagnoses)
        interpretation_rules = {}
        for fname, entry in stratified.items():
            if entry.get("level") in (None, "normal"):
                continue
            rule = self._rule_for(fname)
            if rule:
                interpretation_rules[fname] = rule

        # ── Agent-2 knowledge layer (analysis steps 16-27) ─────────────
        # Population association evidence for out-of-range features — the LLM
        # may cite these only as "consistent with the pattern seen in
        # {disease}-related cases in the reference population", never as a
        # diagnosis (see CARDIOVASCULAR_PROMPT output rule 4).
        disease_associations = self.disease_associations(
            stratified, features, source)

        # Reference-population subtype assignment (nearest z center).
        phenotype_subtype = self.assign_subtype(features, source)

        # Short-window discipline (beats < 20 and/or segments < recommended N)
        window_context = self.window_context(
            source, n_beats_valid if n_beats_valid is not None else n_segments,
            window_n_segments)

        # Between/within-patient direction flips — guards the short window.
        warn = self._scale_warnings()
        scale_warn_feats = self.scale_warning_features(source)
        scale_warnings = {
            "available": bool(scale_warn_feats),
            "features": scale_warn_feats,
            "rule": (warn.get("meta") or {}).get("rule"),
        }

        # Minimal-detectable-change comparison (only when a previous
        # measurement was provided).
        change_reference = self.assess_change(features, previous_features, source)

        return {
            "vascular_age": va,
            "stratified_percentiles": stratified,
            "disease_condition": disease_info,
            "similar_patients": similar,
            "interpretation_rules": interpretation_rules,
            "beat_consistency": beat_consistency,
            "confidence": conf,
            "confidence_note": conf_note,
            "safety_note": "PPG-derived cardiovascular phenotype assessment, not a medical diagnosis.",
            "baseline_source": source,
            "disease_associations": disease_associations,
            "phenotype_subtype": phenotype_subtype,
            "window_context": window_context,
            "scale_warnings": scale_warnings,
            "change_reference": change_reference,
        }
