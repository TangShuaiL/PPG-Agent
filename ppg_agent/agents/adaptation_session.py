"""ReAct dataset-adaptation session for Agent 1.

Interactive, human-in-the-loop parameter tuning for a target dataset:
each round runs the REAL parameterized pipeline on a sample segment, renders
segmentation + feature-extraction figures and a metric panel, and proposes
2-4 concrete parameter-adjustment options. The user picks one (or triggers
batch validation / finalization / revert / quit) and the loop continues.

Objectives (three levels):
  L1 every round  — single-segment quality, coverage, valid ratio, beat CVs
  L2 on demand    — 20-segment batch validation (multiprocessing)
  L3 at finalize  — acceptance gate: split-half reliability + sign-aligned VAI
                    age correlation. Failing the gate refuses the preset save.

The session is a plain state-machine object (matching the repo's synchronous
invoke() patterns); no LangGraph checkpointer involved.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import h5py
from pydantic import BaseModel, Field
from scipy.signal import resample
from scipy.stats import spearmanr

from ppg_agent.config import MODELS_ROOT, PHENOTYPE_DB, REPORTS_ROOT
from ppg_agent.database.connection import phenotype_conn
from ppg_agent.engine.visualization import (
    plot_signal_with_beats,
    plot_feature_extraction_overview,
)
from ppg_agent.llm_service import get_llm
from ppg_agent.engine.signal_engine import process_ppg_segment
from ppg_agent.processing_params import (
    PARAM_SPEC,
    DEFAULT_PARAMS,
    SOURCE_PRESET_MAP,
    list_presets,
    load_preset,
    save_preset,
    validate_params,
)

FS = 500

# ---------------------------------------------------------------------------
# Raw data locations (shared resolution via ppg_agent.config)
# ---------------------------------------------------------------------------

def raw_data_dir(source: str) -> str:
    from ppg_agent.config import raw_data_dir as _rdd
    return str(_rdd(source))


def load_sample_segment(source: str, subject_id: str, seg_idx: int) -> dict:
    """Load one segment (PPG_F resampled to 500Hz) + R-peaks + metadata."""
    fp = os.path.join(raw_data_dir(source), f"{subject_id}.mat")
    with h5py.File(fp, "r") as f:
        subj = f["Subj_Wins"]
        ppg_ref = subj["PPG_F"][0, seg_idx]
        ppg = (ppg_ref.flatten().astype(np.float64)
               if isinstance(ppg_ref, np.ndarray)
               else f[ppg_ref][:].flatten().astype(np.float64))
        rpeaks = None
        if "ECG_RPeaks" in subj:
            rp_ref = subj["ECG_RPeaks"][0, seg_idx]
            rpeaks = (rp_ref.flatten().astype(np.float64)
                      if isinstance(rp_ref, np.ndarray)
                      else f[rp_ref][:].flatten().astype(np.float64))
        age = None
        if "Age" in subj:
            a_ref = subj["Age"][0, 0]
            try:
                age = float(a_ref) if not isinstance(a_ref, np.ndarray) or a_ref.size == 1 \
                    else float(np.asarray(a_ref).flat[0])
            except Exception:
                age = None
    if len(ppg) == 1250:
        ppg = resample(ppg, 5000)
    return {"ppg_500": ppg, "fs": FS, "rpeaks": rpeaks, "age": age,
            "name": f"{source}/{subject_id}/seg{seg_idx}", "path": fp}


# ---------------------------------------------------------------------------
# Metrics & diagnosis
# ---------------------------------------------------------------------------

def _validate_work(task: tuple) -> dict:
    """Module-level worker for ProcessPoolExecutor (picklable)."""
    source, subject_id, seg_idx, params = task
    try:
        sample = load_sample_segment(source, subject_id, seg_idx)
        r = process_ppg_segment(sample["ppg_500"], FS, params=params)
        return {"seg_idx": seg_idx, "quality": r["quality"],
                "tier": r["quality_tier"], "n_valid": r["n_valid"],
                "n_total": r["n_total"]}
    except Exception as e:
        return {"seg_idx": seg_idx, "error": str(e)[:80]}

def segment_metrics(result: dict) -> dict:
    """L1 metric panel from a process_ppg_segment result."""
    cvs = {}
    valid_idx = [i for i, v in enumerate(result["valid_mask"]) if v]
    for name, getter in [
        ("pulse_amp", lambda bt, kp: float(np.max(bt) - np.min(bt))),
        ("t_sys_ms", lambda bt, kp: (kp.get("s", 0) - kp.get("o", 0)) / FS * 1000
         if kp.get("s") is not None and kp.get("o") is not None else np.nan),
        ("beat_len_ms", lambda bt, kp: len(bt) / FS * 1000),
        ("ni_dn", lambda bt, kp: (kp.get("dn") is not None and kp.get("s") is not None
                                  and bt[kp["s"]] != 0)
         and float(bt[kp["dn"]] / bt[kp["s"]]) or np.nan),
    ]:
        vals = [getter(result["beats"][i], result["keypoints"][i]) for i in valid_idx]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= 3:
            cvs[name] = float(np.std(vals) / (np.abs(np.mean(vals)) + 1e-8))
        else:
            cvs[name] = np.nan
    return {
        "quality": result["quality"],
        "tier": result["quality_tier"],
        "coverage": result["coverage"],
        "valid_ratio": result["valid_ratio"],
        "n_total": result["n_total"],
        "n_valid": result["n_valid"],
        "n_rejected_mad": result.get("n_rejected_mad", 0),
        "cvs": cvs,
    }


def diagnose(result: dict) -> dict:
    """Rule-based 6-dimension diagnosis (ported from the legacy tuner)."""
    m = segment_metrics(result)
    issues = []
    q = m["quality"]
    if q < 0.30:
        issues.append({"severity": "critical", "check": "quality",
                       "detail": f"Q={q:.2f} (failed).",
                       "fix": "Try larger changes: lower peak_distance_ms/prominence, validation_mode='lenient'"})
    elif q < 0.60:
        issues.append({"severity": "warning", "check": "quality",
                       "detail": f"Q={q:.2f} (fair).",
                       "fix": "Check rejection breakdown; consider validation_mode='lenient'"})

    n_beats = m["n_total"]
    if n_beats > 25:  # >150bpm equivalent for 10s
        issues.append({"severity": "critical", "check": "beat_count",
                       "detail": f"Over-segmentation: {n_beats} beats in segment",
                       "fix": "Increase peak_distance_ms / peak_prominence / min_beat_len"})
    elif n_beats < 4:
        issues.append({"severity": "critical", "check": "beat_count",
                       "detail": f"Under-segmentation: only {n_beats} beats",
                       "fix": "Decrease peak_distance_ms / peak_prominence"})

    offsets = result.get("onset_offsets", [])
    if offsets:
        off_mean = float(np.mean([abs(o) for o in offsets]))
        n_corrected = sum(1 for o in offsets if o > 0)
        if off_mean > 150:
            issues.append({"severity": "critical", "check": "onset_offset",
                           "detail": f"Onset far from boundary: mean={off_mean:.0f} samples "
                                     f"({n_corrected}/{len(offsets)} corrected)",
                           "fix": "Decrease onset_search_ms"})
        elif off_mean > 100:
            issues.append({"severity": "warning", "check": "onset_offset",
                           "detail": f"Onset on early side: mean={off_mean:.0f} samples",
                           "fix": "Consider slightly lower onset_search_ms"})

    n_total, n_valid = m["n_total"], m["n_valid"]
    if n_total > 0 and (n_total - n_valid) / n_total > 0.5:
        issues.append({"severity": "critical", "check": "rejection_rate",
                       "detail": f"High rejection: {(n_total - n_valid)}/{n_total} beats rejected",
                       "fix": "Increase max_beat_len_ms or use validation_mode='lenient'"})

    if m["coverage"] < 0.40:
        issues.append({"severity": "critical", "check": "coverage",
                       "detail": f"Low coverage: {m['coverage']:.1%}",
                       "fix": "Decrease peak_distance_ms / min_beat_len"})

    cv_amp = m["cvs"].get("pulse_amp", np.nan)
    if not np.isnan(cv_amp) and cv_amp > 0.15:
        issues.append({"severity": "warning", "check": "amp_variation",
                       "detail": f"High amplitude CV: {cv_amp:.3f}",
                       "fix": "Enable mad_reject_k=2.5 or extra_bandpass"})

    criticals = [i for i in issues if i["severity"] == "critical"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    return {
        "issues": issues,
        "n_critical": len(criticals), "n_warning": len(warnings),
        "needs_tuning": len(criticals) > 0 or len(warnings) > 0 or q < 0.60,
    }


# ---------------------------------------------------------------------------
# LLM option proposal
# ---------------------------------------------------------------------------

class ParamChange(BaseModel):
    param: str = Field(description="Parameter key from PARAM_SPEC")
    value: Any = Field(description="New value (must satisfy the parameter spec)")
    rationale: str = Field(description="Why this change, citing observed metrics")
    expected_effect: str = Field(description="Predicted effect on metrics")


class ProposeOptions(BaseModel):
    summary: str = Field(description="One-line human diagnosis of the current state")
    options: list[ParamChange] = Field(description="2-4 parameter adjustment options")


# Static rule-based option library (LLM fallback / mock mode)
RULE_OPTIONS = [
    ("peak_prominence", 0.08, "reject low-amplitude noise peaks",
     "fewer false beats, lower amp CV"),
    ("peak_distance_ms", 400, "reduce over-segmentation",
     "fewer beats, higher valid ratio"),
    ("peak_distance_ms", 200, "reduce under-segmentation",
     "more beats detected"),
    ("validation_mode", "lenient", "reduce validation rejects",
     "fewer rejected beats"),
    ("min_beat_len", 200, "drop short noise fragments",
     "fewer invalid beats"),
    ("max_beat_len_ms", 1600, "accept slow long beats",
     "fewer too-long rejects"),
    ("onset_search_ms", 160, "widen onset search for slow upstrokes",
     "more accurate onsets"),
    ("diastolic_search_ms", 250, "widen diastolic search",
     "more dn/ip/d detections"),
    ("extra_bandpass", True, "cascade 0.5-8Hz bandpass",
     "lower rise-time CV (wearable)"),
    ("mad_reject_k", 2.5, "MAD amplitude outlier rejection",
     "lower amplitude CV"),
]


def _build_rule_options(session: "AdaptationSession", diagnosis: dict) -> list[ParamChange]:
    """Deterministic fallback options from diagnosis fix-hints + static library."""
    opts: list[ParamChange] = []
    used = set()
    # 1. options driven by current issues (fix hints mapped to param changes)
    for issue in diagnosis["issues"][:2]:
        fix = issue["fix"].lower()
        if "peak_distance" in fix and "peak_distance_ms" not in used:
            if "increase" in fix or "↑" in fix:
                opts.append(ParamChange(param="peak_distance_ms",
                                        value=min(600, session.params["peak_distance_ms"] + 100),
                                        rationale=issue["check"], expected_effect=fix))
            else:
                opts.append(ParamChange(param="peak_distance_ms",
                                        value=max(150, session.params["peak_distance_ms"] - 100),
                                        rationale=issue["check"], expected_effect=fix))
            used.add("peak_distance_ms")
        elif "lenient" in fix and "validation_mode" not in used:
            opts.append(ParamChange(param="validation_mode", value="lenient",
                                    rationale=issue["check"], expected_effect=fix))
            used.add("validation_mode")
        elif "mad_reject" in fix and "mad_reject_k" not in used:
            opts.append(ParamChange(param="mad_reject_k", value=2.5,
                                    rationale=issue["check"], expected_effect=fix))
            used.add("mad_reject_k")
        elif "onset_search" in fix and "onset_search_ms" not in used:
            opts.append(ParamChange(param="onset_search_ms", value=80,
                                    rationale=issue["check"], expected_effect=fix))
            used.add("onset_search_ms")
    # 2. fill from static library with params not yet tried
    tried = {r.changed.get("param") for r in session.rounds if r.changed}
    for param, value, rationale, effect in RULE_OPTIONS:
        if len(opts) >= 4:
            break
        if param in used or param in tried:
            continue
        if value == session.params[param]:
            continue
        if validate_params({param: value}):
            continue
        opts.append(ParamChange(param=param, value=value,
                                rationale=rationale, expected_effect=effect))
        used.add(param)
    return opts[:4]


def _llm_propose(session: "AdaptationSession") -> list[ParamChange] | None:
    """Ask the LLM for options; None on any failure (caller falls back to rules)."""
    llm = session.llm
    if llm is None:
        return None
    history_lines = []
    for r in session.rounds[-6:]:
        m = r.metrics
        hist = (f"round {r.round_idx}: params={_compact_params(r.params)} "
                f"Q={m['quality']:.2f} n={m['n_valid']}/{m['n_total']} "
                f"cov={m['coverage']:.2f} ampCV={m['cvs'].get('pulse_amp', float('nan')):.3f}")
        if r.changed:
            hist += f" changed: {r.chosen_label}"
        history_lines.append(hist)
    diag = session.current_diagnosis or {}
    issues_txt = "; ".join(f"{i['severity']}:{i['check']}" for i in diag.get("issues", [])[:4])
    param_txt = "\n".join(
        f"- {k}: default={s.default}, kind={s.kind}, "
        + (f"range=[{s.min},{s.max}]" if s.kind == "range" else f"candidates={s.candidates}")
        for k, s in PARAM_SPEC.items())
    prompt = f"""You are the parameter-adaptation brain of a PPG processing pipeline.

Target sample: {session.sample['name']}
Current round: {session.rounds[-1].round_idx}/{session.max_rounds}
Current params: {session.params}
Latest metrics: quality={session.rounds[-1].metrics['quality']:.3f},
n_valid={session.rounds[-1].metrics['n_valid']}/{session.rounds[-1].metrics['n_total']},
coverage={session.rounds[-1].metrics['coverage']:.3f},
ampCV={session.rounds[-1].metrics['cvs'].get('pulse_amp', float('nan')):.3f}
Diagnosis: {issues_txt or 'none'}
History:
{chr(10).join(history_lines) or '(none)'}

Parameter spec:
{param_txt}

Propose 2-4 parameter adjustments that would most improve extraction quality
for this dataset. Each option changes exactly ONE parameter to a valid value.
Cite the observed metric that motivates each change. Do not repeat parameter
changes already tried in the history."""
    try:
        llm_tools = llm.bind_tools([ProposeOptions])
        resp = llm_tools.invoke(prompt)
        tool_calls = getattr(resp, "tool_calls", None) or []
        for tc in tool_calls:
            if tc.get("name") == "ProposeOptions":
                parsed = ProposeOptions.model_validate(tc.get("args", {}))
                out = []
                for opt in parsed.options:
                    if not validate_params({opt.param: opt.value}):
                        out.append(opt)
                return out
        return None
    except Exception:
        return None


def _compact_params(params: dict, max_len: int = 80) -> str:
    s = " ".join(f"{k}={v}" for k, v in sorted(params.items()))
    return s[:max_len] + ("..." if len(s) > max_len else "")


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class RoundRecord:
    round_idx: int
    params: dict
    metrics: dict
    diagnosis: dict
    changed: dict | None = None
    figures: dict = field(default_factory=dict)
    validate_summary: dict | None = None
    options: list[dict] = field(default_factory=list)
    chosen_label: str | None = None


class AdaptationSession:
    """State-machine ReAct adaptation session (see module docstring)."""

    def __init__(self, source: str, subject_id: str, seg_idx: int,
                 preset: str = "default", save_as: str | None = None,
                 max_rounds: int = 8, seed: int = 42,
                 llm: Any = None, mock_llm: bool = False,
                 gate_n_subjects: int = 30, gate_min_segs: int = 8):
        self.source = source
        self.subject_id = subject_id
        self.seg_idx = seg_idx
        self.preset_name = preset          # starting preset (must exist)
        self.save_as = save_as or preset   # preset name to write at finalize
        self.max_rounds = max_rounds
        self.seed = seed
        if mock_llm:
            llm = None  # rule-based options only
        self.llm = llm
        self.mock_llm = mock_llm
        self.gate_n_subjects = gate_n_subjects
        self.gate_min_segs = gate_min_segs
        self.params: dict = load_preset(preset)
        self.baseline_quality: float | None = None
        self.rounds: list[RoundRecord] = []
        self.sample: dict = load_sample_segment(source, subject_id, seg_idx)
        self.current_diagnosis: dict | None = None
        self.status: str = "running"
        self.acceptance_report: dict | None = None
        self._rng = random.Random(seed)

    # -- internal helpers ----------------------------------------------------

    def _run(self, params: dict) -> dict:
        result = process_ppg_segment(self.sample["ppg_500"], FS, params=params)
        return result

    def _render_figures(self, result: dict, round_idx: int) -> dict:
        tag = f"adaptation_{self.source}_{self.subject_id}_{self.seg_idx}_r{round_idx}"
        seg = plot_signal_with_beats(
            raw=self.sample["ppg_500"], processed=None,
            beats=result["beats"], beat_starts=result["beat_starts"],
            valid_mask=result["valid_mask"], fs=FS,
            keypoints_per_beat=result["keypoints"],
            title=f"{tag} — {result['n_valid']}/{result['n_total']} valid, Q={result['quality']:.2f}",
            filename=f"{tag}_seg")
        feats = plot_feature_extraction_overview(
            result["beats"], result["beat_starts"], result["valid_mask"],
            result["keypoints"], FS,
            title=f"{tag} — feature extraction",
            filename=f"{tag}_feats")
        return {"segmentation": seg["path"], "features": feats["path"]}

    def _new_round(self, params: dict, changed: dict | None, chosen_label: str | None):
        result = self._run(params)
        metrics = segment_metrics(result)
        diag = diagnose(result)
        self.current_diagnosis = diag
        if self.baseline_quality is None:
            self.baseline_quality = metrics["quality"]
        rec = RoundRecord(
            round_idx=len(self.rounds),
            params=dict(params), metrics=metrics, diagnosis=diag,
            changed=changed, chosen_label=chosen_label,
            figures=self._render_figures(result, len(self.rounds)))
        self.rounds.append(rec)
        self.params = dict(params)
        rec.options = self._propose_options(rec)
        return rec

    def _propose_options(self, rec: RoundRecord) -> list[dict]:
        """Propose options for the NEXT step, with guardrails."""
        # Round-limit: only action options
        if rec.round_idx >= self.max_rounds:
            return []
        # mock_llm sets llm=None in __init__, so _llm_propose returns None
        # and the rule-based fallback kicks in — one unified path.
        opts = _llm_propose(self)
        if not opts:
            opts = _build_rule_options(self, rec.diagnosis)
        # Guardrails: dedupe params, validate values, skip already-tried
        tried = {r.changed.get("param") for r in self.rounds if r.changed}
        seen = set()
        clean = []
        for o in opts:
            if o.param in seen or o.param in tried:
                continue
            if validate_params({o.param: o.value}):
                continue
            if o.value == self.params[o.param]:
                continue
            seen.add(o.param)
            tag = "[B-class]" if PARAM_SPEC[o.param].cls == "B" else ""
            clean.append({"param": o.param, "value": o.value,
                          "rationale": o.rationale, "expected": o.expected_effect,
                          "tag": tag})
        # No-degradation guard: offer revert if quality dropped (top priority)
        if (len(self.rounds) >= 2 and
                rec.metrics["quality"] < self.rounds[-2].metrics["quality"] - 0.02):
            clean.insert(0, {"action": "revert", "param": None, "value": None,
                             "rationale": "quality dropped", "expected": "restore previous",
                             "tag": ""})
        return clean[:4]

    # -- public API -----------------------------------------------------------

    def start(self) -> str:
        """Run the baseline round with the starting preset and render its panel."""
        rec = self._new_round(self.params, None, None)
        return self.render_panel(rec)

    def apply(self, choice: str) -> str:
        """Apply a choice letter; returns the panel of the resulting state."""
        choice = choice.strip().lower()
        rec = self.rounds[-1]
        letters = "abcd"
        if choice in letters:
            idx = letters.index(choice)
            if idx >= len(rec.options):
                return f"Invalid option '{choice}'.\n" + self.render_panel(rec)
            opt = rec.options[idx]
            if opt.get("action") == "revert":
                return self.revert()
            old = self.params[opt["param"]]
            new_params = {**self.params, opt["param"]: opt["value"]}
            changed = {"param": opt["param"], "old_value": old,
                       "new_value": opt["value"]}
            new_rec = self._new_round(new_params, changed, choice.upper())
            return self.render_panel(new_rec)
        if choice == "v":
            summary = self.validate()
            rec.validate_summary = summary
            return self.render_panel(rec)
        if choice == "f":
            return self.finalize()
        if choice == "r":
            return self.revert()
        if choice == "q":
            self.status = "aborted"
            return "Session aborted. No preset saved."
        return f"Unknown choice '{choice}'.\n" + self.render_panel(rec)

    def revert(self, round_idx: int | None = None) -> str:
        """Revert to a previous round's params (default: one round back)."""
        if len(self.rounds) < 2:
            return "Nothing to revert to.\n" + self.render_panel(self.rounds[-1])
        target = round_idx if round_idx is not None else self.rounds[-2].round_idx
        target = max(0, min(target, len(self.rounds) - 2))
        prev = self.rounds[target]
        new_rec = self._new_round(prev.params,
                                  {"param": "(revert)", "old_value": None,
                                   "new_value": f"round {target} params"},
                                  "R")
        return self.render_panel(new_rec)

    def validate(self, n: int = 20) -> dict:
        """L2: batch validation over n segments of the session subject."""
        from concurrent.futures import ProcessPoolExecutor

        n = min(n, 40)
        tasks = [(self.source, self.subject_id, seg_idx, self.params)
                 for seg_idx in range(n)]
        with ProcessPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_validate_work, tasks))
        ok = [r for r in results if "error" not in r]
        if not ok:
            return {"error": "all segments failed", "n": n}
        tiers = pd.Series([r["tier"] for r in ok]).value_counts().to_dict()
        mean_q = float(np.mean([r["quality"] for r in ok]))
        verdict = "good" if mean_q >= 0.65 else "needs_tuning"
        return {"n_segments": len(ok), "mean_quality": round(mean_q, 3),
                "tier_distribution": tiers, "verdict": verdict,
                "errors": n - len(ok)}

    # -- finalize & acceptance gate --------------------------------------------

    def _gate_features(self, subject_ids: list[str]):
        """Extract per-subject (a) full medians, (b) split-half medians, (c) ages.

        Split-half: segments sorted by seg_idx; first half and second half
        feature medians per subject — used for within-subject reliability.
        Returns (feats_df, half1_df, half2_df, ages) all indexed by subject_id.
        """
        full_rows, half1_rows, half2_rows = {}, {}, {}
        ages = {}
        for sid in subject_ids:
            segs = self._subject_segment_list(sid)
            seg_feats = []
            for seg_idx in segs[:12]:
                try:
                    sample = load_sample_segment(self.source, sid, seg_idx)
                    r = process_ppg_segment(sample["ppg_500"], FS, params=self.params)
                    valid_feats = [f for f, v in zip(r["features"], r["valid_mask"]) if v and f]
                    if valid_feats:
                        df = pd.DataFrame(valid_feats)
                        seg_feats.append(df.select_dtypes(include=[np.number]).mean())
                    ages[sid] = (sample["age"] if sample.get("age") is not None
                                 else self._age_of(sid))
                except Exception:
                    continue
            if len(seg_feats) < 6:
                continue
            df_all = pd.DataFrame(seg_feats)
            half = len(df_all) // 2
            full_rows[sid] = df_all.median()
            half1_rows[sid] = df_all.iloc[:half].median()
            half2_rows[sid] = df_all.iloc[half:].median()
        if not full_rows:
            empty = pd.DataFrame()
            return empty, empty, empty, pd.Series(dtype=float)
        ages = pd.Series({sid: a for sid, a in ages.items() if pd.notna(a)})
        return (pd.DataFrame(full_rows).T, pd.DataFrame(half1_rows).T,
                pd.DataFrame(half2_rows).T, ages)

    def _subject_segment_list(self, subject_id: str) -> list[int]:
        fp = os.path.join(raw_data_dir(self.source), f"{subject_id}.mat")
        with h5py.File(fp, "r") as f:
            return list(range(f["Subj_Wins"]["PPG_F"].shape[1]))

    def _age_of(self, subject_id: str) -> float | None:
        if self.source in ("aurora", "aurorabp"):
            from ..config import labels_path
            labels = pd.read_csv(labels_path("aurora"))
            v = labels.set_index("filename")["age"].get(subject_id)
            return float(v) if pd.notna(v) else None
        import sqlite3
        conn = sqlite3.connect(str(PHENOTYPE_DB))
        row = conn.execute(
            "SELECT age FROM subjects WHERE subject_id=? AND source=?", (subject_id, self.source)
        ).fetchone()
        conn.close()
        return float(row[0]) if row and row[0] is not None else None

    def run_acceptance_gate(self) -> dict:
        """L3: split-half reliability + sign-aligned VAI age correlation."""
        # Pick gate subjects: train-split subjects with enough segments
        candidates = self._gate_subjects()
        if len(candidates) < self.gate_n_subjects // 2:
            return {"error": "not enough gate subjects",
                    "n_candidates": len(candidates)}
        feats_df, half1_df, half2_df, ages = \
            self._gate_features(candidates[:self.gate_n_subjects])
        if feats_df.empty:
            return {"error": "no features extracted", "n_candidates": len(candidates)}

        # -- split-half reliability (within-subject, across two halves) --
        common = [c for c in feats_df.columns
                  if not any(b in c.lower() for b in
                             ["abp", "bp_", "hr_bpm", "ecg", "rpeak", "beat_ratio", "ptt", "hrv"])]
        rhos = {}
        for c in common:
            g1, g2 = half1_df[c].dropna(), half2_df[c].dropna()
            common_idx = g1.index.intersection(g2.index)
            if len(common_idx) < 8:
                continue
            rho, _ = spearmanr(g1.loc[common_idx], g2.loc[common_idx])
            if not np.isnan(rho):
                rhos[c] = rho
        median_rel = float(np.median(list(rhos.values()))) if rhos else 0.0
        frac_gt_half = float(np.mean([1 for r in rhos.values() if r > 0.5])) if rhos else 0.0

        # -- sign-aligned VAI --
        vai_rho, vai_p, n_vai = self._sign_aligned_vai(feats_df, ages)

        passed = (median_rel >= 0.5) and (vai_rho is not None and vai_rho >= 0.25)
        return {
            "median_split_half_rho": round(median_rel, 3),
            "frac_split_half_gt_05": round(frac_gt_half, 3),
            "n_features": len(rhos),
            "vai_rho_age": round(vai_rho, 3) if vai_rho is not None else None,
            "vai_p": vai_p,
            "n_vai_subjects": n_vai,
            "passed": passed,
        }

    def _gate_subjects(self) -> list[str]:
        """Train-split subjects of this source with enough segments."""
        if self.source in ("aurora", "aurorabp"):
            split = pd.read_csv("<PROJECT_ROOT>/analysis/05_reference_baselines/"
                                "output/tables/aurora_split_assignments.csv")
            train_ids = split[split["split"] == "train"]["subject_id"].tolist()
        else:
            import sqlite3
            conn = sqlite3.connect(str(PHENOTYPE_DB))
            train_ids = [r[0] for r in conn.execute(
                "SELECT subject_id FROM subjects WHERE source=? AND split='train'",
                (self.source,)).fetchall()]
            conn.close()
        self._rng.shuffle(train_ids)
        return train_ids

    def _sign_aligned_vai(self, feats_df: pd.DataFrame, ages: pd.Series) -> tuple:
        """VAI with per-feature directions from a precomputed reference (if available),
        else from an internal half-split of the gate sample."""
        ref_path = Path("<PROJECT_ROOT>/analysis/06_conclusions/"
                        "output/feature_age_correlation.csv")
        col = {"aurora": "aurora", "aurorabp": "aurora",
               "vital": "vital", "mimic": "mimic"}.get(self.source, self.source)
        directions = {}
        if ref_path.exists():
            ref = pd.read_csv(ref_path)
            for _, r in ref.iterrows():
                rho = r.get(col)
                if pd.notna(rho):
                    direction = np.sign(rho) if abs(rho) > 0.05 else 0
                    # reference names carry _mean; gate features may be bare
                    directions[r["feature"] + "_mean"] = direction
                    directions[r["feature"]] = direction
        if not directions:
            # Internal: directions from first half, evaluation on second half
            half = len(feats_df) // 2
            for c in feats_df.columns:
                sub = pd.DataFrame({c: feats_df[c], "age": ages}).dropna()
                if len(sub) < 8:
                    continue
                rho, _ = spearmanr(sub[c].iloc[:half], sub["age"].iloc[:half])
                directions[c] = np.sign(rho) if abs(rho) > 0.05 else 0

        vais, age_vals = [], []
        for sid in feats_df.index:
            if sid not in ages.index or pd.isna(ages[sid]):
                continue
            zs = []
            for c, direction in directions.items():
                if direction == 0 or c not in feats_df.columns:
                    continue
                v = feats_df.loc[sid, c]
                if pd.isna(v) or feats_df[c].std() == 0:
                    continue
                zs.append(direction * (v - feats_df[c].mean()) / feats_df[c].std())
            if len(zs) >= 8:
                vais.append(np.mean(zs))
                age_vals.append(ages[sid])
        if len(vais) < 8:
            return None, None, len(vais)
        rho, p = spearmanr(vais, age_vals)
        return float(rho), float(p), len(vais)

    def finalize(self, force: bool = False) -> str:
        """Run the acceptance gate; save the preset on pass (or force)."""
        report = self.run_acceptance_gate()
        self.acceptance_report = report
        out_dir = Path(REPORTS_ROOT) / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"adaptation_{self.save_as}_acceptance.md"
        lines = [
            f"# Adaptation Acceptance — {self.save_as}",
            f"source: {self.source}  subject: {self.subject_id}  "
            f"rounds: {len(self.rounds)}",
            "",
            "## Gate results",
            f"- split-half median ρ: {report.get('median_split_half_rho')}",
            f"- split-half ρ>0.5 fraction: {report.get('frac_split_half_gt_05')}",
            f"- sign-aligned VAI ρ(age): {report.get('vai_rho_age')}",
            f"- passed: {report.get('passed')}",
            "",
            "## Final params",
            "```yaml",
        ]
        import yaml
        lines.append(yaml.dump(dict(sorted(self.params.items())),
                               default_flow_style=False))
        lines.append("```")
        report_path.write_text("\n".join(lines))

        if report.get("passed") or force:
            save_preset(self.save_as, self.params)
            self.status = "done"
            return (f"Gate {'PASSED' if report.get('passed') else 'FORCED'} — "
                    f"preset '{self.save_as}' saved.\n"
                    f"Report: {report_path}")
        self.status = "done"
        return (f"Gate FAILED: {report.get('median_split_half_rho', 0):.3f} "
                f"(need ≥0.5), VAI ρ={report.get('vai_rho_age')} (need ≥0.25). "
                f"Preset NOT saved. Re-run finalize(force=True) to override.\n"
                f"Report: {report_path}")

    # -- panel rendering -------------------------------------------------------

    def render_panel(self, rec: RoundRecord) -> str:
        line = "─" * 72
        m = rec.metrics
        b_tag = " (baseline)" if rec.round_idx == 0 else ""
        header = (f"Round {rec.round_idx}/{self.max_rounds} — {self.sample['name']}"
                  f"        preset: {self.preset_name}{b_tag}")
        a_params = " ".join(f"{k}={v}" for k, v in rec.params.items()
                            if PARAM_SPEC[k].cls == "A")
        b_params = " ".join(f"{k}={v}" for k, v in rec.params.items()
                            if PARAM_SPEC[k].cls == "B")
        s_params = " ".join(f"{k}={v}" for k, v in rec.params.items()
                            if PARAM_SPEC[k].cls == "S")
        changed_txt = ""
        if rec.changed:
            changed_txt = (f"Changed last round: {rec.chosen_label}) "
                           f"{rec.changed['param']}: {rec.changed['old_value']} → "
                           f"{rec.changed['new_value']}")
        diag_txt = (f"{rec.diagnosis['n_critical']} critical, "
                    f"{rec.diagnosis['n_warning']} warnings")
        if rec.diagnosis["issues"]:
            top = "; ".join(f"{i['severity']}:{i['check']}"
                            for i in rec.diagnosis["issues"][:3])
            diag_txt += f" ({top})"
        delta_txt = ""
        if rec.round_idx >= 1:
            prev = self.rounds[rec.round_idx - 1].metrics
            dq = m["quality"] - prev["quality"]
            dcv = (m["cvs"].get("pulse_amp", np.nan)
                   - prev["cvs"].get("pulse_amp", np.nan))
            color = "" if dq >= 0 else " [DEGRADED]"
            delta_txt = (f"Δ vs round {rec.round_idx-1}: quality {dq:+.3f}, "
                         f"amp CV {dcv:+.3f}{color}")
        cv_txt = " | ".join(f"{k} {v:.3f}" for k, v in m["cvs"].items())
        fig_txt = "\n".join(f"  {k}: {v}" for k, v in rec.figures.items())
        validate_txt = ""
        if rec.validate_summary:
            v = rec.validate_summary
            if "error" in v:
                validate_txt = f"\nValidation: ERROR {v['error']}"
            else:
                validate_txt = (f"\nValidation (n={v['n_segments']}): "
                                f"mean Q={v['mean_quality']}, "
                                f"tiers={v['tier_distribution']}, "
                                f"verdict={v['verdict']}")
        options_txt = ""
        if rec.options and rec.round_idx < self.max_rounds:
            options_txt = "\nOptions (choose by letter):"
            for i, opt in enumerate(rec.options):
                letter = "abcd"[i].upper()
                if opt.get("action") == "revert":
                    options_txt += f"\n  {letter}. Revert to round "
                    options_txt += f"{self.rounds[-2].round_idx} params"
                else:
                    options_txt += (f"\n  {letter}. {opt['param']} "
                                    f"{self.params[opt['param']]} → {opt['value']}"
                                    f" {opt['tag']}")
                    options_txt += f"\n       {opt['rationale']} → {opt['expected']}"
            options_txt += "\n  V. Run batch validation (20 segments)"
        else:
            options_txt = "\nOptions:"
        options_txt += "\n  F. Finalize & save preset (acceptance gate, ~2-3 min)"
        if len(self.rounds) >= 2:
            options_txt += "\n  R. Revert to previous round"
        options_txt += "\n  q. quit without saving"
        prompt_txt = f"\nYour choice [A-D/V/F{'/R' if len(self.rounds) >= 2 else ''}/q]: "

        return (
            f"{line}\n{header}\n{line}\n"
            f"Params:  {a_params}\n         [B] {b_params}\n         [S] {s_params}\n"
            f"{changed_txt}\n{line}\n"
            f"Quality {m['quality']:.3f} ({m['tier']})   beats {m['n_valid']}/{m['n_total']} "
            f"valid   coverage {m['coverage']:.2f}   valid_ratio {m['valid_ratio']:.2f}"
            f"   mad_rejected {m['n_rejected_mad']}\n"
            f"CVs: {cv_txt}\n"
            f"Diagnosis: {diag_txt}\n{delta_txt}"
            f"{validate_txt}\n"
            f"Figures:\n{fig_txt}\n{line}\n"
            f"{options_txt}{prompt_txt}"
        )


def create_adaptation_session(source: str, subject_id: str = "p000001", seg_idx: int = 0,
                              preset: str = "default", save_as: str | None = None,
                              max_rounds: int = 8,
                              llm: Any = None, mock_llm: bool = False, seed: int = 42,
                              **kwargs) -> AdaptationSession:
    """Entry point. llm=None → try the signal_processor LLM from the registry."""
    if llm is None and not mock_llm:
        try:
            from ppg_agent.llm_service import get_llm
            llm = get_llm("signal_processor")
        except Exception:
            llm = None
    return AdaptationSession(source=source, subject_id=subject_id, seg_idx=seg_idx,
                             preset=preset, save_as=save_as, max_rounds=max_rounds,
                             llm=llm, mock_llm=mock_llm, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Dataset-level adaptation session (the ONE session engine used by both the
# Agent-1 graph's adapt branch and the web chat's deterministic layer).
#
# Sessions live in a module-level per-source dict so they survive across web
# requests (a HITL session spans many messages).  ``adapt_pipeline`` is the
# single tool entry point: the graph LLM calls it with semantic actions
# (start/propose/apply/sample/save/cancel/status), and the web layer calls
# the very same actions deterministically for quick actions and "选N".
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, "DatasetAdaptation"] = {}
_ORDER: list[str] = []  # most recently touched first (source keys)


def _touch(source: str) -> None:
    if source in _ORDER:
        _ORDER.remove(source)
    _ORDER.insert(0, source)


def active_adapt_session(source: str = "") -> "DatasetAdaptation | None":
    """Active session for a source; empty source → most recently touched.

    Source is matched strictly: after cancel/save a session is no longer
    'active', and the lookup must NOT fall back to some *other* source's
    active session — that would propagate a foreign state to the frontend.
    """
    if source:
        d = _SESSIONS.get(source)
        return d if d is not None and d.status == "active" else None
    for s in _ORDER:
        d = _SESSIONS.get(s)
        if d is not None and d.status == "active":
            return d
    return None


def get_adapt_session(source: str = "") -> "DatasetAdaptation | None":
    """Any (not only active) session: used after save/cancel to surface the
    final state to the frontend (which must stop rendering active buttons)."""
    if source:
        return _SESSIONS.get(source)
    for s in _ORDER:
        d = _SESSIONS.get(s)
        if d is not None:
            return d
    return None


N_SAMPLES = 3  # sample segments per iteration


def _adapt_summary(dapt: "DatasetAdaptation", prefix: str = "") -> str:
    """Short text summary of the current adaptation round for the LLM."""
    rec = dapt.rounds[-1]
    lines = [prefix]
    for s in rec.samples:
        m = s.metrics
        if "error" in m:
            lines.append(f"  {s.name}: ERROR {m['error']}")
        else:
            lines.append(f"  {s.name}: {m['n_valid']}/{m['n_total']} valid, "
                         f"Q={m['quality']} ({m['tier']})")
    return "\n".join(lines) or f"{dapt.source} 暂无采样"


def _options_text(dapt: "DatasetAdaptation", msg: str) -> str:
    opts = dapt.state_dict().get("pending_options") or []
    if not opts:
        return msg
    lines = [f"{msg}: "]
    for i, o in enumerate(opts, 1):
        desc = o.get("user_desc") or o.get("rationale") or ""
        lines.append(f"  选项{i}: {desc}")
    return "\n".join(lines)


# Plain-language descriptions for rule-fallback options (no parameter names).
USER_DESC_MAP = {
    "peak_distance_ms": "让 beat 分割更齐，减少过碎/误切的分割",
    "peak_prominence": "提高峰检测灵敏度，找回漏检的 beat",
    "min_segment_ms": "放宽最短片段限制，保留更多有效 beat",
    "min_beat_len": "放宽 beat 长度下限，避免误丢短 beat",
    "max_beat_len_ms": "放宽 beat 长度上限，避免误丢长 beat",
    "validation_mode": "放宽 beat 验证标准，减少误剔除",
    "resegment_min_ms": "调整片段合并阈值，让分割更连续",
    "onset_search_ms": "扩大起始点搜索范围，让 o 点标得更准",
    "onset_margin_ms": "微调起始点余量，让 o 点贴紧谷点",
    "onset_rise_check_ms": "调整上升沿校验窗口，o 点更稳定",
    "diastolic_search_ms": "扩大重搏切迹搜索范围，补上 dn 特征点",
    "notch_depth_threshold": "放宽重搏切迹深度要求，更容易标到 dn",
    "extra_bandpass": "加强滤波去噪，波形更平滑、毛刺更少",
    "mad_reject_k": "剔除振幅离群的 beat，波形更整齐、指标更稳",
}


class ParamOption(BaseModel):
    param: str
    value: Any
    rationale: str
    expected_effect: str
    user_desc: str = ""  # 面向用户的通俗描述（用户看图后能理解的选择项）


class AdaptProposeOptions(BaseModel):
    summary: str
    options: list[ParamOption]  # 2-4 options, one parameter change each


@dataclass
class SampleResult:
    subject_id: str
    seg_idx: int
    name: str
    metrics: dict
    figures: dict = field(default_factory=dict)
    waveform: dict | None = None   # unified ECharts payload (replaces PNG)


@dataclass
class AdaptRound:
    idx: int
    params: dict
    samples: list[SampleResult]
    change: dict | None = None
    feedback: str | None = None
    diagnosis: dict | None = None


def _clean_json(value):
    """Recursively replace non-finite floats (NaN/inf) with None — FastAPI
    rejects them in JSON responses (500: 'Out of range float values are not
    JSON compliant')."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    return value


def _subject_pool(source: str) -> list[str]:
    with phenotype_conn() as conn:
        rows = conn.execute(
            "SELECT subject_id FROM subjects WHERE source=? ORDER BY subject_id",
            (source,),
        ).fetchall()
    return [r["subject_id"] for r in rows]


def _build_waveform_payload(ppg: np.ndarray, result: dict) -> dict:
    """Unified ECharts payload replacing the segmentation PNG: waveform
    points (downsampled for display) + beat overlay (boundaries, keypoints,
    validity) + per-beat feature dicts for the feature-curve chart."""
    from ppg_agent.engine.overlay import waveform_payload
    return waveform_payload(ppg, result["beat_starts"], result["valid_mask"],
                            result["keypoints"], result["beats"], FS,
                            max_points=2000,
                            features_list=result.get("features"))


class DatasetAdaptation:
    """One active dataset-adaptation session (per source, in-memory).

    The single session engine for dataset-level parameter tuning: a few
    sample segments per iteration, the LLM proposes 2-4 concrete
    parameter-change options, the user picks one (option-list HITL), and
    figures are re-rendered after each applied change.  The LLM never
    applies a parameter change directly — options are the only way to
    mutate params, so a misread free-text feedback can at worst produce a
    bad option, not a bad edit.

    Preset hygiene: user adaptations are saved as ``user_<source>`` and may
    never overwrite system presets (``default`` is locked; source-mapped
    presets like ``aurora`` are reserved for official adaptation).
    ``cancel()`` removes the user preset entirely when asked.
    """

    def __init__(self, source: str, seed: int = 42):
        self.source = source
        self.params: dict = dict(DEFAULT_PARAMS)
        self.rounds: list[AdaptRound] = []
        self.status: str = "active"
        self._pending_options: list[ParamOption] = []
        self._pending_feedback: str | None = None
        self._rng = random.Random(seed)
        self._pool: list[str] = _subject_pool(source)

    # ------------------------------------------------------------------
    # Public API (called by adapt_pipeline / the deterministic layer)
    # ------------------------------------------------------------------

    def start(self) -> AdaptRound:
        """Baseline round with default params over N sample segments."""
        return self._new_round(None, None)

    def propose(self, feedback: str | None = None) -> tuple[list[dict], str]:
        """LLM proposes 2-4 parameter-change options for the user to pick.

        Feedback is *input* to the proposal, never a direct parameter edit —
        option-list HITL keeps the LLM from mutating params on a misread.
        """
        self._pending_feedback = feedback or None
        diag = self.rounds[-1].diagnosis if self.rounds else None
        options = self._propose_options(feedback, diag)
        self._pending_options = options
        if not options:
            return [], "未能提出合适的参数调整选项，请描述更具体一些"
        return [o.model_dump() for o in options], (
            f"根据你的反馈提出了 {len(options)} 个参数调整选项，请选择一项"
            if feedback else f"基于当前诊断提出 {len(options)} 个参数调整选项，请选择一项")

    def apply_choice(self, idx: int) -> tuple[AdaptRound, str]:
        """Apply the chosen pending option → new round with re-sampled figures."""
        if not self._pending_options:
            return self.rounds[-1], "当前没有待选选项，请先让 Agent 提议选项"
        if not 0 <= idx < len(self._pending_options):
            return self.rounds[-1], f"选项序号不合法（应为 1-{len(self._pending_options)}）"
        change = self._pending_options[idx]
        if validate_params({change.param: change.value}):
            return self.rounds[-1], f"参数 {change.param}={change.value} 不合法"
        old = self.params.get(change.param)
        self.params[change.param] = change.value
        change_dict = {"param": change.param, "old_value": old, "new_value": change.value,
                       "rationale": change.rationale, "expected_effect": change.expected_effect}
        rec = self._new_round(change_dict, self._pending_feedback)
        self._pending_options = []
        self._pending_feedback = None
        tag = "[B-class]" if PARAM_SPEC[change.param].cls == "B" else ""
        summary = f"已应用选项 {idx + 1}: {change.param} {old} → {change.value} {tag}\n{change.rationale}"
        return rec, summary

    def sample_more(self) -> AdaptRound:
        """Re-sample new segments with the current params (no change)."""
        return self._new_round(None, None)

    def save(self, name: str | None = None) -> dict:
        """Save user preset as user_<source> (never overwrites system presets)."""
        preset_name = (name or f"user_{self.source}").strip()
        if not preset_name.startswith("user_"):
            raise ValueError(f"用户适配预设必须以 user_ 前缀命名 (got '{preset_name}')")
        if preset_name in SOURCE_PRESET_MAP.values():
            raise ValueError(f"{preset_name} 是系统预留的预设名，请使用 user_ 前缀")
        info = save_preset(preset_name, self.params)
        self.status = "saved"
        return {"preset": preset_name, "path": info["path"], "params": self.params}

    def cancel(self, remove_preset: bool = False) -> dict:
        """Abort the session; optionally delete the user preset file."""
        if remove_preset:
            name = f"user_{self.source}"
            fp = MODELS_ROOT / "param_presets" / self.source / "user.yaml"
            if fp.exists():
                fp.unlink()
                removed = name
            else:
                removed = None
        else:
            removed = None
        self.status = "aborted"
        return {"status": "aborted", "removed_preset": removed}

    def existing_presets(self) -> list[str]:
        """Presets that represent prior adaptation of this source.

        ``default`` (and other official presets) is the system baseline — it is
        NOT "prior adaptation", so it never appears here; only source-mapped
        official presets (e.g. ``aurora``) and ``user_<source>`` count.
        """
        names = []
        mapped = SOURCE_PRESET_MAP.get(self.source)
        if mapped and mapped != "default" and mapped in list_presets():
            names.append(mapped)
        user = f"user_{self.source}"
        if user in list_presets():
            names.append(user)
        return names

    def state_dict(self) -> dict:
        """Structured state for the chat response / frontend rendering.

        NaN/inf metrics (e.g. CV of a 1-beat sample → 0/0) must be cleaned:
        FastAPI rejects non-finite floats with
        "Out of range float values are not JSON compliant".
        """
        rounds = []
        for r in self.rounds:
            rounds.append({
                "idx": r.idx,
                "change": _clean_json(r.change),
                "feedback": r.feedback,
                "samples": [{"subject_id": s.subject_id, "seg_idx": s.seg_idx,
                             "name": s.name, "metrics": _clean_json(s.metrics),
                             "figures": s.figures, "waveform": s.waveform}
                            for s in r.samples],
            })
        return {"source": self.source, "status": self.status, "params": self.params,
                "rounds": rounds, "existing_presets": self.existing_presets(),
                "pending_options": [o.model_dump() for o in self._pending_options],
                "preset_dir": str(MODELS_ROOT / "param_presets")}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_round(self, change: dict | None, feedback: str | None) -> AdaptRound:
        idx = len(self.rounds)
        samples: list[SampleResult] = []
        first_result = None
        for i in range(N_SAMPLES):
            # deterministic rotation over the subject pool
            subj = self._pool[(idx * N_SAMPLES + i) % len(self._pool)]
            seg_idx = i % 3
            try:
                sample = load_sample_segment(self.source, subj, seg_idx)
                result = process_ppg_segment(sample["ppg_500"], FS,
                                             params=self.params)
                if first_result is None:
                    first_result = result
                metrics = segment_metrics(result)
                waveform = _build_waveform_payload(sample["ppg_500"], result)
            except Exception as e:  # noqa: BLE001 — keep the round alive
                metrics = {"error": str(e)[:120]}
                waveform = None
            samples.append(SampleResult(subject_id=subj, seg_idx=seg_idx,
                                        name=f"{self.source}/{subj}/seg{seg_idx}",
                                        metrics=metrics, waveform=waveform))
        diag = diagnose(first_result) if first_result is not None else None
        rec = AdaptRound(idx=idx, params=dict(self.params),
                         samples=samples, change=change, feedback=feedback,
                         diagnosis=diag)
        self.rounds.append(rec)
        return rec

    def _propose_options(self, feedback: str | None, diag: dict | None) -> list[ParamOption]:
        """LLM proposes 2-4 options; validated + deduped; rule fallback on failure.

        The user speaks in *visual* terms ("beats not on the valleys", "no
        keypoint on the dicrotic notch"); the LLM maps that to a parameter
        direction and writes ``user_desc`` in plain, user-facing language so
        the options are easy to pick while looking at the waveform.
        """
        options: list[ParamOption] = []
        try:
            llm = get_llm("signal_processor")
            llm_tool = llm.bind_tools([AdaptProposeOptions])
            param_lines = "\n".join(
                f"  {k}: {v} ({PARAM_SPEC[k].cls}-class: {PARAM_SPEC[k].desc})"
                for k, v in self.params.items())
            diag_lines = ""
            if diag:
                issues = diag.get("issues", [])
                if issues:
                    diag_lines = "\n".join(f"  - {i.get('severity','')}: {i.get('message','')}"
                                           for i in issues)
            prompt = (
                f"用户反馈（用户在看波形图时提出的观察）：{feedback or '(未提供，请基于诊断提出常规改进选项)'}\n"
                f"当前参数：\n{param_lines}\n"
                f"当前诊断：\n{diag_lines or '  (无异常)'}\n"
                "任务：用户的反馈是看图后的直观描述（如 beat 没分割在谷点、重搏切迹上"
                "没有特征点、波形毛刺多等）。先把反馈推理为对应的参数调节方向（分割类/"
                "特征点类/滤波开关），再提出 2-4 个参数调整选项，每个选项只改一个参数。\n"
                "每个选项的 user_desc 必须用通俗语言描述：用户能看懂\"选了之后波形/特征点"
                "会变成什么样\"（例如\"让 beat 分割更齐，减少误切\"、\"在重搏切迹处补上"
                "特征点标注\"、\"滤掉毛刺让波形更平滑\"）。不要出现参数名。"
            )
            ai = llm_tool.invoke([{"role": "user", "content": prompt}])
            if ai.tool_calls:
                args = ai.tool_calls[0]["args"]
                parsed = AdaptProposeOptions(**args)
                seen: set[str] = set()
                for o in parsed.options[:6]:
                    if o.param not in PARAM_SPEC or o.param in seen:
                        continue
                    if validate_params({o.param: o.value}):
                        continue  # invalid value — drop the option
                    seen.add(o.param)
                    if not o.user_desc:
                        o.user_desc = f"{o.param} → {o.value}"
                    options.append(o)
        except Exception:  # noqa: BLE001 — fall back to rules
            options = []
        if len(options) < 2:
            rule = self._rule_change(feedback)
            if rule is not None:
                options = [ParamOption(param=rule.param, value=rule.value,
                                       rationale=rule.rationale,
                                       expected_effect=rule.expected_effect,
                                       user_desc=USER_DESC_MAP.get(
                                           rule.param, f"{rule.param} → {rule.value}"))]
        return options[:4]

    def _rule_change(self, feedback: str | None) -> ParamChange | None:
        """Keyword-based fallback when the LLM is unavailable."""
        f = feedback or ""
        if any(k in f for k in ("分割", "beat", "拍", "分段", "peak")) and \
                any(k in f for k in ("多", "重复", "过细", "碎")):
            return ParamChange(param="peak_distance_ms",
                               value=min(int(self.params["peak_distance_ms"]) + 50, 600),
                               rationale="用户反馈 beat 过多/过碎", expected_effect="减少分割")
        if any(k in f for k in ("分割", "beat", "拍", "分段", "漏")):
            return ParamChange(param="peak_prominence",
                               value=max(round(self.params["peak_prominence"] - 0.01, 2), 0.01),
                               rationale="用户反馈 beat 漏检", expected_effect="检出更多 beat")
        if any(k in f for k in ("特征点", "onset", "起始", "起点", "关键")):
            return ParamChange(param="onset_search_ms", value=160,
                               rationale="用户反馈特征点位置不准", expected_effect="更宽搜索窗")
        if any(k in f for k in ("重搏", "notch", "双峰", "dicrotic")):
            return ParamChange(param="diastolic_search_ms", value=250,
                               rationale="用户反馈重搏波特征未找到", expected_effect="更宽搜索窗")
        if any(k in f for k in ("噪声", "毛刺", "抖动", "baseline", "基线")):
            return ParamChange(param="extra_bandpass", value=True,
                               rationale="用户反馈噪声大", expected_effect="滤波去噪")
        if not f:
            return ParamChange(param="mad_reject_k", value=2.5,
                               rationale="默认兜底选项: 过滤振幅离群 beat",
                               expected_effect="降低 CV、提升质量")
        return None


def adapt_pipeline(action: str = "status", source: str = "",
                   feedback: str | None = None, choice_idx: int | None = None,
                   remove_preset: bool = False) -> dict:
    """数据集参数适配会话（Agent 1 适配分支与 web 确定性层共用的单工具入口）。

    action:
      start   — 启动会话：采样轮 + 诊断 + 初始选项（已有活跃会话则幂等返回）
      propose — 根据用户反馈重新提议 2-4 个选项（绝不直接修改参数）
      apply   — 应用用户选择的选项（choice_idx 从 1 开始）
      sample  — 换一批 segment 采样（参数不变）
      save    — 保存为 user_<source> 预设（绝不覆盖系统预设）
      cancel  — 中止会话；remove_preset=True 同时删除已保存的用户预设
      status  — 返回当前会话状态（不推进）

    返回 {"status", "message", "state"}：
      status ∈ active / saved / aborted / idle / error
      state  — 前端渲染用的 state_dict（仅在存在会话时返回）。
    会话按 source 存于服务端（跨请求/跨图调用保留）。
    """
    def _ret(dapt: DatasetAdaptation, message: str) -> dict:
        return {"status": dapt.status, "message": message,
                "state": dapt.state_dict()}

    if action == "status":
        dapt = get_adapt_session(source)
        if dapt is None:
            return {"status": "idle", "message": "当前没有适配会话。",
                    "state": None}
        return _ret(dapt, _adapt_summary(dapt, f"{dapt.source} 适配会话 (第 {len(dapt.rounds)} 轮):"))

    if action == "start":
        if not source:
            return {"status": "error", "message": "请先指定要适配的数据集 (source)。",
                    "state": None}
        data_dir = raw_data_dir(source)
        if not Path(data_dir).exists():
            return {"status": "error",
                    "message": (f"数据集 {source} 的原始数据目录不存在: {data_dir}。"
                                f"无法进行适配，请先检查数据是否已上传/挂载。"),
                    "state": None}
        dapt = active_adapt_session(source)
        if dapt is not None:
            _touch(source)
            return _ret(dapt, (f"数据集 {source} 的适配会话已在运行中 "
                               f"(第 {len(dapt.rounds)} 轮)。请基于用户反馈继续。"))
        dapt = DatasetAdaptation(source)
        _SESSIONS[source] = dapt
        _touch(source)
        existing = dapt.existing_presets()
        rec = dapt.start()
        # server-side guarantee: options always exist after start, so the LLM
        # cannot claim proposed options that aren't actually selectable
        options, _msg = dapt.propose()
        head = (f"数据集 {source} 适配会话已开始 (默认参数), 采样 {len(rec.samples)} 段, "
                f"已基于诊断提出 {len(options)} 个参数调整选项:\n")
        for i, o in enumerate(options, 1):
            desc = o.get("user_desc") or o.get("rationale") or ""
            head += f"  选项{i}: {desc}\n"
        head += _adapt_summary(dapt, "采样概况:")
        head = head.rstrip()
        if existing:
            head += (f"\n该数据集已有适配参数: {existing}. "
                     f"请询问用户是否要基于现有适配继续或重新适配.")
        return _ret(dapt, head)

    dapt = active_adapt_session(source)
    if dapt is None:
        return {"status": "idle",
                "message": "还没有进行中的适配会话，请先调用 action='start'。",
                "state": None}
    _touch(dapt.source)

    if action == "propose":
        options, msg = dapt.propose(feedback)
        if not options:
            return _ret(dapt, msg)
        return _ret(dapt, _options_text(dapt, msg))
    if action == "apply":
        if choice_idx is None:
            return {"status": "error", "message": "缺少 choice_idx（用户选择的选项序号，从 1 开始）。",
                    "state": dapt.state_dict()}
        rec, msg = dapt.apply_choice(choice_idx - 1)
        return _ret(dapt, f"{msg}. 新参数下采样 {len(rec.samples)} 段:\n"
                          + _adapt_summary(dapt, "采样概况:"))
    if action == "sample":
        rec = dapt.sample_more()
        return _ret(dapt, f"已更换采样, 当前参数下 {len(rec.samples)} 段:\n"
                          + _adapt_summary(dapt, "采样概况:"))
    if action == "save":
        info = dapt.save()
        return _ret(dapt, (f"已保存用户适配预设: {info['preset']} (path: {info['path']}). "
                           f"该数据集的后续分析与批量提取将自动使用此预设, 无需手动指定. "
                           f"如需撤销, 可删除 {info['preset']} 预设后重新适配."))
    if action == "cancel":
        info = dapt.cancel(remove_preset=remove_preset)
        if info["removed_preset"]:
            return _ret(dapt, f"适配会话已中止, 用户预设 {info['removed_preset']} 已删除")
        return _ret(dapt, "适配会话已中止, 未保存任何参数")
    return {"status": "error",
            "message": f"Unknown action '{action}'. "
                       "Use start / propose / apply / sample / save / cancel / status.",
            "state": None}
