"""Personal longitudinal comparison engine (Agent 2 Phase 1).

All deterministic statistics for the three-layer comparison (beat →
segment → person) live here.  Core semantics (user-confirmed):

* when analysing segment k: **long-term history** = segments 0..k-1,
  **recent history** = the preceding ``recent_k`` segments
  (default = ppg_model causal_attn 流式头 K，manifest 读取，回退 32 ——
  v3.2 口径对齐：两侧引用同一时间尺度)，
  and the current segment never enters its own history;
* k == 0            → ``no_history`` (population reference only);
* 历史门限与 ppg_model 完全对齐（v3.2）：有历史即判定（不再 ≥3 段硬门），
  历史不足时以 ``limited_history`` / ``insufficient_power`` / ``power_note``
  如实标注统计效力下降；ρ 类统计 <20 段照算但标注。
* failed-quality segments are dropped from history.

Fixed design decisions:
① beat statistics use valid beats only (feature non-null), valid_ratio
   reported as a quality note;
② (v3.2 修订) 原「compare 需 ≥3 历史段」硬门移除——门限与 ppg_model 流式头
   对齐（0 历史 → e_k-only / no_history；ρ 类统计效力门 20 保留为标注）;
③ trend: Theil-Sen slope + p plus a plain persistence sentence;
④ beat×BP coupling is de-confounded (beat value minus its segment-level
   mean) with a pooled variant kept as contrast;
⑤ returns plain statistic dicts + verdict enums — narrative text is the
   Agent layer's job.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ppg_agent.config import beat_features_path, segment_features_path

# Segment-level feature columns carry a ``_mean`` suffix; these outcome /
# label columns are not PPG features and stay out of comparisons.
_NON_FEATURE_MEANS = {"abp_sbp_mean", "abp_dbp_mean"}
_MIN_COMPARE_SEGS = 3      # (v3.2) 不再作比较硬门；保留为「可靠基准」注记边界
_MIN_RHO_SEGS = 20         # ρ 类统计效力门（Phase 2-2/3 证据），超标降为标注


def _causal_recent_k() -> int:
    """近期窗宽 = ppg_model causal_attn 流式头 K（manifest 读取，回退 32）。

    口径对齐纪律：Agent 2 的近期/长期历史结构与流式头完全一致
    （近 K 段 + 全量前缀），保证两侧叙事引用同一时间尺度。"""
    try:
        import json as _json
        from .causal_stream import EMB_ROOT
        return int(_json.loads((EMB_ROOT / "manifest.json").read_text())
                   .get("K")) or 32
    except Exception:  # noqa: BLE001 — 产物缺失时回退默认窗宽
        return 32


_ICC_POPULATION_GATE = 0.6  # population-position gate
# percentile-band ordinal for the population-drift contrast (query() bands)
_BAND_ORDINAL = {"P5-": 0, "P5-P25": 1, "P25-P75": 2,
                 "P75-P85": 3, "P85-P95": 4, "P95+": 5}
_MAD_SCALE = 1.4826        # MAD → σ conversion
_OUTLIER_K = 3.0           # median ± 3σ outlier fence
_DRIFT_MIN_EXCEED = 5      # >=5 of the recent 7 segments → drift

# Data-quality gate (2026-08-24, mirrors the step-01 materialization):
# obviously-wrong BP labels are masked to NaN at load time so they never
# enter personal ρ, bp_context or beat×BP coupling.  Raw parquet stays
# untouched — the gate lives in the consumer.
_BP_RANGE = {  # column candidates → (lo, hi) plausibility fence
    "sbp_label": (50.0, 250.0), "abp_sbp": (50.0, 250.0),
    "dbp_label": (30.0, 160.0), "abp_dbp": (30.0, 160.0),
}


def _sanitize_bp(df: pd.DataFrame) -> pd.DataFrame:
    """Mask implausible BP labels (out-of-range or DBP>SBP) to NaN."""
    sbp_col = next((c for c in ("sbp_label", "abp_sbp") if c in df.columns), None)
    dbp_col = next((c for c in ("dbp_label", "abp_dbp") if c in df.columns), None)
    for col in (c for c in (sbp_col, dbp_col) if c):
        lo, hi = _BP_RANGE[col]
        v = df[col]
        df[col] = v.mask((v < lo) | (v > hi))
    if sbp_col and dbp_col:
        s, d = df[sbp_col], df[dbp_col]
        df[dbp_col] = d.mask(s.notna() & d.notna() & (d > s))
    return df


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Robust Theil-Sen slope + Kendall-tau trend p; None on degenerate input."""
    if len(x) < 3:
        return None, None
    from scipy.stats import kendalltau, theilslopes
    try:
        slope = float(theilslopes(y, x)[0])
        p = float(kendalltau(x, y).pvalue)
    except Exception:  # noqa: BLE001
        return None, None
    if not np.isfinite(slope):
        return None, None
    return slope, (p if np.isfinite(p) else None)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return None
    from scipy.stats import spearmanr
    r = spearmanr(x[m], y[m]).correlation
    return float(r) if np.isfinite(r) else None


class PersonalEngine:
    """Subject-level longitudinal statistics over features_v2 parquet.

    Reuses :class:`PercentileEngine` knowledge (MDC / ICC / recommended
    window / stratified percentiles) without touching its query surface.
    """

    def __init__(self, percentile_engine=None):
        if percentile_engine is None:
            from ppg_agent.engine.percentile_engine import PercentileEngine
            percentile_engine = PercentileEngine()
        self.pe = percentile_engine
        self._seg_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._beat_cache: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Data layer
    # ------------------------------------------------------------------

    @staticmethod
    def _source(ds: str, sid: str) -> str:
        """Knowledge-source key; aurora splits by scenario (id prefix)."""
        if ds in ("aurora", "aurorabp"):
            sc = (sid[:1] or "").lower()
            return f"aurora_{sc}" if sc in ("a", "o") else "aurora"
        return ds

    def load_subject_segments(self, ds: str, sid: str) -> pd.DataFrame:
        """All segments of a subject, ordered by seg_idx (fresh + cached)."""
        key = (ds, sid)
        if key not in self._seg_cache:
            path = segment_features_path(ds)
            t = pq.read_table(path, filters=[("subject_id", "=", sid)])
            df = t.to_pandas().sort_values("seg_idx").reset_index(drop=True)
            self._seg_cache[key] = _sanitize_bp(df)
        return self._seg_cache[key]

    def load_subject_beats(self, ds: str, sid: str) -> pd.DataFrame:
        """All beats of a subject from the merged wide table (cached)."""
        key = (ds, sid)
        if key not in self._beat_cache:
            path = beat_features_path(ds)
            t = pq.read_table(path, filters=[("subject_id", "=", sid)])
            self._beat_cache[key] = _sanitize_bp(t.to_pandas())
        return self._beat_cache[key]

    @staticmethod
    def feature_mean_columns(segs: pd.DataFrame) -> list[str]:
        """Segment-level PPG feature columns (``*_mean``, outcomes excluded)."""
        return [c for c in segs.columns
                if c.endswith("_mean") and c not in _NON_FEATURE_MEANS]

    @staticmethod
    def _valid_mask(beats: pd.DataFrame) -> pd.Series:
        """A beat is valid when it carries extracted features (决策①)."""
        if "lasi" in beats.columns:
            return beats["lasi"].notna()
        return pd.Series(True, index=beats.index)

    # ------------------------------------------------------------------
    # PersonalProfile — long-term history baseline
    # ------------------------------------------------------------------

    def build_profile(self, ds: str, sid: str,
                      history_seg_idxs: list[int] | None = None) -> dict:
        """Robust per-feature baseline from long-term history.

        ``history_seg_idxs=None`` → every usable segment of the subject.
        """
        segs = self.load_subject_segments(ds, sid)
        if history_seg_idxs is not None:
            segs = segs[segs["seg_idx"].isin(set(history_seg_idxs))]
        segs = segs[segs.get("quality_tier", "good") != "failed"]
        n = len(segs)
        prof: dict = {
            "subject_id": sid, "dataset": ds,
            "n_segments": n,
            "n_good": int((segs.get("quality_tier") == "good").sum())
            if "quality_tier" in segs else n,
            "seg_range": [int(segs["seg_idx"].min()), int(segs["seg_idx"].max())]
            if n else None,
            "features": {}, "trends": {}, "rho_bp": {},
        }
        if n == 0:
            return prof
        cols = self.feature_mean_columns(segs)
        feats = {}
        for c in cols:
            v = segs[c].dropna().to_numpy(dtype=float)
            if len(v) < _MIN_COMPARE_SEGS:
                continue
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            cv = mad / med if med not in (0, 0.0) else None
            feats[c] = {"n": len(v), "median": round(med, 4),
                        "mad": round(mad, 4), "cv": round(cv, 4) if cv is not None else None}
        prof["features"] = feats

        x = segs["seg_idx"].to_numpy(dtype=float)
        for c in cols:
            if c not in feats:
                continue
            y = segs[c].to_numpy(dtype=float)
            m = np.isfinite(y)
            slope, p = _theil_sen(x[m], y[m])
            if slope is None:
                continue
            # persistence: of the last 5 segments, how many sit above median
            tail = segs[c].dropna().tail(5)
            med = feats[c]["median"]
            above = int((tail > med).sum())
            prof["trends"][c] = {
                "theil_sen_slope": round(slope, 6), "p": p,
                "persistence": f"最近{len(tail)}段中{above}段高于个人中位数",
            }

        # personal within-subject ρ(feature, BP label), >=10 labelled segs
        if "sbp_label" in segs.columns:
            lab = segs[["sbp_label"]].copy()
            labelled = segs[segs["sbp_label"].notna()]
            if len(labelled) >= _MIN_RHO_SEGS:
                for c in cols:
                    sub = labelled[[c, "sbp_label"]].dropna()
                    r = _spearman(sub[c].to_numpy(dtype=float),
                                  sub["sbp_label"].to_numpy(dtype=float))
                    if r is not None:
                        prof["rho_bp"][c] = {"rho": round(r, 3), "n": len(sub)}
            else:
                prof["rho_bp_reason"] = (
                    f"labelled segments {len(labelled)} < {_MIN_RHO_SEGS}")
        return prof

    # ------------------------------------------------------------------
    # compare_segment — dual-reference change detection (axis ①+②)
    # ------------------------------------------------------------------

    def compare_segment(self, ds: str, sid: str, seg_idx: int,
                        recent_k: int | None = None) -> dict:
        """Compare segment ``seg_idx`` against its own past only.

        v3.2 口径对齐：recent_k 默认取 ppg_model 流式头 K；有历史即判定
        （不再 ≥3 段硬门），历史不足以 insufficient_power/power_note 标注。"""
        if recent_k is None:
            recent_k = _causal_recent_k()
        segs = self.load_subject_segments(ds, sid)
        cur_rows = segs[segs["seg_idx"] == seg_idx]
        if cur_rows.empty:
            return {"verdict": "segment_not_found", "seg_idx": seg_idx}
        cur = cur_rows.iloc[0]
        hist_all = segs[(segs["seg_idx"] < seg_idx)
                        & (segs.get("quality_tier", "good") != "failed")]
        n_hist = len(hist_all)
        if n_hist == 0:
            return {"verdict": "no_history", "seg_idx": seg_idx,
                    "hint": "首段无个人参照，仅可给人群参考"}
        recent = hist_all.tail(min(recent_k, n_hist))
        limited = n_hist < recent_k
        src = self._source(ds, sid)
        cols = self.feature_mean_columns(segs)

        out: dict = {
            "verdict": "ok", "seg_idx": seg_idx,
            "n_history": n_hist, "n_recent": len(recent),
            "recent_k": recent_k,
            "limited_history": limited,
            # v3.2：不再拦截，保留为注记；效力门 = _MIN_RHO_SEGS
            "insufficient_history": n_hist < _MIN_COMPARE_SEGS,
            "insufficient_power": n_hist < _MIN_RHO_SEGS,
            "power_note": (f"历史仅{n_hist}段(<{_MIN_RHO_SEGS})，判定效力"
                           "有限，与 ppg_model 流式头 0 历史降级语义对齐："
                           "照常判定但需降权解读")
            if n_hist < _MIN_RHO_SEGS else None,
            "features": {}, "drift": [],
        }
        for c in cols:
            cur_v = cur.get(c)
            if cur_v is None or not np.isfinite(cur_v):
                continue
            hv = hist_all[c].dropna().to_numpy(dtype=float)
            if len(hv) < 1:
                continue
            med = float(np.median(hv))
            mad = float(np.median(np.abs(hv - med)))
            mdc_ref = self.pe._mdc(src, c.removesuffix("_mean")) or {}
            mdc = mdc_ref.get("mdc")
            # personal MAD fallback: 3σ of the personal distribution
            thr = float(mdc) if mdc else _OUTLIER_K * _MAD_SCALE * mad
            if thr <= 0:
                # MAD 退化（历史极少且无 MDC 参照）→ 1% 相对地板，
                # 避免 0 阈值把任何噪声判成 real_change
                thr = 0.01 * abs(med)
            delta = float(cur_v) - med
            v_long = "real_change" if abs(delta) > thr else "stable"
            # recent reference
            rv = recent[c].dropna().to_numpy(dtype=float)
            v_recent = None
            if len(rv) >= 1:
                rmed = float(np.median(rv))
                v_recent = ("real_change"
                            if abs(float(cur_v) - rmed) > thr else "stable")
            out["features"][c] = {
                "value": round(float(cur_v), 4),
                "personal_median": round(med, 4),
                "delta_vs_long": round(delta, 4),
                "delta_vs_recent": round(float(cur_v) - rmed, 4) if len(rv) else None,
                "threshold": round(float(thr), 4),
                "threshold_source": "mdc_reference" if mdc else "personal_mad",
                "verdict_long": v_long, "verdict_recent": v_recent,
                **({"insufficient_power": True}
                   if len(hv) < _MIN_RHO_SEGS else {}),
            }
            # drift: >=5 of recent segments beyond the long-term fence
            over = int((np.abs(rv - med) > thr).sum()) if len(rv) else 0
            if len(rv) >= recent_k and over >= _DRIFT_MIN_EXCEED:
                out["drift"].append({"feature": c, "exceeded": over,
                                     "of": len(rv)})

        out["beat_layer"] = self._compare_beats(ds, sid, seg_idx, cols)
        if not out["beat_layer"].get("available"):
            out["verdict"] = "current_segment_low_quality"
            out["hint"] = ("当前段有效 beat 过少，仅段级特征可对照："
                           + out["beat_layer"].get("reason", ""))
        n_chg = sum(1 for f in out["features"].values()
                    if f["verdict_long"] == "real_change")
        out["n_real_change"] = n_chg
        return out

    def _compare_beats(self, ds: str, sid: str, seg_idx: int,
                       cols: list[str]) -> dict:
        """Axis ①: new segment's beat distribution vs personal history."""
        try:
            beats = self.load_subject_beats(ds, sid)
        except FileNotFoundError:
            return {"available": False, "reason": "no beat table"}
        beat_cols = [c.removesuffix("_mean") for c in cols
                     if c.removesuffix("_mean") in beats.columns]
        cur_b = beats[beats["seg_idx"] == seg_idx]
        hist_b = beats[beats["seg_idx"] < seg_idx]
        if cur_b.empty or hist_b.empty or not beat_cols:
            return {"available": False, "reason": "no beats to compare"}
        cur_valid = cur_b[self._valid_mask(cur_b)]
        hist_valid = hist_b[self._valid_mask(hist_b)]
        if cur_valid.empty or hist_valid.empty:
            return {"available": False, "reason": "no beats to compare"}
        # v3.2 对齐：拍层历史不再 30 拍硬门，有历史即对照，
        # 不足 30 拍以 insufficient_power 标注（效力下降，非不可用）
        power_ok = len(hist_valid) >= 30

        res: dict = {
            "available": True,
            "n_beats_cur": int(len(cur_b)),
            "n_valid_cur": int(len(cur_valid)),
            "valid_ratio": round(len(cur_valid) / max(len(cur_b), 1), 3),
            "n_valid_hist": int(len(hist_valid)),
            "insufficient_power": not power_ok,
            "quantiles": {}, "outlier_rate": {},
        }
        rates = []
        for f in beat_cols:
            hv = hist_valid[f].dropna().to_numpy(dtype=float)
            cv = cur_valid[f].dropna().to_numpy(dtype=float)
            if len(hv) < 1 or len(cv) < 3:
                continue
            med = float(np.median(hv))
            mad = float(np.median(np.abs(hv - med)))
            res["quantiles"][f] = {
                "hist_p50": round(med, 4),
                "cur_p10": round(float(np.percentile(cv, 10)), 4),
                "cur_p50": round(float(np.percentile(cv, 50)), 4),
                "cur_p90": round(float(np.percentile(cv, 90)), 4),
            }
            if mad > 0:
                fence = _OUTLIER_K * _MAD_SCALE * mad
                rate = float((np.abs(cv - med) > fence).mean())
                res["outlier_rate"][f] = round(rate, 4)
                rates.append(rate)
        res["outlier_rate_median"] = round(float(np.median(rates)), 4) if rates else None
        # per-beat BP distribution (vital/mimic only)
        if "abp_sbp" in cur_b.columns:
            sbp = cur_b["abp_sbp"].dropna()
            if len(sbp):
                res["beat_bp"] = {
                    "sbp_median": round(float(sbp.median()), 1),
                    "sbp_iqr": [round(float(sbp.quantile(.25)), 1),
                                round(float(sbp.quantile(.75)), 1)],
                    "dbp_median": round(float(cur_b["abp_dbp"].dropna().median()), 1)
                    if cur_b["abp_dbp"].notna().any() else None,
                }
        return res

    # ------------------------------------------------------------------
    # population_position — axis ③ (gated, dual reference)
    # ------------------------------------------------------------------

    def subject_demographics(self, ds: str, sid: str
                             ) -> tuple[float | None, str | None]:
        """(age, gender) of the subject from the step-01 merged table —
        the SAME source the stratified baselines were built on, so the
        query key matches the reference distribution.  Lazy + cached per
        (ds); missing file or subject → (None, None) → positions fall
        back to the global baseline."""
        if not hasattr(self, "_demo_cache"):
            self._demo_cache: dict = {}
        if ds not in self._demo_cache:
            self._demo_cache[ds] = self._load_demographics(ds)
        return self._demo_cache[ds].get(sid, (None, None))

    @staticmethod
    def _load_demographics(ds: str) -> dict:
        import pyarrow.parquet as pq
        root = Path(__file__).parent.parent.parent / "analysis"
        path = (root / "07_unified_analysis" / "output" /
                "aurora_merged.parquet") if ds == "aurora" else (
                root / "01_data_preparation" / "output" /
                f"{ds}_merged.parquet")
        out: dict = {}
        try:
            t = pq.read_table(path, columns=["subject_id", "age", "gender"])
            df = t.to_pandas().groupby("subject_id", as_index=True).first()
            for sid, row in df.iterrows():
                age = row.get("age")
                gender = row.get("gender")
                # invalid ages (0, negatives, implausible) are treated as
                # MISSING — an invalid value in an age group is worse than
                # a fallback to the global baseline
                out[str(sid)] = (
                    float(age) if age is not None and np.isfinite(age)
                    and 0 < float(age) <= 120 else None,
                    str(gender) if isinstance(gender, str) and gender
                    else None)
        except Exception:  # noqa: BLE001 — optional knowledge, degrade
            pass
        return out

    # ------------------------------------------------------------------
    # age_direction — deterministic trend-consistency annotation
    # ------------------------------------------------------------------

    def _age_patterns(self) -> dict:
        """{source: {feature: {rho, n}}} from the interpretation KB's
        age_patterns — lazy, cached, degrades to {}.  Per-source matching
        is mandatory: the trend sign FLIPS between sources for some
        features (pulse_width_50, lasi)."""
        if getattr(self, "_age_kb", None) is None:
            self._age_kb = {}
            try:
                import json
                kb = json.loads((Path(__file__).parent.parent.parent
                                 / "knowledge"
                                 / "feature_interpretation.json")
                                .read_text())
                for feat, entry in kb.get("features", {}).items():
                    for a in entry.get("age_patterns", []):
                        src, rho = a.get("dataset"), a.get(
                            "rho_age_sex_adjusted")
                        if src and rho is not None:
                            self._age_kb.setdefault(src, {})[feat] = {
                                "rho": float(rho), "n": a.get("n")}
            except Exception:  # noqa: BLE001 — optional knowledge
                pass
        return self._age_kb

    def _age_direction(self, src: str, f: str,
                       band: str | None) -> dict | None:
        """Fixed-wording annotation: does the subject's population band
        sit in the direction the feature tends to move with age in THIS
        source?  Cross-sectional and weak (|ρ| ≲ 0.3) — describes where
        the value sits; never explains the subject's own drift over time
        and never supports a vascular-age estimate."""
        pat = self._age_patterns().get(src, {}).get(f)
        b = _BAND_ORDINAL.get(band or "")
        if pat is None or b is None:
            return None
        rho, n = pat["rho"], pat.get("n")
        direction = "随年龄升高" if rho > 0 else "随年龄降低"
        if b >= 3:                      # P75+ — sits high
            tag = "consistent" if rho > 0 else "opposite"
        elif b <= 1:                    # P25- — sits low
            tag = "consistent" if rho < 0 else "opposite"
        else:                           # P25-P75 — center: nothing to check
            tag = "not_applicable"
        side = "高" if b >= 3 else "低"
        stmt = {
            "consistent": (f"位于同龄组 {band}（偏{side}），与该特征"
                           f"{direction}的人群趋势方向一致"),
            "opposite": (f"位于同龄组 {band}（偏{side}），与该特征"
                         f"{direction}的人群趋势方向相反"),
            "not_applicable": (f"位于同龄组 {band}（居中），年龄趋势"
                               "方向对照不适用"),
        }[tag]
        return {"trend": direction, "rho_age_sex_adjusted": round(rho, 3),
                "n": n, "consistency": tag, "statement": stmt,
                "_provenance": "cross-sectional age trend (age+sex "
                               "adjusted rank ρ, weak effect)"}

    def population_position(self, ds: str, sid: str, seg_idx: int,
                            age: float | None = None,
                            gender: str | None = None,
                            recent_k: int | None = None) -> dict:
        """Dual-reference population positioning against the per-patient-
        median reference distribution (built on the train split).

        * main position: the subject's PERSONAL MEDIAN over the long-term
          history (all preceding segments, failed dropped) — matches the
          reference granularity; gated on ICC(2,k) >= 0.6 (reliability of
          the k-segment median), so low-ICC features unlock as history
          accumulates;
        * recent contrast: the median of the preceding ``recent_k``
          segments, reported only when its own ICC(2,k_recent) >= 0.6;
          features whose recent band sits >=1 band away from the long-term
          band are flagged in ``population_drift``;
        * history < 3 segments: falls back to single-segment positioning
          gated on ICC(1,1) >= 0.6, flagged ``basis='single_segment'`` —
          preserves the "population reference only" mode for new subjects.

        v3.2 alignment: with >=1 history segment the comparison always runs
        (no hard minimum-history gate); ``recent_k`` defaults to the causal
        streaming-head K (manifest), keeping Agent 2 on the same time scale
        as Agent 3.
        """
        if recent_k is None:
            recent_k = _causal_recent_k()
        segs = self.load_subject_segments(ds, sid)
        cur_rows = segs[segs["seg_idx"] == seg_idx]
        if cur_rows.empty:
            return {"verdict": "segment_not_found", "seg_idx": seg_idx}
        cur = cur_rows.iloc[0]
        hist = segs[(segs["seg_idx"] < seg_idx)
                    & (segs.get("quality_tier", "good") != "failed")]
        n_hist = len(hist)
        recent = hist.tail(min(recent_k, n_hist))
        src = self._source(ds, sid)
        # same-age/gender-group positioning: auto-fill the subject's
        # demographics so positions match the stratified baselines
        if age is None or gender is None:
            d_age, d_gender = self.subject_demographics(ds, sid)
            age = age if age is not None else d_age
            gender = gender if gender is not None else d_gender
        out: dict = {
            "seg_idx": seg_idx, "n_history": n_hist,
            "age": age, "gender": gender,
            "age_stratified": age is not None,
            "basis": ("personal_median" if n_hist >= _MIN_COMPARE_SEGS
                      else "single_segment"),
            "positions": {}, "recent_positions": {}, "population_drift": [],
            "suppressed": {},
        }
        for c in self.feature_mean_columns(segs):
            f = c.removesuffix("_mean")
            icc1 = self.pe._icc(src, f)
            if icc1 is None:
                out["suppressed"][f] = "no ICC decomposition for this feature"
                continue
            if n_hist >= _MIN_COMPARE_SEGS:
                hv = hist[c].dropna().to_numpy(dtype=float)
                if len(hv) < _MIN_COMPARE_SEGS:
                    out["suppressed"][f] = "too few usable history values"
                    continue
                k = len(hv)
                icc2 = self.pe._icc_2k(src, f, k)
                if icc2 is None or icc2 < _ICC_POPULATION_GATE:
                    out["suppressed"][f] = (
                        f"median reliability ICC(2,{k})="
                        f"{round(icc2, 3) if icc2 is not None else 'n/a'} "
                        f"< {_ICC_POPULATION_GATE}")
                    continue
                v, basis, rel = float(np.median(hv)), "personal_median", icc2
            else:
                cv = cur.get(c)
                if cv is None or not np.isfinite(cv):
                    continue
                if icc1 < _ICC_POPULATION_GATE:
                    out["suppressed"][f] = (
                        f"single-segment ICC {round(icc1, 3)} "
                        f"< {_ICC_POPULATION_GATE} and history "
                        f"{n_hist} < {_MIN_COMPARE_SEGS}")
                    continue
                v, basis, rel = float(cv), "single_segment", icc1
            q = self.pe.query_stratified(f, v, source=src,
                                         age=age, gender=gender)
            if "error" in q:
                out["suppressed"][f] = q.get("error", "query failed")
                continue
            q.update({"value_basis": basis, "k_used": int(k if basis
                                                           == "personal_median"
                                                           else 1),
                      "reliability_used": round(rel, 3)})
            out["positions"][f] = q
            ad = self._age_direction(src, f, q.get("percentile"))
            if ad:
                q["age_direction"] = ad
            # recent state contrast — only for median-based positions
            if basis != "personal_median" or len(recent) < _MIN_COMPARE_SEGS:
                continue
            rv = recent[c].dropna().to_numpy(dtype=float)
            if len(rv) < _MIN_COMPARE_SEGS:
                continue
            icc2r = self.pe._icc_2k(src, f, len(rv))
            if icc2r is None or icc2r < _ICC_POPULATION_GATE:
                continue
            qr = self.pe.query_stratified(f, float(np.median(rv)), source=src,
                                          age=age, gender=gender)
            if "error" in qr:
                continue
            qr.update({"k_used": len(rv), "reliability_used": round(icc2r, 3)})
            out["recent_positions"][f] = qr
            b_long = _BAND_ORDINAL.get(q["percentile"])
            b_rec = _BAND_ORDINAL.get(qr["percentile"])
            if (b_long is not None and b_rec is not None
                    and abs(b_rec - b_long) >= 1):
                out["population_drift"].append({
                    "feature": f, "long_term_band": q["percentile"],
                    "recent_band": qr["percentile"]})
        if any("age_direction" in p for p in out["positions"].values()):
            out["age_direction_note"] = (
                "age_direction is a CROSS-SECTIONAL population trend "
                "annotation (weak |ρ| ≲ 0.3): it describes where the "
                "value sits, never why it changed over time for this "
                "subject, and never supports a vascular-age estimate")
        out["verdict"] = "ok" if out["positions"] else "all_suppressed"
        return out

    # ------------------------------------------------------------------
    # bp_context — axis ④ (labels as background, never reversed)
    # ------------------------------------------------------------------

    def bp_context(self, ds: str, sid: str, seg_idx: int) -> dict:
        segs = self.load_subject_segments(ds, sid)
        cur_rows = segs[segs["seg_idx"] == seg_idx]
        if cur_rows.empty:
            return {"verdict": "segment_not_found"}
        cur = cur_rows.iloc[0]
        prof = self.build_profile(ds, sid,
                                  history_seg_idxs=[int(i) for i in
                                                    segs.loc[segs["seg_idx"] < seg_idx,
                                                             "seg_idx"]])
        out: dict = {"seg_idx": seg_idx, "measured": {}, "condition_baseline": None,
                     "personal_rho": prof.get("rho_bp", {}),
                     "population_rho_prior": self._rho_bp_top(ds, sid)}
        for k_src, k_out in [("abp_sbp_mean", "abp_sbp"),
                             ("abp_dbp_mean", "abp_dbp"),
                             ("sbp_label", "cuff_sbp"),
                             ("dbp_label", "cuff_dbp")]:
            v = cur.get(k_src)
            if v is not None and np.isfinite(v):
                out["measured"][k_out] = float(v)
        # disease-stratified BP baseline prior (Phase 2 JSON; optional)
        kb = Path(__file__).parent.parent.parent / "knowledge" / "bp_baseline_by_condition.json"
        if kb.exists():
            import json
            try:
                prior = json.loads(kb.read_text()).get(ds, {})
                out["condition_baseline"] = prior or None
            except Exception:  # noqa: BLE001
                pass
        out["verdict"] = "ok"
        return out

    def _rho_bp_top(self, ds: str, sid: str) -> dict | None:
        """Population ρ(feature, BP) prior (weak, for contrast only).

        Prefer population_bp_association.json (step 30): per-patient
        medians, rank partial controlling age+sex, BH-FDR, aurora a/o
        split.  Fall back to pulse_db_correlations.json (pooled
        segment-level Spearman, also a/o split now).
        Returns {feature: {target: rho}} plus '_provenance'.
        """
        import json
        root = Path(__file__).parent.parent.parent / "knowledge"
        source = self._source(ds, sid)   # vital | mimic | aurora_a | aurora_o
        try:
            data = json.loads(
                (root / "population_bp_association.json").read_text())
            pat = data.get("patterns", {}).get(source)
            if pat:
                out: dict = {"_provenance": "per-patient median, rank "
                                            "partial (age+sex), FDR<0.05"}
                for feat, targets in pat.items():
                    out[feat] = {t: v["rho_partial"]
                                 for t, v in targets.items()}
                return out
        except Exception:  # noqa: BLE001
            pass
        try:
            records = json.loads(
                (root / "pulse_db_correlations.json").read_text()
            ).get("stats", [])
            out = {"_provenance": "pooled segments, unadjusted spearman"}
            for r in records:
                if r.get("dataset") != source or r.get("spearman_r") is None:
                    continue
                out.setdefault(r["feature"], {})[r["target"]] = round(
                    float(r["spearman_r"]), 4)
            return out if len(out) > 1 else None
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # beat_bp_association — beat×BP coupling (vital/mimic only)
    # ------------------------------------------------------------------

    def beat_bp_association(self, ds: str, sid: str,
                            seg_idx: int | None = None,
                            max_segments: int | None = None) -> dict:
        """ρ(beat feature, beat SBP).

        seg_idx given  → within that single segment.
        seg_idx=None   → pooled over the subject's segments, both raw
        (pooled) and de-confounded (beat value − its segment mean, 决策④).
        """
        try:
            beats = self.load_subject_beats(ds, sid)
        except FileNotFoundError:
            return {"available": False, "reason": "no beat table"}
        if "abp_sbp" not in beats.columns:
            return {"available": False, "verdict": "not_available",
                    "reason": f"{ds} 无 ABP，beat×BP 耦合不适用"}
        if seg_idx is not None:
            beats = beats[beats["seg_idx"] == seg_idx]
        elif max_segments:
            keep = sorted(beats["seg_idx"].unique())[:max_segments]
            beats = beats[beats["seg_idx"].isin(keep)]
        valid = beats[self._valid_mask(beats)]
        valid = valid[valid["abp_sbp"].notna()]
        if len(valid) < 10:
            return {"available": False, "reason": "too few beats with BP"}
        feat_cols = [c for c in ["lasi", "pulse_amp", "pulse_width_50",
                                 "beat_len_ms", "hr_bpm", "t_sys_ms",
                                 "reflection_index", "crest_factor"]
                     if c in valid.columns]
        sbp = valid["abp_sbp"].to_numpy(dtype=float)
        seg_means = (valid.groupby("seg_idx")["abp_sbp"].transform("mean")
                     .to_numpy(dtype=float))
        sbp_resid = sbp - seg_means
        out: dict = {"available": True, "n_beats": int(len(valid)),
                     "n_segments": int(valid["seg_idx"].nunique()),
                     "pooled": {}, "deconfounded": {}}
        for f in feat_cols:
            x = valid[f].to_numpy(dtype=float)
            r_pool = _spearman(x, sbp)
            if seg_idx is None:
                x_res = x - valid.groupby("seg_idx")[f].transform("mean").to_numpy()
                r_dec = _spearman(x_res, sbp_resid)
            else:
                r_dec = r_pool  # single segment: no between-segment confound
            if r_pool is not None:
                out["pooled"][f] = round(r_pool, 3)
            if r_dec is not None:
                out["deconfounded"][f] = round(r_dec, 3)
        out["verdict"] = "ok"
        return out
