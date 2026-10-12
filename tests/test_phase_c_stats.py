import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"


def assert_finite_json(value):
    if isinstance(value, dict):
        for item in value.values():
            assert_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            assert_finite_json(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def load_stats(filename):
    path = KNOWLEDGE_ROOT / filename
    assert path.exists(), path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "created_at" in payload
    assert "stats" in payload
    assert payload["stats"]
    assert_finite_json(payload)
    return payload["stats"]


def test_reference_stats_schema():
    stats = load_stats("pulse_db_reference_stats.json")
    required = {"dataset", "quality_filter", "feature", "n", "mean", "std", "p05", "p25", "p50", "p75", "p85", "p95"}
    assert required.issubset(stats[0])
    assert {row["dataset"] for row in stats} == {"mimic", "vital"}
    assert {row["quality_filter"] for row in stats} == {"all_success", "quality_ok"}


def test_correlation_stats_schema():
    stats = load_stats("pulse_db_correlations.json")
    required = {"dataset", "quality_filter", "feature", "target", "spearman_r", "p_value", "n"}
    assert required.issubset(stats[0])
    # aurora scenarios a/o are stored separately — never pooled
    assert {"mimic", "vital"}.issubset({row["dataset"] for row in stats})
    assert {"sbp_label", "dbp_label"}.issubset({row["target"] for row in stats})


def test_condition_stats_schema():
    stats = load_stats("pulse_db_condition_stats.json")
    required = {"dataset", "quality_filter", "pattern", "condition", "thresholds", "n", "sbp_median", "sbp_iqr", "sbp_gt_140_ratio"}
    assert required.issubset(stats[0])
    assert {row["dataset"] for row in stats} == {"mimic", "vital"}
    patterns = {row["pattern"] for row in stats}
    assert "ri_high_but_quality_low" in patterns
    # ptt_mean_ms 在现役 features_v2 缓存缺失时 ptt_* 模式被跳过
    # （见 compute_condition_stats.py），存在 ptt_short 时组合模式必然同存
    if "ptt_short" in patterns:
        assert "ri_high_and_ptt_short" in patterns
