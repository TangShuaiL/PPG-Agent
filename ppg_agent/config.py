"""Central configuration for the PPG Agent project.

All project paths should flow through this module. Environment variables may
override the defaults for local machines, servers, or benchmark runs.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


# 数据版本烙印：PPG-Agent 数据层全部从 ppg_model corpus_v2c 派生（单一真源），
# 产物 meta 带 data_version=v2c；v1 产物冻结于各 output/_archive_v1/。
DATA_VERSION = "v2c"

DEFAULT_DATA_ROOT = _first_existing(
    Path("<DATA_ROOT>/data_v2"),  # canonical unified library (v3.1+)
    Path("<DATA_ROOT>/data"),     # legacy data disk (pre-v3.1)
    PROJECT_ROOT.parent / "data",
    PROJECT_ROOT.parents[1] / "data",
    PROJECT_ROOT.parent / "data" / "data",
)

DATA_ROOT = Path(os.environ.get("PPG_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()

# Legacy pre-v3.1 data disk.  Intentional references only: aurora 方案A 特征
# (历史对照/迁移校验)、multi_scenarios（仅存在于旧库）等一律走此常量，
# 禁止在业务脚本中硬编码旧库绝对路径。
LEGACY_DATA_ROOT = Path("<DATA_ROOT>/data")

# Phenotype database (heavy derived asset — lives with the raw data on the
# data disk).  Resolution order: PPG_DB_PATH env override → DATA_ROOT/
# ppg_agent/ → legacy project-local data/ (fresh clones without the disk).
PHENOTYPE_DB = Path(os.environ.get(
    "PPG_DB_PATH",
    _first_existing(
        DATA_ROOT / "ppg_agent" / "ppg_phenotype.db",
        PROJECT_ROOT / "data" / "ppg_phenotype.db",
    ),
)).expanduser()
KNOWLEDGE_ROOT = Path(os.environ.get("PPG_KNOWLEDGE_ROOT", PROJECT_ROOT / "knowledge")).expanduser()
REPORTS_ROOT = Path(os.environ.get("PPG_REPORTS_ROOT", PROJECT_ROOT / "reports")).expanduser()
MODELS_ROOT = Path(os.environ.get("PPG_MODELS_ROOT", PROJECT_ROOT / "models")).expanduser()
MODEL_CACHE_ROOT = Path(
    os.environ.get(
        "PPG_MODEL_CACHE_ROOT",
        _first_existing(PROJECT_ROOT.parents[1] / "models", MODELS_ROOT / "local"),
    )
).expanduser()
MODEL_REGISTRY = Path(
    os.environ.get("PPG_MODEL_REGISTRY", MODELS_ROOT / "model_registry.yaml")
).expanduser()

DEFAULT_FS = 125


# ── BeatMAE 外部依赖收口（v3.3 断联）──────────────────────────────
# 交付形态：代码靠 git（把 BeatMAE clone 到 external/BeatMAE）、权重靠 DOI
# （Zenodo）、数据靠 data_v2 数仓。BeatMAE 仓一行动代码不改，Agent 侧适配。
# 三级解析顺序：BEATMAE_ROOT env 覆盖 → external/BeatMAE（项目内 clone）
# → sibling ppg_model（开发机过渡期兜底）。三者皆无 → 末项（消费方惰性抛
# LiveStreamUnavailable → 上层降级查表，纪律不变）。
def resolve_beatmae_root() -> Path:
    env = os.environ.get("BEATMAE_ROOT")
    if env:
        return Path(env).expanduser()
    return _first_existing(
        PROJECT_ROOT / "external" / "BeatMAE",
        Path("<WORKSPACE>/ppg_model"),
    )


# encoder/头权重本地缓存根（数据盘，与 embeddings/heads 同属共享契约层）。
# resolve_encoder_ckpt() 优先级：PPG_ENCODER_CKPT env > 此目录缓存 > Zenodo 下载。
ENCODER_WEIGHTS_DIR = Path(os.environ.get(
    "PPG_ENCODER_WEIGHTS_DIR",
    DATA_ROOT / "ppg_agent" / "weights",
)).expanduser()

# On-disk directory names under DATA_ROOT (data_v2 layout: mimic/vital/aurora/anker).
# Legacy aliases (mimic3/vitaldb/aurora_dataset) map to the same canonical dirs.
DATASET_DIRS = {
    "mimic": "mimic",
    "mimic3": "mimic",
    "vital": "vital",
    "vitaldb": "vital",
    "aurorabp": "aurora",
    "aurora": "aurora",
    "anker": "anker",
    "multiscenarios": "multi_scenarios",   # 仅存在于旧库
    "ms": "multi_scenarios",
}

CANONICAL_DATASETS = {
    "mimic": "mimic",
    "mimic3": "mimic",
    "vital": "vital",
    "vitaldb": "vital",
    "aurorabp": "aurora",
    "aurora": "aurora",
    "anker": "anker",
    "multiscenarios": "multiscenarios",
    "ms": "multiscenarios",
}


def normalize_dataset(dataset: str) -> str:
    """Return the canonical dataset name (mimic / vital / aurora / anker / multiscenarios)."""
    key = dataset.lower().strip()
    try:
        return CANONICAL_DATASETS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(CANONICAL_DATASETS))
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {choices}") from exc


def dataset_dirname(dataset: str) -> str:
    """Return the on-disk directory name for a dataset."""
    key = dataset.lower().strip()
    try:
        return DATASET_DIRS[key]
    except KeyError:
        canonical = normalize_dataset(dataset)
        return DATASET_DIRS[canonical]


def dataset_root(dataset: str) -> Path:
    return DATA_ROOT / dataset_dirname(dataset)


# All raw directories now resolve under DATA_ROOT (aurora consolidated into
# data_guest/data/aurora_dataset); absolute overrides kept as escape hatch.
RAW_DIR_OVERRIDES: dict[str, str] = {}


def raw_data_dir(dataset: str) -> Path:
    key = dataset.lower().strip()
    if key in RAW_DIR_OVERRIDES:
        return Path(RAW_DIR_OVERRIDES[key])
    return dataset_root(dataset) / "raw"


def feature_cache_dir(dataset: str) -> Path:
    return dataset_root(dataset) / "features_v2"


def labels_path(dataset: str) -> Path:
    return dataset_root(dataset) / "labels.csv"


# 特征文件前缀与 canonical 名不同的数据集（data_v2 沿用 aurorabp_ 前缀）
FEATURE_PREFIXES = {
    "aurora": "aurorabp",
}


def feature_prefix(dataset: str) -> str:
    name = normalize_dataset(dataset)
    return FEATURE_PREFIXES.get(name, name)


def segment_features_path(dataset: str) -> Path:
    return feature_cache_dir(dataset) / f"{feature_prefix(dataset)}_segments.parquet"


def beat_features_path(dataset: str) -> Path:
    """Return the beat-level feature table path.

    Prefers the merged wide table ({prefix}_beats_full.parquet, 57 cols:
    PPG features + keypoints + per-beat BP) over the legacy split tables.
    Falls back to {prefix}_beats.parquet if the merged table is absent
    (e.g. aurora/ms which were not merged).
    """
    key = dataset.lower().strip()
    prefix = feature_prefix(dataset)
    full = feature_cache_dir(dataset) / f"{prefix}_beats_full.parquet"
    if full.exists():
        return full
    return feature_cache_dir(dataset) / f"{prefix}_beats.parquet"


def ensure_project_dirs() -> None:
    """Create lightweight output/config directories used by reports."""
    for path in (REPORTS_ROOT, MODELS_ROOT):
        path.mkdir(parents=True, exist_ok=True)
