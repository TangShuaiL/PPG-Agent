"""weights.py — BeatMAE 权重来源解析（Zenodo 下载 + 校验缓存）。

交付三供应链：代码靠 git、权重靠 DOI、数据靠 data_v2。本模块只管"权重靠 DOI"
这一环——把 encoder（及头）从 Zenodo 记录拉到本地缓存，供实时链与离线重建脚本
共用，彻底消除对 ppg_model/runs 绝对路径的硬依赖。

Zenodo 记录把 encoder + heads 打成单个 `weights.zip`（非散文件），故本模块按
"下载 zip → sha256 校验 → 解包到缓存目录 → 定位内部文件"处理。

解析优先级（resolve_encoder_ckpt）：
    1. PPG_ENCODER_CKPT env —— 开发机/离线场景直接指本地整份 ckpt
       （如 results/pretrain/.../best.pt，含优化器，sha1 与历史烙印一致）；
    2. 缓存目录已解出的 encoder 文件 —— 命中即用，离线可跑；
    3. Zenodo 下载 weights.zip → 校验 → 解包 → 定位。

sha256 校验：期望值取自 external/CONTRACT.md 的 ZENODO_SHA256 表（按 zip 文件名
键入）；下载/命中后校验，不符抛 WeightIntegrityError（绝不静默用坏权重）。首次
下载若契约尚未登记，则打印观测值供回填。
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

from ..config import ENCODER_WEIGHTS_DIR, PROJECT_ROOT

# Zenodo 记录（BeatMAE 权重）；版本号 DOI 定位具体一次发布，concept DOI 跨版本。
ZENODO_RECORD_ID = "23080600"
ZENODO_ARTIFACT = "weights.zip"
# zip 内 encoder 文件名（Zenodo 发布名）。解析时若在 zip 内找不到该精确名，
# 会退化为按 *.pt 且非 heads/ 的唯一顶层权重自动定位。
DEFAULT_ENCODER_FILE = "beatmae_v25c3_s4.pt"
_CONTRACT_PATH = PROJECT_ROOT / "external" / "CONTRACT.md"


class WeightUnavailable(RuntimeError):
    """权重不可得（无缓存 + 下载失败/无网络）→ 上层降级查表。"""


class WeightIntegrityError(RuntimeError):
    """权重 sha256 与契约不符 → 拒绝使用（不静默降级，避免坏权重污染输出）。"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zenodo_assets() -> dict:
    """读 external/CONTRACT.md 里登记的 {filename: sha256} 期望表。

    契约用一行 JSON 便于机读：`<!-- ZENODO_SHA256 {...} -->`；缺失返回 {}
    （= 尚未钉死，首次下载后回填）。"""
    if not _CONTRACT_PATH.exists():
        return {}
    text = _CONTRACT_PATH.read_text(encoding="utf-8")
    marker = "ZENODO_SHA256"
    i = text.find(marker)
    if i < 0:
        return {}
    j = text.find("{", i)
    k = text.find("}", j)
    if j < 0 or k < 0:
        return {}
    try:
        return dict(json.loads(text[j:k + 1]))
    except Exception:  # noqa: BLE001 — 契约表解析失败按"未钉死"处理
        return {}


def _verify_zip(zip_path: Path) -> None:
    got = _sha256(zip_path)
    want = zenodo_assets().get(zip_path.name)
    if want and got.lower() != str(want).lower():
        raise WeightIntegrityError(
            f"{zip_path.name} sha256 不符契约: got={got} want={want}")
    if not want:
        print(f"[weights] 未登记 sha256，观测值请回填 CONTRACT.md: "
              f"{zip_path.name}: {got}")


def _locate(filename: str) -> Path | None:
    """在缓存目录树里找已解出的目标文件（顶层优先，再递归）。"""
    direct = ENCODER_WEIGHTS_DIR / filename
    if direct.exists():
        return direct
    for p in sorted(ENCODER_WEIGHTS_DIR.rglob(filename)):
        return p
    return None


def _pick_encoder(extract_root: Path) -> Path:
    """解包后定位 encoder 权重：优先 DEFAULT_ENCODER_FILE；否则取顶层（非 heads/）
    唯一的 *.pt 文件。头文件在 heads/ 子目录，不会误选。"""
    hit = _locate(DEFAULT_ENCODER_FILE)
    if hit:
        return hit
    cands = [p for p in sorted(extract_root.rglob("*.pt"))
             if "heads" not in p.parts]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise WeightUnavailable(f"weights.zip 内未找到 encoder .pt（{extract_root}）")
    raise WeightUnavailable(f"weights.zip 内 encoder .pt 不唯一: {cands}")


def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:  # noqa: BLE001
        raise WeightUnavailable(f"Zenodo 下载失败 {url}: {e}") from e
    tmp.replace(dst)
    return dst


def _ensure_extracted() -> Path:
    """确保缓存目录内有解出的 encoder：缺 zip 则下载校验，缺解包则解包。"""
    enc = _locate(DEFAULT_ENCODER_FILE) or _pick_encoder_or_missing()
    if enc is not None:
        return enc
    zip_path = ENCODER_WEIGHTS_DIR / ZENODO_ARTIFACT
    if not zip_path.exists():
        url = (f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
               f"/files/{ZENODO_ARTIFACT}?download=1")
        _download(url, zip_path)
    _verify_zip(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ENCODER_WEIGHTS_DIR)
    return _pick_encoder(ENCODER_WEIGHTS_DIR)


def _pick_encoder_or_missing() -> Path | None:
    """缓存里可能已解包但文件名与 DEFAULT 不符：尝试自动定位；找不到返回 None。"""
    if not ENCODER_WEIGHTS_DIR.exists():
        return None
    if _locate(DEFAULT_ENCODER_FILE):
        return _locate(DEFAULT_ENCODER_FILE)
    try:
        return _pick_encoder(ENCODER_WEIGHTS_DIR)
    except WeightUnavailable:
        return None


def resolve_encoder_ckpt(filename: str | None = None,
                         record_id: str = ZENODO_RECORD_ID) -> Path:
    """定位可用的 encoder ckpt，返回本地路径（env > 缓存 > Zenodo 下载解包）。

    找不到且无法下载 → WeightUnavailable（消费方据此降级查表）。"""
    if filename:
        globals()["DEFAULT_ENCODER_FILE"] = filename
    env = os.environ.get("PPG_ENCODER_CKPT")
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise WeightUnavailable(f"PPG_ENCODER_CKPT 指向不存在的文件: {p}")
        return p
    return _ensure_extracted()
