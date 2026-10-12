# -*- coding: utf-8 -*-
"""data/paths — 语料根解析（唯一权威，2026-09-05 收编 6 个脚本的 exists() 探测）。

背景：旧模式 `CORPUS_17 if Path(CORPUS_17).exists() else CORPUS_18` 散在多个
脚本，若某机两条路径同时存在会**静默选错根**（数据根混淆写错折号的前科）。

解析优先级（高 → 低）：
  1. 显式参数 explicit（sh 传参，最强意图）
  2. 环境变量（机器级一次配置）
  3. 候选列表存在性探测（多机部署差异兜底）
任何非显式的选择都打印来源；全部失败则报错并提示指定方式，绝不静默猜。
"""
import os
from pathlib import Path

# 各语料的机器候选（17 机在前 / 18 机在后，与历史脚本一致）
# 旧库 corpus_v2 的候选常量（CORPUS_V2）已于 2026-09-10 删除：全仓零引用，
# 旧库只在历史结果里出现，现役脚本一律走 CORPUS_V2C。
CORPUS_V2C = ['/data1/corpus_v2c',                # 新库（现役）
              '/data2/corpus_v2c']


def resolve_corpus(candidates=None, explicit=None, env_var='PPG_CORPUS_ROOT',
                   tag='corpus'):
    """解析语料根目录，返回 str。candidates: 候选路径列表。"""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f'{tag}: 显式指定的语料根不存在: {p}')
        return str(p)
    env = os.environ.get(env_var)
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f'{tag}: {env_var} 指向不存在的路径: {p}')
        print(f'[paths] {tag} ← {env_var}={p}', flush=True)
        return str(p)
    for c in (candidates or []):
        if Path(c).exists():
            print(f'[paths] {tag} ← 探测命中: {c}', flush=True)
            return c
    raise FileNotFoundError(
        f'{tag}: 候选均不存在 {candidates}（可用 {env_var} 环境变量或 explicit 参数指定）')


# ── 下游探针产物树（与 pretrain 四层树同构，2026-09-05 定稿）──────────
# results/probe/<model>/<ds>/<scope>/<family>/——每实验一目录，
# 结果 json 与头权重同目录（不再分 probe/ 与 probe_heads/ 两处）。
PROBE_ROOT = Path('results/probe')
CACHE_ROOT = Path('results/cache/embeddings')   # 嵌入缓存 <model>__<ds>__<scope>_lab.npz
STREAM_CACHE = Path('results/cache/stream')     # 29 号物化 stream_emb.npy + stream_meta.npz


def probe_dir(name, family=None):
    """实验名 → results/probe/<model>/<ds>/<scope>[/<family>]。

    实验名须为三段 <model>__<ds>__<scope>（双下划线分隔，与四层树
    <模型>/<数据集>/<变体> 对应）；不合规名（历史/临时实验）归 _adhoc/。
    family ∈ {subject, window, stream, zeroshot}，为 None 时返回实验根。
    """
    parts = name.split('__')
    if len(parts) == 3 and all(parts):
        d = PROBE_ROOT.joinpath(*parts)
    else:
        d = PROBE_ROOT / '_adhoc' / name
    return d / family if family else d
