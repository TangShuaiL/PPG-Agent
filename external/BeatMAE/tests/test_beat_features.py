"""40 维拍级形态学特征提取 —— 冒烟与回归。

这些测试有两个明确目的：

1. **防列名漂移**：`FEATURE_GROUPS` 产出的键名必须与
   `preprocess/corpus_build/03_build_features.py::BEAT_FEAT_COLS` 逐字一致 —— 两者一错位，
   语料列名就会静默错配。

2. **防「缺 import 静默退化」**：本模块的代码是从另一个项目按函数切片搬来的，
   切片时若漏掉一个模块级 import（例：`keypoints.py` 曾漏 `import pandas`），
   上游的 try/except 会把 NameError 吞掉，表现为**所有特征都是 None 而不报错**。
   所以 `test_full_chain_produces_real_values` 不只看"跑通"，还断言**值非空**。

   另见下面两个自包含守卫 —— 本仓必须能独立运行，不得隐式依赖仓库外的本地代码。
"""
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess.features import extract_segment_features, FEATURE_GROUPS   # noqa: E402

FS = 500
SEG_LEN = FS * 10


def _target_cols():
    """从 03_build_features.py 直接读目标列名（单一出处，不在这里抄一份）。"""
    src = (ROOT / 'preprocess' / 'corpus_build' / '03_build_features.py') \
        .read_text(encoding='utf-8')
    out = {}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if getattr(t, 'id', '') in ('KP_COLS', 'BEAT_FEAT_COLS'):
                    out[t.id] = ast.literal_eval(n.value)
    return out


def _synthetic_segment(seed=0, hr=1.2):
    """合成 10 s PPG：systolic 峰 + 重搏切迹 + 呼吸漂移。"""
    rng = np.random.default_rng(seed)
    t = np.arange(SEG_LEN) / FS
    sig = np.zeros_like(t)
    for b in range(int(10 * hr)):
        off = b / hr
        m = (t >= off) & (t < off + 1.0 / hr)
        x = t[m] - off
        sig[m] += (np.sin(2 * np.pi * hr * x)
                   + 0.35 * np.sin(2 * np.pi * 2.6 * hr * x + 0.7)) * np.exp(-3.0 * x)
    sig += 0.15 * np.sin(2 * np.pi * 0.25 * t)
    return (sig + rng.normal(0, 0.01, len(t))) / np.std(sig)


# ────────────────────────── 列名契约 ──────────────────────────

def test_feature_names_match_corpus_contract():
    """FEATURE_GROUPS 产出的键名 ⊆ BEAT_FEAT_COLS，且无多余、无拼写漂移。"""
    target = set(_target_cols()['BEAT_FEAT_COLS'])
    produced = {k for g in FEATURE_GROUPS.values() for k in g['features']}
    extra = produced - target
    assert not extra, f'FEATURE_GROUPS 产出了 40 维之外的键名: {sorted(extra)}'
    print(f'  ✓ FEATURE_GROUPS 产出 {len(produced)} 键，全部落在 40 维内')


def test_feature_count_is_40():
    """34（五提取器）+ 6（链上别处产出）= 40。少一个都说明装配漏了。"""
    in_groups = sum(len(g['features']) for g in FEATURE_GROUPS.values())
    outside = {'intensity_ratio_s_d', 'beat_len_ms', 'hr_bpm',
               't_sys_ms', 't_dia_ms', 'ptt_ppg_ms'}
    target = _target_cols()['BEAT_FEAT_COLS']
    assert in_groups + len(outside) == len(target) == 40, \
        f'{in_groups} + {len(outside)} != {len(target)}'
    print(f'  ✓ {in_groups}（组内）+ {len(outside)}（组外）= 40')


# ────────────────────────── 端到端 ──────────────────────────

def _run(seg, **kw):
    """跑一段。abp 传占位：它只喂 segment 级的 abp_* 输出，不影响 40 维。"""
    return extract_segment_features(seg, seg.copy(), None,
                                    subject_id='t0', seg_idx=0, dataset='test',
                                    sbp_label=None, dbp_label=None, fs=FS, **kw)


def test_full_chain_produces_real_values():
    """跑通全链，且**特征值非空** —— 这条才拦得住「缺 import 静默退化」。"""
    out = _run(_synthetic_segment())
    beats = out['beats']
    assert beats, '没有产出任何拍'

    names = _target_cols()['BEAT_FEAT_COLS']
    import pandas as pd
    df = pd.DataFrame(beats)
    missing = [c for c in names if c not in df.columns]
    assert not missing, f'未产出这些列: {missing}'

    # 关键断言：核心特征不能全空（ptt_ppg_ms 因无 ECG 恒为 None，是预期内的）
    core = [c for c in names if c != 'ptt_ppg_ms']
    ratio = df[core].notna().mean().mean()
    assert ratio > 0.5, f'仅 {ratio:.1%} 的特征有值 —— 疑似 import 缺失导致静默退化'
    print(f'  ✓ {len(beats)} 拍，40 维中 {ratio:.1%} 有值')


def test_ptt_is_none_without_ecg():
    """无 R 峰时 ptt_ppg_ms 恒为 None，其余维度不受影响 —— 已在文档中声明。"""
    out = _run(_synthetic_segment(seed=3))
    assert all(b.get('ptt_ppg_ms') is None for b in out['beats'])
    print('  ✓ 无 ECG 时 ptt_ppg_ms 全为 None（预期行为）')


# ────────────────────────── 移植自检 ──────────────────────────

def test_no_out_of_repo_local_modules():
    """运行期守卫：跑完一轮后，不得加载仓库之外的**本地**模块。

    只允许三种来源：仓内、标准库、site-packages。切片搬运时函数体内的 import
    最容易漏，而仓外的本地项目一旦在 sys.path 上就会被静默命中 —— 这里把它挡住。
    """
    import site
    _run(_synthetic_segment(seed=1))

    allow = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    for d in list(site.getsitepackages()) + [site.getusersitepackages()]:
        try:
            allow.append(Path(d).resolve())
        except OSError:
            pass

    leaked = []
    for name, m in list(sys.modules.items()):
        f = getattr(m, '__file__', None)
        if not f:
            continue
        try:
            p = Path(f).resolve()
        except OSError:
            continue
        if p.is_relative_to(ROOT) or any(p.is_relative_to(a) for a in allow):
            continue
        leaked.append(f'{name} → {p}')
    assert not leaked, f'加载了仓库外的本地模块: {leaked}'
    print(f'  ✓ 未加载仓外本地模块（已核 {len(sys.modules)} 个已加载模块）')


def test_no_unknown_external_imports():
    """源码级守卫：`third_party/` 与 `preprocess/features/` 不得 import 未知的顶层模块。

    切片搬运时最容易漏掉**函数体内的** import（`segment.py` 曾漏一处，
    因为测试时把源项目加进了 sys.path 而被掩盖）。这里直接扫源码、不依赖运行环境：
    每个 import 的顶层名必须落在标准库、仓内模块、或已知第三方依赖三者之一。
    """
    KNOWN_THIRD_PARTY = {'numpy', 'pandas', 'scipy', 'biosppy', 'matplotlib',
                         'sklearn', 'seaborn', 'torch', 'yaml', 'h5py'}
    local = {p.stem for p in ROOT.rglob('*.py')}
    local |= {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith('.')}

    bad = []
    for d in ('third_party', 'preprocess/features'):
        for p in (ROOT / d).rglob('*.py'):
            for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
                s = line.strip()
                if s.startswith('#'):
                    continue
                m = re.match(r'(?:from|import)\s+([A-Za-z_]\w*)', s)
                if not m:
                    continue                      # 相对 import（`.x`）必在仓内，跳过
                top = m.group(1)
                if top in sys.stdlib_module_names or top in KNOWN_THIRD_PARTY or top in local:
                    continue
                bad.append(f'{p.relative_to(ROOT)}:{i}  未知 import: {top}')
    assert not bad, '存在未知外部 import:\n  ' + '\n  '.join(bad)
    print('  ✓ 无未知外部 import（third_party/ + preprocess/features/）')
