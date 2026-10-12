#!/usr/bin/env python3
"""把发布用的权重打包成一个可直接上传的目录。

为什么不能直接发训练 checkpoint
────────────────────────────────
`results/pretrain/**/ckpt/best.pt` 有 **两处不能原样发布**：

1. **体积**：68 MB 里 45 MB 是 Adam 优化器状态（`opt`），推理用不到。
   模型本身只有 22.7 MB。
2. **内嵌标识**：`cfg` 与 `meta` 带着训练时的本机信息 ——
   `corpus_root`（集群挂载路径）、`reliability_json`（含用户名与另一个项目名）、
   `config_path`（内部 yaml 路径）、`meta.host.hostname`（机器名）。
   **直接发出去等于把脱敏工作全绕过了** —— 仓库 README 干净，权重里却写着，
   而且 `strings` 就能看到。

下游头（`heads/*.pt`）不含任何路径或标识，实测无敏感词，原样拷贝即可。

产物
────
    <out>/beatmae_v25c3_s4.pt                    编码器（22.7 MB）
    <out>/heads/{sbp,dbp,age}_causal_attn_s4.pt  三个目标的下游头（1.7 MB）
    <out>/README.md                              model card

用法：python tools/pack_weights.py [--out dist/weights]
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

# 论文主模型 / best seed（与 results/paper 各表一致）
ENC = 'results/pretrain/beatmae_v25c3/v2c_3src_mva/anchor_mr065_traverse/ckpt/best.pt'
HEAD_DIR = 'results/probe/beatmae_v25c3/v2c_3src_mva/full/stream/heads'
HEAD_KIND = 'causal_attn'
SEED = 4
TARGETS = ('sbp', 'dbp', 'age')

# 剥壳时**白名单保留**的 cfg 键 —— 其余（路径类）一律丢弃。
# 用白名单而非黑名单：将来 cfg 新增了带路径的键，默认落在「不保留」的安全侧。
KEEP_CFG = ('model', 'data')

# 打包后扫这些模式，命中即失败。
#
# ★ 这里**只写结构性模式，不写具体标识符** —— 本文件随仓发布，把标识符列出来
#   等于把它们塞进一个公开文件里。（tools/deidentify_release.py 正是因此被排除在
#   发布之外。权威的标识符校验在那边，本脚本只负责证明"剥壳确实生效了"。）
#
# 判据充分性：剥壳后 cfg 只剩 model/data 两个键、meta 只剩 created_at，
# 路径与主机名无从残留 —— 所以查路径与这些键，足以证明标识符没跟着出去。
FORBIDDEN = ['/home/', '/mnt/', '/scratch/', '/raid/',
             'corpus_root', 'reliability_json', 'config_path', 'hostname', 'out_dir']


def strip_encoder(src: Path) -> dict:
    """只留推理所需：权重 + 架构 + 输入形状 + 无标识的溯源信息。"""
    o = torch.load(src, map_location='cpu', weights_only=False)
    cfg = o.get('cfg', {})
    return {
        'model': o['model'],                                # 权重本体
        'cfg': {k: cfg[k] for k in KEEP_CFG if k in cfg},   # 架构 + 输入形状
        # 溯源：训练轮次与验证重建误差，不含任何路径/主机
        'provenance': {
            'epoch': o.get('epoch'),
            'opt_step': o.get('opt_step'),
            'val_recon': o.get('val_recon'),
        },
        # meta 只留创建时间；git / host / config_path 全丢（host 里是机器名）
        'created_at': (o.get('meta') or {}).get('created_at'),
    }


MODEL_CARD = """# BeatMAE — pretrained weights

Beat-aligned masked pre-training and longitudinal context representation for
PPG-based physiological estimation. Companion to the repository
[`TangShuaiL/BeatMAE`](https://github.com/TangShuaiL/BeatMAE).

## Files

| File | What it is |
|---|---|
| `beatmae_v25c3_s4.pt` | The pretrained encoder, seed 4 |
| `heads/sbp_causal_attn_s4.pt` | Downstream causal attention head, SBP |
| `heads/dbp_causal_attn_s4.pt` | Downstream causal attention head, DBP |
| `heads/age_causal_attn_s4.pt` | Downstream causal attention head, Age |

## The encoder

```python
ckpt = torch.load('beatmae_v25c3_s4.pt', map_location='cpu')
ckpt['model']          # encoder state_dict
ckpt['cfg']['model']   # {'d_model': 256, 'depth': 6, 'nhead': 8,
                       #  'dec_kind': 'transformer', 'd_dec': 128,
                       #  'dec_depth': 2, 'n_aux_feats': 40}
ckpt['cfg']['data']    # {'beat_len': 128, 'max_beats': 32,
                       #  'sources': ['mimic', 'vital', 'aurora']}
```

Build the model with `models.BeatMAE` and load `ckpt['model']`.

## The heads

Each head file carries its own normalisation constants, so no separate
statistics file is needed:

```python
h = torch.load('heads/sbp_causal_attn_s4.pt', map_location='cpu')
h['state']         # head state_dict
h['mu'], h['sd']   # target normalisation: y = pred * sd + mu
h['kind']          # 'causal_attn'
h['kk']            # history window, in segments
h['target'], h['seed']
```

The head is **causal**: it reads only the subject's own past segments. It is
therefore not a drop-in replacement for a per-segment regressor — feed it a
subject's segments in chronological order.

## Pretraining

- **Corpus**: the training folds of MIMIC-III, VitalDB and Aurora-BP
  (~3.83M segments). `cfg.data.sources` records the three sources.
- **Frozen encoder downstream**: only the lightweight head is trained.
- **Reported on**: the VitalDB and Aurora-BP test folds (folds 1 and 2).

## ⚠ Input requirements

The encoder expects a **0.25–10 Hz bandpassed, baseline-corrected 500 Hz**
signal. This repository's feature pipeline does **no filtering** — handing it
an unfiltered signal produces wrong results **without raising an error**.

## ⚠ Intended use

**Research use only. Not a medical device.** These weights are released to
support reproducible research on PPG representation learning. They have not
been validated for clinical decision-making and must not be used for
diagnosis, treatment, or any application where an error could cause harm.

## Packaging note

The original training checkpoint (68 MB) contained the Adam optimiser state
and training-time machine paths. This release holds only the model weights,
the architecture config, and the input-shape fields; the optimiser state and
all paths are stripped. See `tools/pack_weights.py` in the repository.

## Citation

See the repository README.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='dist/weights')
    a = ap.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    if out.exists():
        sys.exit(f'输出已存在，拒绝覆盖（要重来先手动删）：{out}')

    src = ROOT / ENC
    if not src.exists():
        sys.exit(f'找不到权重：{src}')

    (out / 'heads').mkdir(parents=True)

    # ── 编码器：剥壳 ──
    enc = strip_encoder(src)
    dst = out / 'beatmae_v25c3_s4.pt'
    torch.save(enc, dst)
    a_, b_ = src.stat().st_size, dst.stat().st_size
    print(f'  编码器 {a_/1048576:6.1f} MB → {b_/1048576:6.1f} MB'
          f'（省 {(a_-b_)/1048576:.1f} MB = {(1-b_/a_)*100:.0f}%）')

    # ── 头：原样拷贝（实测无路径/标识）──
    for t in TARGETS:
        h = ROOT / HEAD_DIR / f'{t}_{HEAD_KIND}_s{SEED}.pt'
        if not h.exists():
            sys.exit(f'找不到头文件：{h}')
        shutil.copy2(h, out / 'heads' / h.name)
    print(f'  头文件 {len(TARGETS)} 个 → heads/')

    (out / 'README.md').write_text(MODEL_CARD, encoding='utf-8')

    # ── 校验：对**打包产物本身**扫敏感词，不是对源 ──
    print('\n══ 校验（对打包产物本身）')
    blob = repr(enc) + MODEL_CARD
    for t in TARGETS:
        blob += repr(torch.load(out / 'heads' / f'{t}_{HEAD_KIND}_s{SEED}.pt',
                                map_location='cpu', weights_only=False))
    hits = sorted({w for w in FORBIDDEN if w in blob})
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    print(f'  敏感词命中: {hits if hits else "0 ✓"}')
    print(f'  合计 {size/1048576:.1f} MB → {out}')
    if hits:
        sys.exit('★ 校验未通过，不可发布。')

    print('\n══ 下一步：把该目录上传到 Zenodo（New upload），publish 后拿到 DOI')
    print('   然后在本仓 README 里加一节指向它。')


if __name__ == '__main__':
    main()
