#!/usr/bin/env python3
"""造「无预训练」对照 ckpt：同架构、同 cfg、**权重为默认随机初始化**。

用途：消融表第 4 行 w/o 预训练。我们的下游评估一律**冻结编码器**（探针式），
所以这条臂测的是「预训练学到的表征 vs 随机特征」，**不是**「预训练 vs 端到端
从头训」——后者是另一个量级的实验，本脚本不涉及。写进论文时这个区别必须写明，
否则会被读成端到端对照。

做法（关键：复用与 load_frozen 完全相同的建模路径，不手搓参数）：
  1. 读基准 ckpt（缺省 = ★主力 v25c3/v2c_3src_mva），取出 cfg 与 state_dict；
  2. `torch.manual_seed(SEED)` 后按 load_frozen 的同一表达式新建 BeatMAE——
     构造函数内部的 reset_parameters 即默认初始化，故新模型 = 随机初始化；
  3. 断言新旧 state_dict **键集与形状逐一相同**（保证下游 strict 加载兼容），
     并断言新权重确实不是原权重、且不是常数；
  4. 存 {'model': 新sd, 'cfg': 原cfg, 'meta': 溯源}。

输出落在 results/cache/randinit/ 而**不是** results/pretrain/——后者是权威预训练
权重树（results/pretrain/INDEX.csv 记账），放一个没训过的模型进去会污染审计。

用法:
    python3 tools/make_randinit_ckpt.py
    ... --src results/pretrain/beatmae_v25c3/v2c_3src_mva/anchor_mr065_traverse/ckpt/best.pt \
        --out results/cache/randinit/beatmae_v25c3__randinit/ckpt/best.pt --seed 0
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.beat_mae.beat_mae import BeatMAE, beatmae_arch_from_sd

DEFAULT_SRC = ('results/pretrain/beatmae_v25c3/v2c_3src_mva/'
               'anchor_mr065_traverse/ckpt/best.pt')
DEFAULT_OUT = 'results/cache/randinit/beatmae_v25c3__randinit/ckpt/best.pt'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=0, help='初始化种子（记账用，必须记录）')
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if not src.exists():
        sys.exit(f'基准 ckpt 不存在: {src}')
    if out.exists():
        sys.exit(f'输出已存在，拒绝覆盖: {out}\n  （要重造就先手动删）')

    ck = torch.load(src, map_location='cpu', weights_only=False)
    cfg, sd = ck['cfg'], ck['model']
    mcfg, dcfg = cfg['model'], cfg['data']
    n_ce = int(sd['ce_head.weight'].shape[0]) if 'ce_head.weight' in sd else 0
    print(f'基准: {src}')
    print(f'  arch(从 sd 反推)={beatmae_arch_from_sd(sd)}  n_ce_codes={n_ce}')

    # 与 train/probe.py::load_frozen 的 beatmae 分支逐字同构
    torch.manual_seed(a.seed)
    model = BeatMAE(beat_len=dcfg['beat_len'], d_model=mcfg['d_model'],
                    depth=mcfg['depth'], nhead=mcfg['nhead'],
                    dec_kind=mcfg['dec_kind'], d_dec=mcfg['d_dec'],
                    dec_depth=mcfg['dec_depth'],
                    n_aux_feats=mcfg.get('n_aux_feats', 0),
                    n_ce_codes=n_ce,
                    arch=beatmae_arch_from_sd(sd),
                    dim_ff=mcfg.get('dim_ff'),
                    dropout=mcfg.get('dropout', 0.0))
    new = model.state_dict()

    # ── 自检 1：键集与形状逐一相同（否则下游 strict 加载会崩） ──
    assert set(new) == set(sd), (f'键集不同: 只在新的有 {set(new)-set(sd)}，'
                                 f'只在旧的有 {set(sd)-set(new)}')
    for k in new:
        assert new[k].shape == sd[k].shape, f'{k} 形状不符 {new[k].shape} vs {sd[k].shape}'
    print(f'  自检1 通过: {len(new)} 个张量键集/形状与原 ckpt 完全一致')

    # ── 自检 2：确实没训过（绝大多数张量应与训练后基准不同） ──
    # 注意不能断言"逐位相同的张量为 0"：`encoder.time_enc.freqs` 是**固定的正弦
    # 时间编码缓冲**，不是学出来的，任何初始化下都相同（2026-09-13 实测：它是
    # 唯一一个逐位相同的浮点张量，156/157 其余全不同）。
    n_fl = sum(1 for k in new if new[k].dtype.is_floating_point)
    n_same = [k for k in new if new[k].dtype.is_floating_point
              and torch.equal(new[k], sd[k])]
    n_diff = n_fl - len(n_same)
    n_par = sum(v.numel() for v in new.values())
    print(f'  自检2: 浮点张量 {n_fl} 个，与训练后基准**不同**的有 {n_diff} 个'
          f'（{n_diff/n_fl:.1%}）；逐位相同的 {n_same}；参数量 {n_par:,}')
    if n_diff / n_fl < 0.90:
        sys.exit(f'★ 自检2 未过：只有 {n_diff/n_fl:.1%} 与基准不同，'
                 f'像是没重新初始化；相同的是 {n_same[:5]}')
    if [k for k in n_same if not k.endswith('time_enc.freqs')]:
        sys.exit(f'★ 自检2 未过：除 time_enc.freqs 外还有逐位相同的张量 '
                 f'{[k for k in n_same if not k.endswith("time_enc.freqs")][:5]}')

    # ── 自检 2b：没有"死权重" ──
    # 判据落在**维度**上，不落在模块类型上：常量 bias / 归一化参数 / 固定编码缓冲
    # 全是默认初始化使然（Linear.bias=0、BatchNorm weight=1 bias=0 running_*=0,1、
    # LayerNorm weight=1 bias=0、time_enc.freqs 是正弦缓冲），都正常；
    # 致命的是**权重矩阵（ndim≥2）恒为常数**，那才是死网络。
    # （2026-09-13 先用 endswith('Norm')、再改成 'Norm' in owner，两轮都在绕远路：
    #   BatchNorm1d 结尾是 'Norm1d'，而 LayerNorm/Linear.bias 又各是一类，
    #   按类型枚举永远枚举不完——判据本该是维度。）
    mods = dict(model.named_modules())
    const_w, by_owner = [], {}
    for k in new:
        if not new[k].dtype.is_floating_point or float(new[k].std()) != 0.0:
            continue
        o = type(mods.get(k.rsplit('.', 1)[0])).__name__
        by_owner[o] = by_owner.get(o, 0) + 1
        if new[k].ndim >= 2 and not k.endswith('time_enc.freqs'):
            const_w.append((k, o, tuple(new[k].shape)))
    print(f'  自检2b: 常量张量 {sum(by_owner.values())} 个（按模块类型 {by_owner}）；'
          f'其中**权重矩阵(ndim≥2)恒为常数**的 {len(const_w)} 个')
    if const_w:
        sys.exit(f'★ 自检2b 未过：权重矩阵恒为常数 → 死网络: {const_w[:5]}')

    meta = dict(kind='randinit', src_ckpt=str(src), init_seed=a.seed,
                created_at=datetime.now(timezone.utc).astimezone().isoformat(),
                note='默认随机初始化、从未预训练；下游冻结评估用作 w/o 预训练对照')
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': new, 'cfg': cfg, 'meta': meta}, out)
    print(f'\n→ {out}  ({out.stat().st_size/1e6:.1f} MB)')
    print(f'  meta: {meta}')

    # 回读自检：load_frozen 能原样重建
    from train.probe import load_frozen
    m2, kind, im, mb = load_frozen(dict(ckpt=str(out), kind='beatmae'), 'cpu')
    sd2 = m2.state_dict()
    same = all(torch.equal(sd2[k], new[k]) for k in new)
    print(f'  自检3 load_frozen 回读: kind={kind} input_mode={im} max_beats={mb} '
          f'权重逐位一致={same}')
    if not same:
        sys.exit('★ 自检3 未过')


if __name__ == '__main__':
    main()
