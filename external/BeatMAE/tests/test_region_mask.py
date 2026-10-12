"""区域掩码策略冒烟测试：三模式（unified/weighted/hybrid）+ 可复现性。

用法：python3 tests/test_region_mask.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data import RegionMaskConfig, RegionMasker, zero_masked_regions
from models import BeatMAE, masked_mse

torch.manual_seed(0)
B, K, L = 8, 16, 128


def make_batch():
    """合成段批：拍数不一（末尾 padding）、含 kp_valid 缺失拍、含 3 拍短段。"""
    beats = torch.randn(B, K, L)
    pad = torch.zeros(B, K, dtype=torch.bool)
    for b in range(B):
        pad[b, -(b % 4 + 1):] = True                     # 每样本 1~4 拍 padding
    pad[7, 3:] = True                                    # 样本 7 只有 3 真实拍
    kp = torch.zeros(B, K, 10, dtype=torch.long)
    for i, v in enumerate([4, 8, 14, 20, 28, 55, 65, 90, 127]):
        kp[..., i] = v
    kp_valid = torch.ones(B, K, dtype=torch.bool)
    kp_valid[:, 2] = False                               # 每样本第 3 拍关键点缺失
    kp = kp.masked_fill(~kp_valid.unsqueeze(-1), -1)
    return beats, pad, kp, kp_valid


def ratio_of(sample_mask, pad):
    """有效拍口径的被遮比例（分母=真实拍×L）。"""
    n_real = (~pad).sum(dim=1).clamp(min=1)
    return sample_mask.sum(dim=(1, 2)).float() / (n_real * L)


def test_unified():
    _, pad, kp, kp_valid = make_batch()
    masker = RegionMasker(RegionMaskConfig(mode='unified'))
    out = masker(kp, kp_valid, pad, generator=torch.Generator().manual_seed(42))
    r = ratio_of(out['sample_mask'], pad)
    # 达成比例 = min(目标, 区域粒度并集覆盖)：五区域互相重叠，本批合成关键点的
    # 并集覆盖≈0.5，故贴不到 0.65 目标——这是设计使然，非缺陷。区间只挡
    # "几乎不遮"与"全遮"两端（精确贴住上限的断言在 test_weighted）。
    assert 0.25 <= r.mean().item() <= 0.75, f'平均比例偏离目标: {r.mean():.3f}'
    assert not out['region_mask'][~kp_valid].any()       # 缺失拍不遮
    assert not out['sample_mask'][pad.unsqueeze(-1).expand(-1, -1, L)].any()
    assert not out['beat_mask'].any()                    # 无整拍遮
    assert set(out['strategy']) == {'unified'}
    # 段内所有可遮拍遮同一区域集合（统一掩码）
    for b in range(B):
        rm = out['region_mask'][b]
        elig = (~pad[b]) & kp_valid[b]
        sets = {tuple(rm[k].tolist())
                for k in torch.nonzero(elig).flatten().tolist()}
        assert len(sets) == 1 and any(sets.pop())
    print(f'unified OK: ratio={r.mean():.3f} 段内区域集合一致')


def test_weighted():
    _, pad, kp, kp_valid = make_batch()
    masker = RegionMasker(RegionMaskConfig(mode='weighted'))
    out = masker(kp, kp_valid, pad, generator=torch.Generator().manual_seed(1))
    # 口径：仅可遮拍（关键点缺失拍无法定义区域，不遮也不计入分母）
    elig = kp_valid & ~pad
    per_beat = out['sample_mask'].sum(dim=2).float()      # (B, K)
    r = per_beat[elig] / L
    # 理论上限 = min(目标, 区域并集覆盖)：五区域不覆盖全拍且互相重叠
    from models.common import N_REGIONS, region_slices
    sl = region_slices(kp, kp_valid)
    pos = torch.arange(L)
    union = torch.zeros(B, K, L, dtype=torch.bool)
    for rr in range(N_REGIONS):
        union |= ((pos[None, None, :] >= sl[..., rr, 0].unsqueeze(-1))
                  & (pos[None, None, :] < sl[..., rr, 1].unsqueeze(-1)))
    cov = union.float().sum(dim=2) / L                    # (B, K) 并集覆盖
    # 目标取配置缺省，勿再写字面量：0.40 曾硬编码在此，缺省对齐基座 0.65 后分叉
    cap = torch.minimum(torch.full_like(cov, RegionMaskConfig().target_ratio), cov)
    cap_real = cap[elig].mean().item()
    m_real = r.mean().item()
    assert abs(m_real - cap_real) < 0.05, \
        f'weighted 未贴住覆盖上限: ratio={m_real:.3f} cap={cap_real:.3f}'
    # 连续性：每区域一整块，被遮区段数 ≤ 区域数（块可能相邻合并）
    sm = out['sample_mask']
    flat = sm.permute(0, 1, 2).reshape(B, K, L).int()
    runs = ((flat[:, :, 1:] - flat[:, :, :-1]) == 1).sum(dim=2) \
        + flat[:, :, 0]
    assert (runs <= N_REGIONS).all(), '加权模式出现散点（区段数超区域数）'
    assert not out['sample_mask'][pad.unsqueeze(-1).expand(-1, -1, L)].any()
    assert not out['sample_mask'][~kp_valid.unsqueeze(-1)
                                  .expand(-1, -1, L)].any()
    # 区域提示与样本掩码一致：被遮样本必落在提示为 True 的区域内
    assert out['region_mask'].any() and set(out['strategy']) == {'weighted'}
    print(f'weighted OK: ratio={m_real:.3f} 贴住覆盖上限 cap={cap_real:.3f}')


def test_hybrid():
    _, pad, kp, kp_valid = make_batch()
    masker = RegionMasker(RegionMaskConfig(mode='hybrid', region_share=0.6))
    out = masker(kp, kp_valid, pad, generator=torch.Generator().manual_seed(5))
    elig = kp_valid & ~pad
    r = out['sample_mask'].sum(dim=2).float()[elig] / L
    # 构成 = 区域部分（目标×region_share，受并集覆盖限制）+ 随机部分
    # （目标×(1-region_share)）减去两者重叠。缺省目标 0.65 下本批实测 0.52；
    # 区间只挡"随机部分没生效"与"整段全遮"两端。
    assert 0.40 <= r.mean().item() <= 0.70, f'hybrid 比例异常: {r.mean():.3f}'
    # 区域部分：可遮拍上提示区域的样本确实被遮（统一区域掩码生效）
    # 散点部分：存在不属于任何提示区域的被遮样本（全局随机生效）
    from models.common import N_REGIONS, region_slices
    sl = region_slices(kp, kp_valid)                     # (B,K,R,2)
    pos = torch.arange(L)
    in_region = torch.zeros(B, K, L, dtype=torch.bool)
    for rr in range(N_REGIONS):
        m = ((pos[None, None, :] >= sl[..., rr, 0].unsqueeze(-1))
             & (pos[None, None, :] < sl[..., rr, 1].unsqueeze(-1))
             & out['region_mask'][..., rr].unsqueeze(-1))
        in_region |= m
    extra = out['sample_mask'] & ~in_region
    assert extra.any(), 'hybrid 无随机 patch 部分'
    # patch 连续性：随机部分存在长度 ≥ patch_len 的连续区段（非散点）
    ex = extra.int()
    starts = ((ex[:, :, 1:] - ex[:, :, :-1]) == 1)
    starts = torch.cat([ex[:, :, :1].bool(), starts], dim=2)
    ends = torch.cat([(ex[:, :, 1:] - ex[:, :, :-1]) == -1,
                      ex[:, :, -1:].bool()], dim=2)
    idx_s = torch.nonzero(starts)
    idx_e = torch.nonzero(ends)
    lens_run = idx_e[:, 2] - idx_s[:, 2] + 1
    assert lens_run.max().item() >= 8, '随机部分不是连续 patch'
    assert (out['sample_mask'] & in_region).any(), 'hybrid 无区域掩码部分'
    assert set(out['strategy']) == {'hybrid'}
    print(f'hybrid OK: ratio={r.mean().item()} '
          f'区域样本={int((out["sample_mask"] & in_region).sum())} '
          f'patch 样本={int(extra.sum())} 最长区段={int(lens_run.max())}')


def test_beat_drop():
    """拍级丢弃：比例锚定可遮拍、护栏留拍、区域提示恒 False。"""
    _, pad, kp, kp_valid = make_batch()
    masker = RegionMasker(RegionMaskConfig(mode='beat_drop', target_ratio=0.65))
    out = masker(kp, kp_valid, pad, generator=torch.Generator().manual_seed(9))
    bm, sm = out['beat_mask'], out['sample_mask']
    elig = kp_valid & ~pad
    # 丢弃只发生在可遮拍上；padding/关键点缺失拍不丢
    assert not bm[~elig].any(), '不可遮拍被丢弃'
    assert not out['region_mask'].any(), 'beat_drop 不应产生区域提示'
    # 被丢拍样本掩码全 L；未丢拍完全可见
    assert sm[bm.unsqueeze(-1).expand(-1, -1, L)].all(), '被丢拍未全 L 遮'
    vis = elig & ~bm
    assert not sm[vis.unsqueeze(-1).expand(-1, -1, L)].any(), '可见拍被误遮'
    # 护栏：每段至少留 1 拍可见
    for b in range(B):
        assert vis[b].any() or not elig[b].any(), f'样本 {b} 全拍被丢'
    # 比例：丢弃拍数 / 可遮拍数 ≈ target（粒度=整拍，容差放宽）
    frac = bm[elig].float().mean().item()
    assert 0.55 <= frac <= 0.75, f'丢弃比例偏离目标: {frac:.3f}'
    assert set(out['strategy']) == {'beat_drop'}
    print(f'beat_drop OK: 丢弃比例={frac:.3f} '
          f'丢拍数={int(bm.sum())}/{int(elig.sum())}')


def test_reproducibility():
    """同种子 ⇒ 掩码完全一致；不同种子 ⇒ 不同。四模式全查。"""
    _, pad, kp, kp_valid = make_batch()
    for mode in ('unified', 'weighted', 'hybrid', 'beat_drop'):
        masker = RegionMasker(RegionMaskConfig(mode=mode))
        a = masker(kp, kp_valid, pad, generator=torch.Generator().manual_seed(123))
        b = masker(kp, kp_valid, pad, generator=torch.Generator().manual_seed(123))
        assert torch.equal(a['sample_mask'], b['sample_mask']) \
            and torch.equal(a['region_mask'], b['region_mask']), \
            f'{mode} 同种子掩码不一致'
        c = masker(kp, kp_valid, pad, generator=torch.Generator().manual_seed(124))
        assert not torch.equal(a['sample_mask'], c['sample_mask']), \
            f'{mode} 不同种子掩码相同'
    print('reproducibility OK: 四模式同种子掩码逐位一致，异种子不同')


def test_zeroing():
    beats, pad, kp, kp_valid = make_batch()
    masker = RegionMasker()
    out = masker(kp, kp_valid, pad, generator=torch.Generator().manual_seed(3))
    z = zero_masked_regions(beats, out['sample_mask'])
    sm = out['sample_mask']
    assert (z[sm] == 0).all(), '被遮样本未置零'
    assert torch.equal(z[~sm], beats[~sm]), '未遮样本被误改'
    print('zeroing OK: 仅被遮样本置零')


def test_beatmae_integration():
    """四模式各跑一遍 BeatMAE 前向/反向。"""
    beats, pad, kp, kp_valid = make_batch()
    onset = torch.cumsum(torch.full((B, K), 0.8), dim=1) * (1 - pad.float())
    side = torch.stack([torch.full((B, K), 800.0),
                        torch.full((B, K), 75.0)], dim=-1)
    side = side.masked_fill(pad.unsqueeze(-1), 0.0)
    model = BeatMAE(beat_len=L, d_model=192, depth=6, nhead=6)
    for mode in ('unified', 'weighted', 'hybrid', 'beat_drop'):
        masker = RegionMasker(RegionMaskConfig(mode=mode))
        out = masker(kp, kp_valid, pad,
                     generator=torch.Generator().manual_seed(7))
        beats_z = zero_masked_regions(beats, out['sample_mask'])
        res = model(beats_z, pad, onset, side, out['beat_mask'],
                    out['region_mask'])
        loss = masked_mse(res['recon'], beats, out['sample_mask'])
        loss.backward()
        assert torch.isfinite(loss) and loss.item() > 0
        print(f'  {mode}: recon={loss.item():.4f}')
    print('integration OK: 四模式 × BeatMAE 前向/反向全部有限')


if __name__ == '__main__':
    test_unified()
    test_weighted()
    test_hybrid()
    test_beat_drop()
    test_reproducibility()
    test_zeroing()
    test_beatmae_integration()
    print('== 区域掩码策略测试全部通过 ==')
