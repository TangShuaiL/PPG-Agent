"""PaPaGei P/S 冒烟测试：前向形状、NT-Xent 语义、sVRI 分箱、增强、下游装载。

合成张量部分**永不 skip**（不依赖语料/形态学表）；真实语料那一处单独 `needs_corpus`。
语料与形态学表都不进版本库，无语料的机器上只有最后一条跳过。

盯的是**契约**而不是数字：512 维嵌入口径、`ignore_index=-1` 的三条语义
（不做 anchor / 不当正样本伙伴 / 全无有效标签时返回 0）、
`n_bins` 必须显式传（留默认值 = 改了 yaml 不生效的假旋钮）、
`load_frozen` 对 papagei 只认 raw_waveform。

用法：python3 tests/test_papagei_smoke.py
"""
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from data.corpus_dataset import CorpusDataset, MAX_BEATS
from data.morphology import (LABEL_KEYS, WithMorphology, load_morphology,
                             morph_collate)
from data.paths import CORPUS_V2C, resolve_corpus
from baselines.papagei import (ARCH, GaussianNoise, PapageiP, PapageiS,
                            TSRandomCropPad, assign_svri_bins, build_model,
                            build_transform, name_from_sd, ntxent_loss,
                            svri_bin_edges)

torch.manual_seed(0)

B, L = 4, 1250            # 官方输入：125 Hz × 10 s
D = ARCH['n_classes']     # 512
TEMP = 0.07               # 官方 NTXentLoss 默认温度


# ──────────────────────────── 真实语料（可 skip）────────────────────────────
def _corpus_root():
    try:
        return resolve_corpus(CORPUS_V2C, tag='corpus_v2c')
    except FileNotFoundError:
        return None


CORPUS = _corpus_root()
MORPH = Path(__file__).resolve().parents[1] / 'results/cache/morphology/morphology_v2c.parquet'
needs_corpus = pytest.mark.skipif(
    CORPUS is None or not MORPH.exists(),
    reason='语料 corpus_v2c / 形态学表不可用（都不进版本库，见 README「已知限制」）')


# ──────────────────────────── 模型前向 ────────────────────────────
def test_papagei_p_forward():
    """(B, L) → (emb, out_emb)；emb = dense(out_emb) 且必须是 512 维。"""
    seg = torch.randn(B, L)
    model = PapageiP().eval()
    emb, out_emb = model(seg)
    assert emb.shape == (B, D), f'嵌入口径必须是官方 512 维，实得 {tuple(emb.shape)}'
    assert out_emb.shape[0] == B and out_emb.dim() == 2
    for _, p in model.named_parameters():
        assert p.requires_grad
    # batch 维之外的形状不该被 B 影响（下游按 (B, 512) 直接进头）
    assert PapageiP().eval()(torch.randn(2, L))[0].shape == (2, D)
    # (B, 1, L) 与 (B, L) 同义（_as_bcl 的两种合法输入）
    assert torch.equal(model(seg.unsqueeze(1))[0], emb)
    print(f'P 前向 OK: emb={tuple(emb.shape)} out_emb={tuple(out_emb.shape)}')


def test_papagei_s_forward():
    """S 四输出：dense 头给对比损失，两个 MoE 头各回归一个标量。"""
    seg = torch.randn(B, L)
    model = PapageiS().eval()
    out_class, out_moe1, out_moe2, out = model(seg)
    assert out_class.shape == (B, D)
    assert out_moe1.shape == (B, 1) and out_moe2.shape == (B, 1), \
        'MoE 头是标量回归（官方 L1(ipa) / L1(skewness)）'
    assert out.shape[0] == B and out.dim() == 2
    assert torch.isfinite(out_moe1).all() and torch.isfinite(out_moe2).all()
    print(f'S 前向 OK: class={tuple(out_class.shape)} '
          f'moe1={tuple(out_moe1.shape)} moe2={tuple(out_moe2.shape)}')


def test_embed_segment_matches_forward():
    """下游嵌入口径 = 训练前向的第 0 项，两族都必须一致（否则 ckpt 与下游对不上）。"""
    x = torch.randn(B, L)
    for cls in (PapageiP, PapageiS):
        m = cls().eval()
        assert torch.equal(m.embed_segment(x), m(x)[0]), f'{cls.__name__} 嵌入口径分叉'
    print('embed_segment == forward[0] OK（P/S 同口径）')


def test_build_model_and_name_from_sd():
    """构建与判别同一份权威：两条路径必须互逆（ckpt 无 cfg 时的兜底就靠它）。"""
    for name, cls in (('papagei_p', PapageiP), ('papagei_s', PapageiS)):
        m = build_model(name)
        assert isinstance(m, cls)
        assert name_from_sd(m.state_dict()) == name
    with pytest.raises(ValueError):
        build_model('papagei_x')
    print('build_model / name_from_sd 互逆 OK')


def test_as_bcl_rejects_bad_dim():
    """4 维输入应显式报错，而不是静默按 batch 维对齐算错。"""
    with pytest.raises(ValueError):
        PapageiP().eval()(torch.randn(2, 1, 1, L))


# ──────────────────────────── NT-Xent ────────────────────────────
def _ntxent_ref(emb, labels, tau, ignore_index=None):
    """逐样本循环的手写参照：`ignore_index` 的行既不做 anchor 也不当正样本伙伴。

    与 `baselines/papagei/losses.py` 的向量化实现独立（这里看不出任何 masked_fill），
    用来锁语义，不是重复实现。
    """
    z = torch.nn.functional.normalize(emb.float(), dim=-1)
    sim = z @ z.t() / tau
    ign = (lambda v: ignore_index is not None and int(v) == ignore_index)
    per_row = []
    for i in range(len(labels)):
        if ign(labels[i]):
            continue
        pos = [j for j in range(len(labels))
               if j != i and int(labels[j]) == int(labels[i]) and not ign(labels[j])]
        if not pos:
            continue
        others = [j for j in range(len(labels)) if j != i]
        log_denom = torch.logsumexp(sim[i][others], 0)
        log_pos = torch.logsumexp(sim[i][pos], 0)
        per_row.append(log_denom - log_pos / len(pos))
    return torch.stack(per_row).mean()


def test_ntxent_finite_and_backward():
    emb = torch.randn(B, D, requires_grad=True)
    labels = torch.tensor([0, 0, 3, 3])
    loss = ntxent_loss(emb, labels, temperature=TEMP)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
    # 数值稳定性：全同向量的余弦相似度会爆（sim 最大 ~1/0.07=14.3），
    # 实现里减过 max 再 masked_fill，这里只要求不 NaN。
    same = torch.ones(B, D, requires_grad=True)
    ls = ntxent_loss(same, labels, temperature=TEMP)
    assert torch.isfinite(ls), '归一化后全同向量导致 NaN（logsumexp 未减 max？）'
    print(f'NT-Xent OK: loss={loss.item():.4f} 全同向量={ls.item():.4f}')


def test_ntxent_matches_reference():
    torch.manual_seed(1)
    emb = torch.randn(6, 64)
    labels = torch.tensor([-1, -1, 0, 0, 2, 2])
    got = ntxent_loss(emb, labels, temperature=TEMP, ignore_index=-1)
    want = _ntxent_ref(emb, labels, TEMP, ignore_index=-1)
    assert torch.allclose(got, want, atol=1e-5), f'向量化实现 {got.item():.6f} != 参照 {want.item():.6f}'
    print(f'NT-Xent 参照一致 OK: {got.item():.6f}（6 行中 2 行无标签，不计入均值）')


def test_ntxent_ignore_index_changes_value():
    """ignore_index 必须真的改变数值——否则它又是个改了不生效的假旋钮。"""
    torch.manual_seed(2)
    emb = torch.randn(4, 64)
    labels = torch.tensor([-1, -1, 0, 0])
    a = ntxent_loss(emb, labels, temperature=TEMP, ignore_index=-1)
    b = ntxent_loss(emb, labels, temperature=TEMP, ignore_index=None)
    # -1 当普通标签时，前两行互为唯一正样本 → 参与均值，值必然不同
    assert not torch.allclose(a, b), 'ignore_index 未生效'
    assert torch.allclose(b, _ntxent_ref(emb, labels, TEMP, ignore_index=None), atol=1e-5)
    print(f'ignore_index 生效 OK: -1 参与={b.item():.4f} 跳过={a.item():.4f}')


def test_ntxent_all_ignored_is_zero():
    """整批无有效标签：返回 0（不 NaN），且梯度为 0 而不是断链。"""
    emb = torch.randn(B, 8, requires_grad=True)
    loss = ntxent_loss(emb, torch.full((B,), -1), temperature=TEMP, ignore_index=-1)
    assert loss.item() == 0.0
    loss.backward()
    assert (emb.grad == 0).all()
    print('全无有效标签 → 0 损失、0 梯度 OK')


def test_ntxent_batch1_raises():
    """NT-Xent 需要 ≥2 行（自身除外后无负样本）。"""
    with pytest.raises(AssertionError):
        ntxent_loss(torch.randn(1, 8), torch.tensor([0]))


# ──────────────────────────── sVRI 分箱 ────────────────────────────
def test_svri_bin_edges_n_bins_required():
    """n_bins 是训练超参（唯一权威 = config 的 papagei.bins_svri），必须显式传。

    回归守卫：此前 labels.py 有默认值、调用方按关键字传，签名一改就静默分叉。
    """
    with pytest.raises(TypeError):
        svri_bin_edges(np.array([0.5, 1.0]))


def test_svri_bin_edges_train_rows_only():
    """边界只在训练池上取——验证折的极端值不得进边界（信息泄漏）。"""
    svri = np.array([0.10, 0.90, 0.50, 1.50, 0.20])   # 行 3 是"验证折"的极端值
    tr = svri_bin_edges(svri, 4, rows=np.array([0, 1, 2]))
    assert tr.shape == (5,)
    assert np.allclose(tr, [0.10, 0.30, 0.50, 0.70, 0.90]), tr
    full = svri_bin_edges(svri, 4)
    assert not np.allclose(tr, full), 'rows 参数未生效（整表边界=泄漏）'
    assert tr[-1] < 1.50, '验证折的 1.50 漏进了边界'
    print(f'n_bins 边界 OK: 训练池={tr[-1]:.2f} < 全表={full[-1]:.2f}')


def test_svri_bin_edges_empty_pool_raises():
    """rows 传错（池子里一个合格样本都没有）→ 显式报错，不返回 NaN 边界。"""
    with pytest.raises(ValueError):
        svri_bin_edges(np.array([5.0, -1.0, np.nan]), 8, rows=np.array([0, 1, 2]))


def test_assign_svri_bins_semantics():
    """-1 = 无有效标签：NaN / 落在 (0,2) 外（含等于端点）/ 计算失败。"""
    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    svri = np.array([np.nan, -1.0, 0.0, 0.2, 0.6, 1.4, 2.0, 2.5, 1.999])
    b = assign_svri_bins(svri, edges)
    assert b.tolist() == [-1, -1, -1, 0, 1, 2, -1, -1, 3], b.tolist()
    assert b.dtype == np.int64
    print(f'分箱 OK: {b.tolist()}')


def test_assign_svri_bins_degenerate_edges():
    """退化兜底：合格子集全等值 → w=0、边界全相等，digitize 会越界，靠 clip 收住。"""
    b = assign_svri_bins(np.array([1.0]), np.array([1.0, 1.0, 1.0]))
    assert b.tolist() == [1], '退化边界未兜底（箱号越界）'


# ──────────────────────────── 增强 ────────────────────────────
def test_build_transform_order_and_identity():
    """官方 Compose 顺序：噪声在前、裁剪在后；两个概率都为 0 时逐元素恒等。"""
    t = build_transform()
    assert isinstance(t[0].tf, GaussianNoise) and isinstance(t[1].tf, TSRandomCropPad), \
        '增强顺序与官方 prob_dictionary 的 Compose 不符'
    x = torch.randn(B, L)
    assert torch.equal(t(x), x)
    print('增强顺序 / p=0 恒等 OK')


def test_build_transform_noise_sigma():
    """官方 sigma=0.05，是 z-score 尺度（上游 raw_waveform 已 z-score）。"""
    torch.manual_seed(3)
    x = torch.randn(64, L)
    y = build_transform(g_p=1.0, c_p=0.0)(x)
    assert torch.allclose((y - x).std(), torch.tensor(0.05), atol=0.01), \
        f'噪声幅度 {float((y - x).std()):.4f} != 官方 0.05'


def test_build_transform_per_sample():
    """逐样本独立掷骰（官方 dataset 一次一个样本；本库整批前向但语义等价）。"""
    np.random.seed(0)
    torch.manual_seed(0)
    x = torch.randn(64, L)
    y = build_transform(g_p=0.0, c_p=0.5)(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    changed = (y != x).any(dim=-1)
    assert 0 < int(changed.sum()) < 64, \
        f'p=0.5 时 {int(changed.sum())}/64 行被改动——整批同掷而非逐样本？'
    # 裁剪只删不加：非零处必须与原信号逐点相等
    kept = (y != 0) & (x != 0)
    assert torch.equal(y[kept], x[kept])
    print(f'逐样本增强 OK: {int(changed.sum())}/64 行被裁剪')


# ──────────────────────────── 下游装载 ────────────────────────────
@pytest.mark.parametrize('name,cls', [('papagei_p', PapageiP), ('papagei_s', PapageiS)])
def test_load_frozen_roundtrip(tmp_path, name, cls):
    """ckpt → load_frozen：解析族名、锁定 raw_waveform 口径、strict 加载。"""
    from train.probe import load_frozen
    ck = tmp_path / 'best.pt'
    torch.save({'model': cls().state_dict(), 'cfg': {'papagei': {'model': name}}}, ck)
    model, kind, input_mode, max_beats = load_frozen(
        {'kind': 'papagei', 'ckpt': str(ck)}, torch.device('cpu'))
    assert kind == name, f'--kind papagei 应由 ckpt 自己解析出 {name}，实得 {kind}'
    assert input_mode == 'raw_waveform', 'PaPaGei 只有 raw_waveform 一种口径'
    assert max_beats == MAX_BEATS
    assert not any(p.requires_grad for p in model.parameters()), '下游必须冻结编码器'
    # 显式 kind 与 ckpt 不符 → 报错而不是静默换模型
    with pytest.raises(ValueError):
        load_frozen({'kind': 'papagei_s' if name == 'papagei_p' else 'papagei_p',
                     'ckpt': str(ck)}, torch.device('cpu'))
    with pytest.raises(ValueError):
        load_frozen({'kind': name, 'ckpt': str(ck), 'input_mode': 'beat_resampled'},
                    torch.device('cpu'))
    print(f'load_frozen OK: {name} → kind={kind} input_mode={input_mode}')


def test_load_frozen_without_cfg(tmp_path):
    """ckpt 无 cfg 时按 state_dict 认 MoE（P 的骨干后直连 dense）。"""
    from train.probe import load_frozen
    ck = tmp_path / 'nocfg.pt'
    torch.save({'model': PapageiS().state_dict()}, ck)
    _, kind, _, _ = load_frozen({'kind': 'papagei', 'ckpt': str(ck)}, torch.device('cpu'))
    assert kind == 'papagei_s'
    print('无 cfg 兜底判别 OK: papagei_s')


# ──────────────────────────── 真实语料端到端 ────────────────────────────
@needs_corpus
def test_real_corpus_pretrain_step():
    """真实段波形 → 训练池边界分箱 → P 前向 → NT-Xent 反传（照抄训练口径）。"""
    from torch.utils.data import DataLoader

    base = CorpusDataset(CORPUS, exclude_folds={1, 2}, max_rows=4096,
                         input_mode='raw_waveform')
    morph = load_morphology(str(MORPH))
    n_bins = 8                                   # 唯一权威是 config 的 papagei.bins_svri
    edges = svri_bin_edges(morph['svri'], n_bins, rows=base.rows)
    ds = WithMorphology(base, morph, assign_svri_bins(morph['svri'], edges))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=partial(morph_collate,
                                           keys=LABEL_KEYS + ('svri_bin',)))
    seg, lab = next(iter(loader))
    assert seg.shape == (32, L), f'PaPaGei 只吃 {L} 点原始段，实得 {tuple(seg.shape)}'

    model = PapageiP().train()
    emb, _ = model(seg)
    loss = ntxent_loss(emb, lab['svri_bin'], temperature=TEMP, ignore_index=-1)
    loss.backward()
    assert torch.isfinite(loss)

    bins = lab['svri_bin'].numpy()
    n_anchor = int(((bins >= 0) & (bins < n_bins)).sum())
    assert n_anchor > 0, '一个有效标签都没有（形态学表与 manifest 对错行了？）'
    print(f'真实语料 OK: loss={loss.item():.4f} 有效标签={n_anchor}/32 '
          f'无标签={int((bins < 0).sum())}')


if __name__ == '__main__':
    import tempfile
    torch.manual_seed(0)
    test_papagei_p_forward()
    test_papagei_s_forward()
    test_embed_segment_matches_forward()
    test_build_model_and_name_from_sd()
    test_as_bcl_rejects_bad_dim()
    test_ntxent_finite_and_backward()
    test_ntxent_matches_reference()
    test_ntxent_ignore_index_changes_value()
    test_ntxent_all_ignored_is_zero()
    test_ntxent_batch1_raises()
    test_svri_bin_edges_n_bins_required()
    test_svri_bin_edges_train_rows_only()
    test_svri_bin_edges_empty_pool_raises()
    test_assign_svri_bins_semantics()
    test_assign_svri_bins_degenerate_edges()
    test_build_transform_order_and_identity()
    test_build_transform_noise_sigma()
    test_build_transform_per_sample()
    with tempfile.TemporaryDirectory() as td:
        for nm, cl in (('papagei_p', PapageiP), ('papagei_s', PapageiS)):
            test_load_frozen_roundtrip(Path(td), nm, cl)
        test_load_frozen_without_cfg(Path(td))
    if CORPUS is None or not MORPH.exists():
        print('== 合成部分全部通过；真实语料/形态学表不可用，跳过端到端 ==')
    else:
        test_real_corpus_pretrain_step()
        print('== PaPaGei 冒烟测试全部通过 ==')
