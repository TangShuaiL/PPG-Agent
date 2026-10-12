"""数据层真实数据测试：corpus_v2c → RegionMasker → BeatMAE 端到端。

语料不进版本库——无语料的机器上本文件整体 skip（而非 ERROR）。
语料根走 data.paths 权威解析，可用 PPG_CORPUS_ROOT 环境变量覆盖。

用法：python3 tests/test_corpus_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from data import CorpusDataset, RegionMasker, beat_collate, zero_masked_regions
from data.paths import CORPUS_V2C, resolve_corpus
from models import BeatMAE, masked_mse


def _corpus_root():
    """解析语料根；探测不到返回 None（供模块级 skipif 用）。"""
    try:
        return resolve_corpus(CORPUS_V2C, tag='corpus_v2c')
    except FileNotFoundError:
        return None


ROOT = _corpus_root()
pytestmark = pytest.mark.skipif(
    ROOT is None,
    reason='语料 corpus_v2c 不可用（语料不进版本库，见 README「已知限制」）')


def _first_batch(ds):
    """取一个真实 batch（fixture 与脚本模式共用，避免两处各建一次 loader）。"""
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        collate_fn=beat_collate, num_workers=0)
    return next(iter(loader))


# pytest 模式的 fixture（脚本模式 __main__ 仍直接传参，双模式兼容）
@pytest.fixture(scope='module')
def ds():
    return CorpusDataset(ROOT, max_rows=5000)


@pytest.fixture(scope='module')
def batch(ds):
    return _first_batch(ds)


def test_single_item(ds):
    assert len(ds) > 0
    it = ds[0]
    K = it['beats'].shape[0]
    assert it['beats'].shape == (K, 128)
    assert it['keypoints'].shape == (K, 10)
    assert np.isfinite(it['beats']).all()
    # 关键点：有效坐标在 [0,127]，无效为 -1
    kp9 = it['keypoints'][:, :9]
    vmask = np.broadcast_to(it['kp_valid'][:, None], kp9.shape)
    assert ((kp9 >= 0) & (kp9 <= 127))[vmask].all()
    assert (kp9[~it['kp_valid']] == -1).all()
    # onset 升序且在 10s 内
    assert (np.diff(it['onset']) > 0).all() and it['onset'].max() < 10.0
    # 侧信息合理范围
    assert (it['side'][:, 0] > 100).all() and (it['side'][:, 0] < 2100).all()
    hr = it['side'][:, 1]
    assert ((hr == 0) | ((hr >= 20) & (hr <= 200))).all()
    assert it['feat'].shape == (K, 40)
    print(f'item OK: K={K} source={it["source"]} fold={it["fold"]} '
          f'kp_valid={it["kp_valid"].mean():.2f} feat_valid={it["feat_valid"].mean():.2f}')


def test_index_filter(ds):
    """严格模式：排除测试折后样本量减少。"""
    n_all = len(ds)
    ds_tr = CorpusDataset(ROOT, exclude_folds={1, 2}, max_rows=5000)
    assert len(ds_tr) < n_all, '折过滤未生效'
    folds = np.unique(ds_tr.fold)
    assert 1 not in folds and 2 not in folds
    ds_m = CorpusDataset(ROOT, sources={'mimic'}, max_rows=5000)
    assert set(ds_m.source) == {'mimic'}
    print(f'filter OK: full={n_all} train(fold!=1,2)={len(ds_tr)} mimic-only={len(ds_m)}')


def test_beat_stats(ds):
    """坐标基准修正后的拍统计回归（中位拍数≈13、kp_valid 高）。"""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), 80, replace=False)
    ks, kpv, hrs = [], [], []
    for i in idx:
        it = ds[int(i)]
        ks.append(it['beats'].shape[0])
        kpv.append(it['kp_valid'].mean())
        hr = it['side'][:, 1]
        hrs.extend(hr[hr > 0].tolist())
    ks = np.array(ks)
    assert ks.mean() > 8, f'平均拍数过低（坐标换算异常？）: {ks.mean():.1f}'
    assert np.mean(kpv) > 0.6, f'kp_valid 均值过低: {np.mean(kpv):.3f}'
    hrs = np.array(hrs)
    assert 40 < np.median(hrs) < 120, f'HR 中位数异常: {np.median(hrs):.1f}'
    print(f'stats OK: K均值={ks.mean():.1f} kp_valid={np.mean(kpv):.3f} '
          f'HR中位={np.median(hrs):.1f}bpm')


def test_collate_and_loader(ds, batch):
    B, K = batch['beats'].shape[:2]
    assert batch['beat_pad_mask'].shape == (B, K)
    n_real = int((~batch['beat_pad_mask']).sum())
    assert n_real == sum(ds[i]['beats'].shape[0] for i in range(8))
    assert torch.isfinite(batch['beats']).all()
    print(f'collate OK: batch=({B},{K},128) pad比例={batch["beat_pad_mask"].float().mean():.2f}')


def test_end_to_end(batch):
    """真实 batch → RegionMasker → BeatMAE → 损失反传。"""
    masker = RegionMasker()
    gen = torch.Generator().manual_seed(0)
    m = masker(batch['keypoints'], batch['kp_valid'],
               batch['beat_pad_mask'], generator=gen)
    ratio = m['sample_mask'].sum().float() / (
        (~batch['beat_pad_mask']).sum() * 128)
    assert 0.15 < ratio < 0.70, f'真实数据掩码比例异常: {ratio:.3f}'
    beats_z = zero_masked_regions(batch['beats'], m['sample_mask'])

    model = BeatMAE(beat_len=128, d_model=192, depth=6, nhead=6,
                    n_aux_feats=40)
    out = model(beats_z, batch['beat_pad_mask'], batch['onset'],
                batch['side'], m['beat_mask'], m['region_mask'])
    loss = masked_mse(out['recon'], batch['beats'], m['sample_mask'])
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    print(f'end-to-end OK: mask_ratio={ratio:.3f} strategy={m["strategy"]} '
          f'recon={loss.item():.4f}')


if __name__ == '__main__':
    if ROOT is None:
        sys.exit('语料 corpus_v2c 不可用；可用 PPG_CORPUS_ROOT 环境变量指定。')
    ds = CorpusDataset(ROOT, max_rows=5000)
    test_single_item(ds)
    test_index_filter(ds)
    test_beat_stats(ds)
    batch = _first_batch(ds)
    test_collate_and_loader(ds, batch)
    test_end_to_end(batch)
    print('== 数据层真实数据冒烟测试全部通过 ==')
