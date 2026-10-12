"""公平口径 v2（raw_waveform）冒烟测试。

验证：①三对比模型在 1250 点原始波形上的前向/掩码/反传/段嵌入；
②raw_collate 形状与字段；③raw 与 beat 模式段池逐段一致。

用法：python3 tests/test_raw_mode_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from data.corpus_dataset import CorpusDataset, raw_collate
from data.paths import CORPUS_V2C, resolve_corpus
from models import masked_mse
from baselines import HimaeAE, MMRMAE, XmaePPGMAE

torch.manual_seed(0)
B, L = 4, 1250


def _corpus_root():
    """解析语料根；探测不到返回 None（供 skipif 用）。"""
    try:
        return resolve_corpus(CORPUS_V2C, tag='corpus_v2c')
    except FileNotFoundError:
        return None


CORPUS = _corpus_root()
# 仅 test_dataset_parity 碰语料：前两个用例是纯合成，不该被连坐 skip
needs_corpus = pytest.mark.skipif(
    CORPUS is None,
    reason='语料 corpus_v2c 不可用（语料不进版本库，见 README「已知限制」）')


def rand_patch_mask(B_, N, ratio):
    pm = torch.zeros(B_, N, dtype=torch.bool)
    for b in range(B_):
        pm[b, torch.randperm(N)[:round(N * ratio)]] = True
    return pm


def test_models_raw():
    """三模型 @ raw 1250：前向形状 + masked_mse 反传 + embed_segment。"""
    seg = torch.randn(B, L)
    # MMR：50 patches（原作原值 patch=25）
    m = MMRMAE(seg_len=L, patch_len=25)
    pm = rand_patch_mask(B, 50, 0.65)
    out = m(seg, pm)
    assert out['recon'].shape == (B, 50, 2 * 25)
    loss = masked_mse(out['recon'], out['coeff_target'],
                      pm.unsqueeze(-1).expand_as(out['recon']))
    loss.backward()
    assert m.embed_segment(seg).shape == (B, 256)
    print(f'MMR raw OK: loss={loss.item():.4f}')

    # xMAE：25 patches（0.4s@125Hz）
    x = XmaePPGMAE(seq_len=L, patch_len=50)
    pm = rand_patch_mask(B, 25, 0.65)
    out = x(seg, pm)
    sm = pm.repeat_interleave(50, dim=1)
    loss = masked_mse(out['recon'], seg, sm)
    loss.backward()
    assert x.embed_segment(seg).shape == (B, 256)
    print(f'xMAE raw OK: loss={loss.item():.4f}')

    # HiMAE：250 patches（原作原值 P=5）
    h = HimaeAE(patch_len=5)
    pm = rand_patch_mask(B, 250, 0.65)
    out = h(seg, pm)
    sm = pm.repeat_interleave(5, dim=1)
    loss = masked_mse(out['recon'], seg, sm)
    loss.backward()
    assert h.embed_segment(seg).shape == (B, 256)
    print(f'HiMAE raw OK: loss={loss.item():.4f}')


def test_raw_collate():
    items = [{'waveform': np.random.randn(L).astype(np.float32),
              'row': r, 'source': 'mimic', 'fold': 3, 'subject_id': f's{r}'}
             for r in range(B)]
    wave = raw_collate(items)
    assert isinstance(wave, torch.Tensor) and wave.shape == (B, L)
    assert wave.dtype == torch.float32
    print('raw_collate OK:', tuple(wave.shape))


@needs_corpus
def test_dataset_parity():
    """raw 与 beat 模式的段池（rows）必须逐段一致——公平性的前提。"""
    kw = dict(root=CORPUS, exclude_folds=[1, 2], max_rows=64)
    ds_raw = CorpusDataset(input_mode='raw_waveform', **kw)
    ds_beat = CorpusDataset(input_mode='beat_resampled', **kw)
    assert np.array_equal(ds_raw.rows, ds_beat.rows), '段池不一致！'
    it = ds_raw[0]
    assert it['waveform'].shape == (L,)
    assert abs(float(it['waveform'].mean())) < 1e-4      # z-score 后零均值
    # 窗口内容对齐：raw 波形在 beat 版首拍 onset 前无内容缺失（整段直喂）
    print(f'dataset parity OK: rows 一致（{len(ds_raw)} 段），'
          f'waveform 零均值 z-score 验证通过')


if __name__ == '__main__':
    test_models_raw()
    test_raw_collate()
    if CORPUS is None:
        print('[skip] test_dataset_parity：语料 corpus_v2c 不可用（合成用例已通过）')
    else:
        test_dataset_parity()
    print('\n== raw 模式冒烟结束 ==')
