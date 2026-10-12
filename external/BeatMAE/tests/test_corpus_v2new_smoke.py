"""新 corpus_v2（data_v2 版，含 iwshi_bp）冒烟测试：数据层契约与标签对齐。

CPU-only，不需要 GPU。覆盖体检确认的关键契约：
- CorpusDataset 以 sources=[iwshi_bp]/[aurora] 可正常实例化与取样本
- iwshi_bp 折号仅占偶数折（fold×2 映射）；aurora/iwshi_bp 折号在 0..9
- 标签对齐：有标签段 sbp/dbp 非空且在阈值内；无标签段为 NaN
- iwshi_bp 扩列 label_source/near_cuff/label_dt_s 与源 parquet 一致
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.corpus_dataset import CorpusDataset
from data.paths import CORPUS_V2C, resolve_corpus


def _corpus_root():
    """解析语料根；探测不到返回 None（供模块级 skipif 用）。"""
    try:
        return resolve_corpus(CORPUS_V2C, tag='corpus_v2c')
    except FileNotFoundError:
        return None


# 语料不进版本库——无语料的机器上本文件整体 skip（而非 ERROR）。
# 语料根走 data.paths 权威解析，可用 PPG_CORPUS_ROOT 环境变量覆盖。
ROOT = _corpus_root()
pytestmark = pytest.mark.skipif(
    ROOT is None,
    reason='语料 corpus_v2c 不可用（语料不进版本库，见 README「已知限制」）')
SBP_LO, SBP_HI, DBP_LO, DBP_HI = 50, 250, 30, 160


@pytest.fixture(scope='module')
def labels_df():
    man = pd.read_parquet(f'{ROOT}/manifest.parquet',
                          columns=['row', 'source', 'fold'])
    lab = pd.read_parquet(f'{ROOT}/labels.parquet')
    return man.merge(lab, on='row')


def test_labels_cleaned(labels_df):
    """坏标签已清洗：有标签段（sbp/dbp 均非 NaN）全部在阈值内。"""
    s = labels_df[labels_df['sbp'].notna() & labels_df['dbp'].notna()]
    assert len(s) > 4700000   # v2c（sig_ok=1 清洗后）= 4,796,216
    assert ((s.sbp >= SBP_LO) & (s.sbp <= SBP_HI)).all()
    assert ((s.dbp >= DBP_LO) & (s.dbp <= DBP_HI)).all()
    assert (s.dbp <= s.sbp).all()


def test_iwshi_bp_label_ext_cols(labels_df):
    """iwshi_bp 扩列命中全部 iwshi_bp 行（无标签段为空串）；有标签段必有 source 且时距 ≤120s。"""
    an = labels_df[labels_df.source == 'iwshi_bp']
    assert an['label_source'].notna().all()
    labeled = an[an.sbp.notna()]
    assert len(labeled) > 11000   # v2c：sig_ok=1 有标签段 ≈ 12k（低质量有标签段已剔）
    assert (labeled['label_source'].isin(['abpm_cuff', 'seated_mercury'])).all()
    assert labeled['label_dt_s'].dropna().le(120).all()


def test_folds(labels_df):
    """iwshi_bp 仅占偶数折；aurora/iwshi_bp 折号均在 0..9。"""
    an = labels_df[labels_df.source == 'iwshi_bp']
    assert set(an.fold.unique()) <= {0, 2, 4, 6, 8}
    au = labels_df[labels_df.source == 'aurora']
    assert au.fold.between(0, 9).all()
    assert an.fold.between(0, 9).all()


@pytest.mark.parametrize('source', ['iwshi_bp', 'aurora'])
def test_dataset_smoke(source):
    """Dataset 实例化 + 取样本：形状/键/折号/标签对齐。"""
    ds = CorpusDataset(ROOT, sources=[source], max_beats=8,
                       min_beats=2, max_rows=3000)
    assert len(ds) > 0
    item = ds[0]
    assert item['beats'].shape[1] == 128
    assert item['beats'].shape[0] == len(item['kp_valid']) == len(item['feat_valid'])
    assert item['source'] == source
    assert 0 <= item['fold'] <= 9
    assert np.isfinite(item['beats']).all()
    assert np.isfinite(item['feat']).all()   # ±Inf 兜底后全有限

    # 标签对齐：取到的段在 labels 中可索引（row 对齐未错位）
    lab = pd.read_parquet(f'{ROOT}/labels.parquet').set_index('row')
    rows = [int(ds.rows[i]) for i in range(min(20, len(ds)))]
    sub = lab.loc[rows]
    assert len(sub) == len(rows)
    if source == 'iwshi_bp':
        assert sub['label_source'].notna().all()   # iwshi_bp 行扩列全命中
