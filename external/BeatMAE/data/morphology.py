"""morphology：PaPaGei 形态学伪标签表的加载与挂载。

`scripts/compute_morphology.py` 产出的 parquet 按 `row` 与 manifest 对齐，而
manifest 的 `row` 恰好是 0..N-1 的稠密行号（已实测），所以展成稠密数组后可以
按 row 直接下标查表——不需要 join，也不需要按各折/各源重建索引。

**本模块只管"挂标签"，不做分箱。** sVRI 的 8 等宽分箱是训练超参（边界必须
来自训练切分，否则验证/测试折的信息会漏进边界），落在 `baselines/papagei/labels.py`
由训练脚本按自己的池子现算。这里原样交出三个测量值与两个失败标志。

缺失行（如 `--limit` 试跑产出的半张表）一律填 NaN / fail=True——宁可让下游把
它当"无标签"，也不能静默当成有效测量值。

**表路径不在本模块**：产者默认值 = `scripts/compute_morphology.py::DEFAULT_OUT`，
供给训练的是 config 的 `morphology:`（17 机路径不同，必须覆盖）。
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ['FLOAT_KEYS', 'BOOL_KEYS', 'LABEL_KEYS',
           'load_morphology', 'WithMorphology', 'morph_collate']

FLOAT_KEYS = ('svri', 'skewness', 'ipa')
BOOL_KEYS = ('svri_fail', 'ipa_fail')
LABEL_KEYS = FLOAT_KEYS + BOOL_KEYS


def load_morphology(path: str) -> dict:
    """读形态学表 → 稠密数组字典（键见 LABEL_KEYS），下标 = manifest 行号。

    数组长度 = max(row)+1（manifest 行号 0..N-1 稠密，见模块 docstring）。
    """
    df = pd.read_parquet(path)
    rows = df['row'].to_numpy(np.int64)
    n = int(rows.max()) + 1
    out = {}
    for k in FLOAT_KEYS:
        a = np.full(n, np.nan, np.float32)
        a[rows] = df[k].to_numpy(np.float32)
        out[k] = a
    for k in BOOL_KEYS:
        a = np.ones(n, bool)                 # 缺失行 = fail（保守）
        a[rows] = df[k].to_numpy(bool)
        out[k] = a
    return out


class WithMorphology(Dataset):
    """在 CorpusDataset 之上按 row 挂形态学伪标签（PaPaGei 专用视图）。

    只加键、不改段：底层仍返回同一段波形，因此**段池与另三族逐段一致**。
    另两个挂件：
      - `svri_bin`：外部按训练池算好边界后 set（`baselines/papagei/labels.py`），
        是 PaPaGei 的对比标签；未设时 __getitem__ 不输出该键（提取嵌入等推理
        路径不需要标签，也就不会因为缺表而报错）。
      - 属性透传：SourceBalancedSampler 要读 `.source`，其余同。
    """

    def __init__(self, base: Dataset, morph: dict, svri_bin: np.ndarray | None = None):
        self.base = base
        self.morph = morph
        self.svri_bin = svri_bin

    def __len__(self):
        return len(self.base)

    def __getattr__(self, k):
        # __getattr__ 只在常规查找失败时触发；self.base 由 __init__ 正常设置，
        # 但 unpickle / 半初始化时会走到这里 → 必须挡住，否则无限递归
        if k in ('base', 'morph', 'svri_bin'):
            raise AttributeError(k)
        return getattr(self.base, k)

    def __getitem__(self, i):
        it = dict(self.base[i])
        r = int(it['row'])
        for k, a in self.morph.items():
            it[k] = a[r]
        if self.svri_bin is not None:
            it['svri_bin'] = int(self.svri_bin[r])
        return it


def morph_collate(batch: list, keys=LABEL_KEYS) -> tuple:
    """→ (seg (B, L) float32, labels dict[str, Tensor])。

    波形部分与 `raw_collate` 同（固定 1250 点，无需 padding）；标签逐样本堆叠。
    """
    seg = torch.from_numpy(
        np.stack([np.asarray(it['waveform'], dtype=np.float32) for it in batch]))
    lab = {k: torch.from_numpy(np.stack([it[k] for it in batch])) for k in keys}
    return seg, lab
