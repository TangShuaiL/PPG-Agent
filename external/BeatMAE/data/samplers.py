"""samplers：源平衡采样（mimic/vital/aurora 三库统一预训练）。

策略（v1_plan §5.2 / 源平衡决策）：
- 每步按源权重抽源，再在该源内有放回抽段；
- 权重默认 sqrt(n_source)：防止 mimic（264 万）淹没 aurora（3.2 万），
  又不至于把 aurora 过采样到失真（sqrt 后 aurora:~1% → 抽样占比 ~6%）。
"""
import numpy as np
from torch.utils.data import Sampler

__all__ = ['SourceBalancedSampler', 'EpochTraverseSampler']

SOURCE_WEIGHT_MODES = ('sqrt', 'equal', 'natural')


class SourceBalancedSampler(Sampler):
    """按源权重混合的无限采样器（每 epoch 产出一个长度确定的索引序列）。

    Args:
        dataset:   需带 .source 数组（CorpusDataset）
        n_samples: 每 epoch 抽样总数（= steps_per_epoch × batch_size）
        mode:      'sqrt'=√样本数加权（默认） / 'equal'=三源等权 /
                   'natural'=按自然占比
        seed:      随机种子；每 epoch 用 set_epoch 更新
    """

    def __init__(self, dataset, n_samples: int, mode: str = 'sqrt',
                 seed: int = 0):
        assert mode in SOURCE_WEIGHT_MODES
        self.dataset = dataset
        self.n_samples = int(n_samples)
        self.mode = mode
        self.seed = seed
        self.epoch = 0

        src = np.asarray(dataset.source)
        self.sources = np.unique(src)
        self.idx_by_src = {s: np.nonzero(src == s)[0] for s in self.sources}
        counts = np.array([len(self.idx_by_src[s]) for s in self.sources],
                          dtype=np.float64)
        if mode == 'sqrt':
            w = np.sqrt(counts)
        elif mode == 'equal':
            w = np.ones_like(counts)
        else:
            w = counts
        self.src_prob = w / w.sum()
        self.counts = counts

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        src_draw = rng.choice(len(self.sources), size=self.n_samples,
                              p=self.src_prob)
        out = np.empty(self.n_samples, dtype=np.int64)
        for si, s in enumerate(self.sources):
            pos = np.nonzero(src_draw == si)[0]
            if pos.size == 0:
                continue
            pool = self.idx_by_src[s]
            out[pos] = pool[rng.integers(0, pool.size, size=pos.size)]
        return iter(out.tolist())

    def __len__(self):
        return self.n_samples

    def stats(self) -> dict:
        return {s: float(p) for s, p in zip(self.sources, self.src_prob)}


class EpochTraverseSampler(Sampler):
    """全局无放回 shuffle 遍历采样器（标准 epoch 语义，v1 补充模式）。

    与 SourceBalancedSampler（有放回流式，单轮覆盖 ~17%）的区别：
    - 每 epoch 恰好完整遍历全池一遍（无放回，覆盖率 100%）；
    - 全局 permutation 保证同受试者的段随机散布：manifest 按源排序
      导致同受试者段在索引上连续，permutation 后与任意指定段相邻的
      概率 ≈ 2·n_subj/N（最坏 <0.3%），落入同一 batch 的概率更低，
      无需额外的受试者级去相关；
    - 源占比退为自然占比（aurora ~0.8%）——但每段每 epoch 恰好见
      1 次，小库覆盖均匀性反而优于流式（流式下 aurora 1.37 遍/epoch
      而 mimic 仅 0.15 遍）。源平衡与无放回互斥，使用本模式时在
      实验记录中注明该差异。

    Args:
        dataset: 任意 Dataset（仅用 len）
        seed:    随机种子；每 epoch 用 set_epoch 更新（seed+epoch，可复现）
    """

    def __init__(self, dataset, seed: int = 0):
        self.dataset = dataset
        self.n = len(dataset)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        return iter(rng.permutation(self.n).tolist())

    def __len__(self):
        return self.n
