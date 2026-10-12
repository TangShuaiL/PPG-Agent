"""PaPaGei 的 sVRI 分箱（对比学习的正样本定义）。

官方出处 `dataset.py`（`PPGDatasetLabelsArray.__init__`）：

    svri = np.digitize(df['svri'].values, bins=self.bin_data(df['svri'].values, bins_svri)) - 1
    # bin_data: np.arange(min, max + w, w)，w = (max - min) / num_buckets
    # bins_svri = 8（training_pospair.py / training_mt.py 两份脚本都是 8）

**等宽 8 箱，边界取传入 df 的 min/max。** 关键在"传入的 df 是什么"：官方先把
df 交给 `harmonize_datasets`，那里按 `svri∈(0,2) & skewness∈(-3,3) & ipa∈(-10,10)`
滤过一遍，所以官方边界 = **合格子集的 min/max**，而不是全池的。

本库不丢段（见 `scripts/compute_morphology.py` 模块 docstring 偏差 1：段池要与
xMAE / HiMAE / MMR 三族同构），于是复现官方几何的做法是：
  - 边界只在合格子集 `(0,2)` 上取（同一批点、同一几何）；
  - 落在 `(0,2)` 外的段（本库实测 ~2.7%）与 svri 计算失败的段（~0.3%）
    标 **-1 = 无有效标签**：它们照常进前向、照常当负样本，但既不做 anchor
    （没有正样本可拉）也不进任何箱（不会被误当成极值段的同箱正样本）。
    `ntxent_loss` 对 `n_pos == 0` 的 anchor 本来就是跳过。
    这样"带标签的子集"与官方**逐箱同构**，多出来的段只是额外的负样本。

P 的 `bins_skewness=5`（skewness 也分箱）在 `training_pospair.py` 里算完却**从未
进损失**——`criterion(embeddings, svri)` 只吃 svri，故本库不移植那段死代码。
S 的 `bins_skewness=0` 表示 skewness 用**原始值**（MoE 第二头回归它），
也不需要分箱。两族真正用到的标签只有：svri 箱（对比）+ 原始 ipa/skewness（S 的辅助回归）。
"""
import numpy as np

__all__ = ['SVRI_RANGE', 'svri_bin_edges', 'assign_svri_bins']

SVRI_RANGE = (0.0, 2.0)    # 官方 harmonize_datasets 的合格区间


def svri_bin_edges(svri: np.ndarray, n_bins: int,
                   rows: np.ndarray | None = None,
                   lo: float = SVRI_RANGE[0],
                   hi: float = SVRI_RANGE[1]) -> np.ndarray:
    """合格子集的 min/max 等宽 n_bins 箱边界（官方 `bin_data`）→ (n_bins+1,)。

    n_bins: **必传，不设默认值**——它是训练超参，唯一权威是 config 的
            `papagei.bins_svri`（官方两份脚本都是 8）。留个默认值等于两处各说
            各话，改了 yaml 却不生效。
    rows: 参与定边界的行号（**必须传训练池的行号**——边界是训练超参，把验证/
          测试折算进来就是信息泄漏）。None = 用全表（只适合调试）。
    """
    v = np.asarray(svri, np.float64)
    if rows is not None:
        v = v[np.asarray(rows, np.int64)]
    v = v[np.isfinite(v) & (v > lo) & (v < hi)]
    if v.size == 0:
        raise ValueError(f'定边界失败：({lo}, {hi}) 内没有样本（rows 传错？）')
    mn, mx = float(v.min()), float(v.max())
    w = (mx - mn) / n_bins
    return mn + w * np.arange(n_bins + 1)


def assign_svri_bins(svri: np.ndarray, edges: np.ndarray,
                     lo: float = SVRI_RANGE[0],
                     hi: float = SVRI_RANGE[1]) -> np.ndarray:
    """→ (N,) int64 箱号，**-1 = 无有效标签**（NaN / 落在合格区间外）。

    与官方 `np.digitize(x, edges) - 1` 同式；差别只在官方把区间外的段丢掉，
    本库给它们 -1（见模块 docstring）。
    """
    edges = np.asarray(edges, np.float64)
    v = np.asarray(svri, np.float64)
    b = np.digitize(v, edges) - 1
    valid = np.isfinite(v) & (v > lo) & (v < hi)
    # clip 只为退化情形兜底：合格子集若全等值（w=0，边界全相等），digitize 会把
    # v==edge 判到 n_bins（越界）。正常语料 w>0，这一行不生效。
    return np.where(valid, np.clip(b, 0, len(edges) - 2), -1).astype(np.int64)
