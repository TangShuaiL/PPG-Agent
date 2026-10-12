"""PaPaGei 对比损失：多正样本 NT-Xent（对比模型）。

官方两份脚本都实例化 `pytorch_metric_learning.losses.NTXentLoss()`（temperature
默认 0.07），本库不引这个依赖，按同样语义本地实现。

**正样本由 sVRI 分箱定义，不由数据增强的两次视图定义。** 这一点容易搞错：官方
`dataset.py` 里确有双视图类 `PPGDatasetVanillaSimCLR`，但 P/S 两份训练脚本用的都是
`PPGDatasetLabelsArray`，其 `__getitem__` 只返回**一个**增强视图。于是"正样本对"=
**同一 batch 内 sVRI 落到同一分箱的段**，负样本 = 其余。所以这本质是 SupCon
（Khosla et al. 2020）而不是 SimCLR 双视图对比——代价是 batch 内必须有同箱样本，
收益是标签给了更强的正样本信号。`bins_svri=8`（见 `scripts/compute_morphology.py`）。

**官方 P 的 MultiSimilarityMiner 没有移植**：`training_pospair.py` 里写的是
`criterion(embeddings, svri, hard_pairs)`，而 PML 的签名是
`forward(embeddings, labels, ref_emb=None, ref_labels=None)` —— 第三位收的是
`ref_emb`，不是 miner 产出的索引元组。也就是说那个 miner 要么直接报错、要么静默
不生效，官方**实际**优化的就是纯 NT-Xent。本实现只移植实际生效的目标函数；
若后续要做"难例挖掘"消融，另加 `MultiSimilarityLoss` 更干净（见 docs 里的待办）。
"""
import torch
import torch.nn.functional as F


def ntxent_loss(emb: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.07,
                ignore_index: int | None = None) -> torch.Tensor:
    """多正样本 NT-Xent / SupCon。

    emb:          (B, d) 未归一化嵌入（内部 L2 归一化，对齐 PML 默认的 CosineSimilarity）
    labels:       (B,)   长整型，正样本 = 同标签（同 sVRI 箱）
    ignore_index: 该标签值表示"无有效标签"（`baselines/papagei/labels.py` 用 -1 标
                  落在 sVRI 合格区间外 / 算失败的段）。这些样本**仍当负样本**参与
                  分母，但既不做 anchor 也不与彼此互为正样本——不加这一条，所有
                  -1 的段会因"标签相等"被当成同一个箱互相拉近，等于凭空造出一个
                  几万段的伪簇。None = 关（官方原式，标签全部有效）。
    返回标量。逐 anchor 形式：

        L_i = logsumexp_{j≠i}(sim_ij / T) − mean_{p∈P_i}(sim_ip / T)

    即 PML `NTXentLoss` 的算法（分母含正样本、按正样本**个数**平均而非求和；
    SimCLR 原式是 `−log(Σ_pos / Σ_all)`，两者差一个 per-anchor 的 log|P_i| 常数）。
    batch 内没有正样本的 anchor 跳过（不计入均值，也不产生梯度）。
    """
    B = emb.shape[0]
    assert B > 1, 'NT-Xent 需要 batch ≥ 2'
    z = F.normalize(emb.float(), dim=-1)
    sim = z @ z.t() / temperature                                  # (B, B)

    self_mask = torch.eye(B, dtype=torch.bool, device=emb.device)
    ok = (labels != ignore_index) if ignore_index is not None \
        else torch.ones_like(labels, dtype=torch.bool)
    pos = ((labels[:, None] == labels[None, :]) & ~self_mask
           & ok[:, None] & ok[None, :])
    sim = sim.masked_fill(self_mask, float('-inf'))
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()       # 数值稳定
    sim = sim.masked_fill(self_mask, float('-inf'))

    log_denom = torch.logsumexp(sim, dim=1)                        # 排除自身
    log_pos_sum = torch.logsumexp(sim.masked_fill(~pos, float('-inf')), dim=1)
    n_pos = pos.sum(dim=1)
    valid = (n_pos > 0) & ok
    if not valid.any():
        return emb.sum() * 0.0
    return (log_denom[valid] - log_pos_sum[valid] / n_pos[valid]).mean()
