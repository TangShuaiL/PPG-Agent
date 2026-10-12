"""论文产物的共享口径（**非绘图代码**）。

2026-09-21：论文出图改用 `results/paper/code/make_content_figs.py`（作者新版，
含 KDE 边缘分布、log2 k 轴、经 CVD 校验的配色），`tools/` 下的三个旧出图脚本退役。
但其中两个常量集合是**口径而非绘图代码**，表格/数据生成器要用，故独立成模块 ——
避免「删绘图时把口径一起删掉」，也避免两处各写一份。

| 口径 | 消费者 |
|---|---|
| `BUCKETS` / `load_k` | `tools/make_paper_tables.py`（history 表） |
| `BINS_BP` / `BINS_AGE` / `FINE_STEP` / `XVIEW_BP` | `tools/export_figure_data.py` |
"""
import numpy as np


# ── 历史累积的分桶口径 ──────────────────────────────────────────────
# 从 tools/plot_history_gain.py 抽出（2026-09-21）：论文出图改用
# results/paper/code/ 后该绘图脚本退役，但**桶定义与 k 映射是口径不是绘图代码** ——
# make_paper_tables.py 的 history 表要用，故独立成模块。
BUCKETS = {
    'vital': [(0, 1, 'k=1'), (1, 4, 'k2–4'), (4, 16, 'k5–16'),
              (16, 64, 'k17–64'), (64, 10**9, 'k65+')],
    'aurora': [(0, 1, 'k=1'), (1, 4, 'k2–4'), (4, 8, 'k5–8'),
               (8, 16, 'k9–16'), (16, 32, 'k17–32'), (32, 10**9, 'k33+')],
}


def load_k(meta_path):
    """(受试者, seg_idx) → 段在该人完整段表里的序位 k。

    注意：本 stream 是 2026-09-11 跨源同名修复**之前**materialize 的旧版（meta 里
    没有 sub_src 键），键只能用 sub_names。若 sub_names 有重名，两个人会被并成
    同一条段表——下面会把重名情况打出来，为 0 则无此问题。
    """
    z = np.load(meta_path, allow_pickle=True)
    nm, segi, bnd = z['sub_names'].astype(str), z['seg_idx'], z['bnd']
    dup = len(nm) - len(set(nm))
    if dup:
        print(f'  [warn] stream_meta 有 {dup} 个重名受试者，其 k 会被合并计算')
    key2k = {}
    for p in range(len(bnd) - 1):
        a, b = bnd[p], bnd[p + 1]
        for rank, j in enumerate(np.argsort(segi[a:b], kind='stable')):
            key2k[(nm[p], int(segi[a + j]))] = rank
    return key2k, dup


# ── 分布图的分箱域 ──────────────────────────────────────────────────
# 从 tools/plot_data_distribution.py 抽出（同上）—— export_figure_data.py 要用。
# 核心约束：numpy 的 histogram 把「恰好等于末边界」的值归入最后一桶；若末边界落在
# 数据内部，该规则就会真的触发，而计数型导出**无法区分** 204.0 与 204.5（同一格），
# 两条路径必然对不上。故末边界取 258（> sbp max 249.8）使其永不触发。
# **显示范围另用 XVIEW_BP 裁**，裁掉的量标在图上 —— 裁剪是显式选择，不是静默丢数据。
BINS_BP = np.arange(30, 261, 3.0)      # SBP/DBP 共用一个轴；末边界 258


BINS_AGE = np.arange(15, 96, 1.5)


FINE_STEP = 1.0                        # 导出用的细粒度（必须整除 BINS_BP 步长）


XVIEW_BP = (30, 204)                   # 显示裁剪（沿用旧版观感），非分箱域
