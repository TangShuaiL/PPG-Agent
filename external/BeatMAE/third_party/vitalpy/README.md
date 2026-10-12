# third_party/vitalpy — VitalPy 派生代码

本目录是 [VitalPy](https://github.com/SCAI-Lab/VitalPy) 的**派生副本**，为 BeatMAE 的
40 维拍级形态学特征服务。**它的存在是整仓采用 GPL-3.0 的原因。**

## 出处

| | |
|---|---|
| 上游 | VitalPy: A Vital Signal Analysis Package |
| 版权 | Copyright (c) SCAI-Lab, ETH Zürich |
| 作者 | Ana Cisnal, Yanke Li, Bertram Fuchs, Mehdi Ejtehadi, Robert Riener, Diego Paez-Granados |
| 许可 | **GNU GPL v3.0**（全文见仓库根 `LICENSE`） |
| 转承 | BeatMAE ← VitalPy `src/vitalpython/ppg/` |

**引用时请引 VitalPy 原文**：

> A. Cisnal, Y. Li, B. Fuchs, M. Ejtehadi, R. Riener and D. Paez-Granados,
> "Robust Feature Selection for BP Estimation in Multiple Populations: Towards
> Cuffless Ambulatory BP Monitoring," *IEEE J. Biomed. Health Inform.*,
> doi: 10.1109/JBHI.2024.3411693.

## 许可

GPL-3.0 的 copyleft 要求：分发包含 GPL 代码的作品，整体须以 GPL-3.0 分发。
**本目录即是整仓采用 GPL-3.0 的原因。**

## 目录内容与修改清单

本目录只收 40 维特征闭包所需的 7 个文件。每个文件顶部有 GPL-3.0 §5 要求的
版权与修改声明；下表是汇总。

| 文件 | 相对 VitalPy 原件的改动 |
|---|---|
| `params.py` | 2 行 — 仅 import 路径 |
| `utils.py` | 2 行 — import 路径；`seaborn` 改为惰性导入（仅调试函数 `get_mean_df` 用） |
| `preprocess/validate.py` | 2 行 — 仅 import 路径 |
| `features/utils.py` | **4 行 + 2 处功能性修改**（见下）；`sklearn` 改为惰性导入 |
| `features/features_config.py` | 25 行 — import 路径；**删除 `TEMPORAL_FEATURES` 列表** |
| `features/statistical_features.py` | **0 行（逐字未改）** |
| `features/spectral_features.py` | **0 行（逐字未改）** |

### 两处功能性修改（`features/utils.py`）

这两处**不是** BeatMAE 改的，是这份代码在到达本仓之前的那一次转承中引入的；
相对于 VitalPy 原件属行为差异，在此点名以免读者比对时困惑：

1. `extract_key_points` 增加「systolic 峰落在边缘 → 该拍分割不当」的提前返回守卫
2. `extract_key_points` 在「无相对极大值」时回退到整段 systolic 导数的 `argmax`

### 未搬运的部分及原因

| 未搬 | 原因 |
|---|---|
| `preprocess/filter_baseline.py` | 只被 `filter_ppg` 使用，而 `filter_ppg` 不在 40 维闭包内。副产品：去掉了外部依赖 `BaselineRemoval` |
| `features/temporal_features.py` | `extract_key_temporal` 自包含，不调用它。故 `features_config.py` 中对应的 star-import 与 `TEMPORAL_FEATURES` 列表一并删除 |
| `utils.py` 的 `plot_*` | **保留**（`plot_key_points_dev` 会被 `extract_key_points(show=True)` 调到），但 matplotlib 已是本仓依赖，seaborn 改为惰性导入 |

## 依赖

- 必需：`numpy` `pandas` `scipy`
- **`biosppy`** —— `features/spectral_features.py` 的 `fsqi` 用 `biosppy.signals.ecg`
- 惰性（只在调试分支触发）：`matplotlib` `seaborn` `scikit-learn`

## 入口

读者通常不需要直接调用本目录 —— 用 `preprocess/features/` 的
`extract_segment_features()`，或 `scripts/extract_beat_features.py`。
