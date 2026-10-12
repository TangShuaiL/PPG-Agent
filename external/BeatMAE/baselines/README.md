# baselines/ — 对比模型（**非本文贡献**）

本目录只放论文里作对照的四个模型族。**本文模型（BeatMAE）在 `models/`**，
两边分开摆放是刻意的：主干与对照的边界一眼可见。

| 目录 | 模型 | 预训练入口 |
|---|---|---|
| `himae/` | HiMAE —— 层级 CNN U-Net MAE | `train/pretrain_baseline.py` |
| `mmr/` | MMR —— 小波系数图 + ViT 非对称 MAE | 同上 |
| `xmae/` | xMAE —— 只取 PPG 通路（原作是 PPG→ECG 跨模态） | 同上 |
| `papagei/` | PaPaGei P / S —— 形态学伪标签引导的对比学习 | `train/pretrain_papagei.py` |

## ⚠ 来源逐族不同：两族自写、两族有上游

**都不是作者的官方代码**，但四族的来路并不一样，不要一概而论：

| 族 | 来源 | 上游许可 |
|---|---|---|
| HiMAE / MMR | 按论文描述**独立实现**（上游未公开代码，无参考实现可依） | — |
| xMAE | **复现自其公开实现**：`utils/model_arch/xmae.py` 的 `SinglePath`，只取 PPG encoder 侧。编码器的构造块（ConvBNAct / InceptionBlock1D / U-Net stem / transformer block）与其公开实现同构，两个顶层类（`XmaePpgEncoder` / `XmaePPGMAE`）为本库自写 | 上游未附许可声明 |
| PaPaGei P/S | 复现自其官方仓库，按本库惯例重写为独立实现 | **BSD-3-Clause-Clear (© 2024 Nokia)**，署名见各文件头 |

**本目录不 import `references/`**（那目录是上游参考副本，在 `.gitignore` 里，
clone 下来跑不了）。

## 对比的口径：统一协议，不是「官方权重」

**所有模型（含 BeatMAE）在统一协议下训练与评测**：同语料、同折划分、同探针头、
同 best-seed 口径。**对比的说服力来自这个统一协议，而非「用了各方官方权重」** ——
后者反而会因各论文的数据集、划法、指标不同而失去可比性。

因此：**本目录复现出的数值可能与各官方发布不一致，这属预期**，不构成对比口径的一部分。

## 用法

```python
from baselines import HimaeAE, MMRMAE, XmaePPGMAE   # 与 models/ 的用法一致
from baselines.papagei import build_model, build_transform, ntxent_loss
```

`train/probe.py` / `scripts/extract_embeddings.py` 的 `--kind` 仍用
`himae | mmr | xmae | papagei` 这些名字，未随目录变动。
