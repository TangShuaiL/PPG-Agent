"""对比模型（baselines）—— 论文复现，**非本文贡献**。

与 `models/` 分开摆放是刻意的：`models/` 只放本文模型（BeatMAE）与共享组件，
本目录只放对照用的四个模型族，读者一眼能看出边界。

⚠ 这四族都是**独立重实现**，不是各作者的官方代码
    HiMAE / MMR   按论文描述实现
    xMAE          复现自其 `utils/model_arch/xmae.py`（只取 PPG 通路）
    PaPaGei P/S   复现自其官方仓库（© 2024 Nokia, BSD-3-Clause-Clear），
                  按本库惯例重写为独立实现，不 import 官方代码

**所有模型（含 BeatMAE）在统一协议下训练与评测**——同语料、同折划分、同探针头、
同 best-seed 口径。**对比的说服力来自这个统一协议，而非「用了官方权重」**；
本目录的数值因此可能与各官方发布不一致，属预期，不构成对比口径的一部分。

目录：
    himae/    层级 CNN U-Net MAE
    mmr/      MMR 小波系数图 + ViT MAE
    xmae/     xMAE PPG encoder（只取 PPG 通路）
    papagei/  PaPaGei P/S（形态学伪标签引导的对比学习）

用法与本文模型一致，只是改从这里取：
    from baselines import HimaeAE, MMRMAE, XmaePPGMAE
"""
from .himae import HimaeAE, HimaeDecoder, HimaeEncoder
from .mmr import MMRMAE, MmrDecoder, MmrViTEncoder, build_coeff_map, haar_dwt
from .papagei import (PapageiP, PapageiS,
                      build_model as papagei_build_model,
                      name_from_sd as papagei_name_from_sd,
                      build_transform as papagei_transform, ntxent_loss)
from .xmae import XmaePPGMAE, XmaePpgEncoder

__all__ = [
    'HimaeEncoder', 'HimaeDecoder', 'HimaeAE',
    'haar_dwt', 'build_coeff_map', 'MmrViTEncoder', 'MmrDecoder', 'MMRMAE',
    'XmaePpgEncoder', 'XmaePPGMAE',
    'PapageiP', 'PapageiS', 'papagei_build_model',
    'papagei_name_from_sd', 'papagei_transform', 'ntxent_loss',
]
