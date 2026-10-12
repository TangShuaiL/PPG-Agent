# -*- coding: utf-8 -*-
"""models/probe_heads — 下游探针头模型库（2026-09-05 从训练脚本抽取集中）。

原则：模型结构定义集中在 models/，训练/评估脚本只 import。
模块属性名与抽取前完全一致（V/U/q/head/W/a/mlp/net/cores），旧权重的
state_dict 键名不受影响。

| 文件 | 类 | 来源 | 协议 |
|------|-----|------|------|
| mlp.py          | MLPProbe              | train/probe.py（主口径 head=mlp） | 2 层 MLP，Adam 1e-3，早停 patience=20 |
| attn_pool.py    | AttnPoolProbe         | train/probe.py（head=attn）       | ABMIL 门控注意力池化（Ilse 2018） |
| causal_stream.py| CausalAttn / CausalDropLong / CausalDropRecent / SegMLP | 段级流式探针 + 消融臂 | 双参照拼接 / 各砍一条参照 / 无历史对照 + 前缀均值 |
"""
from models.probe_heads.mlp import MLPProbe
from models.probe_heads.attn_pool import AttnPoolProbe
from models.probe_heads.causal_stream import (CausalAttn, CausalDropLong,
                                              CausalDropRecent, SegMLP)

__all__ = ['MLPProbe', 'AttnPoolProbe', 'CausalAttn', 'CausalDropLong',
           'CausalDropRecent', 'SegMLP']
