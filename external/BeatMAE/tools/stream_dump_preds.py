# -*- coding: utf-8 -*-
"""tools/stream_dump_preds.py — 存量 stream 实验预测明细回填（免训练纯前向）。

对 28 号已完成（或部分完成）的实验，用已存头权重对测试集逐段前向，
把预测明细写入 stream/preds/{t}__{var}.csv.gz（列: target,variant,seed,
sid,seg_idx,src,y_true,y_pred），对齐 subject preds.csv 落盘规范。
不重训、不改 results.json。数据划分与 28 号完全一致（协议见 train/stream_probe.py：
split_full_corpus(rng 1234 / TEST_FOLDS)），mu/sd 取头文件内存储值；头缺失的组合跳过。

2026-09-10 收编：先前靠 importlib 借道 28 号、并改 `_m28.META / _m28.EMBF` 换数据源，
现直接 import 协议库，预测路径与 28 号共用 head_predict 一份实现。

用法: python tools/stream_dump_preds.py exp=<model>__<ds>__<scope> [exp=...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from train import stream_probe as sp
from data.paths import probe_dir


def run(exp):
    sdir = sp.resolve_sdir(exp)                  # 与 28 号同口径（含旧根目录布局兼容）
    out_dir = probe_dir(exp, 'stream')
    pred_dir = out_dir / 'preds'
    pred_dir.mkdir(parents=True, exist_ok=True)
    data = sp.load_all(sdir)
    _, _, te_p = sp.split_full_corpus(data.fold, data.bnd)
    print(f'{exp}: 测试 {len(te_p)} 人 → {pred_dir}/')
    for ti, t in enumerate(sp.TARGETS):
        for var in sp.ALL_VARIANTS:
            acc = []
            for seed in (sp.SEEDS if var != 'w1_ridge' else (0,)):
                hp = out_dir / 'heads' / sp.head_name(t, var, seed)
                if not hp.exists():
                    print(f'  [缺] {t}/{var}/s{seed} 头不存在，跳过')
                    continue
                ev = sp.eval_variant(torch.load(hp, map_location=sp.DEV), var,
                                     data, te_p, ti, seed, with_src=True)
                acc.append(ev.row)
            if acc:
                fp = sp.dump_preds(pred_dir, t, var, acc)
                print(f'  {t}/{var}: {len(acc[-1][-1])} 段/seed × {len(acc)} seed → {fp.name}')


if __name__ == '__main__':
    for a in sys.argv[1:]:
        if a.startswith('exp='):
            run(a.split('=', 1)[1])
