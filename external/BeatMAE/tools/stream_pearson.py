# -*- coding: utf-8 -*-
"""tools/stream_pearson.py — stream 探针 Pearson r 重算（免训练，复用已存头权重）。

协议全部来自 train/stream_probe.py（2026-09-10 收编：此前靠 importlib 借道 28 号、
并改 `_m28.META / _m28.EMBF` 换数据源，现在直接 import）。对已训头逐 seed 重算
段级 r 与受试者级 r，把 seg_r / sub_r 增量写回各模型 stream results.json。
不改动任何 MAE 字段，不训练任何模型。

注：seg_r 与 28 号 `--backfill-r` 同规则（逐 seed 算 Pearson 再平均；跨 seed 池化
    会得到不同的数，勿混），也已逐位对齐 _tables/final_full_stream_all_models.csv 的
    r 列（v2.5c 0.605 / beatmae_v2_3src 0.582 / himae_3src 0.483 ...）。
    sub_r 是受试者级相关（人内 median 池化后跨人相关），仓内暂无消费方
    （32/36 号与 _tables 都不读），保留备受试者级分析用。

用法：python tools/stream_pearson.py exp=<model>__<ds>__<scope> [exp=... ...]
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from train import stream_probe as sp
from data.paths import probe_dir


def main():
    exps = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('exp=')]
    if not exps:
        sys.exit('用法: python tools/stream_pearson.py exp=<model>__<ds>__<scope> ...')
    for exp in exps:
        sdir = sp.resolve_sdir(exp)                  # 与 28 号同口径（含旧根目录布局兼容）
        out_dir = probe_dir(exp, 'stream')
        head_dir, rj = out_dir / 'heads', out_dir / 'results.json'
        results = json.load(open(rj))
        if 'results' in results and 'meta' in results:
            results = results['results']              # 旧版包裹格式（beatmae 4src）
        data = sp.load_all(sdir)
        _, _, te_p = sp.split_full_corpus(data.fold, data.bnd)

        for ti, t in enumerate(sp.TARGETS):
            for var in sp.ALL_VARIANTS:
                if f'{t}/{var}' not in results:
                    print(f'[{exp}] 缺 {t}/{var}，跳过'); continue
                seg_rs, sub_rs = [], []
                for seed in (sp.SEEDS if var != 'w1_ridge' else (0,)):
                    hp = head_dir / sp.head_name(t, var, seed)
                    if not hp.exists():
                        print(f'[{exp}] 缺头 {hp.name}，该 seed 跳过'); continue
                    pr, gt, ws, _ = sp.head_predict(torch.load(hp, map_location=sp.DEV),
                                                    var, data, te_p, ti)
                    # 逐 seed：段级 = 全体测试段一次相关；受试者级 = 跨人中位数相关
                    seg_rs.append(sp.pearson(pr, gt))
                    sub_rs.append(sp.subject_r(pr, ws, data.y, data.bnd, ti))
                results[f'{t}/{var}']['seg_r'] = round(float(np.nanmean(seg_rs)), 3)
                results[f'{t}/{var}']['sub_r'] = round(float(np.nanmean(sub_rs)), 3)
                print(f'[{exp}] {t}/{var}: seg_r {results[f"{t}/{var}"]["seg_r"]:.3f} '
                      f'sub_r {results[f"{t}/{var}"]["sub_r"]:.3f}')
        json.dump(results, open(rj, 'w'), indent=1)
        print(f'[{exp}] 已写回 {rj}')


if __name__ == '__main__':
    main()
