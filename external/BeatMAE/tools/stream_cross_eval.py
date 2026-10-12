# -*- coding: utf-8 -*-
"""tools/stream_cross_eval.py — stream 头跨域零样本评估（免训练纯前向）。

把 <head_exp> 训练好的 stream 头（三库口径，如 v2c_4src_mvaiwb__full）直接前向到
另一个流式物化（如 iwshi_bp 跨域测试），产出 seg/sub/curve 指标 + 预测明细
落盘（dump_preds，含 sid/seg_idx/src）。不训练、不改动任何已有结果。

数据划分: 跨域零样本语义下 data 物化的全部受试者都是测试集（无头训练）；
mu/sd 取头文件内存储值（三库训练时的归一化参数，输出已还原 raw 尺度）。

2026-09-10 收编：先前靠 importlib 借道 28 号、并改 `_m28.META / _m28.EMBF` 换数据源，
现直接 import 协议库（train/stream_probe.py），头文件名/分桶/落盘列布局共用一份实现。

用法: python tools/stream_cross_eval.py head=<model>__<ds>__<scope> data=<model>__<ds>__<scope>
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from train.run_meta import write_meta
from train import stream_probe as sp
from data.paths import probe_dir


def run(head_exp, data_dir):
    sdir = sp.resolve_sdir(data_dir, legacy_fallback=False)
    out_exp = head_exp.rsplit('__', 1)[0] + '__' + data_dir.rsplit('__', 1)[1]   # <model>__<head_ds>__<data_scope>
    out_dir = probe_dir(out_exp, 'stream')
    pred_dir = out_dir / 'preds'
    pred_dir.mkdir(parents=True, exist_ok=True)
    head_dir = probe_dir(head_exp, 'stream') / 'heads'
    data = sp.load_all(sdir)
    te_p = np.arange(len(data.bnd) - 1)          # 零样本：全部受试者都是测试
    print(f'{head_exp} 的头 → {data_dir} 物化（{len(te_p)} 人全部测试）| 输出 {out_dir}/')
    results = {}
    for ti, t in enumerate(sp.TARGETS):
        for var in sp.ALL_VARIANTS:
            accs_seg, accs_sub, times, curve, rows = [], [], [], {}, []
            for seed in (sp.SEEDS if var != 'w1_ridge' else (0,)):
                hp = head_dir / sp.head_name(t, var, seed)
                if not hp.exists():
                    print(f'  [缺] {t}/{var}/s{seed} 头不存在，跳过')
                    continue
                ck = torch.load(hp, map_location=sp.DEV)
                t0 = time.time()
                ev = sp.eval_variant(ck, var, data, te_p, ti, seed, with_src=True)
                accs_seg.append(ev.seg); accs_sub.append(ev.sub)
                times.append(time.time() - t0)
                if var == 'causal_attn' and seed == 0:
                    curve = {name: round(v, 3)
                             for name, v in sp.bucket_mae(ev.pr, ev.gt, ev.wp).items()}
                rows.append(ev.row)
            if not rows:
                continue
            fp = sp.dump_preds(pred_dir, t, var, rows)
            # 跨 seed 池化成一次相关（与 28 号「逐 seed 再平均」不同，这是本工具的口径）
            gt_all = np.concatenate([r[-2] for r in rows])
            pr_all = np.concatenate([r[-1] for r in rows])
            r_p = sp.seg_mae_r(pr_all, gt_all)[1]
            results[f'{t}/{var}'] = dict(seg=[round(a, 3) for a in accs_seg],
                                         sub=[round(a, 3) for a in accs_sub],
                                         seg_mean=round(float(np.mean(accs_seg)), 3),
                                         sub_mean=round(float(np.mean(accs_sub)), 3),
                                         seg_r=None if np.isnan(r_p) else round(r_p, 3),
                                         sec=round(float(np.mean(times)), 1), curve=curve)
            r_ = results[f'{t}/{var}']
            print(f'  {t}/{var}: seg {r_["seg_mean"]} sub {r_["sub_mean"]} r {r_["seg_r"]} '
                  f'| {r_["sec"]}s/run | 明细 {fp.name}')
    json.dump(results, open(out_dir / 'results.json', 'w'), indent=1)
    write_meta(out_dir, cfg=None, config_path=None)
    print(f'完成 → {out_dir / "results.json"}')


if __name__ == '__main__':
    args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
    run(args['head'], args['data'])
