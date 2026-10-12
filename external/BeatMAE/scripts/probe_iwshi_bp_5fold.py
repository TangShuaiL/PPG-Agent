#!/usr/bin/env python3
"""37 号：iwshi_bp 域内五折 stream 探针——主表②。协议实现全在 train/stream_probe.py。

2026-09-10 收编：此前靠 `importlib.import_module('28_causal_stream_probe')` 借道 28 号
并改 `_m28.META / _m28.EMBF` 换数据源，现在直接 import 协议库，本文件只剩
「五折切分 + 逐折落盘」两件编排事。与 28 号的实质差别只有「试哪些折」。

协议（2026-09-06 定稿，补齐实验矩阵空格子 #4/#5）：
  数据 = 29 号物化 {m}__v2c_4src_mvaiwb__iwshi_bp（4src global 编码器 × iwshi_bp 段，46 人 / 23,506 段）。
  五折 CV：受试者折 = fold//2 ∈ {0..4}（fold 为偶数 0/2/4/6/8），每折 8~10 人测试，
          其余人训练，再按人切 15% 做早停验证（rng 1234+k 确定性，至少 2 人）。
  query 段：BP = 袖带标签段（NaN 段只进历史不进损失）；age = 全量段（人级属性铺满，
          与三库协议对齐，2026-09-06 回填后覆盖 100%）。
  变体/超参与 28 号逐项一致：K=32、LR=3e-3、Huber、patience=10、NN 5 种子、
          ridge α∈{0.1..1000} 验证折选优；与 28 号唯一的参数差别是 ridge 不吃
          1/5 人抽样（折内人少，不需要控内存）。mu/sd 每折按训练人重算（域内归一化）。
  输出: probe/<exp>/stream5f/{results.json, heads/f{k}_{t}_{var}_s{seed}.pt, preds/}
        指标按折报 mean±std；curve 为各折全部 seed 的段级误差按 k 分桶后跨 seed 平均。
用法: python probe_iwshi_bp_5fold.py exp=<model>__v2c_4src_mvaiwb__iwshi_bp [variant ...]
      缺省实验 beatmae_v2__v2c_4src_mvaiwb__iwshi_bp；缺省跑主表三变体。
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from train import stream_probe as sp
from data.paths import probe_dir


def main():
    exp = 'beatmae_v2__v2c_4src_mvaiwb__iwshi_bp'
    sdir = None
    variants = []
    for a in sys.argv[1:]:
        if a.startswith('exp='):
            exp = a.split('=', 1)[1]
        elif a.startswith('dir='):
            sdir = Path(a.split('=', 1)[1])
        else:
            variants.append(a)
    variants = variants or list(sp.VARIANTS)
    unknown = [v for v in variants if v not in sp.ALL_VARIANTS]
    if unknown:
        sys.exit(f'未知变体 {unknown}；合法值: {" / ".join(sp.ALL_VARIANTS)}')
    sdir = sdir or sp.resolve_sdir(exp, legacy_fallback=False)   # 只认 per-exp 物化目录

    out_dir = probe_dir(exp, 'stream5f')
    head_dir = out_dir / 'heads'
    pred_dir = out_dir / 'preds'
    for d in (out_dir, head_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = sp.load_all(sdir)
    pf = data.fold[data.bnd[:-1]] // 2                  # 五折号 0..4
    folds = sorted(set(pf.tolist()))
    print(f'五折人数: { {k: int((pf==k).sum()) for k in folds} }')

    rj = out_dir / 'results.json'
    results = json.load(open(rj)) if rj.exists() else {}
    if results:
        print(f'[断点] 已有 {len(results)} 组结果，未完成的继续跑')

    for k in folds:
        tr_p, va_p, te_p = sp.split_five_fold(data.fold, data.bnd, k)
        for ti, t in enumerate(sp.TARGETS):
            ytr = np.concatenate([data.y[data.bnd[p]:data.bnd[p+1], ti] for p in tr_p])
            mu, sd = np.nanmean(ytr), np.nanstd(ytr) + 1e-8
            for var in variants:
                key = f'f{k}/{t}/{var}'
                if key in results:
                    print(f'── {key} ──  跳过（断点已有）')
                    continue
                t0 = time.time()
                accs_seg, accs_sub, accs_r, all_rows = [], [], [], []
                curve_acc = {name: [] for _, _, name in sp.BUCKETS}
                kk, shuf = sp.variant_hparams(var)
                for seed in (sp.SEEDS if var != 'w1_ridge' else (0,)):
                    hp = head_dir / sp.head_name(t, var, seed, fold=k)
                    if not hp.exists():
                        if var == 'w1_ridge':
                            # 与 28 号同口径，但不抽样：五折内训练人本就少，全量入闭式解
                            w, alpha = sp.fit_ridge(data.emb, data.y, data.bnd,
                                                    tr_p, va_p, ti)
                            sp.save_ridge_head(hp, w, alpha, mu, sd, t, fold=k)
                        else:
                            # 嵌入维随族而变（PaPaGei 512 / 另四族 256），从数据取
                            model = sp.make_model(var, data.eg.shape[1]).to(sp.DEV)
                            sp.train_variant(model, data.eg, data.y, data.bnd,
                                             tr_p, va_p, ti, seed, kk=kk, shuf=shuf)
                            sp.save_variant_head(hp, var, model, mu, sd, t,
                                                 seed, kk, shuf, fold=k)
                    ev = sp.eval_variant(torch.load(hp, map_location=sp.DEV), var,
                                         data, te_p, ti, seed, with_src=False)
                    all_rows.append(ev.row)
                    accs_seg.append(ev.seg); accs_sub.append(ev.sub); accs_r.append(ev.r)
                    if var == 'causal_attn':
                        for name, v in sp.bucket_mae(ev.pr, ev.gt, ev.wp).items():
                            curve_acc[name].append(v)
                fp = sp.dump_preds(pred_dir, t, var, all_rows, fold=k, src=False, append=True)
                curve = {name: round(float(np.mean(v)), 3) for name, v in curve_acc.items() if v}
                results[key] = dict(seg=[round(float(a), 3) for a in accs_seg],
                                    sub=[round(float(a), 3) for a in accs_sub],
                                    seg_mean=round(float(np.mean(accs_seg)), 3),
                                    seg_std=round(float(np.std(accs_seg)), 3),
                                    sub_mean=round(float(np.mean(accs_sub)), 3),
                                    sub_std=round(float(np.std(accs_sub)), 3),
                                    seg_r=round(float(np.nanmean(accs_r)), 3),
                                    curve=curve, sec=round(time.time() - t0, 1))
                r = results[key]
                print(f'f{k} {t}/{var}: seg {r["seg_mean"]}±{r["seg_std"]} sub {r["sub_mean"]}±{r["sub_std"]} '
                      f'r {r["seg_r"]} | {r["sec"]}s | 明细 {fp.name}'
                      + (f' | curve {curve}' if curve else ''), flush=True)
                json.dump(results, open(rj, 'w'), indent=1)
    json.dump(results, open(rj, 'w'), indent=1)
    print(f'完成 → {rj} | 头 {len(list(head_dir.glob("*.pt")))} 个')


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print(f'用法: python probe_iwshi_bp_5fold.py [variant ...]   # variant ∈ {" / ".join(sp.ALL_VARIANTS)}，缺省主表三变体')
        sys.exit(0)
    main()
