#!/usr/bin/env python3
"""28 号：段级流式探针（全库折）——主表①。协议实现全部在 train/stream_probe.py。

2026-09-10 收编：这套协议原先只住在本文件里，37 号与 tools/ 三个脚本靠
importlib 借道复用、并直接改本模块的全局变量换数据源。现已提到
train/stream_probe.py，本文件退化为薄入口，只负责编排三件事：
「全库折 + 断点续跑 + 结果落盘」。改协议请改 train/stream_probe.py。

协议: 训练域 mimic+vital+aurora（iwshi_bp 纯测试不训）；测试折 fold∈{1,2}；
      受试者级隔离；每受试者 cap 512 段保序均匀采样（29 号物化）；raw 段级标签，
      NaN 段只做历史；梯度变体 5 种子，闭式单跑；早停 patience=10 on val seg-MAE。
输出: probe/<model>/<ds>/<scope>/stream/{results.json, heads/, preds/}
      段级 MAE（新轴）+ 受试者 MAE（median 聚合，部署口径）+ 段级 r + 冷启动曲线。
      seg_r = 逐 seed 算 Pearson r 再取均值（跨 seed 池化会得到不同的数，勿混）——
      与 37 号的 seg_r 同规则，也是 _tables/final_full_stream_all_models.csv 的 r 列来源。
用法: python probe_full_stream.py exp=<model>__<ds>__<scope> [variant ...]
      缺省实验 beatmae_v2__v2c_4src_mvaiwb__full；缺省跑主表三变体。
      python probe_full_stream.py --backfill-r   # 一次性：给早期结果补 seg_r（见 backfill_r）
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from train.run_meta import write_meta
from train import stream_probe as sp
from data.paths import probe_dir


def backfill_r():
    """一次性回填：给早期 stream/results.json 补 seg_r 字段（当时 28 号还没写）。

    r 无法从 JSON 反推（只有 seg 的 MAE 列表），但可从 preds/*.csv.gz 逐 seed 重算。
    已逐位验证：重算值 == _tables/final_full_stream_all_models.csv 的 r 列
    （v2.5c 0.605 / beatmae_v2_3src 0.582 / himae_3src 0.483 ... 全中）。
    幂等：已有 seg_r 的键跳过；只增字段，不改任何既有数字。约 2GB preds，数分钟级。
    """
    root = Path(__file__).resolve().parents[1]
    n_key = n_file = n_skip = 0
    for rj in sorted(root.glob('results/probe/**/stream/results.json')):
        rel = rj.parent.parent.relative_to(root)
        results = json.load(open(rj))
        pred_dir = rj.parent / 'preds'
        changed = False
        for key in sorted(results):
            rec = results[key]
            if not isinstance(rec, dict) or 'seg_r' in rec:
                continue
            t, var = key.split('/')
            gz = pred_dir / f'{t}__{var}.csv.gz'
            if not gz.exists():
                print(f'[skip] {rel} {key}: 缺 preds/{gz.name}，无法重算')
                n_skip += 1
                continue
            r, ns, per = sp.r_from_preds(gz)
            rec['seg_r'] = r
            changed = True
            n_key += 1
            print(f'  {rel} {key}: seg_r={r}（{ns} 个 seed: {per}）', flush=True)
        if changed:
            json.dump(results, open(rj, 'w'), indent=1)
            n_file += 1
    print(f'\n回填 {n_key} 个键 / {n_file} 个 results.json'
          + (f'；{n_skip} 个键因缺 preds 跳过' if n_skip else ''))


def main():
    # 传参: exp=<model>__<ds>__<scope> 实验名（缺省主实验）；dir=物化目录（缺省 STREAM_CACHE/<exp>）；
    #       其余位置参数 = variants 子集
    exp = 'beatmae_v2__v2c_4src_mvaiwb__full'
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
    # 输出路径（2026-09-05 二次定稿）：与 pretrain 树同构，结果与头同目录——
    # probe/<model>/<ds>/<scope>/stream/{results.json, heads/, preds/}
    out_dir = probe_dir(exp, 'stream')
    head_dir, pred_dir = out_dir / 'heads', out_dir / 'preds'
    for d in (out_dir, head_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = sp.load_all(sdir or sp.resolve_sdir(exp))
    tr_p, va_p, te_p = sp.split_full_corpus(data.fold, data.bnd)
    print(f'人: train {len(tr_p)} / val {len(va_p)} / test {len(te_p)}')
    # 断点续跑：results.json 每变体组完成即增量落盘，重启后跳过已完成变体
    rj = out_dir / 'results.json'
    results = json.load(open(rj)) if rj.exists() else {}
    if 'results' in results and 'meta' in results:
        results = results['results']   # 旧版包裹格式（beatmae 4src）解包为扁平结构
    if results:
        print(f'[断点] 已有 {len(results)} 个变体结果，未完成的继续跑')

    for ti, t in enumerate(sp.TARGETS):
        ytr = np.concatenate([data.y[data.bnd[p]:data.bnd[p+1], ti] for p in tr_p])
        mu, sd = np.nanmean(ytr), np.nanstd(ytr) + 1e-8
        for var in variants:
            r0 = results.get(f'{t}/{var}')
            # 变体级断点：已完成则整组跳过（causal_attn 缺 curve 视为旧版结果，仍重跑补 curve）
            if r0 is not None and (var != 'causal_attn' or 'curve' in r0):
                print(f'── {t}/{var} ──  跳过（断点已有结果）')
                continue
            kk, shuf = sp.variant_hparams(var)
            accs_seg, accs_sub, accs_r, times, rows = [], [], [], [], []
            curve = {}
            for seed in (sp.SEEDS if var != 'w1_ridge' else (0,)):
                t0 = time.time()
                hp = head_dir / sp.head_name(t, var, seed)
                if var == 'w1_ridge' and not hp.exists():
                    # 段级闭式 ridge：α 用 val 选（1/5 训练人采样控内存），闭式单跑无 seed 概念
                    w, alpha = sp.fit_ridge(data.emb, data.y, data.bnd, tr_p, va_p, ti, stride=5)
                    sp.save_ridge_head(hp, w, alpha, mu, sd, t)
                elif hp.exists():
                    if var != 'w1_ridge':
                        # seed 级断点：头已存则免训练直接评估（秒级 predict 换分钟级 train）
                        print(f'  [断点] {t}/{var}/s{seed} 复用已训头')
                else:
                    # 嵌入维随族而变（PaPaGei 512 / 另四族 256），从数据取，不许靠头的默认值
                    model = sp.make_model(var, data.eg.shape[1]).to(sp.DEV)
                    sp.train_variant(model, data.eg, data.y, data.bnd,
                                     tr_p, va_p, ti, seed, kk=kk, shuf=shuf)
                    sp.save_variant_head(hp, var, model, mu, sd, t, seed, kk, shuf)
                ev = sp.eval_variant(torch.load(hp, map_location=sp.DEV), var,
                                     data, te_p, ti, seed)
                rows.append(ev.row)
                accs_seg.append(ev.seg); accs_sub.append(ev.sub)
                accs_r.append(ev.r); times.append(time.time() - t0)
                if var == 'causal_attn' and seed == 0:
                    curve = {k: round(v, 3)
                             for k, v in sp.bucket_mae(ev.pr, ev.gt, ev.wp).items()}
            fp = sp.dump_preds(pred_dir, t, var, rows)
            print(f'  明细 → {fp.name}')
            results[f'{t}/{var}'] = dict(seg=[round(float(a), 3) for a in accs_seg],
                                         sub=[round(float(a), 3) for a in accs_sub],
                                         seg_mean=round(float(np.mean(accs_seg)), 3),
                                         sub_mean=round(float(np.mean(accs_sub)), 3),
                                         seg_r=round(float(np.nanmean(accs_r)), 3),
                                         sec=round(float(np.mean(times)), 1), curve=curve)
            r = results[f'{t}/{var}']
            print(f'{t}/{var}: seg {r["seg_mean"]} (n={len(accs_seg)}) sub {r["sub_mean"]} '
                  f'r {r["seg_r"]} | {r["sec"]}s/run' + (f' | curve {r["curve"]}' if r["curve"] else ''))
            json.dump(results, open(out_dir / 'results.json', 'w'), indent=1)   # 增量落盘（断点续跑点）
    json.dump(results, open(out_dir / 'results.json', 'w'), indent=1)
    write_meta(out_dir, cfg=None, config_path=None)
    n_heads = len(list(head_dir.glob('*.pt')))
    print(f'完成 → {out_dir / "results.json"} | 头权重 {n_heads} 个 → {head_dir}/')


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print(f'用法: python probe_full_stream.py [variant ...]   # variant ∈ {" / ".join(sp.ALL_VARIANTS)}，缺省主表三变体')
        print('      python probe_full_stream.py --backfill-r   # 给早期结果补 seg_r（幂等）')
        sys.exit(0)
    if '--backfill-r' in sys.argv:
        backfill_r()
        sys.exit(0)
    main()
