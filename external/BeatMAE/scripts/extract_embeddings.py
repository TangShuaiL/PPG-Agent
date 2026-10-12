#!/usr/bin/env python3
"""v2c 新库段嵌入提取（自定义输出名，不撞旧库缓存）。

用途：zero-shot（旧模型 × 新库段）与新库预训练模型 × 新库的嵌入提取。
输出：results/cache/embeddings/<name>_lab.npz（已过滤有标签行）。

用法:
  python extract_embeddings.py --name=myexp --ckpt=results/pretrain/<...>/ckpt/best.pt \
         --sources=iwshi_bp,aurora [--kind=beatmae] [--patch-len=16] [--input-mode=raw_waveform]

  权重路径：results/pretrain/。

注意：--kind himae/mmr/xmae 必须显式给 --patch-len（脚本不猜，raw 口径下
himae 是 5 而非 16，见上）。

--kind=papagei（PaPaGei P/S，对比模型）：**不要给 --patch-len / --input-mode**
（它只有 raw_waveform 一种口径，固定 1250 点，无拍操作）。P/S 从 ckpt 自己认，
认出来的族名会打印出来并写进 npz 的 kind 字段，事后可查当时跑的是哪一支。
例：--name=papagei_s__v2c_3src_avi__full \
     --ckpt=results/pretrain/papagei_s/v2c_3src_avi/full/ckpt/best.pt
（槽位名自证源集（2026-09-12 起）：`v2c_3src_avi` = iwshi_bp+vital+aurora，与 v2.5c 主表同源；
  `v2c_3src_mva` = mimic+vital+aurora，`v2c_4src_mvaiwb` = 全四源——三者**互不相同**，写错会静默换源。）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
import pandas as pd
import torch

from data.corpus_dataset import CorpusDataset
from data.paths import CACHE_ROOT, CORPUS_V2C, resolve_corpus
from train.probe import load_frozen, embed_batch, beat_collate_meta, raw_collate_meta


CORPUS = resolve_corpus(CORPUS_V2C, tag='corpus_v2c')

# 拍数口径必须与 ckpt 训练时一致，否则嵌入的拍窗不同、且不会报错：
# 由 load_frozen 从 ckpt 内嵌 cfg 的 data.max_beats 取出返回（缺省 = 模块常量
# MAX_BEATS），本处只透传，勿再写字面量（2026-09-11 收编）。


@torch.no_grad()
def run(exp):
    device = 'cuda'
    model, kind, input_mode, max_beats = load_frozen(exp, device)
    im = exp.get('input_mode') or input_mode
    # 打印 load_frozen 解析出的**生效值**（不是命令行原样）：papagei 的 P/S 与
    # raw_waveform 口径都是从 ckpt 里认出来的，不打出来事后无从对照
    print(f'模型: {kind} | 口径: {im}'
          + (f' | 拍上限: {max_beats}' if im == 'beat_resampled' else ''), flush=True)
    ds = CorpusDataset(CORPUS, sources=exp['sources'],
                       max_beats=max_beats,
                       input_mode=im)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False,
        # raw_waveform 口径 item 无 'beats' 键，必须配 raw collate（aurora_raw 提取曾因此崩退）
        collate_fn=raw_collate_meta if im == 'raw_waveform' else beat_collate_meta,
        num_workers=8)
    embs, sids, rows, folds = [], [], [], []
    t0, done = time.time(), 0
    for bc in loader:
        embs.append(embed_batch(model, kind, bc, device, True).numpy())
        sids.extend(bc['meta']['subject_id'])
        rows.extend(bc['meta']['row'])
        folds.extend(bc['meta']['fold'])
        done += len(bc['meta']['subject_id'])
        if done % 25600 == 0:
            print(f'  {done:,} 段 | {done/(time.time()-t0):.0f} 段/s', flush=True)
    emb = np.concatenate(embs).astype(np.float32)
    sids = np.array(sids)
    rows = np.asarray(rows, np.int64)
    folds = np.asarray(folds, np.int64)
    lab = pd.read_parquet(Path(CORPUS) / 'labels.parquet')
    keep = np.isin(rows, lab['row'].to_numpy())
    out = CACHE_ROOT
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{exp["name"]}_lab.npz'
    np.savez(path, emb=emb[keep], subject_id=sids[keep], row=rows[keep],
             fold=folds[keep], kind=kind)
    print(f'{exp["name"]}: {keep.size:,} 段 → 有标签 {int(keep.sum()):,} → {path}',
          flush=True)


def main():
    ap = argparse.ArgumentParser(description='v2c 段嵌入提取')
    ap.add_argument('--name', required=True,
                    help='实验名（输出 results/cache/embeddings/<name>_lab.npz）')
    ap.add_argument('--ckpt', required=True, help='预训练权重路径')
    ap.add_argument('--sources', default=None,
                    help='逗号分隔源过滤，如 iwshi_bp,aurora；缺省=全库')
    ap.add_argument('--kind', default='beatmae',
                    help='模型族：beatmae | himae | mmr | xmae | papagei'
                         '（papagei = PaPaGei P/S，族内细分由 ckpt 自认，默认 beatmae）')
    ap.add_argument('--patch-len', type=int, default=None,
                    help='himae=16 / mmr=32 / xmae=64 / raw_waveform-himae=5'
                         '（beatmae 留空）——显式给，脚本不猜')
    ap.add_argument('--input-mode', default=None,
                    choices=['beat_resampled', 'raw_waveform'],
                    help='缺省=由 ckpt 配置推断（见 train/probe.py::load_frozen）')
    args = ap.parse_args()

    if args.kind in ('himae', 'mmr', 'xmae') and args.patch_len is None:
        ap.error(f'--kind {args.kind} 需 --patch-len'
                 f'（himae=16 / mmr=32 / xmae=64；raw_waveform 口径 himae=5）')
    ck = Path(args.ckpt)
    if not ck.exists():
        ap.error(f'--ckpt 不存在：{ck}\n'
                 f'  提示：权威权重树在 results/pretrain/（INDEX.csv 可查，'
                 f'四层 <family>/<corpus>/<variant>/ckpt/best.pt）')

    exp = dict(name=args.name, ckpt=args.ckpt, kind=args.kind,
               sources=args.sources.split(',') if args.sources else None)
    if args.patch_len is not None:
        exp['patch_len'] = args.patch_len
    if args.input_mode is not None:
        exp['input_mode'] = args.input_mode
    print(f'语料: {CORPUS}', flush=True)
    run(exp)


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    main()
