#!/usr/bin/env python3
"""PaPaGei P/S 对比学习预训练（形态学伪标签引导）。

与另三个对比族（HiMAE / MMR / xMAE 走 train/pretrain_baseline.py）的差别在
**方法轴线**上：不做掩码重建，而是取 **同 batch 内 sVRI 同箱的段互为为正样本**
做多正样本 NT-Xent（SupCon，不是 SimCLR 双视图——见 baselines/papagei/losses.py），
S 另加两个 MoE 头回归 IPA / skewness。段池、采样器、日志与 ckpt 契约与 baseline
逐项同形，主表才能并列。

**相对官方 recipe 的唯一偏差是训练预算**：官方 P 15000 / S 10000 epoch × bs=128，
本脚本 30 epoch × 10000 步 × bs=64（= 另三族口径，为的是同池同预算可比）。
其余全部照官方：Adam(lr=1e-4, weight_decay=0)、常数 LR、无 warmup、无裁剪、
fp32 损失、增强概率取官方 prob_dictionary 的非零项。

用法：python train/pretrain_papagei.py config/base/papagei.yaml \
          papagei.model=papagei_s data.sources=[iwshi_bp,vital,aurora] \
          out_dir=results/pretrain/papagei_s/v2c_3src_avi/full
槽位名自证源集（2026-09-12 起）：v2c_3src_avi=iwshi_bp+vital+aurora（与 v2.5c 主表同源），
v2c_3src_mva=mimic+vital+aurora，v2c_4src_mvaiwb=全四源——三者互不相同，写错会静默换源。
"""
import argparse, sys, time, random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.beat_ops import SEG_LEN
from data.corpus_dataset import CorpusDataset
from data.morphology import (LABEL_KEYS, WithMorphology, load_morphology,
                             morph_collate)
from data.samplers import EpochTraverseSampler, SourceBalancedSampler
from baselines.papagei import build_model, build_transform, ntxent_loss
from baselines.papagei.labels import SVRI_RANGE, assign_svri_bins, svri_bin_edges
from train.pretrain import load_config
from train.run_meta import ckpt_meta, log, write_meta

# 段长口径权威在 data.beat_ops（此处只 import，勿再写字面量）。
# PaPaGei 只有 raw_waveform 一种输入：原始 10 s 段 @125 Hz，无拍操作、无掩码。
SEG_RAW = SEG_LEN
NO_LABEL = -1                      # labels.py 的"无有效标签"哨兵，须与损失一致

# morph_collate 比 LABEL_KEYS 多交一列 svri_bin（对比标签）
COLLATE = partial(morph_collate, keys=LABEL_KEYS + ('svri_bin',))


# build_model（papagei_p | papagei_s → 模型）在 baselines/papagei/model.py，与下游
# train/probe.py::load_frozen 共用同一份判别，勿在本文件另写一份


def masked_l1(pred, target, keep=None):
    """L1(pred, target)；keep=False 的样本不计入均值（全被屏蔽时返回 0 且无梯度）。

    官方 `criterion2(ipa_pred, ipa.unsqueeze(-1))` 对全部样本求均值；`keep` 就是
    官方 `clean_ipa_only=True` 才有的屏蔽分支（本库默认关，见 ipa_valid_only）。
    """
    d = (pred.float() - target.float().unsqueeze(-1)).abs()
    if keep is None:
        return d.mean()
    if not bool(keep.any()):
        return d.sum() * 0.0
    return d[keep].mean()


def compute_loss(model, seg, lab, pap, is_s):
    """前向 + 损失 → (total, parts)。parts 全是 detach 过的标量，只用于记录。

    对比项与辅助项一律在 **fp32** 下算：官方全程 fp32，而 NT-Xent 的 logsumexp
    与逐 anchor 归一化在 fp16 下会明显失真，所以 autocast 只包前向（embedding
    出来即 `.float()`）。这也是本族 fp16 只省前向显存/时间的由来。
    """
    if is_s:
        emb, ipa_pred, skew_pred, _ = model(seg)
    else:
        emb, _ = model(seg)

    contrast = ntxent_loss(emb, lab['svri_bin'], temperature=pap['temperature'],
                           ignore_index=NO_LABEL)
    parts = {'contrast': contrast.detach()}
    if not is_s:
        return contrast, parts

    alpha = float(pap['alpha'])
    keep = ~lab['ipa_fail'] if pap['ipa_valid_only'] else None
    ipa_l = masked_l1(ipa_pred, lab['ipa'], keep)
    # S 的第二头回归 skewness（官方变量名叫 sqi，实为 bins_skewness=0 的原始偏度）
    skew_l = F.l1_loss(skew_pred.float(), lab['skewness'].float().unsqueeze(-1))
    parts.update(ipa=ipa_l.detach(), skew=skew_l.detach())
    # 官方：loss = alpha*contrastive + (1-alpha)/2*ipa + (1-alpha)/2*sqi
    return alpha * contrast + (1 - alpha) / 2 * (ipa_l + skew_l), parts


def bin_report(bins, n_bins, rows):
    """**该池子**各箱段数的一行摘要（-1 单列）——标签分布落日志，便于事后溯源。

    bins 是全库稠密数组，必须按 rows 取池子：两池的分布差别很大（尤其单源
    训练池），拿全库分布看训练池会看不出"某箱在这池里其实几乎是空的"。
    """
    b = bins[np.asarray(rows, np.int64)] + 1        # 哨兵 -1 → 下标 0
    cnt = np.bincount(b, minlength=n_bins + 1)      # 下标 0..n_bins
    n = max(int(cnt.sum()), 1)
    body = ' '.join(f'{c * 100.0 / n:.1f}' for c in cnt[1:])
    return (f'无标签 {cnt[0]:,d} ({cnt[0] * 100.0 / n:.2f}%) | '
            f'箱占比 [{body}] %')


@torch.no_grad()
def do_val(model, val_loader, device, pap, is_s, fp16):
    """验证损失 = **纯对比项**（P/S 同式可比）；S 的辅助项一并回传只作记录。

    验证不做增强：增强是随机噪声，会让早停判据抖动（官方无验证集，此处以
    "干净且可复现"为准）。训练侧测的是带增强的损失，天然更大，两者不并列看。
    """
    model.eval()
    tot, n = {}, 0
    for seg, lab in val_loader:
        if seg.shape[0] < 2:               # NT-Xent 需 ≥2（loader 已 drop_last，兜底）
            continue
        seg = seg.to(device, non_blocking=True)
        lab = {k: v.to(device, non_blocking=True) for k, v in lab.items()}
        with autocast('cuda', dtype=torch.float16, enabled=fp16):
            _, parts = compute_loss(model, seg, lab, pap, is_s)
        B = seg.shape[0]
        for k, v in parts.items():
            tot[k] = tot.get(k, 0.0) + float(v) * B
        n += B
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    # 与姊妹入口 train/pretrain.py、train/pretrain_baseline.py 同形：
    # 位置参数 config + 未识别项作 overrides（`a.b=v` 形式，见 load_config）
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    tr, pap = cfg['train'], cfg['papagei']
    is_s = pap['model'] == 'papagei_s'
    seed = tr['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # out_dir 在基座里写的是 `out_dir: null`（键存在、值为 None），.get 的默认值
    # 不可达 → 真实入口一律是命令行 override（同 baseline，见 config/README.md）
    out_dir = Path(cfg.get('out_dir', 'results/pretrain/papagei'))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ckpt').mkdir(exist_ok=True)
    log_path = out_dir / 'train.log'
    _log = lambda msg: log(msg, log_path)

    meta = write_meta(out_dir, cfg, sys.argv[1])
    git = meta['git']
    meta_lite = ckpt_meta(meta)
    _log(f'== run_meta == git={git["commit"][:8] if git["commit"] else "未提交"}'
         f' dirty={git["dirty"]} | host={meta["host"]["hostname"]}'
         f' gpu={meta["host"].get("gpu", "?")}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', 'PaPaGei 训练需要 GPU'
    _log(f'== 设备 == {torch.cuda.get_device_name(0)}')

    # ── 数据 ──
    # 形态学表按 manifest row 稠密查表（见 data/morphology.py），段池与另三族逐段一致
    morph = load_morphology(cfg['morphology'])
    sources = cfg['data'].get('sources')
    train_base = CorpusDataset(cfg['corpus_root'], exclude_folds=cfg['test_folds'],
                               sources=sources, input_mode='raw_waveform')
    val_base = CorpusDataset(cfg['corpus_root'], include_folds=cfg['test_folds'],
                             sources=sources, max_rows=cfg['val']['max_segments'],
                             input_mode='raw_waveform')

    # sVRI 箱边界是训练超参：**只用训练池的行号定边界**，验证池各自另算一份，
    # 否则测试折的 sVRI 分布会漏进边界（官方每个 split 的 Dataset 各有一份 df，
    # 天然如此；本库必须显式分池，见 baselines/papagei/labels.py）。
    n_bins = int(pap['bins_svri'])          # 唯一权威（labels.py 不设默认值）
    tr_edges = svri_bin_edges(morph['svri'], n_bins, rows=train_base.rows)
    va_edges = svri_bin_edges(morph['svri'], n_bins, rows=val_base.rows)
    tr_bins = assign_svri_bins(morph['svri'], tr_edges)
    va_bins = assign_svri_bins(morph['svri'], va_edges)
    train_ds = WithMorphology(train_base, morph, tr_bins)
    val_ds = WithMorphology(val_base, morph, va_bins)

    if tr['sampler_mode'] == 'traverse':
        sampler = EpochTraverseSampler(train_ds, seed=seed)
        spe = (len(train_ds) + tr['batch_size'] - 1) // tr['batch_size']
    else:
        sampler = SourceBalancedSampler(
            train_ds, n_samples=tr['steps_per_epoch'] * tr['batch_size'],
            mode=tr['sampler_mode'], seed=seed)
        spe = tr['steps_per_epoch']
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=tr['batch_size'], sampler=sampler, collate_fn=COLLATE,
        num_workers=tr['num_workers'], pin_memory=True)
    # drop_last：NT-Xent 在 B=1 时既无负样本也无正样本（整行被 self_mask 遮成
    # -inf），尾批必须丢掉——否则 val 池大小除不尽就崩
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=tr['batch_size'], shuffle=False, collate_fn=COLLATE,
        num_workers=tr['num_workers'], drop_last=True)

    # 源组合必须打进日志：槽位名 2026-09-12 起自证源集（`v2c_3src_mva` = mimic+vital+aurora、
    # `v2c_4src_mvaiwb` = 全库含 iwshi_bp、`v2c_3src_avi` = iwshi_bp+vital+aurora），但日志是权威
    # 落据，仍逐次打印，事后不必靠名字推断（2026-09-11 就是靠 himae 的 train.log 才定案）。
    _log(f'== 数据 == train={len(train_ds):,d} val={len(val_ds):,d} '
         f'sources={"全部" if sources is None else "+".join(sources)} '
         f'model={pap["model"]} 段长={SEG_RAW} sampler={tr["sampler_mode"]} 每epoch={spe}步')
    _log(f'== sVRI 箱 == {n_bins} 等宽 / 合格区间 {SVRI_RANGE} | '
         f'训练池 边界=[{", ".join(f"{e:.4f}" for e in tr_edges)}] '
         f'{bin_report(tr_bins, n_bins, train_base.rows)}')
    _log(f'            验证池 边界=[{", ".join(f"{e:.4f}" for e in va_edges)}] '
         f'{bin_report(va_bins, n_bins, val_base.rows)}')

    # ── 模型 ──
    model = build_model(pap['model']).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    aug = pap['aug'][pap['model']]
    transform = build_transform(g_p=aug['g_p'], c_p=aug['c_p'])
    _log(f'== 模型 == {pap["model"]} {n_params:.2f}M 参数（base_filters=32 n_block=18 '
         f'n_classes=512{"，3 专家 MoE" if is_s else ""}）')
    _log(f'== 增强 == g_p={aug["g_p"]} c_p={aug["c_p"]}（官方 prob_dictionary 的非零项）')

    opt = torch.optim.Adam(model.parameters(), lr=tr['lr'],
                           weight_decay=tr.get('weight_decay', 0.0))
    if tr.get('scheduler', 'none') == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr['epochs'] * spe)
    else:
        sched = None                       # 官方常数 LR

    def cur_lr():
        return sched.get_last_lr()[0] if sched else tr['lr']

    scaler = GradScaler('cuda', enabled=tr['fp16'])
    _log(f'== 优化 == Adam(lr={tr["lr"]:.1e}, wd={tr.get("weight_decay", 0.0)}) '
         f'scheduler={tr.get("scheduler", "none")} clip_grad={tr.get("clip_grad", 0.0)}')
    if is_s:
        _log(f'== 损失 == {pap["alpha"]}·NT-Xent + {(1 - pap["alpha"]) / 2:.2f}·L1(IPA) '
             f'+ {(1 - pap["alpha"]) / 2:.2f}·L1(skewness)（官方 training_mt.py；'
             f'ipa_valid_only={pap["ipa_valid_only"]}）')
    else:
        _log(f'== 损失 == NT-Xent(sVRI {n_bins} 箱, T={pap["temperature"]})')
    _log(f'== 预算 == {tr["epochs"]} epoch × {spe} 步 × bs={tr["batch_size"]}'
         f'（= 另三族口径）；官方 recipe 为 P 15000 / S 10000 epoch × bs=128，'
         f'这是本族相对官方唯一的偏差轴')
    _log(f'== 开训 == {"遍历" if tr["sampler_mode"] == "traverse" else "流式"}'
         f' | fp16={tr["fp16"]}（仅前向；损失在 fp32）')

    best_val = float('inf')
    stale = 0
    patience = tr.get('early_stop_patience', 10)
    gstep = 0
    t0 = time.time()

    for epoch in range(tr['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        ema_loss, ema_c = 0.0, 0.0

        for step, (seg, lab) in enumerate(loader):
            if gstep == 0:
                assert seg.shape[1] == SEG_RAW, \
                    f'PaPaGei 只吃 {SEG_RAW} 点原始段，实得 {seg.shape[1]}'
            # 增强在 CPU 上做（官方在 dataset 里逐样本做，语义一致）：整批进模型前
            # 一次变换，省掉 GPU 上 64 次同步切片；不参与计算图
            with torch.no_grad():
                seg = transform(seg).to(device, non_blocking=True)
            lab = {k: v.to(device, non_blocking=True) for k, v in lab.items()}

            with autocast('cuda', dtype=torch.float16, enabled=tr['fp16']):
                loss, parts = compute_loss(model, seg, lab, pap, is_s)

            scaler.scale(loss / tr['grad_accum']).backward()
            if (step + 1) % tr['grad_accum'] == 0:
                scaler.unscale_(opt)
                if tr.get('clip_grad', 0.0) > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), tr['clip_grad'])
                scaler.step(opt)
                scaler.update()
                if sched is not None:
                    sched.step()
                opt.zero_grad(set_to_none=True)

            cur_c = float(parts['contrast'])
            if gstep == 0:                 # 首步直接取原值，免得 EMA 从 0 爬升
                ema_loss, ema_c = loss.item(), cur_c
            else:
                ema_loss = 0.95 * ema_loss + 0.05 * loss.item()
                ema_c = 0.95 * ema_c + 0.05 * cur_c
            gstep += 1

            if gstep % cfg['log_every'] == 0:
                segs_s = gstep * tr['batch_size'] / max(time.time() - t0, 1e-6)
                extra = ''.join(f' {k}={float(v):.4f}' for k, v in parts.items()
                                if k != 'contrast')
                _log(f'[e{epoch} s{step}] loss={ema_loss:.4f} contrast={ema_c:.4f}{extra} '
                     f'lr={cur_lr():.2e} {segs_s:.0f} 段/s')

            # ── 验证 / 早停（与 baseline 同判据：只用 val_loss）──
            if gstep % tr['val_every'] == 0:
                vp = do_val(model, val_loader, device, pap, is_s, tr['fp16'])
                val_loss = vp['contrast']
                _log(f'[val e{epoch} s{gstep}] contrast={val_loss:.4f}'
                     + ''.join(f' {k}={v:.4f}' for k, v in vp.items() if k != 'contrast'))
                if val_loss < best_val:
                    best_val, stale = val_loss, 0
                    torch.save({'model': model.state_dict(), 'epoch': epoch,
                                'step': gstep, 'val_loss': val_loss,
                                'opt': opt.state_dict(), 'cfg': cfg, 'meta': meta_lite},
                               out_dir / 'ckpt' / 'best.pt')
                    _log(f'best.pt updated (contrast={val_loss:.4f})')
                else:
                    stale += 1
                    _log(f'stale {stale}/{patience}')
                    if patience > 0 and stale >= patience:
                        _log(f'== 提前停止 == best val contrast={best_val:.4f} '
                             f'（{time.time() - t0:.0f}s）')
                        return
                model.train()

        vp = do_val(model, val_loader, device, pap, is_s, tr['fp16'])
        val_loss = vp['contrast']
        _log(f'[val e{epoch} end] contrast={val_loss:.4f}'
             + ''.join(f' {k}={v:.4f}' for k, v in vp.items() if k != 'contrast'))
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'step': gstep, 'val_loss': val_loss,
                        'opt': opt.state_dict(), 'cfg': cfg, 'meta': meta_lite},
                       out_dir / 'ckpt' / 'best.pt')
            _log(f'best.pt updated (contrast={val_loss:.4f})')
        else:
            stale += 1
            _log(f'stale {stale}/{patience}')
            if patience > 0 and stale >= patience:
                _log(f'== 提前停止 == best val contrast={best_val:.4f} '
                     f'（{time.time() - t0:.0f}s）')
                return

    _log(f'== 训练完成 == best val contrast={best_val:.4f} 用时 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
