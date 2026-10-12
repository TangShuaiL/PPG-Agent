#!/usr/bin/env python3
"""对比模型（HiMAE / MMR / xMAE PPG）预训练脚本。

与 BeatMAE 共享同一语料管线（CorpusDataset + SourceBalancedSampler），
将拍级张量拼接为定长波形后做 patch 级随机掩码重建。
用法：python train/pretrain_baseline.py config/base/himae.yaml [overrides...]
      （himae / mmr / xmae 三族各一份 base 配置，用法见 config/README.md）
"""
import argparse, sys, time, random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.corpus_dataset import (CorpusDataset, beat_collate, raw_collate,
                                 MAX_BEATS, BEAT_LEN)
from data.samplers import EpochTraverseSampler, SourceBalancedSampler
from train.pretrain import load_config
from train.run_meta import ckpt_meta, log, write_meta
from models import masked_mse
from baselines import HimaeAE, MMRMAE, XmaePPGMAE

# ──────────────────── 波形提取 ────────────────────
# 拍数/拍长口径权威在 data.corpus_dataset（本处只 import，勿再写字面量）。
# 生效拍数 = yaml 的 data.max_beats，main() 读出后经 partial 传进 collate；
# WAVE_LEN 是基座口径（32×128）下的窗长，非基座时以 main() 的 wave_len 为准。
WAVE_LEN = MAX_BEATS * BEAT_LEN  # 4096
SEG_RAW = 1250                   # 原始 10s 段 @125Hz（raw_waveform 口径）


def waveform_collate(batch, max_beats=MAX_BEATS, wave_len=WAVE_LEN):
    """beat_collate → 拼接为定长波形 (B, wave_len)。"""
    bc = beat_collate(batch)
    B = len(batch)
    beats = bc['beats']                     # (B, K, L)  K ≤ max_beats
    K = beats.shape[1]
    if K < max_beats:
        padded = torch.zeros(B, max_beats, BEAT_LEN)
        padded[:, :K] = beats
        beats = padded
    return beats.reshape(B, wave_len)       # (B, 4096) @ 基座口径


def random_patch_mask(B, N, mask_ratio, generator):
    """均匀随机 patch 掩码 (B, N) bool，True=被遮。每样本被遮数一致。"""
    n_mask = round(N * mask_ratio)
    pm = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        idx = torch.randperm(N, generator=generator)[:n_mask]
        pm[b, idx] = True
    return pm


# ──────────────────── 模型工厂 ────────────────────
# 注：此处原有 MODEL_REGISTRY 字典，已于 2026-09-10 删除——全仓零引用，
# 且与下面 build_model() 重复（后者还要按族传不同构造参数，字典版覆盖不到）。


def build_model(name, seg_len, patch_len):
    if name == 'himae':
        return HimaeAE(patch_len=patch_len)
    elif name == 'mmr':
        return MMRMAE(seg_len=seg_len, patch_len=patch_len)
    elif name == 'xmae':
        return XmaePPGMAE(seq_len=seg_len, patch_len=patch_len)
    else:
        raise ValueError(f'未知模型: {name}')


# ──────────────────── 训练主循环 ────────────────────
def main():
    # 与姊妹入口 train/pretrain.py 同形：位置参数 config + 未识别项作 overrides
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)
    seed = cfg['train']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 默认值只在配置里根本没有 out_dir 键时生效。基座写的是 `out_dir: null`
    # （键存在、值为 None），此时 .get 返回 None、本默认值不可达 → 真实入口
    # 一律是命令行 override。见 config/README.md「配套约束」一节的核查表。
    out_dir = Path(cfg.get('out_dir', 'results/pretrain/baseline'))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ckpt').mkdir(exist_ok=True)
    log_path = out_dir / 'train.log'
    _log = lambda msg: log(msg, log_path)

    # 运行元数据（git + 环境 + 生效配置快照）；baseline best.pt 此前无 cfg，一并补齐
    meta = write_meta(out_dir, cfg, sys.argv[1])
    git = meta['git']
    meta_lite = ckpt_meta(meta)
    _log(f'== run_meta == git={git["commit"][:8] if git["commit"] else "未提交"}'
         f' dirty={git["dirty"]} | host={meta["host"]["hostname"]}'
         f' gpu={meta["host"].get("gpu", "?")}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', '对比模型训练需要 GPU'
    _log(f'== 设备 == {torch.cuda.get_device_name(0)}')

    # ── 数据 ──
    model_name = cfg['baseline']['model']
    # 公平口径 v2：raw_waveform=原始 125Hz 波形直喂（对比模型推荐）；
    # beat_resampled=拍定长重采样波形（v1 旧口径，保留对照）
    input_mode = cfg['baseline'].get('input_mode', 'beat_resampled')
    # 拍数取 yaml 生效值（2026-09-11 前本脚本硬编码 32，基座里的 data.max_beats
    # 是假旋钮；train/pretrain.py 一直读 cfg，现两边一致）。缺省 = 模块常量。
    max_beats = int(cfg['data'].get('max_beats', MAX_BEATS))
    wave_len = max_beats * BEAT_LEN
    collate = (raw_collate if input_mode == 'raw_waveform'
               else partial(waveform_collate, max_beats=max_beats, wave_len=wave_len))
    sources = cfg['data'].get('sources')
    train_ds = CorpusDataset(cfg['corpus_root'], exclude_folds=cfg['test_folds'],
                             sources=sources, max_beats=max_beats,
                             input_mode=input_mode)
    val_ds = CorpusDataset(cfg['corpus_root'], include_folds=cfg['test_folds'],
                           sources=sources, max_rows=cfg['val']['max_segments'],
                           max_beats=max_beats, input_mode=input_mode)
    tr = cfg['train']
    if tr['sampler_mode'] == 'traverse':
        # 标准 epoch 遍历模式：全局无放回 shuffle，epoch 长度 = 全池步数，
        # 覆盖率 100%；源占比为自然比例（与流式 sqrt 源平衡的差异需在实验记录注明）
        sampler = EpochTraverseSampler(train_ds, seed=tr['seed'])
        spe = (len(train_ds) + tr['batch_size'] - 1) // tr['batch_size']
    else:
        sampler = SourceBalancedSampler(
            train_ds, n_samples=tr['steps_per_epoch'] * tr['batch_size'],
            mode=tr['sampler_mode'], seed=tr['seed'])
        spe = tr['steps_per_epoch']
    loader = torch.utils.data.DataLoader(train_ds, batch_size=tr['batch_size'],
                                         sampler=sampler, collate_fn=collate,
                                         num_workers=tr['num_workers'], pin_memory=True)
    _log(f'== 数据 == train={len(train_ds):,d} val={len(val_ds):,d} model={model_name} '
         f'input_mode={input_mode} max_beats={max_beats} sampler={tr["sampler_mode"]} '
         f'每epoch={spe}步')

    # ── 模型 ──
    seg_len = SEG_RAW if input_mode == 'raw_waveform' else wave_len
    patch_len = cfg['baseline']['patch_len']
    model = build_model(model_name, seg_len, patch_len).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    _log(f'== 模型 == {model_name} {n_params:.2f}M 参数 patch_len={patch_len}')

    opt = torch.optim.AdamW(model.parameters(), lr=tr['lr'], weight_decay=tr['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr['epochs'] * spe)
    scaler = GradScaler('cuda', enabled=tr['fp16'])

    mask_ratio = cfg['baseline']['mask_ratio']
    n_patches = seg_len // patch_len
    _log(f'== 掩码 == ratio={mask_ratio} n_patches={n_patches}')
    _log(f'== 开训 == {tr["epochs"]} epoch × {spe} 步'
         f'（{"遍历" if tr["sampler_mode"] == "traverse" else "流式"}）'
         f' | bs={tr["batch_size"]} fp16={tr["fp16"]}')

    best_val = float('inf')
    stale = 0
    patience = tr.get('early_stop_patience', 10)
    gstep = 0
    t0 = time.time()

    for epoch in range(tr['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        ema_loss = 0.0

        for step, waveform in enumerate(loader):
            gen = torch.Generator().manual_seed(seed * 1000003 + gstep)
            waveform = waveform.to(device, non_blocking=True)

            # 随机 patch 掩码
            pm = random_patch_mask(waveform.shape[0], n_patches, mask_ratio, gen).to(device)

            with autocast('cuda', dtype=torch.float16, enabled=tr['fp16']):
                out = model(waveform, pm)

                # 损失计算（因模型而异）
                if model_name == 'mmr':
                    # MMR 重建目标是 DWT 系数
                    target = out['coeff_target']              # (B, N, S*P)
                    pm_exp = pm.unsqueeze(-1).expand_as(target)
                    loss = masked_mse(out['recon'], target, pm_exp)
                else:
                    # HiMAE / xMAE 重建目标是原始波形
                    recon = out['recon']                      # (B, L)
                    # patch_mask → sample_mask
                    sm = pm.repeat_interleave(patch_len, dim=1)
                    loss = masked_mse(recon, waveform, sm)

            scaler.scale(loss / tr['grad_accum']).backward()

            if (step + 1) % tr['grad_accum'] == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tr['clip_grad'])
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

            ema_loss = 0.95 * ema_loss + 0.05 * loss.item()
            gstep += 1

            if gstep % cfg['log_every'] == 0:
                segs_s = (gstep) * tr['batch_size'] / max(time.time() - t0, 1e-6)
                _log(f'[e{epoch} s{step}] loss={ema_loss:.4f} '
                    f'lr={sched.get_last_lr()[0]:.2e} {segs_s:.0f} 段/s')

            # ── 验证 ──
            if gstep % cfg['train']['val_every'] == 0:
                val_loss = do_val(model, val_ds, device, model_name, patch_len,
                                  mask_ratio, n_patches, cfg, seed, collate)
                _log(f'[val e{epoch} s{gstep}] loss={val_loss:.4f}')
                if val_loss < best_val:
                    best_val = val_loss
                    stale = 0
                    torch.save({'model': model.state_dict(), 'epoch': epoch,
                                'step': gstep, 'val_loss': val_loss,
                                'opt': opt.state_dict(),
                                'cfg': cfg, 'meta': meta_lite},
                               out_dir / 'ckpt' / 'best.pt')
                    _log(f'best.pt updated (loss={val_loss:.4f})')
                else:
                    stale += 1
                    _log(f'stale {stale}/{patience}')
                    if patience > 0 and stale >= patience:
                        _log(f'== 提前停止 == best val loss={best_val:.4f}')
                        return
                model.train()

        # epoch 末验证
        val_loss = do_val(model, val_ds, device, model_name, patch_len,
                          mask_ratio, n_patches, cfg, seed, collate)
        _log(f'[val e{epoch} end] loss={val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'step': gstep, 'val_loss': val_loss,
                        'opt': opt.state_dict(),
                        'cfg': cfg, 'meta': meta_lite},
                       out_dir / 'ckpt' / 'best.pt')
            _log(f'best.pt updated (loss={val_loss:.4f})')
        else:
            stale += 1
            _log(f'stale {stale}/{patience}')
            if patience > 0 and stale >= patience:
                _log(f'== 提前停止 == best val loss={best_val:.4f}')
                return

    _log(f'== 训练完成 == best val loss={best_val:.4f}')


@torch.no_grad()
def do_val(model, val_ds, device, model_name, patch_len, mask_ratio, n_patches, cfg, seed,
           collate):
    model.eval()
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg['train']['batch_size'],
                                             shuffle=False, collate_fn=collate,
                                             num_workers=cfg['train']['num_workers'])
    total_loss, n = 0.0, 0
    gen = torch.Generator().manual_seed(777)
    for waveform in val_loader:
        waveform = waveform.to(device, non_blocking=True)
        pm = random_patch_mask(waveform.shape[0], n_patches, mask_ratio, gen).to(device)
        with autocast('cuda', dtype=torch.float16, enabled=cfg['train']['fp16']):
            out = model(waveform, pm)
            if model_name == 'mmr':
                target = out['coeff_target']
                pm_exp = pm.unsqueeze(-1).expand_as(target)
                loss = masked_mse(out['recon'], target, pm_exp)
            else:
                sm = pm.repeat_interleave(patch_len, dim=1)
                loss = masked_mse(out['recon'], waveform, sm)
        total_loss += loss.item() * waveform.shape[0]
        n += waveform.shape[0]
    return total_loss / max(n, 1)


if __name__ == '__main__':
    main()
