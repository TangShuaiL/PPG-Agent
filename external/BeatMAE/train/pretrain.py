#!/usr/bin/env python
"""pretrain.py — BeatMAE 预训练（第 1 轮：严格模式，排除测试折 1/2）。

用法（配置基座 = config/base/beatmae.yaml；差量一律命令行 key=value，
`load_config` 只认 `a.b=v` 形式，勿用 `--out_dir xxx`）：
    python3 train/pretrain.py config/base/beatmae.yaml \
        out_dir=results/pretrain/<family>/<corpus>/<variant> data.sources=null
    必传 override：out_dir（基座不给默认值）

组件：
- 数据：CorpusDataset（memmap）+ SourceBalancedSampler（sqrt 源平衡）
- 掩码：RegionMasker 三模式（cfg mask.mode：unified/weighted/hybrid，
  每步新种子 generator=seed×1000003+gstep，同种子可完全复现）
- 损失：L_recon（被遮样本 MSE）+ aux_weight × L_aux（ICC 加权 SmoothL1，z 参数化）
  + 可选 ce_aux：V3-B 拍形态离散码 CE（线性头读 token，只算可见拍，类别加权）
- 精度：fp16 AMP + GradScaler（V100）；梯度累积 + 裁剪
- 验证：折 1/2 上的掩码重建损失（固定掩码种子，可跨轮对比）
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import (CorpusDataset, EpochTraverseSampler, RegionMaskConfig,
                  RegionMasker, SourceBalancedSampler, beat_collate,
                  zero_masked_regions)
from data.corpus_dataset import BEAT_LEN
from models import BeatMAE, aux_loss, beat_codes, ce_loss, masked_mse
from train.run_meta import ckpt_meta, write_meta


def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    for ov in overrides:
        k, v = ov.split('=', 1)
        keys = k.split('.')
        d = cfg
        for kk in keys[:-1]:
            d = d.setdefault(kk, {})
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        d[keys[-1]] = v
    return cfg


def build_icc_weights(reliability_json, names, gate):
    """40 列辅助特征 → ICC 权重（mimic+vital 均值，低于 gate 置 0）。"""
    rel = json.load(open(reliability_json))
    w = []
    for n in names:
        iccs = [rel[s].get(n) for s in ('mimic', 'vital') if n in rel.get(s, {})]
        w.append(float(np.mean(iccs)) if iccs else 0.0)
    w = torch.tensor(w, dtype=torch.float32)
    w[w < gate] = 0.0
    if (w > 0).sum() == 0:                    # 全被门控掉则退化为均匀
        w = torch.ones_like(w)
    return w / w.sum() * (w > 0).sum()        # 保持激活维平均权重=1


@torch.no_grad()
def validate(model, loader, masker, device, fp16, n_batches):
    model.eval()
    gen = torch.Generator().manual_seed(777)   # 固定验证掩码，跨轮可比
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        m = masker(batch['keypoints'], batch['kp_valid'],
                   batch['beat_pad_mask'], generator=gen)
        beats_z = zero_masked_regions(batch['beats'], m['sample_mask'])
        beats = batch['beats'].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=fp16):
            out = model(beats_z.to(device), batch['beat_pad_mask'].to(device),
                        batch['onset'].to(device), batch['side'].to(device),
                        m['beat_mask'].to(device), m['region_mask'].to(device))
            loss = masked_mse(out['recon'], beats,
                              m['sample_mask'].to(device))
        tot += loss.item()
        n += 1
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--resume', default=None)
    args, overrides = ap.parse_known_args()
    cfg = load_config(args.config, overrides)

    # ---------------- 全局随机种子（所有随机源的唯一入口） ----------------
    seed = cfg['train']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)               # python 内置 random（防御性，当前未用）
    det = bool(cfg['train'].get('deterministic', False))
    if det:                         # GPU 逐位复现模式（慢，调试/消融用）
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cfg['out_dir'])
    (out_dir / 'ckpt').mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train.log'

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    # ---------------- 运行元数据（git + 环境 + 生效配置快照） ----------------
    meta = write_meta(out_dir, cfg, args.config)
    git = meta['git']
    meta_lite = ckpt_meta(meta)     # ckpt 用轻量版（cfg 已单独存于 ckpt）
    log(f'== run_meta == git={git["commit"][:8] if git["commit"] else "未提交"}'
        f' dirty={git["dirty"]} | host={meta["host"]["hostname"]}'
        f' gpu={meta["host"].get("gpu", "?")}')

    # 设备确认（沙箱环境下 /dev/nvidia* 不可见会静默退回 CPU，必须显式检查）
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用：拒绝在 CPU 上跑预训练'
                           '（若确认环境无 GPU，请检查是否在沙箱内运行）')
    log(f'== 设备 == cuda:{torch.cuda.current_device()} '
        f'{torch.cuda.get_device_name()}')
    log(f'== 随机 == seed={seed} deterministic={det} '
        f'（掩码/采样/Dataset 窗口均显式种子驱动）')

    # ---------------- 数据 ----------------
    tr = cfg['train']
    sources = cfg['data'].get('sources')     # 单源专用模型：['mimic'] 等；None=三库
    # 拍长口径单点：yaml 的 data.beat_len 必须同时喂给 Dataset、掩码器与模型。
    # 此前只喂了模型，另两处回落常量 BEAT_LEN——yaml 调成非 128 会形状对不上
    # （2026-09-11 收编为单一局部变量；缺省 = 模块常量 BEAT_LEN）
    beat_len = cfg['data'].get('beat_len', BEAT_LEN)
    log(f'== 数据 == root={cfg["corpus_root"]} sources={sources or "全部"} '
        f'beat_len={beat_len}')
    train_ds = CorpusDataset(cfg['corpus_root'],
                             exclude_folds=cfg['test_folds'],
                             sources=sources,
                             max_rows=cfg['data'].get('max_rows'),
                             max_beats=cfg['data']['max_beats'],
                             beat_len=beat_len)
    val_ds = CorpusDataset(cfg['corpus_root'],
                           include_folds=cfg['test_folds'],
                           sources=sources,
                           max_rows=cfg['val']['max_segments'],
                           max_beats=cfg['data']['max_beats'],
                           beat_len=beat_len)
    if tr['sampler_mode'] == 'traverse':
        # 标准 epoch 遍历模式：全局无放回 shuffle，epoch 长度 = 全池步数，
        # 覆盖率 100%；源占比为自然比例（与流式 sqrt 源平衡的差异需在实验记录注明）
        sampler = EpochTraverseSampler(train_ds, seed=tr['seed'])
        spe = math.ceil(len(train_ds) / tr['batch_size'])   # 每 epoch 实际步数
    else:
        sampler = SourceBalancedSampler(
            train_ds, n_samples=tr['steps_per_epoch'] * tr['grad_accum']
            * tr['batch_size'], mode=tr['sampler_mode'], seed=tr['seed'])
        spe = tr['steps_per_epoch']
    train_loader = DataLoader(train_ds, batch_size=tr['batch_size'],
                              sampler=sampler, collate_fn=beat_collate,
                              num_workers=tr['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tr['batch_size'],
                            shuffle=False, collate_fn=beat_collate,
                            num_workers=2, pin_memory=True)
    log(f'train={len(train_ds)} 段 | val={len(val_ds)} 段 | '
        f'sampler={tr["sampler_mode"]}'
        + (f' 源抽样权重={sampler.stats()}'
           if hasattr(sampler, 'stats') else
           f' 每 epoch={spe} 步（全池遍历，无放回）'))

    # ---------------- 辅助任务资产 ----------------
    stats = np.load(cfg['feat_stats'])
    feat_mean = torch.from_numpy(stats['mean']).to(device)
    feat_std = torch.from_numpy(stats['std']).to(device)
    feat_names = [str(s) for s in stats['names']]
    icc_w = build_icc_weights(cfg['reliability_json'], feat_names,
                              tr['icc_gate']).to(device)
    n_active = int((icc_w > 0).sum())
    # 分母取特征表实际维数（此前写死 40，n_aux_feats≠40 时日志说谎）
    log(f'辅助任务: {n_active}/{len(feat_names)} 维特征通过 ICC≥{tr["icc_gate"]} 门控')

    # ---------------- V3-B 拍形态离散码本（可选，cfg ce_aux 段） ----------------
    ce_cfg = cfg.get('ce_aux') or {}
    ce_centroids, ce_w = None, None
    if ce_cfg.get('n_codes', 0):
        cb = np.load(ce_cfg['codebook'])
        ce_centroids = torch.from_numpy(cb['centroids']).to(device)
        ce_w = torch.from_numpy(cb['class_weights']).float().to(device)
        log(f'== V3-B == 形态码本 {ce_cfg["n_codes"]} 类 '
            f'(weight={ce_cfg["weight"]}) ← {ce_cfg["codebook"]}')

    # ---------------- 模型与优化 ----------------
    mc = cfg['model']
    model = BeatMAE(beat_len=beat_len, d_model=mc['d_model'],
                    depth=mc['depth'], nhead=mc['nhead'],
                    dec_kind=mc['dec_kind'], d_dec=mc['d_dec'],
                    dec_depth=mc['dec_depth'],
                    n_aux_feats=mc['n_aux_feats'],
                    n_ce_codes=ce_cfg.get('n_codes', 0),
                    dim_ff=mc.get('dim_ff'),
                    dropout=mc.get('dropout', 0.0)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=tr['lr'],
                            weight_decay=tr['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=tr['fp16'])
    _mask_cfg = dict(cfg.get('mask', {}))
    _mask_cfg.pop('ladder', None)          # ladder 是 pretrain 层调度参数，不进 dataclass
    _mask_cfg.setdefault('beat_len', beat_len)   # 与 Dataset/模型同源（见上）
    masker = RegionMasker(RegionMaskConfig(**_mask_cfg))
    # 课程掩码（v2.7h）：mask_ladder={start, end, ramp_epochs}——训练步内线性升
    # target_ratio；验证始终用基准 ratio（cfg.mask.target_ratio），保证跨 epoch
    # 的 val_recon 可比（早停依据不受课程影响）
    _ladder = cfg.get('mask', {}).get('ladder')
    mask_ladder = None
    if _ladder:
        mask_ladder = (_ladder['start'], _ladder['end'],
                       _ladder['ramp_epochs'] * spe)
        log(f'== 课程掩码 == ratio {mask_ladder[0]}→{mask_ladder[1]} '
            f'线性 ramp {mask_ladder[2]} 步（验证固定基准 '
            f'{masker.cfg.target_ratio}）')
    _base_ratio = masker.cfg.target_ratio
    log(f'== 掩码 == mode={masker.cfg.mode} '
        f'target_ratio={masker.cfg.target_ratio} '
        f'region_share={masker.cfg.region_share}')

    total_opt_steps = tr['epochs'] * spe

    def lr_lambda(step):
        if step < tr['warmup_steps']:
            return step / max(tr['warmup_steps'], 1)
        p = (step - tr['warmup_steps']) / max(
            total_opt_steps - tr['warmup_steps'], 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_epoch, opt_step = 0, 0
    best_val = float('inf')
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        start_epoch, opt_step = ck['epoch'] + 1, ck['opt_step']
        best_val = ck.get('val_recon', float('inf'))
        log(f'已恢复 {args.resume} (epoch={start_epoch} step={opt_step} '
            f'best_val={best_val:.4f})')

    # ---------------- 验证与早停 ----------------
    patience = tr.get('early_stop_patience', 0)   # 连续 N 次验证无改善则停；0=关
    bad_checks = 0

    def do_val(gstep, epoch):
        """跑一次验证；改善则刷写 best.pt；返回是否触发早停。"""
        nonlocal best_val, bad_checks
        cur_ratio = masker.cfg.target_ratio          # 课程模式下暂存训练值
        masker.cfg.target_ratio = _base_ratio        # 验证固定基准比例
        v = validate(model, val_loader, masker, device, tr['fp16'],
                     cfg['val']['n_batches'])
        masker.cfg.target_ratio = cur_ratio          # 恢复课程当前值
        improved = v < best_val
        if improved:
            best_val, bad_checks = v, 0
        else:
            bad_checks += 1
        log(f'[val e{epoch} s{gstep}] recon={v:.4f} best={best_val:.4f}'
            + ('  ↑best' if improved else f'  stale {bad_checks}/{patience}'))
        if improved:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(),
                        'scaler': scaler.state_dict(),
                        'epoch': epoch, 'opt_step': opt_step,
                        'val_recon': v, 'cfg': cfg, 'meta': meta_lite},
                       out_dir / 'ckpt' / 'best.pt')
            log(f'best.pt 已更新 (val recon={v:.4f})')
        model.train()
        return patience > 0 and bad_checks >= patience

    # ---------------- 训练循环 ----------------
    log(f'== 开训 == {n_params:.2f}M 参数 | {tr["epochs"]} epoch × '
        f'{spe} 步（{"遍历" if tr["sampler_mode"] == "traverse" else "流式"}） | '
        f'bs={tr["batch_size"]} × accum={tr["grad_accum"]} | fp16={tr["fp16"]}')
    model.train()
    t0 = time.time()
    start_gstep = start_epoch * spe                    # resume 后吞吐基线修正
    stop = False
    for epoch in range(start_epoch, tr['epochs']):
        sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        ema_recon, ema_aux, ema_ce = 0.0, 0.0, 0.0
        for step, batch in enumerate(train_loader):
            gstep = epoch * spe + step
            if mask_ladder is not None:              # 课程掩码：随步线性升
                r0, r1, ramp = mask_ladder
                masker.cfg.target_ratio = r0 + (r1 - r0) * min(
                    1.0, (gstep - start_gstep) / max(ramp, 1))
            gen = torch.Generator().manual_seed(tr['seed'] * 1000003 + gstep)
            m = masker(batch['keypoints'], batch['kp_valid'],
                       batch['beat_pad_mask'], generator=gen)
            beats_z = zero_masked_regions(batch['beats'], m['sample_mask'])
            beats = batch['beats'].to(device, non_blocking=True)
            feat_z = ((batch['feat'].to(device) - feat_mean) / feat_std
                      ).clamp(-5, 5)
            aux_valid = (batch['feat_valid']
                         & ~batch['beat_pad_mask']).to(device)
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16, enabled=tr['fp16']):
                out = model(beats_z.to(device, non_blocking=True),
                            batch['beat_pad_mask'].to(device, non_blocking=True),
                            batch['onset'].to(device, non_blocking=True),
                            batch['side'].to(device, non_blocking=True),
                            m['beat_mask'].to(device),
                            m['region_mask'].to(device))
                l_recon = masked_mse(out['recon'], beats,
                                     m['sample_mask'].to(device))
                l_aux = aux_loss(out['aux'], feat_z, aux_valid, icc_w,
                                 beat_mask=m['beat_mask'].to(device))
                loss = l_recon + tr['aux_weight'] * l_aux
                if ce_centroids is not None:
                    # 目标取自原始波形（答案侧，不进 encoder 输入）；只算可见拍
                    ce_tgt = beat_codes(beats, ce_centroids)
                    ce_valid = (~batch['beat_pad_mask'].to(device)
                                & ~m['beat_mask'].to(device))
                    l_ce = ce_loss(out['ce_logits'], ce_tgt, ce_valid, ce_w)
                    loss = loss + ce_cfg['weight'] * l_ce
            scaler.scale(loss / tr['grad_accum']).backward()

            ema_recon = 0.95 * ema_recon + 0.05 * l_recon.item()
            ema_aux = 0.95 * ema_aux + 0.05 * l_aux.item()
            if ce_centroids is not None:
                ema_ce = 0.95 * ema_ce + 0.05 * l_ce.item()

            if (step + 1) % tr['grad_accum'] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               tr['clip_grad'])
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

            if gstep % cfg['log_every'] == 0:
                segs_s = (gstep - start_gstep + 1) * tr['batch_size'] / max(
                    time.time() - t0, 1e-6)
                log(f'[e{epoch} s{step}] recon={ema_recon:.4f} '
                    f'aux={ema_aux:.4f}'
                    + (f' ce={ema_ce:.3f}' if ce_centroids is not None else '')
                    + f' lr={sched.get_last_lr()[0]:.2e} '
                    f'{segs_s:.0f} 段/s')

            if gstep > 0 and gstep % cfg['ckpt']['every'] == 0:
                p = out_dir / 'ckpt' / f'step{gstep:07d}.pt'
                torch.save({'model': model.state_dict(),
                            'opt': opt.state_dict(),
                            'sched': sched.state_dict(),
                            'scaler': scaler.state_dict(),
                            'epoch': epoch, 'opt_step': opt_step,
                            'cfg': cfg}, p)
                log(f'ckpt 保存 {p}')

            if gstep > 0 and gstep % tr['val_every'] == 0:
                if do_val(gstep, epoch):
                    stop = True
                    break

        # epoch 末固定存一份
        p = out_dir / 'ckpt' / f'epoch{epoch:03d}.pt'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch, 'opt_step': opt_step, 'cfg': cfg}, p)
        glast = (epoch + 1) * spe - 1
        if do_val(glast, epoch):
            stop = True
        log(f'[epoch {epoch} 完成] ckpt={p}')
        if stop:
            break

    # 正常收尾：断点备份（step/epoch）已完成崩溃恢复使命，只留 best.pt。
    # 异常崩溃时走不到这里，备份仍在，可 resume 后再清。
    ck = out_dir / 'ckpt'
    leftovers = sorted(ck.glob('step*.pt')) + sorted(ck.glob('epoch*.pt'))
    if leftovers:
        freed = sum(p.stat().st_size for p in leftovers)
        for p in leftovers:
            p.unlink()
        log(f'断点备份清理: {len(leftovers)} 个 step/epoch ckpt 已删 '
            f'(释放 {freed / 1073741824:.1f}GB, best.pt 保留)')

    log(f'== 预训练{"提前停止" if stop else "完成"} == '
        f'best val recon={best_val:.4f} (ckpt/best.pt)')


if __name__ == '__main__':
    main()
