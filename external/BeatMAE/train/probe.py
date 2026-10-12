#!/usr/bin/env python3
"""probe.py — 嵌入提取与探针的公用库函数。

现役调用面（全部在此）：
    scripts/extract_embeddings.py   load_frozen / embed_batch / beat_collate_meta / raw_collate_meta
    tools/make_randinit_ckpt.py     load_frozen

段级流式探针（论文口径）在 `train/stream_probe.py`；本模块不含任何探针训练逻辑。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.corpus_dataset import (beat_collate, raw_collate, MAX_BEATS, BEAT_LEN)
from models import BeatMAE
from baselines import (HimaeAE, MMRMAE, XmaePPGMAE,
                       papagei_build_model, papagei_name_from_sd)
from models.beat_mae.beat_mae import beatmae_arch_from_sd

# 拍数/拍长口径权威在 data.corpus_dataset（本处只 import，勿再写字面量）：
# 32 拍 × 128 = 4096 定长波形，与 pretrain_baseline 一致；raw 口径为 1250 原始段
WAVE_LEN = MAX_BEATS * BEAT_LEN
SEG_RAW = 1250


# ──────────────────────── collate ────────────────────────
def _to_waveform(beats):
    """(B, K, L) → (B, WAVE_LEN)，K 右侧补零到 MAX_BEATS（对齐预训练分布）。"""
    B, K, L = beats.shape
    if K < MAX_BEATS:
        padded = torch.zeros(B, MAX_BEATS, L, dtype=beats.dtype)
        padded[:, :K] = beats
        beats = padded
    return beats.reshape(B, -1)


def beat_collate_meta(batch):
    """beat_collate + 转波形（baseline 用），统一携带 meta。"""
    bc = beat_collate(batch)
    bc['waveform'] = _to_waveform(bc['beats'])
    return bc


def raw_collate_meta(batch):
    """raw_collate + meta（raw 口径对比模型用）：原始 1250 点波形直喂。"""
    return {'waveform': raw_collate(batch),
            'meta': {'subject_id': [it['subject_id'] for it in batch],
                     'row': [it['row'] for it in batch],
                     'fold': [it['fold'] for it in batch]}}


# ──────────────────────── 模型工厂 ────────────────────────
def load_frozen(exp, device):
    """按 kind 构建冻结编码器并加载权重。

    可接受 kind：beatmae | himae | mmr | xmae | papagei[/_p|/_s]。
    papagei 例外——返回的 kind 是**解析后的** `papagei_p` / `papagei_s`（见下），
    调用方拿它当"实际跑了哪一支"的记账值即可。

    返回 (model, kind, input_mode, max_beats)：max_beats 取自 ckpt 内嵌 cfg
    （缺省 = 模块常量 MAX_BEATS），调用方必须用它建 CorpusDataset——否则
    ckpt 训练口径与嵌入口径不符（拍窗不同），且不会报错（2026-09-11 收编）。
    input_mode 是**最终生效值**（papagei 恒为 raw_waveform），调用方照此选 collate。
    """
    kind = exp['kind']
    ck = torch.load(exp['ckpt'], map_location=device, weights_only=False)
    sd = ck['model']
    _ck_cfg = ck.get('cfg') or {}
    max_beats = _ck_cfg.get('data', {}).get('max_beats', MAX_BEATS)

    if kind == 'beatmae':
        mcfg = ck['cfg']['model']
        dcfg = ck['cfg']['data']
        # V3-B ckpt 带 ce_head，按 sd 键自动识别以保持 strict 加载兼容
        n_ce = int(sd['ce_head.weight'].shape[0]) if 'ce_head.weight' in sd else 0
        model = BeatMAE(beat_len=dcfg['beat_len'], d_model=mcfg['d_model'],
                        depth=mcfg['depth'], nhead=mcfg['nhead'],
                        dec_kind=mcfg['dec_kind'], d_dec=mcfg['d_dec'],
                        dec_depth=mcfg['dec_depth'],
                        n_aux_feats=mcfg.get('n_aux_feats', 0),
                        n_ce_codes=n_ce,
                        arch=beatmae_arch_from_sd(sd),
                        dim_ff=mcfg.get('dim_ff'),
                        dropout=mcfg.get('dropout', 0.0))
        input_mode = 'beat_resampled'
    elif kind in ('papagei', 'papagei_p', 'papagei_s'):
        # PaPaGei P/S：只有 raw_waveform 一种口径（1250 点原始段，无拍操作，
        # 见 config/base/papagei.yaml）。`--kind papagei` 时 P/S 由 ckpt 自己说了算
        # （内嵌 cfg['papagei']['model']，缺 cfg 则按 state_dict 认 MoE），
        # 显式给 papagei_p / papagei_s 时校验一致——判别唯一权威在 baselines/papagei/。
        # **返回解析后的族名**（papagei_p / papagei_s）而不是笼统的 papagei：
        # 这个值会写进嵌入缓存的 kind 字段，是事后回溯"当时跑的是哪一支"的唯一凭据。
        pname = (_ck_cfg.get('papagei') or {}).get('model') or papagei_name_from_sd(sd)
        if kind != 'papagei' and kind != pname:
            raise ValueError(f'--kind={kind} 与 ckpt 里的模型 {pname} 不符')
        model = papagei_build_model(pname)
        input_mode = 'raw_waveform'
        if exp.get('input_mode') not in (None, 'raw_waveform'):
            raise ValueError(f'papagei 只有 raw_waveform 一种口径，'
                             f'实得 input_mode={exp["input_mode"]!r}')
        kind = pname
    else:
        # 公平口径 v2：input_mode 优先取 exp 显式指定，其次 ckpt cfg，
        # 最后默认 beat_resampled（旧口径 ckpt 向后兼容）
        bcfg = _ck_cfg.get('baseline', {}) if isinstance(_ck_cfg, dict) else {}
        input_mode = exp.get('input_mode',
                             bcfg.get('input_mode', 'beat_resampled'))
        seg_len = SEG_RAW if input_mode == 'raw_waveform' else WAVE_LEN
        if kind == 'himae':
            model = HimaeAE(patch_len=exp['patch_len'])
        elif kind == 'mmr':
            model = MMRMAE(seg_len=seg_len, patch_len=exp['patch_len'])
        elif kind == 'xmae':
            model = XmaePPGMAE(seq_len=seg_len, patch_len=exp['patch_len'])
        else:
            raise ValueError(f'未知 kind: {kind}')

    model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, kind, input_mode, max_beats


@torch.no_grad()
def embed_batch(model, kind, bc, device, fp16):
    """一批 → (B, d) 段嵌入。"""
    if kind == 'beatmae':
        with torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
            return model.embed_segment(
                bc['beats'].to(device), bc['beat_pad_mask'].to(device),
                bc['onset'].to(device), bc['side'].to(device)).float().cpu()
    else:
        # himae/mmr/xmae（拍拼接 4096 点）与 papagei（原始 1250 点）同走 waveform 的
        # embed_segment，但**嵌入维不同**：前四族 256，papagei 512（官方 n_classes=512）。
        # 下游头的输入宽度必须随之而变——写死 256 会在 papagei 上崩（2026-09-11 实测）
        with torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
            return model.embed_segment(bc['waveform'].to(device)).float().cpu()
