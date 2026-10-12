"""beatmae_encoder.py — BeatMAE 冻结编码器推理装配壳（Agent 侧，替代已退役的 ppg_model serve/encoder.py）。

断联背景：BeatMAE 开源清理时删除了 serve/ 推理服务层（论文仓只留训练/评估），
而其实时链运行时依赖（PPGEncoder）本就属于**消费方**职责。故在 PPG-Agent 侧
重写一个最小装配壳：只用 BeatMAE 仓的公开面（models/ 模型结构 + ckpt 内 cfg），
BeatMAE 一行动代码不改。

口径与 probe / 旧 serve/encoder.py 严格逐位一致：
    建 BeatMAE（arch 由 beatmae_arch_from_sd 自动识别）→ load_state_dict → eval
    → 冻结 requires_grad=False → embed_arrays: autocast(CUDA fp16) 走
    model.embed_segment → float().cpu()（掩码均值池化段嵌入）。

只实现低层 embed_arrays（波形已切好/定长重采样/逐段 z-score 的输入）——
实时链的表示层由 live_stream.represent（走 BeatMAE data/beat_ops）负责，
不需要旧 serve 的 embed_rows/CorpusDataset。

依赖纪律：torch / BeatMAE 仓 / ckpt 任一缺失都在构造期抛异常，由调用方
（live_stream._get_encoder）统一包成 LiveStreamUnavailable → 上层降级查表。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from ..config import resolve_beatmae_root


def _derive_run_name(ckpt_path: Path) -> str:
    """encoder_version 的 run 标签：Zenodo 打包名（beatmae_v25c3_s4.pt）取文件干；
    训练目录约定 .../<run>/ckpt/best.pt 取 run 目录名。"""
    stem = ckpt_path.stem
    if stem == "best":
        return ckpt_path.parents[1].name
    return stem


class BeatmaeEncoder:
    """BeatMAE 段嵌入推理封装（冻结、eval）。

    Args:
        ckpt_path: encoder ckpt 路径（内含 model state_dict + cfg，同 probe.load_frozen；
            Zenodo 脱敏版与训练整份均可——二者权重 bit-identical）。
        root:      BeatMAE 仓根；缺省走 config.resolve_beatmae_root() 三级解析。
        device:    'cuda'/'cpu'，默认可用即 cuda。
        fp16:      CUDA 上启用 fp16 autocast（与 probe 一致；CPU 忽略）。
        run_name:  encoder_version 的 run 标签，缺省由 ckpt 文件名推导。
    """

    def __init__(self, ckpt_path: str | Path, root: str | Path | None = None,
                 device: str | None = None, fp16: bool = True,
                 run_name: str | None = None):
        self.ckpt_path = Path(ckpt_path).resolve()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = bool(fp16) and self.device.type == "cuda"

        root = Path(root) if root else resolve_beatmae_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from models import BeatMAE                              # BeatMAE 公开面
        from models.beat_mae.beat_mae import beatmae_arch_from_sd
        from train.run_meta import ckpt_sha1

        ck = torch.load(self.ckpt_path, map_location=self.device,
                        weights_only=False)
        mcfg, dcfg = ck["cfg"]["model"], ck["cfg"]["data"]
        self.model = BeatMAE(
            beat_len=dcfg["beat_len"], d_model=mcfg["d_model"],
            depth=mcfg["depth"], nhead=mcfg["nhead"],
            dec_kind=mcfg["dec_kind"], d_dec=mcfg["d_dec"],
            dec_depth=mcfg["dec_depth"],
            n_aux_feats=mcfg.get("n_aux_feats", 0),
            arch=beatmae_arch_from_sd(ck["model"]))
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        sha = ckpt_sha1(self.ckpt_path)
        run = run_name or _derive_run_name(self.ckpt_path)
        self.info = {
            "kind": "beatmae",
            "run": run,
            "ckpt": str(self.ckpt_path),
            "ckpt_sha1_12": sha,
            "d_model": mcfg["d_model"],
            "sources": dcfg.get("sources"),
            "encoder_version": f"{run}@{sha}",
        }

    @property
    def d_model(self) -> int:
        return self.info["d_model"]

    @torch.no_grad()
    def embed_arrays(self, beats: torch.Tensor, beat_pad_mask: torch.Tensor,
                     onset: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """低层接口：(B,K,L) 等已预处理张量 → (B, d) 段嵌入（CPU float32）。"""
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model.embed_segment(
                beats.to(self.device), beat_pad_mask.to(self.device),
                onset.to(self.device), side.to(self.device))
        return out.float().cpu()

    def embed_arrays_grad(self, beats: torch.Tensor, beat_pad_mask: torch.Tensor,
                          onset: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """可微通道：与 ``embed_arrays`` 同前向（``model.embed_segment``），但
        保留 autograd 图、用 fp32（不 autocast / 不 no_grad）、输出留在
        ``self.device``。调用方需先对 ``beats`` 设 ``requires_grad_(True)``（已
        在 device 上）。用于 L2 输入归因：参数仍冻结（requires_grad=False），
        但输入梯度可回传。fp32 与线上 fp16 逐位不同——归因取梯度方向，
        接受此差异以保证反传数值稳定。"""
        return self.model.embed_segment(
            beats, beat_pad_mask, onset, side)
