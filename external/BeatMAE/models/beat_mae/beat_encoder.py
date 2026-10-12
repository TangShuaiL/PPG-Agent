"""beat_mae 编码器：每拍 → CNN stem → token（v2 主架构）。

接口约定：
    beats:         (B, K, L) 拍波形（逐段 z-score 后）
    beat_pad_mask: (B, K) bool，True=padding 拍
    onset_times:   (B, K) 拍起始秒数
    side:          (B, K, 2) 侧信息 [beat_len_ms, hr_bpm]（无效拍补 0）
    beat_mask:     (B, K) bool，True=该拍被遮（MAE 训练时传入）
    region_mask:   (B, K, R) bool，True=该区域被遮
输出：
    tokens: (B, K, d_model)；token_pad_mask: (B, K) bool

v2 变更（相对 v1）：
    - stem 重写：渐进下采样 + BN + BasicBlock 残差精炼 + 多尺度池化拼接收口，
      移除 Flatten-Linear 瓶颈（v1 的 32:1 单步维度压缩）；
    - d_model 默认 256（与 MMR/xMAE 基线对齐）；
    - 整拍遮的 mask token 补齐 TimeEncoding + 侧信息（v1 遗漏：被遮拍丢失
      时间/节律先验，decoder 端相对位置感知失效）。三者均为遮罩不变量。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import N_REGIONS, TimeEncoding, TransformerStack


class BasicBlock1d(nn.Module):
    """同通道 k3 残差块（双层卷积 + 恒等捷径）。"""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(x + h)


class BeatTokenEncoder(nn.Module):
    """每拍 → CNN stem → token。

    arch='v2'（默认）：v2 整拍 CNN 多尺度 stem（当前主架构）；
    arch='v1'：v1 Flatten-Linear stem（仅用于加载旧 checkpoint，如 mr065 轮）。

    两级遮挡处理（对应 region_mask 定稿方案）：
    - 整拍遮（beat_mask）：token 替换为可学习 mask_token + TimeEncoding
      + 侧信息 + region-type 提示；
    - 拍内部分区域遮（仅 region_mask）：保留波形嵌入（输入侧被遮区域已由
      数据层置 0），叠加 region-type 提示告知缺的是哪类区域。
    """

    def __init__(self, beat_len: int = 128, d_model: int = 256, depth: int = 6,
                 nhead: int = 8, dim_ff: int | None = None, dropout: float = 0.0,
                 side_dim: int = 2, arch: str = 'v2'):
        super().__init__()
        self.d_model = d_model
        self.arch = arch
        if arch == 'v1':
            # v1 stem：三段卷积 + Flatten-Linear 单步压缩（与旧 ckpt 权重对齐）
            self.stem = nn.Sequential(
                nn.Conv1d(1, 48, 7, padding=3), nn.GELU(),
                nn.Conv1d(48, 96, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(96, d_model, 3, stride=2, padding=1), nn.GELU())
            self.pool_n = (beat_len + 3) // 4
            self.stem_norm = nn.LayerNorm(d_model * self.pool_n)
            self.stem_proj = nn.Linear(d_model * self.pool_n, d_model)
        else:
            mid_ch = d_model // 2
            # v2 stem：1→48→96→mid_ch→d_model，时间维 128→64→32→16，无 Flatten
            self.stem1 = nn.Sequential(
                nn.Conv1d(1, 48, 7, padding=3), nn.BatchNorm1d(48), nn.GELU())
            self.stem2 = nn.Sequential(
                nn.Conv1d(48, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96), nn.GELU())
            self.stem3 = nn.Sequential(
                nn.Conv1d(96, mid_ch, 3, stride=2, padding=1),
                nn.BatchNorm1d(mid_ch), nn.GELU())
            self.stem4 = nn.Sequential(
                nn.Conv1d(mid_ch, d_model, 3, stride=2, padding=1),
                nn.BatchNorm1d(d_model), nn.GELU())
            self.refine = BasicBlock1d(d_model)
            # 多尺度收口：深层(d) + 中层(mid_ch) 各自归一化后拼接投影（跨分辨率
            # 跳连，防止细粒度特征在深层被稀释）
            self.mid_norm = nn.LayerNorm(mid_ch)
            self.deep_norm = nn.LayerNorm(d_model)
            self.stem_proj = nn.Linear(d_model + mid_ch, d_model)

        self.side_proj = nn.Sequential(
            nn.Linear(side_dim, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model))
        self.time_enc = TimeEncoding(d_model)
        self.region_type_emb = nn.Embedding(N_REGIONS + 1, d_model)  # idx R=padding 位
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.encoder = TransformerStack(d_model, depth, nhead, dim_ff, dropout)

    def _stem(self, x):
        """整拍 CNN 主干：返回 (深层特征, 中层特征)。"""
        h = self.stem1(x)                               # (n, 48, 128)
        h = self.stem2(h)                               # (n, 96, 64)
        mid = self.stem3(h)                             # (n, mid_ch, 32)
        h = self.stem4(mid)                             # (n, d, 16)
        h = self.refine(h)                              # (n, d, 16)
        return h, mid

    def forward(self, beats, beat_pad_mask=None, onset_times=None,
                side=None, beat_mask=None, region_mask=None):
        B, K, L = beats.shape
        if beat_pad_mask is None:
            beat_pad_mask = beats.new_zeros(B, K, dtype=torch.bool)

        x = beats.reshape(B * K, 1, L)
        if self.arch == 'v1':
            h = self.stem(x).flatten(1)                 # (n, d*pool_n)
            h = self.stem_proj(self.stem_norm(h)).reshape(B, K, -1)
        else:
            deep, mid = self._stem(x)
            h = self.stem_proj(torch.cat([
                self.deep_norm(deep.mean(dim=-1)),
                self.mid_norm(mid.mean(dim=-1))], dim=-1)).reshape(B, K, -1)

        if side is not None:
            h = h + self.side_proj(side)
        if onset_times is not None:
            h = h + self.time_enc(onset_times)

        reg_sum = None
        if region_mask is not None:
            reg = self.region_type_emb.weight[:N_REGIONS]          # (R, d)
            reg_sum = region_mask.to(h.dtype) @ reg                # (B, K, d)

        if reg_sum is not None:
            # 拍内部分区域遮：波形嵌入 + region-type 提示
            partial = region_mask.any(dim=-1)
            if beat_mask is not None:
                partial = partial & ~beat_mask
            h = torch.where(partial.unsqueeze(-1), h + reg_sum, h)

        if beat_mask is not None:
            # 整拍遮：mask_token + region-type 提示（v1）；v2 追加 TimeEncoding
            # + 侧信息（均为遮罩不变量，不引入被遮波形值的信息）
            rep = self.mask_token.expand(B, K, -1)
            if self.arch != 'v1':
                if onset_times is not None:
                    rep = rep + self.time_enc(onset_times)
                if side is not None:
                    rep = rep + self.side_proj(side)
            if reg_sum is not None:
                rep = rep + reg_sum
            h = torch.where(beat_mask.unsqueeze(-1), rep, h)

        tokens = self.encoder(h, key_padding_mask=beat_pad_mask)
        return tokens, beat_pad_mask
