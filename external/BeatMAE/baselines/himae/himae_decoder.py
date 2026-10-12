"""HiMAE 镜像 CNN 解码器（复现自 himae.md 规格）。

与编码器对称：线性上采样（长度精确对齐到对应跳连）→
与跳连特征拼接 → 2×Conv1d(k5)+BN+GELU。
通道：[128, 64, 32, 16, 1]，末级输出层换 Tanh（论文要求 [-1,1]）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEC_CHANNELS = (128, 64, 32, 16, 1)


class HimaeDecoderBlock(nn.Module):
    """线性上采样对齐跳连长度 → 拼接 → 2×Conv(k5)。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, last: bool = False):
        super().__init__()
        self.last = last
        self.conv1 = nn.Conv1d(in_ch + skip_ch, out_ch, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2)
        if not last:
            self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.Tanh() if last else nn.GELU()

    def forward(self, x, skip, target_len):
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act(x) if self.last else self.act(self.bn2(x))


class HimaeDecoder(nn.Module):
    """镜像解码器。

    Args:
        bott:    (B, c_last, L_last) 瓶颈特征
        skips:   编码器各级跳连（浅→深对应 enc 顺序）
        lens:    各级跳连长度
    """

    def __init__(self, enc_channels=(16, 32, 64, 128, 256),
                 dec_channels=DEC_CHANNELS):
        super().__init__()
        n = len(dec_channels)
        self.blocks = nn.ModuleList()
        for i in range(n):
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i - 1]
            skip_ch = enc_channels[-(i + 1)]
            self.blocks.append(HimaeDecoderBlock(
                in_ch, skip_ch, dec_channels[i], last=(i == n - 1)))

    def forward(self, bott, skips, lens):
        x = bott
        for i, blk in enumerate(self.blocks):
            j = len(skips) - 1 - i                     # 从最深层往外
            x = blk(x, skips[j], lens[j])
        return x                                       # (B, 1, L)
