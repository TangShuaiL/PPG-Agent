"""HiMAE 层级 CNN 编码器（复现自 himae.md 规格）。

5 个 Encoder Block：每块 2×Conv1d(k=5) + BatchNorm + GELU，
块间以 stride=2 卷积下采样（长度减半）。
通道：[16, 32, 64, 128, 256]。

为适配任意段长（本库 1250，论文 1000），各级输出长度在
forward 中记录并返回，供解码器精确还原。
"""
import torch.nn as nn

CHANNELS = (16, 32, 64, 128, 256)


class HimaeEncoderBlock(nn.Module):
    """2×Conv1d(k5) + BN + GELU，随后 stride-2 下采样卷积。

    forward 返回 (下采样前特征, 下采样后特征)：前者供解码器跳连。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.down = nn.Conv1d(out_ch, out_ch, 5, stride=2, padding=2)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        pre = self.act(self.bn2(self.conv2(x)))        # 跳连用（长度不变）
        return pre, self.down(pre)


class HimaeEncoder(nn.Module):
    """层级下采样编码器。

    Returns:
        skips:  list[(B, c_i, L_i)]，各级下采样前特征（跳连）
        bott:   (B, c_last, L_last)，最后一级下采样后特征
        lens:   list[int]，各级跳连长度（解码器还原用）
    """

    def __init__(self, in_ch: int = 1, channels=CHANNELS):
        super().__init__()
        self.blocks = nn.ModuleList([
            HimaeEncoderBlock(in_ch if i == 0 else channels[i - 1], c)
            for i, c in enumerate(channels)])

    def forward(self, x):
        skips, lens = [], []
        for blk in self.blocks:
            pre, x = blk(x)
            skips.append(pre)
            lens.append(pre.shape[-1])
        return skips, x, lens
