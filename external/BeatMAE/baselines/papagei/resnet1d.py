"""PaPaGei 骨干 ResNet1D / ResNet1DMoE（对比模型）。

复现自 PaPaGei 官方仓库 `references/papagei-foundation-model/models/resnet.py`
（© 2024 Nokia, BSD-3-Clause-Clear；上游架构又源自 Shenda Hong 2019 的 1-D ResNet）。
按本库惯例重写为独立实现 —— 不 import `references/`，那目录在 .gitignore 里，
clone 下来跑不了，且开源归属要干净。

数学结构与官方逐行对齐，仅两处非数学改动：去掉 `verbose` 打印、去掉官方文件里
与网络无关的 numpy/tqdm/matplotlib 导入。

三处官方「怪」但**必须原样保留**的语义（改了结构就与官方不对等）：

1. `MyMaxPool1dPadSame` 的 padding 用 `self.stride = 1` 算，但真正下采样的是
   `nn.MaxPool1d` 的默认 `stride = kernel_size`。所以它**不是**恒等池化，而是
   「核长 = block stride 的 SAME 长度池化，再按核长跳步」。
2. `BasicBlock` 的 downsample 分支里 identity 走上面这个池化；而主路径「长度减半」
   实际发生在同 block 的 `conv1(stride=2)` 上。两者输出长度恰好相等
   （官方 padding 公式在偶数长度上恰好补 1 点），shortcut 才没炸。
3. 通道扩展时 identity 用「转置 → F.pad 两侧补零 → 转置」，**不是** 1×1 conv。

out_channels 轨迹（base_filters=32, n_block=18, increasefilter_gap=4,
downsample_gap=2）：末块 = 512 = n_classes。1250 点进去，9 次 stride=2
下采样后时间维剩 3，`mean(-1)` 得 (B, 512)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """SAME padding 的 Conv1d：用显式 F.pad 而非 conv 的 padding 参数。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, groups=groups)

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        return self.conv(F.pad(x, (pad_left, p - pad_left), 'constant', 0))


class MyMaxPool1dPadSame(nn.Module):
    """SAME 的 MaxPool1d。注意 self.stride=1 只用于算 padding，池化本身用
    nn.MaxPool1d 的默认 stride=kernel_size（见模块 docstring 第 1 条）。"""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        return self.max_pool(F.pad(x, (pad_left, p - pad_left), 'constant', 0))


class BasicBlock(nn.Module):
    """ResNet basic block：bn-relu-dropout-conv ×2 + 恒等 shortcut。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups,
                 downsample, use_bn, use_do, is_first_block=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        # 官方语义：本块主路径的 stride 由 downsample 决定；不下采样时强制 1
        self.stride = stride if downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels, out_channels, kernel_size,
                                     stride=self.stride, groups=groups)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(out_channels, out_channels, kernel_size,
                                     stride=1, groups=groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x

        out = x
        if not self.is_first_block:          # 首块不吃 bn1/relu1/do1（官方做法）
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.max_pool(identity)
        if self.out_channels != self.in_channels:   # 通道零填充，非 1×1 conv
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            identity = F.pad(identity, (ch1, self.out_channels - self.in_channels - ch1),
                             'constant', 0)
            identity = identity.transpose(-1, -2)

        return out + identity


def _build_blocks(base_filters, kernel_size, stride, groups, n_block,
                  downsample_gap, increasefilter_gap, use_bn, use_do):
    """官方 block 堆叠规则：每 downsample_gap 块下采样一次，每 increasefilter_gap
    块通道翻倍。返回 (ModuleList, 末块 out_channels)。"""
    blocks = nn.ModuleList()
    out_channels = base_filters
    for i_block in range(n_block):
        is_first_block = (i_block == 0)
        downsample = (i_block % downsample_gap == 1)
        if is_first_block:
            in_channels = out_channels = base_filters
        else:
            in_channels = int(base_filters * 2 ** ((i_block - 1) // increasefilter_gap))
            out_channels = (in_channels * 2
                            if (i_block % increasefilter_gap == 0) else in_channels)
        blocks.append(BasicBlock(in_channels, out_channels, kernel_size, stride, groups,
                                 downsample, use_bn, use_do, is_first_block))
    return blocks, out_channels


class ResNet1D(nn.Module):
    """PaPaGei-P 的骨干（官方 `training_pospair.py` 用的类）。

    forward(x) → (out, out_emb)：
      out     = dense(out_emb)，(B, n_classes)=512，即官方 `outputs[0]`，对比损失
                与下游嵌入都用它（`use_projection=False`）；
      out_emb = 末层特征全局均值池化，也是 (B, 512)（末块通道数恰等于 n_classes）。
    另支持官方的 use_mt_regression 变体（多任务回归头），本库 PaPaGei-P 不用。
    """

    def __init__(self, in_channels=1, base_filters=32, kernel_size=3, stride=2,
                 groups=1, n_block=18, n_classes=512, downsample_gap=2,
                 increasefilter_gap=4, use_bn=True, use_do=True,
                 use_mt_regression=False, use_projection=False):
        super().__init__()
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_mt_regression = use_mt_regression
        self.use_projection = use_projection

        self.first_block_conv = MyConv1dPadSame(in_channels, base_filters,
                                                kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()

        self.basicblock_list, out_channels = _build_blocks(
            base_filters, kernel_size, stride, groups, n_block,
            downsample_gap, increasefilter_gap, use_bn, use_do)

        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)

        if use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, 128))
        if use_mt_regression:
            self.mt_regression = nn.Sequential(
                nn.Linear(n_classes, n_classes // 2), nn.BatchNorm1d(n_classes // 2),
                nn.Linear(n_classes // 2, n_classes // 4),
                nn.BatchNorm1d(n_classes // 4),
                nn.Linear(n_classes // 4, 1))

    def forward(self, x):
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        for block in self.basicblock_list:
            out = block(out)
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out_emb = out.mean(-1)

        out = self.projector(out_emb) if self.use_projection else self.dense(out_emb)
        if self.use_mt_regression:
            return out, self.mt_regression(out_emb), out_emb
        return out, out_emb


class ResNet1DMoE(nn.Module):
    """PaPaGei-S 的骨干（官方 `training_mt.py` 用的类），ResNet1D + 两个 MoE 回归头。

    forward(x) → (out_class, out_moe1, out_moe2, out)：
      out_class = dense(out_emb)，对比损失 + 下游嵌入都用它；
      out_moe1 / out_moe2 = 两个 3 专家 MoE 头，分别回归 IPA 与 skewness
                            （官方 S 的 L1 辅助项）；
      第 4 项是池化后的骨干特征（官方 return 的是它，不是 out_class）。

    注意官方 `resnet.py` 里 MoE 两个头结构不同：head1 无 dropout、head2 多一层
    Dropout(0.3)。照抄。
    """

    def __init__(self, in_channels=1, base_filters=32, kernel_size=3, stride=2,
                 groups=1, n_block=18, n_classes=512, n_experts=3,
                 downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True,
                 use_projection=False):
        super().__init__()
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_projection = use_projection
        self.n_experts = n_experts

        self.first_block_conv = MyConv1dPadSame(in_channels, base_filters,
                                                kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()

        self.basicblock_list, out_channels = _build_blocks(
            base_filters, kernel_size, stride, groups, n_block,
            downsample_gap, increasefilter_gap, use_bn, use_do)

        if use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, 128))
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)

        self.expert_layers_1 = nn.ModuleList([
            nn.Sequential(nn.Linear(out_channels, out_channels // 2), nn.ReLU(),
                          nn.Linear(out_channels // 2, 1))
            for _ in range(n_experts)])
        self.gating_network_1 = nn.Sequential(nn.Linear(out_channels, n_experts),
                                              nn.Softmax(dim=1))
        self.expert_layers_2 = nn.ModuleList([
            nn.Sequential(nn.Linear(out_channels, out_channels // 2), nn.ReLU(),
                          nn.Dropout(0.3), nn.Linear(out_channels // 2, 1))
            for _ in range(n_experts)])
        self.gating_network_2 = nn.Sequential(nn.Linear(out_channels, n_experts),
                                              nn.Softmax(dim=1))

    def forward(self, x):
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        for block in self.basicblock_list:
            out = block(out)
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out).mean(-1)          # (B, out_channels)

        out_class = self.projector(out) if self.use_projection else self.dense(out)

        e1 = torch.stack([e(out) for e in self.expert_layers_1], dim=1)
        out_moe1 = torch.sum(self.gating_network_1(out).unsqueeze(2) * e1, dim=1)
        e2 = torch.stack([e(out) for e in self.expert_layers_2], dim=1)
        out_moe2 = torch.sum(self.gating_network_2(out).unsqueeze(2) * e2, dim=1)

        return out_class, out_moe1, out_moe2, out
