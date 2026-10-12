"""papagei：PaPaGei P/S 复现（对比模型）。

与另三个族（xMAE / HiMAE / MMR）的区别在**方法轴线**上：那三族是掩码重建，
PaPaGei 是**形态学伪标签引导的对比学习**——正样本由 sVRI 分箱定义，不做任何掩码。
主表里要给这一条留位置。
"""
from .augment import GaussianNoise, RandomApply, TSRandomCropPad, build_transform
from .labels import (SVRI_RANGE, assign_svri_bins, svri_bin_edges)
from .losses import ntxent_loss
from .model import ARCH, PapageiP, PapageiS, build_model, name_from_sd
from .resnet1d import ResNet1D, ResNet1DMoE

__all__ = ['ResNet1D', 'ResNet1DMoE', 'PapageiP', 'PapageiS', 'ARCH',
           'build_model', 'name_from_sd',
           'GaussianNoise', 'TSRandomCropPad', 'RandomApply', 'build_transform',
           'ntxent_loss', 'SVRI_RANGE', 'svri_bin_edges',
           'assign_svri_bins']
