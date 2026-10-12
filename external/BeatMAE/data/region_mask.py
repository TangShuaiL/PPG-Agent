"""区域掩码策略：关键点锚定 + 静态重要性加权，三种模式可切换。

设计定稿（2026-08-26，RegionMaskConfig.mode 三策略切换）：
- unified（默认）：段内按静态权重随机抽区域集合，所有有效拍遮同一批
  区域；贪婪最近选择防止大区域过冲
- weighted：按静态权重分配被遮面积——每个区域遮一整块连续面积，
  长度 = round(target_ratio × w_r × L)，块起始位置随机（跨拍共享）
- hybrid：region_share 的预算走统一区域掩码，其余预算在每拍上随机撒
  连续小 patch（长度 patch_len，逐拍独立，训练局部插值与抗噪鲁棒性）
- beat_drop（v2.1 消融）：拍级丢弃——按 target_ratio 随机抽可遮拍整拍
  丢弃（sample_mask 全 L + beat_mask=True），剩余拍完全可见；考察
  "跨拍插值/节律重建"与"拍内修补"两种监督粒度的下游差异

共同约束：
- 比例锚定"有效拍预算"：分母 = 可遮拍 × L（padding 排除）
- kp_valid=0 的拍无法定义区域，保持可见；padding 拍永不遮
- beat_mask：unified/weighted/hybrid 下恒 False（兼容接口保留）；
  beat_drop 下 True=该拍整拍丢弃（编码器走 mask_token 遮罩不变量路径）
- 种子体系：全部随机性来自 __call__ 传入的 generator；调用方按步构造
  （训练：seed×1000003+gstep）或固定（验证：777），同种子 ⇒ 掩码完全复现
"""
from dataclasses import dataclass

import torch

from models.common import N_REGIONS, region_slices

__all__ = ['RegionMaskConfig', 'RegionMasker', 'zero_masked_regions']

MODES = ('unified', 'weighted', 'hybrid', 'beat_drop', 'anchor')


@dataclass
class RegionMaskConfig:
    # 缺省 = 基座口径，与 config/base/*.yaml 的 mask.target_ratio 一致
    # （2026-09-11 由 0.40 对齐；0.40 是 v1 遗留值，已无任何配置在用）
    target_ratio: float = 0.65          # 目标被遮样本比例（有效拍口径）
    mode: str = 'unified'               # unified | weighted | hybrid | beat_drop | anchor
    region_share: float = 0.6           # hybrid：分配给统一区域掩码的预算比例
    patch_len: int = 8                  # hybrid：随机 patch 的连续长度
    # 区域静态权重（R1~R5）：R4 重搏切迹区血压最敏感给最高
    region_weights: tuple = (0.20, 0.15, 0.15, 0.35, 0.15)
    beat_len: int = 128


class RegionMasker:
    """三模式区域掩码器；全部随机性集中于传入的 generator，便于复现。

    输入（均为 batch 张量）：
        keypoints:     (B, K, 10) 拍内坐标（已换算到 L=beat_len 尺度）
        kp_valid:      (B, K) bool，True=9 关键点完整
        beat_pad_mask: (B, K) bool，True=padding 拍
        generator:     torch.Generator，本步掩码随机源
    输出 dict：
        region_mask: (B, K, R) bool，True=该区域被遮（weighted 模式下为
                     "区域含被遮样本"的提示；beat_drop 下恒 False）
        beat_mask:   (B, K) bool，True=整拍丢弃（仅 beat_drop / anchor 产生）
        sample_mask: (B, K, L) bool，True=该样本点被遮（损失与置零的权威口径）
        strategy:    list[str]，当前 mode 名
    """

    def __init__(self, cfg: RegionMaskConfig | None = None):
        self.cfg = cfg or RegionMaskConfig()
        assert self.cfg.mode in MODES, f'未知掩码模式: {self.cfg.mode}'
        if self.cfg.mode == 'hybrid':
            assert 0.0 < self.cfg.region_share <= 1.0, \
                'hybrid 的 region_share 应在 (0, 1]'
        w = torch.tensor(self.cfg.region_weights, dtype=torch.float)
        assert w.shape[0] == N_REGIONS and (w > 0).all()
        self._base_w = w / w.sum()

    # ------------------------------------------------------------ 入口
    @torch.no_grad()
    def __call__(self, keypoints: torch.Tensor, kp_valid: torch.Tensor,
                 beat_pad_mask: torch.Tensor | None = None,
                 generator: torch.Generator | None = None) -> dict:
        B, K, _ = keypoints.shape
        L = self.cfg.beat_len
        if beat_pad_mask is None:
            beat_pad_mask = keypoints.new_zeros(B, K, dtype=torch.bool)

        eligible = kp_valid & ~beat_pad_mask              # 可遮拍
        region_mask = keypoints.new_zeros(B, K, N_REGIONS, dtype=torch.bool)
        beat_mask = keypoints.new_zeros(B, K, dtype=torch.bool)
        sl = region_slices(keypoints, kp_valid)           # (B, K, R, 2)
        ln = (sl[..., 1] - sl[..., 0]).clamp(min=0).float()   # (B, K, R)

        if self.cfg.mode == 'beat_drop':
            sample_mask = keypoints.new_zeros(B, K, L, dtype=torch.bool)
            for b in range(B):
                el = eligible[b]
                if el.any():
                    self._drop_beats(b, el, beat_mask, sample_mask, generator)
        elif self.cfg.mode == 'anchor':
            # anchor（v2.5c 消融，移植自 xMAE mask_mode=anchor）：随机起点保留
            # 一段连续可见锚点，其余有效拍整拍遮。与 beat_drop 同走
            # mask_token + TimeEncoding + 侧信息的遮罩不变量路径。
            sample_mask = keypoints.new_zeros(B, K, L, dtype=torch.bool)
            for b in range(B):
                el = eligible[b]
                if el.any():
                    self._mask_anchor(b, el, beat_mask, sample_mask, generator)
        elif self.cfg.mode == 'weighted':
            sample_mask = keypoints.new_zeros(B, K, L, dtype=torch.bool)
            for b in range(B):
                el = eligible[b]
                if el.any():
                    self._mask_weighted(b, el, sl, ln, sample_mask, generator)
            # 区域提示：区域含被遮样本即标记（供编码器 region 提示注入）
            region_mask = self._region_hint_from_samples(sample_mask, sl, L)
        else:
            region_target = self.cfg.target_ratio
            if self.cfg.mode == 'hybrid':
                region_target *= self.cfg.region_share
            for b in range(B):
                el = eligible[b]
                if el.any():
                    self._select_unified(b, el, ln, region_mask,
                                         region_target, generator)
            sample_mask = self._region_to_samples(keypoints, kp_valid,
                                                  region_mask, sl)
            if self.cfg.mode == 'hybrid':
                for b in range(B):
                    el = eligible[b]
                    if el.any():
                        self._scatter_random(b, el, sample_mask, generator)

        sample_mask = sample_mask & ~beat_pad_mask.unsqueeze(-1)
        return {'beat_mask': beat_mask, 'region_mask': region_mask,
                'sample_mask': sample_mask,
                'strategy': [self.cfg.mode] * B}

    # ------------------------------------------------- 拍级丢弃（beat_drop）
    def _drop_beats(self, b, el, beat_mask, sample_mask, generator):
        """拍级丢弃：按 target_ratio 随机抽可遮拍整拍丢弃（v2.1 消融）。

        n_drop = round(target × K_eligible)，护栏保证每段至少留 1 拍可见
        （K_eligible == 1 时整段不丢）；被丢拍 sample_mask 全 L 置 True
        （重建损失权威口径，编码器输入由 zero_masked_regions 置零、
        token 走 mask_token + TimeEncoding + 侧信息）。region 提示不设：
        整拍丢弃无区域特异性，region_mask 恒 False。
        """
        el_idx = torch.nonzero(el).flatten()
        n_drop = int(round(self.cfg.target_ratio * len(el_idx)))
        n_drop = min(n_drop, len(el_idx) - 1)             # 至少留 1 拍可见
        if n_drop <= 0:
            return
        sel = el_idx[torch.randperm(len(el_idx),
                                    generator=generator)[:n_drop]]
        beat_mask[b, sel] = True
        sample_mask[b, sel, :] = True

    # ------------------------------------------------- 连续可见锚点（anchor）
    def _mask_anchor(self, b, el, beat_mask, sample_mask, generator):
        """anchor：随机起点保留一段连续可见拍，其余可遮拍整拍遮（v2.5c）。

        n_vis = round((1-target) × K_eligible)，护栏至少留 1 拍可见
        （K_eligible == 1 时整段不遮）；可见锚点取 el_idx 的连续切片
        （nonzero 升序 → 切片即时间连续）。被遮拍 sample_mask 全 L 置 True，
        region 提示恒 False（整拍遮无区域特异性）。
        """
        el_idx = torch.nonzero(el).flatten()          # 升序 → 时间连续
        n_vis = int(round((1.0 - self.cfg.target_ratio) * len(el_idx)))
        n_vis = max(1, min(n_vis, len(el_idx)))
        if n_vis >= len(el_idx):
            return                                     # 全可见，不遮
        start = int(torch.randint(0, len(el_idx) - n_vis + 1, (1,),
                                  generator=generator))
        keep = torch.ones(len(el_idx), dtype=torch.bool)
        keep[start:start + n_vis] = False
        sel = el_idx[keep]
        beat_mask[b, sel] = True
        sample_mask[b, sel, :] = True

    # ------------------------------------------------- 统一区域抽取
    def _select_unified(self, b, el, ln, region_mask, target, generator):
        """按权重逐次抽区域直至达到预算比例；贪婪最近选择防止过冲。

        抽中的区域在全部可遮拍上同遮（统一掩码，防邻拍抄答案）。
        """
        L = self.cfg.beat_len
        lens_r = ln[b][el].mean(dim=0)                    # (R,) 区域均长
        w = self._base_w.clone()
        w[lens_r <= 0] = 0.0
        total = max(int(el.sum()) * L, 1)
        masked = 0.0
        while masked / total < target and w.sum() > 0:
            r = int(torch.multinomial(w, 1, generator=generator))
            cand = masked + float(ln[b][el, r].sum())
            if masked > 0 and abs(cand / total - target) > \
                    abs(masked / total - target):
                w[r] = 0.0                                # 加上会离目标更远，跳过
                continue
            region_mask[b][el, r] = True                  # 全部可遮拍同遮
            masked = cand
            w[r] = 0.0

    # ------------------------------------------------- 权重面积分配（连续块）
    def _mask_weighted(self, b, el, sl, ln, sample_mask, generator):
        """每个区域按权重分配被遮面积：区域内遮一整块连续面积，长度=配额。

        配额算法：总预算 = target_ratio × L；短区域（含互相重叠的可用容量）
        先截断，剩余按容量二次分配；上限为区域并集覆盖。连续块的起始位置
        随机抽取（种子驱动、跨拍共享），优先选已遮样本少的窗口避免重叠。
        """
        L = self.cfg.beat_len
        el_idx = torch.nonzero(el).flatten()
        lens = ln[b][el].mean(dim=0)                      # (R,) 区域均长（浮点）
        budget = self.cfg.target_ratio * L
        w = self._base_w
        alloc = torch.minimum(w * budget, lens)
        resid = budget - float(alloc.sum())
        if resid > 0:                                     # 短区域截断剩余二次分配
            room = lens - alloc
            if float(room.sum()) > 0:
                alloc = alloc + room * (resid / float(room.sum()))
                alloc = torch.minimum(alloc, lens)
        st_all = sl[b][el_idx, :, 0]                      # (E, R)
        en_all = sl[b][el_idx, :, 1].clamp(max=L)         # (E, R)
        for r in range(N_REGIONS):
            k_r = min(int(round(float(alloc[r]))), int(lens[r]))
            if k_r <= 0:
                continue
            self._place_block_in_region(sample_mask[b], el_idx, st_all[:, r],
                                        en_all[:, r], int(lens[r]), k_r,
                                        generator)

    def _place_block_in_region(self, sm_b, el_idx, st, en, len_r, k, generator):
        """在区域内放一块长 k 的连续面积：起始位置随机，优先已遮少的窗口。"""
        L = self.cfg.beat_len
        k = min(k, len_r)
        n_cand = len_r - k + 1
        local = torch.arange(len_r, device=st.device)
        pos_all = (st.unsqueeze(1) + local.unsqueeze(0)).clamp(max=L - 1)
        valid = pos_all < en.unsqueeze(1)                 # (E, len_r)
        occ = (sm_b[el_idx].gather(1, pos_all) & valid)   # (E, len_r)
        win_occ = occ.unfold(1, k, 1).sum(dim=(0, 2))     # (n_cand,) 窗口占用数
        cand_order = torch.randperm(n_cand, generator=generator)
        s = int(cand_order[torch.argsort(win_occ[cand_order], stable=True)][0])
        offs = torch.arange(s, s + k, device=st.device)
        pos2 = st.unsqueeze(1) + offs.unsqueeze(0)        # (E, k)
        valid2 = pos2 < en.unsqueeze(1)
        pos2 = pos2.clamp(max=L - 1)
        rows = el_idx.unsqueeze(1).expand_as(pos2)[valid2]
        sm_b[rows, pos2[valid2]] = True

    # ------------------------------------------------- 全局随机 patch（hybrid）
    def _scatter_random(self, b, el, sample_mask, generator):
        """在可遮拍上随机撒连续小 patch（hybrid 的随机部分，逐拍独立）。

        每拍预算 = target×(1-region_share)×L 样本，拆为长度 patch_len 的
        连续块，起始位置逐拍随机。
        """
        L = self.cfg.beat_len
        beat_budget = int(round(L * self.cfg.target_ratio
                                * (1.0 - self.cfg.region_share)))
        p = min(self.cfg.patch_len, L)
        n_patches = max(int(round(beat_budget / p)), 0)
        if n_patches <= 0:
            return
        el_idx = torch.nonzero(el).flatten()
        starts = torch.randint(0, L - p + 1, (len(el_idx), n_patches),
                               generator=generator)       # (E, n_patches)
        offs = torch.arange(p, device=starts.device)
        pos = (starts.unsqueeze(-1) + offs).clamp(max=L - 1)  # (E, n, p)
        rows = el_idx[:, None, None].expand_as(pos)
        sample_mask[b][rows.reshape(-1), pos.reshape(-1)] = True

    # ------------------------------------------------------------ 工具
    def _region_to_samples(self, keypoints, kp_valid, region_mask, sl):
        """region_mask → 样本级掩码 (B, K, L)。"""
        B, K, _ = keypoints.shape
        L = self.cfg.beat_len
        pos = torch.arange(L, device=keypoints.device)
        sample_mask = keypoints.new_zeros(B, K, L, dtype=torch.bool)
        for r in range(N_REGIONS):
            sel = region_mask[..., r]                     # (B, K)
            st = sl[..., r, 0].clamp(0, L)                # (B, K)
            en = sl[..., r, 1].clamp(0, L)
            m = (pos[None, None, :] >= st.unsqueeze(-1)) & \
                (pos[None, None, :] < en.unsqueeze(-1))
            sample_mask |= m & sel.unsqueeze(-1)
        return sample_mask

    def _region_hint_from_samples(self, sample_mask, sl, L):
        """weighted 模式：区域内有样本被遮即标记区域提示。"""
        pos = torch.arange(L, device=sample_mask.device)
        region_mask = sample_mask.new_zeros(sample_mask.shape[0],
                                            sample_mask.shape[1], N_REGIONS)
        for r in range(N_REGIONS):
            st = sl[..., r, 0].clamp(0, L)
            en = sl[..., r, 1].clamp(0, L)
            inside = (pos[None, None, :] >= st.unsqueeze(-1)) & \
                     (pos[None, None, :] < en.unsqueeze(-1))
            region_mask[..., r] = (sample_mask & inside).any(dim=-1)
        return region_mask


def zero_masked_regions(beats: torch.Tensor,
                        sample_mask: torch.Tensor) -> torch.Tensor:
    """把被遮输入样本置 0（模型看不到被遮波形）。

    beats: (B, K, L)；sample_mask: (B, K, L) bool（权威掩码口径）。
    返回新张量，不修改输入。
    """
    return beats.masked_fill(sample_mask, 0.0)
