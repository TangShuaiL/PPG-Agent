"""共享组件：五功能区域定义与切片。

五功能区域（v1_plan §4.1），基于 9 关键点的样本坐标（拍内相对，125Hz）。
关键点列序：o,a,md,b,s,dn,ip,d,v,start_sample（keypoints.npz）
"""
import torch

KP_O, KP_A, KP_MD, KP_B, KP_S, KP_DN, KP_IP, KP_D, KP_V = range(9)

# 区域边界：(起点关键点, 终点关键点)；R2 为 s 邻域单独处理
REGION_BOUNDS = [
    (KP_O, KP_S),    # R1 收缩上升支
    (KP_S, KP_S),    # R2 收缩峰区（s 邻域，特殊处理）
    (KP_S, KP_DN),   # R3 舒张早期下降
    (KP_DN, KP_IP),  # R4 重搏切迹区（dn~ip）
    (KP_D, KP_V),    # R5 舒张末期
]
N_REGIONS = 5
R2_HALF_WIDTH_MS = 30  # R2 邻域半宽（毫秒）


def region_slices(keypoints: torch.Tensor, kp_valid: torch.Tensor,
                  fs: float = 125.0) -> torch.Tensor:
    """由关键点生成每拍 5 区域的样本切片 [start, end)。

    Args:
        keypoints: (B, K, 10) int，前 9 列为关键点样本坐标，缺失=-1
        kp_valid:  (B, K) bool，True=该拍 9 关键点完整
    Returns:
        slices: (B, K, N_REGIONS, 2) long，[start, end)；
                无效拍或退化区域 start>=end（调用方需跳过）
    """
    B, K, _ = keypoints.shape
    half = int(R2_HALF_WIDTH_MS / 1000.0 * fs)
    slices = keypoints.new_zeros(B, K, N_REGIONS, 2)
    for r, (a, b) in enumerate(REGION_BOUNDS):
        st = keypoints[..., a]
        en = keypoints[..., b]
        if r == 1:  # R2: s 邻域
            st = keypoints[..., KP_S] - half
            en = keypoints[..., KP_S] + half
        slices[..., r, 0] = st
        slices[..., r, 1] = en
    # 无效拍全部置退化
    slices = torch.where(kp_valid[..., None, None], slices,
                         torch.zeros_like(slices))
    slices = slices.clamp(min=0)
    return slices
