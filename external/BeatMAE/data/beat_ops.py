"""beat_ops：段 → 拍的提取、定长重采样、关键点坐标换算。

流程（v1_plan §5.1 方案 A）：
    段波形 (1250,) @125Hz → 逐段 z-score → 按 beat_start_sample 切拍
    → 拍长过滤 [min, max] → 线性插值定长 L=128
    → 关键点坐标等比换算到 L 尺度（无效保持 -1）

关键点列序：o,a,md,b,s,dn,ip,d,v,start_sample（与 keypoints.npz 一致）
"""
import numpy as np

FS = 125.0            # 语料统一采样率
SEG_LEN = 1250        # 段长（10s）
KP_COLS = 9           # 前 9 列为拍内关键点坐标
START_COL = 9         # 第 10 列为拍在段内的起始样本

__all__ = ['FS', 'SEG_LEN', 'KP_COLS', 'START_COL', 'zscore', 'extract_beats',
           'resample_beat', 'rescale_keypoints', 'beat_side_info']


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """逐段 z-score（std 退化时不缩放）。"""
    mu = x.mean()
    sd = x.std()
    if sd < eps:
        return x - mu
    return (x - mu) / sd


def extract_beats(starts: np.ndarray, min_len: int = 20,
                  max_len: int = 250) -> np.ndarray:
    """由每拍起始样本计算拍区间，返回通过长度过滤的拍索引。

    Args:
        starts: (K,) 每拍在段内的起始样本（升序）
    Returns:
        keep: 通过过滤的拍索引；同时可经 beat_bounds 取 [start, end)
    """
    K = starts.shape[0]
    if K == 0:
        return np.zeros(0, dtype=np.int64)
    ends = np.empty(K, dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = SEG_LEN                                    # 末拍到段尾
    lens = ends - starts
    ok = (lens >= min_len) & (lens <= max_len) & \
         (starts >= 0) & (ends <= SEG_LEN)
    return np.nonzero(ok)[0]


def beat_bounds(starts: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """保留拍的 (start, end) 区间 (K', 2)。"""
    K = starts.shape[0]
    ends = np.empty(K, dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = SEG_LEN
    return np.stack([starts[keep], ends[keep]], axis=1)


def resample_beat(beat: np.ndarray, L: int = 128) -> np.ndarray:
    """线性插值重采样到定长 L。"""
    n = beat.shape[0]
    if n == L:
        return beat.astype(np.float32, copy=False)
    xq = np.linspace(0.0, 1.0, L)
    xp = np.linspace(0.0, 1.0, n)
    return np.interp(xq, xp, beat).astype(np.float32)


def rescale_keypoints(kp: np.ndarray, lens: np.ndarray, L: int = 128,
                      kp_valid: np.ndarray | None = None) -> np.ndarray:
    """拍内关键点坐标等比换算：原拍长 → 定长 L。

    Args:
        kp: (K, 9) 原坐标（无效=-1）
        lens: (K,) 原拍长（样本）
    Returns:
        (K, 9) long，无效或原坐标 <0 保持 -1，其余截断到 [0, L-1]
    """
    out = np.full(kp.shape, -1, dtype=np.int64)
    valid = kp >= 0
    if kp_valid is not None:
        valid &= kp_valid[:, None]
    if valid.any():
        scale = L / np.maximum(lens, 1).astype(np.float64)
        scaled = np.round(kp.astype(np.float64) * scale[:, None])
        scaled = np.clip(scaled, 0, L - 1).astype(np.int64)
        out[valid] = scaled[valid]
    return out


def beat_side_info(starts_kept: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """侧信息 (K, 2)：[beat_len_ms, hr_bpm]。

    hr 用"本拍到下一拍的间期"估计；末拍沿用前一间期；单拍置 0。
    """
    K = starts_kept.shape[0]
    side = np.zeros((K, 2), dtype=np.float32)
    side[:, 0] = lens / FS * 1000.0
    if K >= 2:
        rr = np.diff(starts_kept) / FS                    # 秒
        rr = np.clip(rr, 0.3, 3.0)                        # 20~200bpm 护栏
        hr = 60.0 / rr
        side[:-1, 1] = hr
        side[-1, 1] = hr[-1]
    return side
