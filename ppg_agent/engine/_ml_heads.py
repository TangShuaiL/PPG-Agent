"""_ml_heads.py — BeatMAE 岭回归三头（SBP/DBP/Age）的轻量加载器。

与 ppg_model/serve/heads.py 的固化格式一一对应（linear_heads_subject.npz）；
此处只做推理，固化/训练协议见 ppg_model 仓库。纯 numpy，无 torch 依赖。
"""
from pathlib import Path

import numpy as np


class LinearHeads:
    """冻结三头：嵌入 → SBP/DBP/Age 预测（内部自动标准化）。"""

    def __init__(self, npz_path: str | Path):
        z = np.load(npz_path, allow_pickle=True)
        self.targets = [str(t) for t in z["targets"]]
        self.mu = z["mu"]
        self.sd = z["sd"]
        self._w = {t: z[f"{t}_w"] for t in self.targets}
        self._b = {t: float(z[f"{t}_b"]) for t in self.targets}
        self.test_folds = [int(f) for f in z["test_folds"]]
        self.n_subjects = int(z["n_subjects"])
        self.metrics = {
            t: {"MAE": float(z[f"{t}_MAE"]), "Pearson_r": float(z[f"{t}_r"])}
            for t in self.targets if f"{t}_MAE" in z
        }
        self.encoder_version = (str(z["encoder_version"])
                                if "encoder_version" in z else None)
        self.npz_path = str(npz_path)

    def predict(self, emb: np.ndarray) -> dict[str, float]:
        """(d,) 或 (1,d) 嵌入 → {target: 预测值}。"""
        x = np.asarray(emb, np.float64).reshape(1, -1)
        xz = (x - self.mu) / self.sd
        return {t: float(xz @ self._w[t] + self._b[t]) for t in self.targets}
