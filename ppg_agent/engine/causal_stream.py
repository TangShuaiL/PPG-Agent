"""causal_stream.py — Agent 3 主口径推理引擎（BeatMAE v2.5c × causal_attn 流式头）。

数据面（ppg_model 侧固化，本模块纯 numpy 只读消费，无 torch 依赖）：
    $PPG_DATA_ROOT/ppg_agent/embeddings/
    ├── causal_heads.npz        # serve/heads_causal.py 固化（90 头 + mu/sd）
    ├── manifest.json           # 参数结构 / K / fold 语义 / 官方指标 / 自检
    └── <encoder_version>/embeddings.npy + meta.parquet + build_info.json

查表口径（与 ppg_model 28/37 号评估协议对齐）：
    第 k 段预测 = MLP([e_k ; c_recent ; c_long])
    c_recent = 近 K=32 历史段加性注意力池化（头内 W/a）
    c_long   = 前缀均值 mean(e_{<k})
- 受试者段嵌入按 seg_idx 保序（因果铁律），逐段预测 → 最优单 seed（seed=4）
  头×折集成（见 INFER_SEEDS）
  → 受试者 median（sub 主口径，先预测后池化）。
- 选头：source=='anker' → 五折号 pf = fold//2，用 f{pf} 折头（训练时未见该折）；
  fold 缺失 → 5 折 ensemble（新用户策略）；其余三库 → full 单组头。
- 指标叙事：sub 口径官方 MAE 取自 manifest.official_metrics。

纪律：模型输出是群体级统计估计（置信度见 manifest 指标），不作诊断、
不作心血管风险结论；与 Agent 2 的纵向变化叙事严格分离。
"""
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_ROOT

EMB_ROOT = Path(DATA_ROOT) / "ppg_agent" / "embeddings"
ANKER_FOLDS = (0, 1, 2, 3, 4)
# 推理/归因使用的 seed 集合。manifest 官方口径为 5-seed mean (seeds[0..4])，
# 经与用户确认（2026-10）改为只用最优单一种子 seed=4；fold 维度不变。
# 如需回到 5-seed 集成，改回 INFER_SEEDS = (0, 1, 2, 3, 4) 即可。
INFER_SEEDS = (4,)

_LOCK = threading.Lock()
_ENGINE: "CausalStreamEngine | None" = None


def _erf(x: np.ndarray) -> np.ndarray:
    """精确 erf：优先 scipy，缺失退回 math.erf 逐元素（体积小，速度可接受）。"""
    try:
        from scipy.special import erf as _serf
        return _serf(x)
    except ImportError:
        import math
        return np.array([math.erf(v) for v in np.ravel(x)]).reshape(x.shape)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _gelu_grad(x):
    """``_gelu``（精确 erf 版）的导数：0.5(1+erf(x/√2)) + x·φ(x)，
    φ 为标准正态 pdf。供头侧解析反传到嵌入梯度（L2 归因）。"""
    x = np.asarray(x, np.float64)
    return (0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
            + x * np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))


def _softmax_last(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class CausalStreamEngine:
    """causal_attn 流式头查表引擎（只读、惰性加载、进程内单例）。

    与 ppg_model serve/heads_causal.py 共享 npz 契约：
        anker_params (5,3,5,P) / anker_mu,sd (5,3,5)
        full_params  (3,5,P)   / full_mu,sd  (3,5)
        targets (3,)；manifest.param_names / param_shapes / K / d_model
    """

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else EMB_ROOT
        self.npz_path = self.root / "causal_heads.npz"
        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"causal_heads.npz 不存在: {self.npz_path} "
                "(需从 ppg_model results/serve/ 同步并固化)")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        z = np.load(self.npz_path, allow_pickle=False)
        self.targets = [str(t) for t in z["targets"]]
        self._anker_params, self._anker_mu, self._anker_sd = \
            z["anker_params"], z["anker_mu"], z["anker_sd"]
        self._full_params, self._full_mu, self._full_sd = \
            z["full_params"], z["full_mu"], z["full_sd"]
        self.K = int(self.manifest["K"])
        self.d_model = int(self.manifest["d_model"])
        self.encoder_version = str(self.manifest["encoder_version"])
        self.data_version = str(self.manifest.get("data_version", "v2c"))
        self.infer_seeds = tuple(INFER_SEEDS)
        self._names = self.manifest["param_names"]
        self._shapes = [tuple(s) for s in self.manifest["param_shapes"]]
        self._weights: dict[tuple, dict] = {}
        # 嵌入产物（与 MlPredictEngine 同根同版本解析）
        self.emb_root = self.root
        self.version_dir = self._resolve_version(self.emb_root)
        self.meta: pd.DataFrame = pd.read_parquet(self.version_dir / "meta.parquet")
        self.emb: np.ndarray = np.load(self.version_dir / "embeddings.npy",
                                       mmap_mode="r")
        bv = self.manifest.get("encoder_version")
        if bv and str(bv) != self.version_dir.name:
            raise RuntimeError(
                f"头/嵌入版本不一致: heads={bv} vs embeddings={self.version_dir.name}")
        self._idx_cache: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def _resolve_version(root: Path) -> Path:
        """取唯一版本目录；多个时 PPG_ENCODER_VERSION 优先，
        否则对齐 manifest.encoder_version 声明的版本（旧版本目录保留作
        历史对照，不应阻塞主口径启动）。"""
        import os
        want = os.environ.get("PPG_ENCODER_VERSION")
        versions = sorted(p for p in root.iterdir() if p.is_dir())
        if want:
            d = root / want
            if not d.is_dir():
                raise FileNotFoundError(f"PPG_ENCODER_VERSION={want} 不存在于 {root}")
            return d
        if len(versions) == 1:
            return versions[0]
        mf = root / "manifest.json"
        if mf.exists():
            declared = str(json.loads(mf.read_text()).get("encoder_version", ""))
            d = root / declared
            if declared and d.is_dir():
                return d
        raise RuntimeError(f"嵌入版本不唯一 {[v.name for v in versions]}，"
                           "请设置 PPG_ENCODER_VERSION")

    def _subject_rows(self, source: str, subject_id: str) -> np.ndarray:
        key = (source, subject_id)
        if key not in self._idx_cache:
            m = self.meta
            idx = m.index[(m["source"] == source) & (m["subject_id"] == subject_id)]
            self._idx_cache[key] = idx.to_numpy()
        return self._idx_cache[key]

    # ────────────────────────── 头参数 ──────────────────────────
    def _head(self, domain: str, k: int, t: str, s: int) -> dict:
        """头参数视图 → {W_w, W_b, a, m1_w, m1_b, m2_w, m2_b, m3_w, m3_b}。"""
        key = (domain, k, t, s)
        if key not in self._weights:
            ti = self.targets.index(t)
            vec = (self._anker_params[k, ti, s] if domain == "anker"
                   else self._full_params[ti, s])
            sd = dict(zip(self._names, self._shapes))
            off, w = 0, {}
            for name, shape in zip(self._names, self._shapes):
                n = int(np.prod(shape)) if shape else 1
                w[name] = vec[off:off + n].reshape(shape)
                off += n
            self._weights[key] = {
                "W_w": w["W.weight"], "W_b": w["W.bias"], "a": w["a"],
                "m1_w": w["mlp.0.weight"], "m1_b": w["mlp.0.bias"],
                "m2_w": w["mlp.2.weight"], "m2_b": w["mlp.2.bias"],
                "m3_w": w["mlp.4.weight"], "m3_b": w["mlp.4.bias"],
            }
        return self._weights[key]

    def _mu_sd(self, domain: str, k: int, t: str, s: int) -> tuple[float, float]:
        ti = self.targets.index(t)
        if domain == "anker":
            return float(self._anker_mu[k, ti, s]), float(self._anker_sd[k, ti, s])
        return float(self._full_mu[ti, s]), float(self._full_sd[ti, s])

    # ────────────────────────── numpy 前向 ──────────────────────────
    def _forward_std(self, x: np.ndarray, h: dict,
                     return_alpha: bool = False):
        """x (N,d) 保序段嵌入 → 标准化 z 段预测 (N,)。历史窗因果右对齐。

        ``return_alpha=True`` 时额外返回近窗注意力 (N,K)（第 n 行 = 第 n 段对
        其前序 ≤K 段的池化权重），供归因用；不改变预测数值。"""
        N, d = x.shape
        K = self.K
        idx = np.arange(N)[:, None] - np.arange(K)[None, :] - 1   # (N,K) 依赖索引
        kmask = idx >= 0
        win = np.zeros((N, K, d), np.float32)
        win[kmask] = x[np.clip(idx, 0, None)[kmask]]
        s = np.tanh(win @ h["W_w"].T + h["W_b"]) @ h["a"]         # (N,K)
        s = np.where(kmask, s, -1e9)
        cnt = kmask.sum(1)
        alpha = np.where(cnt[:, None] > 0, _softmax_last(s), 0.0)
        c_recent = np.einsum("nk,nkd->nd", alpha, win)
        c_long = np.zeros_like(x)
        csum = np.cumsum(x, axis=0)
        if N > 1:
            c_long[1:] = csum[:-1] / np.arange(1, N)[:, None]
        hcat = np.concatenate([x, c_recent, c_long], axis=1)      # (N,3d)
        z = _gelu(hcat @ h["m1_w"].T + h["m1_b"])
        z = _gelu(z @ h["m2_w"].T + h["m2_b"])
        pred = (z @ h["m3_w"].T + h["m3_b"]).ravel()
        return (pred, alpha) if return_alpha else pred

    def _forward_std_last_row(self, x: np.ndarray, h: dict,
                              return_grad: bool = False):
        """仅算末行（当前段）的标准化预测，数值与 ``_forward_std(x, h)[-1]``
        等价（复用同一 cumsum / 右对齐窗口 / softmax 原语），但避免构造
        (N,K,d) 全量窗口张量——将单段预测从 O(N·K·d) 降到 O(K·d)（缓解
        长史受试者（2 万+段）的推理瓶颈）。仅用于只需当前段预测的场景。

        ``return_grad=True`` 时返回 ``(pred, alpha_last, dhcat)``：
        ``alpha_last`` (K,) 为末行近窗注意力权重，``dhcat`` (3d,) 为
        ∂(z 预测)/∂hcat 的三段拼接 [∂/∂e_k | ∂/∂c_recent | ∂/∂c_long]。
        前 d 维即 ∂pred/∂e_k（历史窗与 e_k 无关）；后 2d 维供历史段
        归因（∂pred/∂c_recent、∂pred/∂c_long）。解析反传 MLP，与预测
        共用同一中间量。"""
        N, d = x.shape
        K = self.K
        # c_long：与 _forward_std 同 cumsum 原语→csum[N-2]/(N-1)，逐位一致
        if N > 1:
            c_long = np.cumsum(x[:N - 1], axis=0)[-1] / (N - 1)
        else:
            c_long = np.zeros(d, np.float32)
        # 末行窗口：j 槽 = x[N-2-j]（idx>=0），右对齐同 _forward_std
        idx = (N - 1) - np.arange(K) - 1
        kmask = idx >= 0
        win = np.zeros((K, d), np.float32)
        win[kmask] = x[np.clip(idx, 0, None)[kmask]]
        s = np.tanh(win @ h["W_w"].T + h["W_b"]) @ h["a"]
        s = np.where(kmask, s, -1e9)
        alpha = _softmax_last(s) if kmask.any() else np.zeros(K, np.float32)
        c_recent = alpha @ win
        hcat = np.concatenate([x[N - 1], c_recent, c_long])[None]   # (1,3d)
        a1 = hcat @ h["m1_w"].T + h["m1_b"]
        z1 = _gelu(a1)
        a2 = z1 @ h["m2_w"].T + h["m2_b"]
        z2 = _gelu(a2)
        pred = float((z2 @ h["m3_w"].T + h["m3_b"]).ravel()[0])
        if not return_grad:
            return pred
        # 解析反传：∂pred_z/∂hcat (3d)——供当前段(e_k)与历史段(c_recent/c_long)归因
        dz2 = h["m3_w"]                                    # (1,H2)
        da2 = dz2 * _gelu_grad(a2)                          # (1,H2)
        dz1 = da2 @ h["m2_w"]                               # (1,H)
        da1 = dz1 * _gelu_grad(a1)                          # (1,H)
        dhcat = da1 @ h["m1_w"]                             # (1,3d)
        return (pred, np.asarray(alpha, np.float32),
                np.asarray(dhcat[0], np.float32))

    def current_embed_grad(self, x: np.ndarray, domain: str,
                           fold_ids: list[int], target: str) -> np.ndarray:
        """∂(真实值预测)/∂e_k → (d,) float32（对 fold×seed 平均，逐头乘自身 sd）。

        ``e_k = x[-1]`` 为当前段嵌入；历史窗 c_recent/c_long 仅由前序段
        决定（与 e_k 无关），故梯度只经 hcat 前 d 维回传，与预测共用同一
        中间量。mu 为常数偏置、不贡献梯度。供 L2 将头侧解析梯度
        作为 grad_outputs 喂 encoder autograd，得到 ∂pred/∂波形。"""
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        d = self.d_model
        acc = np.zeros(d, np.float64)
        cnt = 0
        for k in fold_ids:
            for s in self.infer_seeds:
                h = self._head(domain, k, target, s)
                _mu, sd = self._mu_sd(domain, k, target, s)
                _pred, _alpha, dhcat = self._forward_std_last_row(
                    x, h, return_grad=True)
                acc += sd * dhcat[:d]                      # 前 d 维 = ∂pred_z/∂e_k
                cnt += 1
        return (acc / max(cnt, 1)).astype(np.float32)

    def embed_grads_for_rows(self, x: np.ndarray, domain: str,
                             fold_ids: list[int], target: str,
                             rows: list[int]) -> dict[int, np.ndarray]:
        """当前段（末行）预测对若干历史段嵌入的梯度 ∂pred/∂e_r → {r: (d,) float32}。

        e_r（r<N-1 的历史段）影响当前预测有两条通道：
          c_recent（近窗注意力池化，权重 α_r；仅最近 K 段非零）+
          c_long（全前缀均值，每段 1/M）。
        近似：将注意力权重 α_r 视为常数（忽略 e_r 改变自身 score 的
        二阶项），故 ∂pred/∂e_r ≈ sd·(α_r·∂z/∂c_recent + (1/M)·∂z/∂c_long)，
        对 fold×seed 平均。窗外行（r<M-K）α_r=0→仅 c_long 项（精确）。
        x 须为完整序列（末行=当前段），保证 c_long 为真实前缀均值。"""
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        N, d = x.shape
        M = N - 1
        K = self.K
        rows = [int(r) for r in rows]
        acc = {r: np.zeros(d, np.float64) for r in rows}
        cnt = 0
        for k in fold_ids:
            for s in self.infer_seeds:
                h = self._head(domain, k, target, s)
                _mu, sd = self._mu_sd(domain, k, target, s)
                _pred, alpha, dhcat = self._forward_std_last_row(
                    x, h, return_grad=True)
                dc_rec = dhcat[d:2 * d]
                dc_long = dhcat[2 * d:3 * d]
                for r in rows:
                    if not (0 <= r <= M - 1):
                        continue
                    j = M - 1 - r                          # 窗口位（0=最近）
                    a = float(alpha[j]) if 0 <= j < K else 0.0
                    acc[r] += sd * (a * dc_rec + (1.0 / max(M, 1)) * dc_long)
                cnt += 1
        return {r: (acc[r] / max(cnt, 1)).astype(np.float32) for r in rows}

    # ────────────────────────── 查表口径 ──────────────────────────
    def _fold_ids(self, domain: str, fold) -> list[int]:
        if domain != "anker":
            return [0]
        try:
            f = int(fold)
        except (TypeError, ValueError):
            return list(ANKER_FOLDS)      # 新用户：无折号 → 5 折 ensemble
        pf = f // 2
        return [pf] if 0 <= pf <= 4 else list(ANKER_FOLDS)

    def predict_subject(self, source: str, subject_id: str) -> dict:
        """受试者级流式预测（主口径：seg_idx 保序 → causal_attn → median）。"""
        rows = self._subject_rows(source, subject_id)
        if len(rows) == 0:
            raise KeyError(f"受试者不在嵌入表: {source}/{subject_id} "
                           "(可能未过语料质量/拍数过滤)")
        m = self.meta.loc[rows]
        domain = "anker" if source == "anker" else "full"
        order = np.argsort(m["seg_idx"].to_numpy(), kind="stable")
        x = np.ascontiguousarray(np.asarray(self.emb[rows], dtype=np.float32)[order])
        fold = m["fold"].iloc[0] if "fold" in m and len(m) else None

        fold_ids = self._fold_ids(domain, fold)
        seg = {t: np.zeros(len(x)) for t in self.targets}
        cnt = 0
        for k in fold_ids:
            for s in self.infer_seeds:
                for t in self.targets:
                    h = self._head(domain, k, t, s)
                    mu, sd = self._mu_sd(domain, k, t, s)
                    seg[t] += self._forward_std(x, h) * sd + mu
                cnt += 1
        preds, mae = {}, {}
        off = self.manifest["official_metrics"][domain]
        for t in self.targets:
            seg_t = seg[t] / cnt
            preds[t] = float(np.median(seg_t))
            if domain == "anker" and fold_ids != list(ANKER_FOLDS):
                mae[t] = off[f"f{fold_ids[0]}"][t]["sub_mean"]
            else:
                mae[t] = float(np.mean(
                    [off[f"f{k}"][t]["sub_mean"] for k in ANKER_FOLDS])
                    if domain == "anker" else off[t]["sub_mean"])
        return {
            "predictions": {t: round(preds[t], 1) for t in self.targets},
            "pool": "subject_stream",
            "n_segments": int(len(x)),
            "segment_level": False,
            "encoder_version": self.encoder_version,
            "data_version": self.data_version,
            "engine": "causal_stream",
            "domain": domain,
            "fold_used": (None if fold_ids == list(ANKER_FOLDS) else fold_ids[0]),
            "n_heads": cnt,
            "confidence": {
                "kind": "population_test_mae_sub",
                **{f"{t}_mae": mae[t] for t in self.targets},
                "note": ("causal_attn 流式主口径（受试者 median）官方 sub 口径 "
                         "MAE；ankera 单折头=该折 held-out 五折口径，三库=full "
                         "单组头口径。非诊断结论，仅供横向参考。"),
            },
        }


    # ────────────────────────── 段级流式原语 ──────────────────────────
    def segment_embedding(self, source: str, subject_id: str,
                          seg_idx: int) -> np.ndarray:
        """单段嵌入查表 (d,)；段不在表时 KeyError。"""
        rows = self._subject_rows(source, subject_id)
        if len(rows) == 0:
            raise KeyError(f"受试者不在嵌入表: {source}/{subject_id} "
                           "(可能未过语料质量/拍数过滤)")
        m = self.meta.loc[rows]
        sel = m[m["seg_idx"] == int(seg_idx)]
        if sel.empty:
            raise KeyError(f"段不在嵌入表: {source}/{subject_id}/seg{seg_idx}")
        return np.asarray(self.emb[sel.index[0]], dtype=np.float32)

    def subject_history_embeddings(self, source: str, subject_id: str,
                                   seg_idx: int) -> np.ndarray:
        """前序段嵌入 (M,d)，seg_idx 保序（因果铁律）；无历史返回 (0,d)。"""
        rows = self._subject_rows(source, subject_id)
        if len(rows) == 0:
            return np.zeros((0, self.d_model), dtype=np.float32)
        m = self.meta.loc[rows]
        hist = m[m["seg_idx"] < int(seg_idx)].sort_values("seg_idx")
        if hist.empty:
            return np.zeros((0, self.d_model), dtype=np.float32)
        return np.asarray(self.emb[hist.index], dtype=np.float32)

    def subject_history(self, source: str, subject_id: str,
                        seg_idx: int) -> tuple[np.ndarray, list[int]]:
        """(前序段嵌入 (M,d), 对应 seg_idx 升序 list)；无历史 → ((0,d), [])。

        与 subject_history_embeddings 同因果口径，额外带回 seg_idx 供
        近窗注意力归因将权重映回具体历史段。"""
        rows = self._subject_rows(source, subject_id)
        if len(rows) == 0:
            return np.zeros((0, self.d_model), dtype=np.float32), []
        m = self.meta.loc[rows]
        hist = m[m["seg_idx"] < int(seg_idx)].sort_values("seg_idx")
        if hist.empty:
            return np.zeros((0, self.d_model), dtype=np.float32), []
        return (np.asarray(self.emb[hist.index], dtype=np.float32),
                [int(s) for s in hist["seg_idx"].tolist()])

    def collect_history_attention(self, x: np.ndarray, domain: str,
                                  fold_ids: list[int],
                                  hist_seg_idxs: list[int], *,
                                  top: int = 12) -> dict:
        """当前段（末行）对前序历史段的近窗注意力归因→ {target: [{seg_idx, weight}]}。

        x = [历史(按 seg_idx 保序, M 行); 当前段]，行 r∈[0,M-1] ↔ hist_seg_idxs[r]。
        末行窗口位 j 映到行 r = M-1-j（最近历史在 j=0），仅最前 min(K,M)
        段有非零权（c_recent 只看近 K 段）；权重对 fold×seed 平均，每个
        target 归一化至和≈过1（softmax 性质保留）。"""
        N = int(x.shape[0])
        M = N - 1
        if M < 1 or not hist_seg_idxs or len(hist_seg_idxs) != M:
            return {}
        K = self.K
        # 仅需当前段（末行）对前序的注意力：c_recent 只覆盖近 K 段，更早历史
        # 权重恒为 0。截断到末尾 K+1 行，数值与全量逐位一致，复杂度 O(N)→O(K)
        # （避免 mimic 等超长史受试者（2 万+段）在此处整段前向而卡顿）。
        if N > K + 1:
            x = x[N - (K + 1):]
            hist_seg_idxs = hist_seg_idxs[len(hist_seg_idxs) - K:]
            N = int(x.shape[0])
            M = N - 1
        out: dict = {}
        for t in self.targets:
            wsum = np.zeros(M, np.float64)
            n = 0
            for k in fold_ids:
                for s in self.infer_seeds:
                    h = self._head(domain, k, t, s)
                    _, alpha = self._forward_std(x, h, return_alpha=True)
                    a_last = alpha[N - 1]                          # (K,)
                    for j in range(min(K, M)):
                        wsum[M - 1 - j] += float(a_last[j])
                    n += 1
            wsum /= max(n, 1)
            pairs = [{"seg_idx": int(hist_seg_idxs[r]),
                      "weight": round(float(wsum[r]), 4)}
                     for r in range(M) if wsum[r] > 1e-4]
            pairs.sort(key=lambda d: -d["weight"])
            out[t] = pairs[:top]
        return out

    def predict_segment_from_sequence(self, x: np.ndarray, source: str,
                                      fold=None, *,
                                      engine_label: str = "causal_stream",
                                      hist_seg_idxs: list[int] | None = None) -> dict:
        """嵌入序列 (N,d)（按 seg_idx 保序，末行=当前段）→ 流式头当前段预测。

        实时链（live_stream）与查表降级（predict_segment_lookup）共用；
        因果约定与 28/37 号评估协议一致：末段只看自身之前的历史。
        置信度用段级口径 seg_mean（受试者 median 用 sub_mean，不适用）。"""
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError(f"嵌入序列为空或形状非法: {x.shape}")
        domain = "anker" if source == "anker" else "full"
        fold_ids = self._fold_ids(domain, fold)
        acc = {t: 0.0 for t in self.targets}
        cnt = 0
        for k in fold_ids:
            for s in self.infer_seeds:
                for t in self.targets:
                    h = self._head(domain, k, t, s)
                    mu, sd = self._mu_sd(domain, k, t, s)
                    acc[t] += float(self._forward_std_last_row(x, h)) * sd + mu
                cnt += 1
        off = self.manifest["official_metrics"][domain]
        mae = {}
        for t in self.targets:
            if domain == "anker" and fold_ids != list(ANKER_FOLDS):
                mae[t] = off[f"f{fold_ids[0]}"][t]["seg_mean"]
            elif domain == "anker":
                mae[t] = float(np.mean(
                    [off[f"f{k}"][t]["seg_mean"] for k in ANKER_FOLDS]))
            else:
                mae[t] = off[t]["seg_mean"]
        out = {
            "predictions": {t: round(acc[t] / cnt, 1) for t in self.targets},
            "pool": "segment_stream",
            "segment_level": True,
            "n_segments": int(x.shape[0]),
            "n_history": int(x.shape[0] - 1),
            "encoder_version": self.encoder_version,
            "data_version": self.data_version,
            "engine": engine_label,
            "domain": domain,
            "fold_used": (None if fold_ids == list(ANKER_FOLDS)
                          else fold_ids[0]),
            "n_heads": cnt,
            "confidence": {
                "kind": "population_test_mae_seg",
                **{f"{t}_mae": mae[t] for t in self.targets},
                "note": ("causal_attn 流式头段级口径官方 seg_mean MAE；"
                         "段级预测噪声大于受试者 median，非诊断结论，仅供横向参考。"),
            },
        }
        if hist_seg_idxs:
            att = self.collect_history_attention(x, domain, fold_ids,
                                                 hist_seg_idxs)
            if att:
                out["history_attention"] = att
                out["history_attention_note"] = (
                    "c_recent 近窗注意力权重：当前段本次估计参考了哪些前序"
                    "历史段；属群体模型"
                    "依赖度证据，非个体生理病因，不作诊断。")
        return out

    def predict_segment_lookup(self, source: str, subject_id: str,
                               seg_idx: int) -> dict:
        """查表降级：历史+当前段全部取自离线嵌入表，流式头出当前段预测。

        live_stream 不可用（无 torch / ckpt 缺失 / 波形读取失败）时的口径
        一致降级路径 —— 同头同嵌入，与实时链逐位等价（见
        scripts/validate_live_stream.py）。"""
        rows = self._subject_rows(source, subject_id)
        if len(rows) == 0:
            raise KeyError(f"受试者不在嵌入表: {source}/{subject_id} "
                           "(可能未过语料质量/拍数过滤)")
        m = self.meta.loc[rows]
        seg_idx = int(seg_idx)
        cur = m[m["seg_idx"] == seg_idx]
        if cur.empty:
            raise KeyError(f"段不在嵌入表: {source}/{subject_id}/seg{seg_idx}")
        fold = m["fold"].iloc[0] if "fold" in m and len(m) else None
        hist, hist_idx = self.subject_history(source, subject_id, seg_idx)
        x = np.concatenate([
            hist,
            np.asarray(self.emb[cur.index[0]], dtype=np.float32)[None],
        ], axis=0)
        out = self.predict_segment_from_sequence(
            x, source, fold=fold, engine_label="causal_stream_lookup",
            hist_seg_idxs=hist_idx)
        out["history_source"] = "table"
        return out


def get_causal_stream_engine() -> "CausalStreamEngine":
    """进程内单例（线程安全；产物缺失时抛 FileNotFoundError 由上层降级）。"""
    global _ENGINE
    if _ENGINE is None:
        with _LOCK:
            if _ENGINE is None:
                _ENGINE = CausalStreamEngine()
    return _ENGINE
