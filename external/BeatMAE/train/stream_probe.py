#!/usr/bin/env python3
"""train/stream_probe.py — 段级流式探针协议库（28 号 / 37 号 / tools 共用一份实现）。

2026-09-10 收编：这套协议原先只住在 scripts/probe_full_stream.py 里，37 号与
tools/ 三个脚本用 `importlib.import_module('28_causal_stream_probe')` 借道复用，
并靠 `_m28.META = ...` 直接改模块级全局来换数据源——隐式且有顺序依赖。现按
「协议只写一遍」收到本模块：28/37 退化为薄入口（差别只剩「试哪些折」），tools
直接 import，不再借道 scripts/。改这里 = 改所有消费者，这是有意的。

协议（各函数内详述）：
  头输入 = [e_k ; c_recent ; c_long]；c_recent = 近 K=32 段加性注意力池化
  （无 gap/无门控/单头），c_long = 前缀均值（无参、因果）。
  变体 w1_ridge（段级闭式 ridge，W=1 重演）/ seg_mlp（无历史 MLP 对照）/
  causal_attn（主形态）为主表三形态；causal_shuf / causal_k8 为消融哨兵。
  训练域 mimic+vital+aurora（iwshi_bp 纯测试不训）；受试者级隔离；raw 段级标签
  （NaN 段只做历史不进损失）；梯度变体 5 种子、闭式单跑；早停 patience=10
  on val seg-MAE。
  seg_r = 逐 seed 算 Pearson r 再取平均（跨 seed 池化会得到不同的数，勿混）——
  也是 _tables/final_*_all_models.csv 的 r 列来源。
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

# 探针头结构集中在 models/（2026-09-06 修复：seg_mlp 变体此前误用 CausalAttn，
# SegMLP 无历史对照从未生效）
from models.probe_heads import CausalAttn, CausalDropLong, CausalDropRecent, SegMLP
from data.windows import recent_history   # 近 kk 段因果历史取窗已集中 data/
from data.paths import STREAM_CACHE

K = 32                                   # 近期历史窗长（主形态）
LR, WD = 3e-3, 1e-4
SEEDS = (0, 1, 2, 3, 4)
TARGETS = ('sbp', 'dbp', 'age')          # 三目标规范序，**也是 y 的列序契约**：
#   写侧 scripts/materialize_stream.py 从这里 import（不再各写一份），并把快照存进
#   stream_meta.npz 的 targets 键；本模块 load_all 读时校验。偏离即静默错位，见 load_all。
VARIANTS = ('w1_ridge', 'seg_mlp', 'causal_attn')   # 主表三变体（缺省全跑）
ABLATIONS = ('causal_shuf', 'causal_k8',
             'causal_droplong', 'causal_droprecent')  # 消融哨兵：需显式点名，不进主表缺省
ALL_VARIANTS = VARIANTS + ABLATIONS                 # 合法变体全集（CLI 校验 / tools 枚举共用）
TEST_FOLDS = {1, 2}
DEV = 'cuda'
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
BUCKETS = [(0, 1, 'k=1'), (1, 4, 'k2-4'), (4, 16, 'k5-16'),
           (16, 64, 'k17-64'), (64, 10**9, 'k65+')]   # 冷启动曲线分桶（段内位置 k）


@dataclass
class StreamData:
    """一次物化的全部读侧数据（load_all 返回）。"""
    y: np.ndarray          # (n_seg, 3) raw 段级标签，NaN = 无标签段（只进历史）
    fold: np.ndarray       # (n_seg,) 段所属受试者的折号
    bnd: np.ndarray        # (n_sub+1,) 每人段区间：人 p 的段 = [bnd[p], bnd[p+1])
    emb: np.ndarray        # (n_seg, 256) float32 嵌入（CPU 原样）
    eg: torch.Tensor       # 同上，GPU 常驻（全库折 ~4.7GB）
    sub_names: np.ndarray  # 段级受试者名（预测明细落盘用）
    seg_idx: np.ndarray    # 段级段序号
    src: np.ndarray        # 段级来源库（bytes，落盘时 decode）


def resolve_sdir(exp, legacy_fallback=True):
    """实验名 → 物化目录（29 号 `--exp` 产出 STREAM_CACHE/<exp>/）。

    legacy_fallback: 旧根目录布局（直接把物化摊在 STREAM_CACHE/ 下）兼容开关，
    28 号与 tools 需要；37 号只认 per-exp 目录。
    """
    d = STREAM_CACHE / exp
    if (d / 'stream_meta.npz').exists():
        return d
    if legacy_fallback:
        return Path(STREAM_CACHE)
    raise FileNotFoundError(f'物化不存在: {d}/stream_meta.npz（先跑 materialize_stream.py）')


def load_all(sdir, dev=DEV):
    """吃 29 号物化文件：嵌入已按 (subject, seg_idx) 保序排列 + cap 采样。
    人 p 的段就是 emb[bnd[p]:bnd[p+1]]，零索引换算。
    旧版把 sub_names/seg_idx/src 挂在模块级、靠改全局换数据源，已收编为显式返回。
    y 是裸 (N,3) 数组、不带列名，下游按 TARGETS 用 y[:, ti] 取——故先验列序：
    物化文件带 targets 快照时须与本模块 TARGETS 一致，否则 y 会静默错位（不报错、只是数字全错）。
    2026-09-11 之前物化无此键，那些文件的列序由同一份 TARGETS 写出，故缺失时回落放行。"""
    t0 = time.time()
    m = np.load(Path(sdir) / 'stream_meta.npz', allow_pickle=True)
    emb = np.load(Path(sdir) / 'stream_emb.npy')
    if 'targets' in m:
        got = tuple(str(t) for t in m['targets'])
        if got != TARGETS:
            raise ValueError(
                f'y 列序不符：物化 {got} vs 协议 {TARGETS}（{sdir}）——'
                f'该物化不可复用，重跑 scripts/materialize_stream.py')
    y_f, fold_f, bnd = m['y'], m['fold'].astype(int), m['bnd']
    n_sub = len(bnd) - 1
    print(f'数据: {len(y_f)} 段 / {n_sub} 人 | 物化加载 {time.time()-t0:.0f}s')
    return StreamData(y=y_f, fold=fold_f, bnd=bnd, emb=emb,
                      eg=torch.as_tensor(np.ascontiguousarray(emb), device=dev),
                      sub_names=m['sub_names'], seg_idx=m['seg_idx'], src=m['src'])


# ── 两种折划分（28 与 37 的唯一实质差别，故并排放便于对照）────────────────

def split_full_corpus(fold_f, bnd, test_folds=TEST_FOLDS, va_frac=0.15, seed=1234):
    """全库折（28 号主表①口径）：受试者折 ∈ test_folds 的全部为测试折；
    其余人 rng(seed) 置换后前 va_frac 做早停验证、末段做训练。
    返回 (tr_p, va_p, te_p)。"""
    n = len(bnd) - 1
    person_fold = fold_f[bnd[:-1]]               # 受试者折 = 首段折
    perm = np.random.default_rng(seed).permutation(n)
    te_p = perm[np.isin(person_fold[perm], list(test_folds))]
    rest = perm[~np.isin(person_fold[perm], list(test_folds))]
    nva = int(len(rest) * va_frac)
    return rest[nva:], rest[:nva], te_p


def split_five_fold(fold_f, bnd, k, va_frac=0.15, seed=1234):
    """五折之一（37 号主表②口径）：受试者折 fold//2 == k 的为测试折，其余人
    rng(seed+k) 置换后切 va_frac 验证（至少 2 人）。返回 (tr_p, va_p, te_p)。

    fold//2 而非 fold：iwshi_bp 物化的 fold 为偶数 0/2/4/6/8，除 2 得五折号 0..4。
    """
    pf = fold_f[bnd[:-1]] // 2
    te_p = np.flatnonzero(pf == k)
    rest = np.random.default_rng(seed + k).permutation(np.flatnonzero(pf != k))
    nva = max(2, int(len(rest) * va_frac))
    return rest[nva:], rest[:nva], te_p


# ── 批量组装 / 训练 / 前向 ──────────────────────────────────────────────

def build_batch(Eg, y_f, bnd, pids, ti, kk=K, shuf=False):
    """收集一批人的全部有效标签位置 → (ek, win, kmask, clong, y, wsub)
    kk: 近期历史窗长；shuf=True 时人内段序随机置换（反时间对照：历史≠真实过去，
        (段,标签)对仍保持对齐，故只摧毁时间结构，不引入标签错配）"""
    eks, wins, kmasks, cls, ys, wsubs = [], [], [], [], [], []
    for p in pids:
        b0, b1 = bnd[p], bnd[p + 1]
        m = b1 - b0
        E = Eg[b0:b1]                                # (m,256) 连续块直取
        yt = torch.as_tensor(y_f[b0:b1, ti], device=DEV)
        if shuf:
            perm = torch.randperm(m, device=DEV)
            E, yt = E[perm], yt[perm]                # 时间结构摧毁，(段,标签)对保持
        clong = (E.cumsum(0) - E) / torch.clamp(torch.arange(1, m + 1, device=DEV) - 1, min=1)[:, None]
        valid = ~torch.isnan(yt)
        vpos = valid.nonzero().squeeze(1)
        if len(vpos) == 0:
            continue
        # 近 kk 段因果历史窗（padded unfold，协议见 data/windows.py）
        winsAll, kmaskAll = recent_history(E, kk)   # (m, kk, d) / (m, kk)
        eks.append(E[vpos]); wins.append(winsAll[vpos]); kmasks.append(kmaskAll[vpos])
        cls.append(clong[vpos]); ys.append(yt[vpos])
        wsubs.append(torch.full((len(vpos),), p, dtype=torch.long, device=DEV))
    if not eks:
        return None
    return (torch.cat(eks), torch.cat(wins), torch.cat(kmasks), torch.cat(cls),
            torch.cat(ys), torch.cat(wsubs))


def train_variant(model, Eg, y_f, bnd, tr_p, va_p, ti, seed, kk=K, shuf=False):
    """训练一个梯度变体：AdamW + Huber(delta=1)、批按 16384 段装人、早停 patience=10
    on val seg-MAE（raw 尺度）。标签标准化 mu/sd 只由训练人算。返回 best val MAE。"""
    torch.manual_seed(seed); np.random.seed(seed)
    # 训练折标签标准化（用训练人）
    ytr = np.concatenate([y_f[bnd[p]:bnd[p+1], ti] for p in tr_p])
    mu, sd = np.nanmean(ytr), np.nanstd(ytr) + 1e-8
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    hub = nn.HuberLoss(delta=1.0)
    best, bad, best_state = 1e9, 0, None
    rng = np.random.default_rng(seed)
    ep_t0 = time.time()
    for ep in range(100):
        model.train()
        order = rng.permutation(len(tr_p))
        i, nb = 0, 0
        while i < len(order):
            batch_p, npos = [], 0
            while i < len(order) and npos < 16384:
                batch_p.append(order[i]); npos += bnd[order[i]+1] - bnd[order[i]]; i += 1
            bt = build_batch(Eg, y_f, bnd, batch_p, ti, kk=kk, shuf=shuf)
            if bt is None: continue
            ek, win, kmask, clong, y, _ = bt
            pred = model(ek, win, kmask, clong)
            loss = hub(pred, (y - mu) / sd)
            opt.zero_grad(); loss.backward(); opt.step(); nb += 1
        # val seg MAE（raw 尺度）
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for j in range(0, len(va_p), 32):
                bt = build_batch(Eg, y_f, bnd, va_p[j:j+32], ti, kk=kk, shuf=shuf)
                if bt is None: continue
                ek, win, kmask, clong, y, _ = bt
                preds.append(model(ek, win, kmask, clong).cpu().numpy() * sd + mu)
                gts.append(y.cpu().numpy())
        mae = np.abs(np.concatenate(preds) - np.concatenate(gts)).mean()
        print(f'[ti={ti} seed={seed}] ep{ep} val {mae:.3f} bad {bad} {time.time()-ep_t0:.1f}s', flush=True)
        if mae < best - 1e-4:
            best, bad = mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10: break
    if best_state: model.load_state_dict(best_state)
    return best


def predict(model, Eg, y_f, bnd, pids, ti, mu, sd, kk=K, shuf=False):
    """已训模型 → (pr, gt, ws, wp)：预测/真值/受试者段号/段内位置（raw 尺度还原）。"""
    model.eval(); preds, gts, wsub, wpos = [], [], [], []
    with torch.no_grad():
        for j in range(0, len(pids), 32):
            bt = build_batch(Eg, y_f, bnd, pids[j:j+32], ti, kk=kk, shuf=shuf)
            if bt is None: continue
            ek, win, kmask, clong, y, ws = bt
            preds.append(model(ek, win, kmask, clong).cpu().numpy() * sd + mu)
            gts.append(y.cpu().numpy()); wsub.append(ws.cpu().numpy())
            for p in pids[j:j+32]:
                b0, b1 = bnd[p], bnd[p+1]
                m = b1 - b0
                v = ~np.isnan(y_f[b0:b1, ti])
                wpos.append(np.arange(m)[v])
    return np.concatenate(preds), np.concatenate(gts), np.concatenate(wsub), np.concatenate(wpos)


# ── 头权重读写（ridge / NN 同一命名与落盘口径）──────────────────────────

def variant_hparams(var):
    """变体 → (kk, shuf)：causal_k8 改窗长、causal_shuf 置换人内段序，其余取主形态。"""
    return (8 if var == 'causal_k8' else K), (var == 'causal_shuf')


def make_model(var, d):
    """变体 + 嵌入维 → 新模型：seg_mlp 为无历史对照，其余 NN 变体用 CausalAttn 主形态。

    未知变体直接报错。此前是静默落到 CausalAttn，于是 `--help` 这种被误当成
    位置参数的旗标会训出一整套垃圾头（2026-09-10 在 iwshi_bp/stream5f 发现）。
    d 必须由调用方给：训练侧 = `Eg.shape[1]`，评估侧 = `head_dim(var, sd)`。
    头里不再有 d=256 这条暗路（见 head_dim 的说明）。
    """
    if var not in ALL_VARIANTS:
        raise ValueError(f'未知变体 {var!r}；合法值: {" / ".join(ALL_VARIANTS)}')
    if var == 'seg_mlp':
        return SegMLP(d=d)
    if var == 'causal_droplong':
        return CausalDropLong(d=d)
    if var == 'causal_droprecent':
        return CausalDropRecent(d=d)
    return CausalAttn(d=d)


def head_dim(var, sd):
    """已存头权重 → 头部输入维（= 该族嵌入维），从权重反推，不写字面量。

    嵌入维随族而变：beatmae/himae/mmr/xmae 是 256，PaPaGei 是 **512**
    （官方 n_classes=512，下游取 dense(out_emb)，见 baselines/papagei/）。凡是
    写死 256 的地方，PaPaGei 一律崩在
    `mat1 and mat2 shapes cannot be multiplied (…x512 and 256x128)`
    （2026-09-11 冒烟实测：#45 就是撞上这个）。训练侧从数据取维、评估侧从权重
    取维，两处都不靠默认值。
    """
    if var == 'seg_mlp':
        return int(sd['mlp.0.weight'].shape[1])
    if var in ('causal_droplong', 'causal_droprecent'):
        # 砍掉一条参照后 MLP 输入是 2d（[e_k; c_recent] 或 [e_k; c_long]），
        # 且 causal_droprecent **没有 W**，落到下面的 W.weight 分支会 KeyError
        return int(sd['mlp.0.weight'].shape[1]) // 2
    return int(sd['W.weight'].shape[1])


def head_name(t, var, seed, fold=None):
    """头文件名：28 号 <t>_<var>_s<seed>.pt；37 号带折号 f<k>_<t>_<var>_s<seed>.pt。"""
    return f'{f"f{fold}_" if fold is not None else ""}{t}_{var}_s{seed}.pt'


def save_variant_head(path, var, model, mu, sd, t, seed, kk, shuf, fold=None):
    d = {'kind': var, 'state': {k: v.cpu() for k, v in model.state_dict().items()},
         'mu': mu, 'sd': sd, 'kk': kk, 'shuf': shuf, 'target': t, 'seed': seed}
    if fold is not None:
        d['fold'] = fold
    torch.save(d, path)


def save_ridge_head(path, w, alpha, mu, sd, t, fold=None):
    d = {'kind': 'ridge', 'w': w, 'alpha': alpha, 'mu': mu, 'sd': sd,
         'target': t, 'seed': 0}
    if fold is not None:
        d['fold'] = fold
    torch.save(d, path)


def fit_ridge(emb, y_f, bnd, tr_p, va_p, ti, alphas=ALPHAS, stride=1):
    """段级闭式 ridge：α 在验证折上选优。stride>1 时每 stride 人取一人控内存
    （全库折训练人数万级段，28 号用 5 抽样）。返回 (w, alpha)。"""
    Xtr = np.concatenate([emb[bnd[p]:bnd[p+1]] for p in tr_p[::stride]])
    ytr = np.concatenate([y_f[bnd[p]:bnd[p+1], ti] for p in tr_p[::stride]])
    ok = ~np.isnan(ytr); Xtr, ytr = Xtr[ok], ytr[ok]
    Xva = np.concatenate([emb[bnd[p]:bnd[p+1]] for p in va_p])
    yva = np.concatenate([y_f[bnd[p]:bnd[p+1], ti] for p in va_p])
    okv = ~np.isnan(yva); Xva, yva = Xva[okv], yva[okv]
    best = (1e9, None, None)
    for al in alphas:
        A = Xtr.T @ Xtr + al * np.eye(Xtr.shape[1], dtype=np.float32)
        w = np.linalg.solve(A, Xtr.T @ ytr)
        mae = np.abs(Xva @ w - yva).mean()
        if mae < best[0]:
            best = (mae, w, al)
    return best[1], best[2]


# ── 评估（ridge 与 NN 同一出口，口径不再靠两处人眼比）────────────────────

def head_predict(ck, var, data, pids, ti, dev=DEV):
    """已存头 → 测试段预测。返回 (pr, gt, ws, wp)，ridge 与 NN 同口径：
    人按 pids 顺序、人内按段序升序拼接；gpos = bnd[ws] + wp 即全局段序号。"""
    if var == 'w1_ridge':
        w, emb, y_f, bnd = ck['w'], data.emb, data.y, data.bnd
        prs, gts, subs, poss = [], [], [], []
        for p in pids:
            b0, b1 = bnd[p], bnd[p + 1]
            yp = y_f[b0:b1, ti]; v = ~np.isnan(yp)
            if not v.any():
                continue
            prs.append(emb[b0:b1][v] @ w); gts.append(yp[v])
            subs.append(np.full(int(v.sum()), p)); poss.append(np.arange(b1 - b0)[v])
        return (np.concatenate(prs), np.concatenate(gts),
                np.concatenate(subs), np.concatenate(poss))
    model = make_model(var, head_dim(var, ck['state'])).to(dev)
    model.load_state_dict(ck['state'])
    return predict(model, data.eg, data.y, data.bnd, pids, ti, ck['mu'], ck['sd'],
                   kk=ck.get('kk', K), shuf=ck.get('shuf', False))


def seg_mae_r(pr, gt):
    """段级 (MAE, r)。r = 全体测试段一次 Pearson；段数 ≤2 时无意义记 NaN。"""
    return (float(np.abs(pr - gt).mean()),
            float(np.corrcoef(gt, pr)[0, 1]) if len(gt) > 2 else float('nan'))


def subject_median_mae(pr, ws, y_f, bnd, ti):
    """受试者级 MAE（部署口径）：人内预测取 median 与真值 median 比，再跨人平均。
    与 results.json 的 sub 字段、_tables 宽表同规则。"""
    aerr = []
    for p in np.unique(ws):
        m = ws == p
        gtp = y_f[bnd[p]:bnd[p+1], ti]
        aerr.append(abs(np.median(pr[m]) - np.nanmedian(gtp)))
    return float(np.mean(aerr))


def pearson(a, b):
    """带退化保护的 Pearson r：样本 <3 或任一列方差 ~0 时记 NaN。

    与 seg_mae_r 内联的 corrcoef 的唯一差别就是这层保护——受试者级样本只有
    几十人，退化情形真会出现，故 tools 的受试者级 r 走这里。"""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def subject_median_pairs(pr, ws, y_f, bnd, ti):
    """受试者级配对：人内 median(预测) vs median(真值)，按受试者号升序。
    与 subject_median_mae 同一次遍历、同一口径，只是取配对而非绝对差。"""
    ps, ts = [], []
    for p in np.unique(ws):
        m = ws == p
        ps.append(np.median(pr[m]))
        ts.append(np.nanmedian(y_f[bnd[p]:bnd[p+1], ti]))
    return np.asarray(ps, np.float64), np.asarray(ts, np.float64)


def subject_r(pr, ws, y_f, bnd, ti):
    """受试者级 r：跨人 median 配对后的 Pearson r（部署口径的相关版）。
    只是分析字段——主表与 wide 表报的都是段级 r，勿混。"""
    return pearson(*subject_median_pairs(pr, ws, y_f, bnd, ti))


class EvalResult(NamedTuple):
    """一个已存头在一个测试折上的评估结果（eval_variant 返回）。"""
    seg: float          # 段级 MAE
    sub: float          # 受试者级 MAE（median 聚合）
    r: float            # 段级 Pearson r（全体测试段一次相关）
    row: tuple          # 预测明细列元组，逐 seed append 成列表后喂 dump_preds
    wp: np.ndarray      # 每段段内位置（冷启动分桶用）
    pr: np.ndarray      # 预测（raw 尺度）
    gt: np.ndarray      # 真值（raw 尺度）


def eval_variant(ck, var, data, te_p, ti, seed, with_src=True, dev=DEV):
    """一个已存头在一个测试折上的完整评估（免训练纯前向）。

    row 的列布局随 with_src 二选一，逐位对应两个调用方的落盘格式：
      True  → (seed, sid, seg_idx, src, y_true, y_pred)（28 号/tools 口径）
      False → (seed, sid, seg_idx, y_true, y_pred)（37 号逐折追加、无 src 列）
    """
    pr, gt, ws, wp = head_predict(ck, var, data, te_p, ti, dev=dev)
    gpos = data.bnd[ws] + wp                     # 全局段序号
    seg_mae, seg_r = seg_mae_r(pr, gt)
    sub_mae = subject_median_mae(pr, ws, data.y, data.bnd, ti)
    cols = (np.full(len(pr), seed, np.int16), data.sub_names[ws], data.seg_idx[gpos])
    if with_src:
        cols += (np.char.decode(data.src[gpos], 'utf-8'),)
    return EvalResult(seg_mae, sub_mae, seg_r, cols + (gt, pr), wp, pr, gt)


def bucket_mae(pr, gt, wp, buckets=BUCKETS):
    """段级误差按「段内位置 k」分桶（冷启动曲线）：返回 {桶名: MAE}，空桶不出现。
    28/tools 逐 seed 取 seed0 的值；37 号按折收集后再平均（聚合在调用方）。"""
    out = {}
    for lo, hi, name in buckets:
        m = (wp >= lo) & (wp < hi)
        if m.sum():
            out[name] = float(np.abs(pr[m] - gt[m]).mean())
    return out


# ── 预测明细落盘 / 回读 ────────────────────────────────────────────────

def dump_preds(pred_dir, t, var, rows, fold=None, src=True, append=False):
    """预测明细落盘（对齐 subject preds.csv 落盘规范，2026-09-06 补齐 stream 覆盖缺口）。
    列布局逐位保持两个调用方的原样，勿合并：
      28 号/tools: target,variant,seed,sid,seg_idx,src,y_true,y_pred（整写）
      37 号:       target,variant,fold,seed,sid,seg_idx,y_true,y_pred（逐折追加、无 src）
    用途：事后单域/单源拆分、散点图、误差分析——免重算前向。
    rows = 逐 seed 列数组元组的列表（元组内容见 eval_variant）。"""
    import pandas as pd
    cols = ([np.concatenate(xs) for xs in zip(*rows)] if len(rows) > 1
            else list(rows[0]))
    df = pd.DataFrame({'seed': cols[0], 'sid': cols[1], 'seg_idx': cols[2]})
    if src:
        df['src'] = cols[3]
    df['y_true'], df['y_pred'] = cols[-2], cols[-1]
    if fold is not None:
        df.insert(0, 'fold', fold)
    df.insert(0, 'variant', var)
    df.insert(0, 'target', t)
    fp = pred_dir / f'{t}__{var}.csv.gz'
    # 注：不指定 float_format 也是无损的——float32 列走 numpy 最短往返 repr
    #（实测 113.81767 读回位模式一致），故此处保持默认，勿加 %.9g（会多出无谓位数）。
    params = dict(index=False, compression='gzip')
    if append:
        params.update(mode='a', header=not fp.exists())
    df.to_csv(fp, **params)
    return fp


def r_from_preds(gz):
    """从预测明细重算段级 r：逐 seed 在全体行（跨源汇总）上算 Pearson r，再取均值。
    返回 (r, seed 数, 逐 seed r)。与 28 号 preds 内存态、37 号 seg_r 同规则。
    用于 28 号 `--backfill-r`：早期 results.json 没写 seg_r，只有 MAE 列表，
    r 无法从 JSON 反推，但可以从 preds/*.csv.gz 逐 seed 重算。"""
    import pandas as pd
    df = pd.read_csv(gz, usecols=['seed', 'y_true', 'y_pred'])
    rs = []
    for _, g in df.groupby('seed'):
        gt, pr = g.y_true.to_numpy(np.float64), g.y_pred.to_numpy(np.float64)
        rs.append(float(np.corrcoef(gt, pr)[0, 1]) if len(gt) > 2 else np.nan)
    return round(float(np.nanmean(rs)), 3), len(rs), [round(x, 4) for x in rs]
