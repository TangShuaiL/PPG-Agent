# BeatMAE 外部依赖契约（v3.3 断联）

本目录（`external/BeatMAE/`）是从远程仓 **clone** 进来的第三方模型仓，**只读消费、
不修改、不入 PPG-Agent 版本库**（`.gitignore` 已排 `external/*`）。本文件是唯一的
入库锚点，记录两仓之间的契约边界。任何 bump 上游或权重都要落到这里。

## 交付三供应链

| 供应链 | 载体 | 获取方式 |
|---|---|---|
| 代码 | BeatMAE 仓 | `git clone git@github.com:TangShuaiL/BeatMAE.git external/BeatMAE` |
| 权重 | Zenodo 记录 | `ppg_agent/engine/weights.py` 自动下载 + sha256 校验（缓存到 `DATA_ROOT/ppg_agent/weights/`） |
| 数据 | data_v2 数仓 | 环境前提（原始生理数据不入公开归档） |

## 上游仓钉版

- **repo**: `git@github.com:TangShuaiL/BeatMAE.git`
- **clone 路径**: `external/BeatMAE`
- **pinned HEAD sha**: `2ef2640`（"更新权重 DOI：Zenodo 记录已发布"）
- **Agent 侧消费的公开面**（BeatMAE 若删除以下任一路径即视为契约破坏，须重估适配）：
  - `data/beat_ops.py`（实时链表示层）
  - `models/`（`BeatMAE` / `embed_segment` / `beatmae_arch_from_sd`）
  - `train/run_meta.py: ckpt_sha1`（encoder 版本烙印）
  - `data/corpus_dataset.py`（`CorpusDataset` / `beat_collate`，离线全量嵌入重算用）
  - `models/probe_heads/` + `train/stream_probe.py`（头结构 + 头拟合协议）

## 主力模型口径（v3.3 定案）

- **主力 encoder = Zenodo 发布的 `beatmae_v25c3_s4`（3src：mimic+vital+aurora）**。
  历史遗留的 `beatmae_v25c_traverse_v2new`（4src，ckpt sha1 `cb89b83cf320`）**已退役**，
  不再是实时链/查表的真源；旧 embeddings/heads 目录保留作历史对照。
- **encoder 整份 ckpt（未剥壳）sha1[:12] = `8896d8940db8`**（源
  `results/pretrain/beatmae_v25c3/v2c_3src_mva/anchor_mr065_traverse/ckpt/best.pt`）。
- **Zenodo 脱敏版**（68MB→22.8MB，剥优化器+路径，163 张量权重 bit-identical；整记录
  打成单个 `weights.zip`，内含 `weights/beatmae_v25c3_s4.pt` + `weights/heads/*`）：
  - encoder: `beatmae_v25c3_s4.pt`，**脱敏文件 sha1[:12] = `007e0d24af9a`**（≠ 整份 ckpt
    `8896d8940db8`，因剥了优化器/路径）。
  - **全链烙印 `encoder_version = beatmae_v25c3_s4@007e0d24af9a`**（embeddings 目录名
    + manifest.encoder_version + 运行期装载 encoder 派生值三者必一致）。
  - heads: `heads/{sbp,dbp,age}_causal_attn_s4.pt`（单 seed，full 域；外部人零拟合直接跑）
- **Zenodo DOI**: 版本 `10.5281/zenodo.23080600` / concept `10.5281/zenodo.23080599`
- **Zenodo record id**（weights.py 下载用）: `23080600`

## encoder_version 一致性红线

`live_stream._get_encoder()` 装载 encoder 时把 `ckpt_sha1` 推出的 `encoder_version`
与 causal `manifest.encoder_version` 烙印比对，不一致 → 显式告警（不静默降级）。
**重建流程**：`build_ml_embeddings.py`（重出 embeddings.npy）→ `build_causal_heads.py`
（在 BeatMAE 已拟合头上**打包**（不重拟合）+ 更新 manifest）→ `validate_live_stream.py`
全绿 → 才可合入。

## 权重完整性表（sha256，供 weights.py 机读校验）

首次下载后由 `weights.py` 打印观测值，维护者回填此 JSON；空值=尚未钉死（不阻断，仅提示）。

<!-- ZENODO_SHA256 {"weights.zip": "16e23391afaae26f7a77c1849b25c46839976cd2c7fa6ef788a644e442d77089"} -->

zip 内各产物 sha256（解包后参考值；`weights.py` 机读校验以 zip 为粒度）：

| 文件 | sha256 |
|---|---|
| `beatmae_v25c3_s4.pt` | `f67418e7812c7e7cdb4f95cd07b7ff34759e4297845c672af20b0a1a0043b679` |
| `heads/sbp_causal_attn_s4.pt` | `e0ef0649f863d6ee4734148d3fdedbd375600f961e53d3cb0c416198c0cacdcd` |
| `heads/dbp_causal_attn_s4.pt` | `0606e0b61b69227cbde9ae31ff6ca63cc87ed346930ebb8eb95f0dbec19bc9c5` |
| `heads/age_causal_attn_s4.pt` | `0530da2f76ebd49f17b1bb5a3cd8c014c93d2bcce9253418df6dd255d0b460b2` |
