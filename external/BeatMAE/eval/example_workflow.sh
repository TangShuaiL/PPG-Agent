#!/bin/bash
# eval/example_workflow.sh — 新实验全流程模板：预训练权重 → 嵌入提取 → 段级流式探针
# 全程零改 Python：所有参数走命令行，产物自动归档 results/
#
# 用法:
#   EXP=myexp CKPT=results/pretrain/beatmae_v2/v2c_4src_mvaiwb/<variant>/ckpt/best.pt \
#     bash eval/example_workflow.sh            # 全库提取
#   EXP=myexp_iwshi_bp CKPT=... bash eval/example_workflow.sh iwshi_bp   # 单源提取
set -e
PY=${PY:-python3}   # 与 example_matrix.sh 同规
EXP=${EXP:?需要 EXP=实验名}
CKPT=${CKPT:?需要 CKPT=预训练权重路径（results/pretrain/ 四层树下）}
export PPG_CORPUS_ROOT=${PPG_CORPUS_ROOT:-/data2/corpus_v2c}

# ── 1. 嵌入提取 → results/cache/embeddings/${EXP}_lab.npz（大体积缓存留 results/cache/）──
if [ -n "$1" ]; then
  $PY scripts/extract_embeddings.py --name="$EXP" --ckpt="$CKPT" --sources="$1"
else
  $PY scripts/extract_embeddings.py --name="$EXP" --ckpt="$CKPT"
fi
CACHE="results/cache/embeddings/${EXP}_lab.npz"

# ── 2. 段级流式（主口径）：29 号物化 → 28 号三变体评估 ──────────────────────
#    产物 → results/probe/<model>/<ds>/<scope>/stream/（段级 MAE/r，逐 seed 算 r 再平均）
$PY scripts/materialize_stream.py --exp="$EXP" --npz="$CACHE"
$PY scripts/probe_full_stream.py exp="$EXP" dir=results/cache/stream/"$EXP"

echo "══ 完成。产物统一在 results/probe/<model>/<ds>/<scope>/（段级主口径 = stream/）══"
