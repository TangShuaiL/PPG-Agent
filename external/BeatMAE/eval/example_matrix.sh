#!/bin/bash
# eval/example_matrix.sh — 统一战役模板：全部下游任务 = sh + 传参，零改代码
#
# 任务 → 入口对照（2026-09-10 收敛后；段级为主口径）:
#   嵌入提取     scripts/extract_embeddings.py: --name/--ckpt/--sources（新组合免改代码）
#   段级流式     scripts/materialize_stream.py 物化 + scripts/probe_full_stream.py
#                （exp= 传实验名；结果+头 → probe/<model>/<ds>/<scope>/stream/）
#   五折段级     scripts/probe_iwshi_bp_5fold.py: 复用同一协议，受试者五折 CV
#                （→ …/stream5f/，报折间 std）
#   （已退役归档：窗口 attn → …/window/；iwshi_bp 零样本 → …/zeroshot/）
#
# 用法: bash eval/example_matrix.sh [stream|extract]   # 缺省 stream；换实验加 EXP=<名>
# 解释器/语料根都可用环境变量覆盖；缺省解释器 = README「环境」一节的那个 conda env。
set -e
PY=${PY:-python3}
export PPG_CORPUS_ROOT=${PPG_CORPUS_ROOT:-/data2/corpus_v2c}

case "${1:-stream}" in

# ── 段级流式（变体可选: w1_ridge seg_mlp causal_attn causal_k8 causal_shuf）──
#    缺省只跑主表三变体；causal_k8/causal_shuf 为消融哨兵，需点名
#    前置：29 号先把 <EXP>_lab.npz 物化成 results/cache/stream/<EXP>/
stream)
  if [ -n "$EXP" ]; then
    $PY scripts/probe_full_stream.py exp="$EXP" w1_ridge seg_mlp causal_attn
  else
    $PY scripts/probe_full_stream.py w1_ridge seg_mlp causal_attn   # 28 号内建缺省实验
  fi
  ;;

# ── 嵌入提取（新实验第一跳：新权重免改代码）─────────────────────────────
extract)
  $PY scripts/extract_embeddings.py --name=my_new_exp \
      --ckpt=results/pretrain/beatmae_v2/v2c_4src_mvaiwb/unified_mr065_traverse/ckpt/best.pt \
      --sources=iwshi_bp
  ;;

*) echo "用法: bash eval/example_matrix.sh [stream|extract]"; exit 1 ;;
esac
