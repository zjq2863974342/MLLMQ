#!/usr/bin/env bash
# ============================================================================
# 副机夜间总驱动（2026-11-01 修订）：
#   1) 等当前 T2I / Exp2 评测结束
#   2) P0-2 补跑：干净口径 25/tag 的 block/awq 四格（scalar 由主机负责）
#   3) GAMQ 理解侧：7 benchmarks（4 卡按任务并行）
#   4) GAMQ COCO-val 5k 生成 + 4 路评分
#
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_clean_remote.sh \
#     > ../output/logs/clean_remote_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

log(){ echo -e "[$(date '+%F %T')] $1"; }

log "===== 副机夜间驱动：等 T2I / Exp2 结束 ====="
sleep 60
while pgrep -f "[r]un_exp3_t2i_mixed.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_exp2_eval.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_exp2_eval_gpu23.sh" >/dev/null 2>&1; do
  log "等待 T2I / Exp2 ..."; sleep 120
done

log "===== 1/3 P0-2 补跑：干净口径 block/awq（25/tag）====="
ONLY="block_w4 block_w3 awq_w4 awq_w3" GPUS="0 1 2 3" bash scripts/run_clean_comparison.sh

log "===== 2/3 GAMQ 理解侧：7 benchmarks ====="
if python -c "import lmms_eval" >/dev/null 2>&1; then
  GPUS="0 1 2 3" bash scripts/run_exp3_understanding_mixed.sh
else
  log "SKIP 理解侧：本机缺少 lmms_eval（不安装、不阻塞，交给主机或后续补）"
fi

log "===== 3/3 GAMQ COCO-val 5k + 评分 ====="
if python -c "import hpsv2, ImageReward" >/dev/null 2>&1; then
  GPUS="0 1 2 3" bash scripts/run_exp3_coco_mixed.sh
else
  log "SKIP COCO 评分全套：缺 hpsv2/ImageReward；仅生成图像（不评分）"
  SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache/mllmq/janus_mixed_w34_spread_k12.pt
  OUTD=/nas3/kangjia/mllmq/output/janus-pro-7b/coco_val/gamt_k12
  mkdir -p "$OUTD/coco_val"
  gpus=(0 1 2 3); i=0; pids=()
  for g in "${gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=$g python eval_generate_quant.py \
      --model_path /nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B \
      --benchmark coco_val --output_dir "$OUTD" --num_samples 1 \
      --quant_method mllmq_mixed --w_bit 3 --scale_path "$SCALE" \
      --shard_id "$i" --num_shards 4 \
      > /nas3/kangjia/mllmq/output/logs/exp3_coco_gen_s${i}.log 2>&1 &
    pids+=($!); i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
  log "COCO 生成完成，评分待主机补"
fi

log "ALL DONE"