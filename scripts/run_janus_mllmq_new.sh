#!/bin/bash
# Janus MLLMQ 新版 Scale Cache 全量测评 — 双队列并行
# Q1 (GPU0): MAE (共享分母) — mae_w4 → mae_w8 → mae_w3
# Q2 (GPU1): MSE (逐模态归一化) — mse_w4 → mse_w8 → mse_w3
#
# 启动: bash scripts/run_janus_mllmq_new.sh
# 监控: tail -f output/logs/janus_mae_new.log
#        tail -f output/logs/janus_mse_new.log
# 停止: pkill -f "eval_benchmarks.*mllmq_mae\|mllmq_mse"

set -euo pipefail

cd /nas3/kangjia/mllmq/MLLMQ

export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/eval
TASKS=mme_local,textvqa_val_local,mmmu_val,mathvista_testmini_local,pope_random_local,pope_popular_local,pope_adversarial_local
LOG_DIR=/nas3/kangjia/mllmq/output/logs

mkdir -p "$LOG_DIR"

# ===== Q1 (GPU0): MAE =====
(
  for w in 4 8 3; do
    echo "=== Q1-MAE W${w}A16 $(date) ==="
    CUDA_VISIBLE_DEVICES=0 python eval_benchmarks.py \
      --model_type janus --model_path "$MODEL" \
      --method mllmq --method_tag mllmq_mae --w_bit $w \
      --scale_path scale_cache/mllmq/janus_mae_w${w}a16.pt \
      --tasks "$TASKS" --output_base "$OUT" --device cuda:0 --log_samples
    echo "=== Q1-MAE W${w}A16 DONE $(date) ==="
  done
  echo "=== Q1-MAE ALL DONE $(date) ==="
) > "$LOG_DIR/janus_mae_new.log" 2>&1 &

# ===== Q2 (GPU1): MSE =====
(
  sleep 120  # 错峰 2min，避免同时 apply_quant VRAM 峰值叠加
  for w in 4 8 3; do
    echo "=== Q2-MSE W${w}A16 $(date) ==="
    CUDA_VISIBLE_DEVICES=1 python eval_benchmarks.py \
      --model_type janus --model_path "$MODEL" \
      --method mllmq --method_tag mllmq_mse --w_bit $w \
      --scale_path scale_cache/mllmq/janus_mse_w${w}a16.pt \
      --tasks "$TASKS" --output_base "$OUT" --device cuda:0 --log_samples
    echo "=== Q2-MSE W${w}A16 DONE $(date) ==="
  done
  echo "=== Q2-MSE ALL DONE $(date) ==="
) > "$LOG_DIR/janus_mse_new.log" 2>&1 &

echo "Q1-MAE (GPU0): PID=$!"
echo "Q2-MSE (GPU1): PID=$(( $! + 1 )) (approx)"
echo ""
echo "监控: tail -f $LOG_DIR/janus_mae_new.log"
echo "       tail -f $LOG_DIR/janus_mse_new.log"
echo "进度: watch -n 30 'find $OUT -path \"*mllmq_m[ae][se]_w*\" -name \"*_results.json\" ! -path \"*bak_old_calib*\" | wc -l'"