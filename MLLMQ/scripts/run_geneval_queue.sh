#!/bin/bash
# GenEval 自动队列 — 双 GPU 串行，每轮 2 进程，支持断点续跑
# Usage: nohup bash scripts/run_geneval_queue.sh > output/logs/geneval_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/geneval
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
COOLDOWN=90

run_cell() {
  local tag=$1 method=$2 w=$3 scale=$4
  local outd="$OUT/$tag"
  local logf="$LOG/geneval_${tag}.log"
  local img_count=$(find "$outd" -name '*.png' 2>/dev/null | wc -l)

  if [ "$img_count" -ge 2212 ]; then
    echo "[$(date +%H:%M:%S)] ✅ $tag: $img_count/2212 — already done, skip"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] 🔄 $tag: $img_count/2212 — starting (resume)..."
  local cmd=(python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval
             --output_dir "$outd" --num_samples 4 --quant_method "$method" --w_bit "$w")
  if [ -n "$scale" ]; then
    cmd+=(--scale_path "$scale")
  fi
  "${cmd[@]}" > "$logf" 2>&1
  local rc=$?
  img_count=$(find "$outd" -name '*.png' 2>/dev/null | wc -l)
  if [ "$img_count" -ge 2212 ]; then
    echo "[$(date +%H:%M:%S)] ✅ $tag: $img_count/2212 — done (rc=$rc)"
  else
    echo "[$(date +%H:%M:%S)] ❌ $tag: $img_count/2212 — FAILED (rc=$rc)"
  fi
  return $rc
}

# ============ W8 Batch (rtn_w8 done) ============
run_cell awq_w8 awq 8 "$SCALE/awq/janus_awq_w8a16.pt" &
PID0=$!
sleep $COOLDOWN

run_cell mbq_block_w8 mbq 8 "$SCALE/mbq/janus_mbq_block_w8a16.pt" &
PID1=$!
wait $PID0; wait $PID1
sleep $COOLDOWN

run_cell mbq_scalar_w8 mbq 8 "$SCALE/mbq/janus_mbq_scalar_w8a16.pt" &
PID0=$!
sleep $COOLDOWN

run_cell mllmq_mae_w8 mbq 8 "$SCALE/mllmq/janus_mae_w8a16.pt" &
PID1=$!
wait $PID0; wait $PID1
sleep $COOLDOWN

run_cell mllmq_mse_w8 mbq 8 "$SCALE/mllmq/janus_mse_w8a16.pt"
echo "[$(date +%H:%M:%S)] ✅ W8 batch complete"

# ============ W4 Batch (rtn_w4 partial resume) ============
run_cell rtn_w4 rtn 4 "" &
PID0=$!
sleep $COOLDOWN

run_cell awq_w4 awq 4 "$SCALE/awq/janus_awq_w4a16.pt" &
PID1=$!
wait $PID0; wait $PID1
sleep $COOLDOWN

run_cell mbq_block_w4 mbq 4 "$SCALE/mbq/janus_mbq_block_w4a16.pt" &
PID0=$!
sleep $COOLDOWN

run_cell mbq_scalar_w4 mbq 4 "$SCALE/mbq/janus_mbq_scalar_w4a16.pt" &
PID1=$!
wait $PID0; wait $PID1
sleep $COOLDOWN

run_cell mllmq_mae_w4 mbq 4 "$SCALE/mllmq/janus_mae_w4a16.pt" &
PID0=$!
sleep $COOLDOWN

run_cell mllmq_mse_w4 mbq 4 "$SCALE/mllmq/janus_mse_w4a16.pt" &
PID1=$!
wait $PID0; wait $PID1
echo "[$(date +%H:%M:%S)] ✅ W4 batch complete"

# ============ W3 Batch (rtn_w3 partial resume) ============
run_cell rtn_w3 rtn 3 "" &
PID0=$!
sleep $COOLDOWN

run_cell awq_w3 awq 3 "$SCALE/awq/janus_awq_w3a16.pt" &
PID1=$!
wait $PID0; wait $PID1
sleep $COOLDOWN

run_cell mbq_block_w3 mbq 3 "$SCALE/mbq/janus_mbq_block_w3a16.pt" &
PID0=$!
sleep $COOLDOWN

run_cell mbq_scalar_w3 mbq 3 "$SCALE/mbq/janus_mbq_scalar_w3a16.pt" &
PID1=$!
wait $PID0; wait $PID1
sleep $COOLDOWN

run_cell mllmq_mae_w3 mbq 3 "$SCALE/mllmq/janus_mae_w3a16.pt" &
PID0=$!
sleep $COOLDOWN

run_cell mllmq_mse_w3 mbq 3 "$SCALE/mllmq/janus_mse_w3a16.pt" &
PID1=$!
wait $PID0; wait $PID1
echo "[$(date +%H:%M:%S)] ✅ W3 batch complete"

echo "========================================="
echo "[$(date +%H:%M:%S)] 🎉 GenEval all 19 configs done!"
echo "========================================="