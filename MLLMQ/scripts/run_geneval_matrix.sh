#!/bin/bash
# GenEval full matrix driver — 19 configs, 2 GPU parallel
# Usage: bash scripts/run_geneval_matrix.sh
# No set -e — single cell failure should not terminate the batch

export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT_BASE=/nas3/kangjia/mllmq/output/janus-pro-7b/geneval
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
SCRIPT=/nas3/kangjia/mllmq/MLLMQ/eval_generate_quant.py
TARGET=2212  # 553 prompts × 4 images

# Config list: tag quant_method w_bit scale_path (empty for rtn/fp16)
CONFIGS=(
  "fp16       fp16  -  -"
  "rtn_w8     rtn   8  -"
  "rtn_w4     rtn   4  -"
  "rtn_w3     rtn   3  -"
  "awq_w8     awq   8  $SCALE/awq/janus_awq_w8a16.pt"
  "awq_w4     awq   4  $SCALE/awq/janus_awq_w4a16.pt"
  "awq_w3     awq   3  $SCALE/awq/janus_awq_w3a16.pt"
  "mbq_block_w8   mbq  8  $SCALE/mbq/janus_mbq_block_w8a16.pt"
  "mbq_block_w4   mbq  4  $SCALE/mbq/janus_mbq_block_w4a16.pt"
  "mbq_block_w3   mbq  3  $SCALE/mbq/janus_mbq_block_w3a16.pt"
  "mbq_scalar_w8  mbq  8  $SCALE/mbq/janus_mbq_scalar_w8a16.pt"
  "mbq_scalar_w4  mbq  4  $SCALE/mbq/janus_mbq_scalar_w4a16.pt"
  "mbq_scalar_w3  mbq  3  $SCALE/mbq/janus_mbq_scalar_w3a16.pt"
  "mllmq_mae_w8   mbq  8  $SCALE/mllmq/janus_mae_w8a16.pt"
  "mllmq_mae_w4   mbq  4  $SCALE/mllmq/janus_mae_w4a16.pt"
  "mllmq_mae_w3   mbq  3  $SCALE/mllmq/janus_mae_w3a16.pt"
  "mllmq_mse_w8   mbq  8  $SCALE/mllmq/janus_mse_w8a16.pt"
  "mllmq_mse_w4   mbq  4  $SCALE/mllmq/janus_mse_w4a16.pt"
  "mllmq_mse_w3   mbq  3  $SCALE/mllmq/janus_mse_w3a16.pt"
)

is_done() {
  local tag=$1
  local count=$(find "$OUT_BASE/$tag" -name '*.png' 2>/dev/null | wc -l)
  [ "$count" -ge "$TARGET" ]
}

run_config() {
  local tag=$1 method=$2 wbit=$3 scale=$4 gpu=$5
  local outdir="$OUT_BASE/$tag"
  local logfile="$LOG/geneval_${tag}.log"

  echo "[$(date '+%H:%M:%S')] Starting $tag on GPU$gpu ..."

  local scale_arg=""
  [ "$scale" != "-" ] && scale_arg="--scale_path $scale"

  CUDA_VISIBLE_DEVICES=$gpu nohup python "$SCRIPT" \
    --model_path "$MODEL" --benchmark geneval \
    --output_dir "$outdir" --num_samples 4 \
    --quant_method "$method" --w_bit "$wbit" \
    $scale_arg \
    > "$logfile" 2>&1 &
  echo "  PID: $!"
}

# Main loop
PENDING=()
for cfg in "${CONFIGS[@]}"; do
  read -r tag method wbit scale <<< "$cfg"
  if is_done "$tag"; then
    echo "[$(date '+%H:%M:%S')] ✅ $tag: already complete"
    continue
  fi
  count=$(find "$OUT_BASE/$tag" -name '*.png' 2>/dev/null | wc -l)
  echo "[$(date '+%H:%M:%S')] ⬜ $tag: $count/$TARGET — queued"
  PENDING+=("$cfg")
done

echo ""
echo "=== ${#PENDING[@]} configs pending ==="

for cfg in "${PENDING[@]}"; do
  read -r tag method wbit scale <<< "$cfg"

  # Wait for a free GPU
  while true; do
    gpu0_busy=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    gpu1_busy=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)

    if [ "$gpu0_busy" -eq 0 ]; then
      GPU=0; break
    elif [ "$gpu1_busy" -eq 0 ]; then
      GPU=1; break
    fi
    sleep 30
  done

  run_config "$tag" "$method" "$wbit" "$scale" "$GPU"
  sleep 90  # Stagger: avoid simultaneous model loading VRAM spike
done

echo ""
echo "=== All configs launched. Waiting for completion... ==="

# Wait for all background jobs
wait

echo ""
echo "=== Final status ==="
for cfg in "${CONFIGS[@]}"; do
  read -r tag _ __ ___ <<< "$cfg"
  count=$(find "$OUT_BASE/$tag" -name '*.png' 2>/dev/null | wc -l)
  if [ "$count" -ge "$TARGET" ]; then
    echo "  ✅ $tag: $count/$TARGET"
  else
    echo "  ⬜ $tag: $count/$TARGET"
  fi
done