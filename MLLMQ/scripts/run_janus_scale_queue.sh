#!/usr/bin/env bash
# Janus-Pro scale search queue — dual GPU
# GPU0: mllmq_mae (shared-denom)  → W4, W8, W3
# GPU1: mllmq_mse (per-modality)   → W4, W8, W3
#
# Usage: bash scripts/run_janus_scale_queue.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
CALIB_DATA=/nas3/kangjia/mllmq/data/coco/janus_mixed_calib_256.json
IMAGE_FOLDER=/nas3/kangjia/mllmq/data/coco/train2017
N_SAMPLES=256
LOG_DIR=/nas3/kangjia/mllmq/output/logs

mkdir -p "$LOG_DIR" scale_cache/mllmq

export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

run_cell() {
    local gpu=$1 method=$2 wbit=$3 loss=$4 tag=$5
    local scale_path="scale_cache/mllmq/janus_${tag}_w${wbit}a16.pt"
    local output_path="/nas3/kangjia/mllmq/output/janus-pro-7b/models/janus_${tag}_w${wbit}a16"
    local log="$LOG_DIR/scale_janus_${tag}_w${wbit}.log"

    echo "============================================================"
    echo "[$(date +%H:%M:%S)] GPU${gpu} | ${tag} W${wbit}A16"
    echo "  scale_path: ${scale_path}"
    echo "  log: ${log}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$gpu python main_quant.py \
        --model_path "$MODEL_PATH" \
        --device "cuda:0" \
        --calib_data coco \
        --data_path "$CALIB_DATA" \
        --image_folder "$IMAGE_FOLDER" \
        --n_samples $N_SAMPLES \
        --method "$method" \
        --w_bit "$wbit" --a_bit 16 --w_group 128 \
        --run_process \
        --scale_path "$scale_path" \
        --output_path "$output_path" \
        --loss_mode "$loss" \
        2>&1 | tee "$log"

    echo "[$(date +%H:%M:%S)] GPU${gpu} | ${tag} W${wbit}A16 DONE (rc=$?)"
}

# ================================================================
# GPU0: mllmq_mae (shared-denom)
# ================================================================
{
    run_cell 0 mllmq 4 mae      mae
    run_cell 0 mllmq 8 mae      mae
    run_cell 0 mllmq 3 mae      mae
    echo "=== GPU0: all mllmq_mae done ==="
} 2>&1 | tee "$LOG_DIR/queue_janus_gpu0.log" &

PID0=$!
echo "GPU0 queue PID: $PID0"

sleep 60  # stagger GPU memory peaks

# ================================================================
# GPU1: mllmq_mse (per-modality / mae_balanced)
# ================================================================
{
    run_cell 1 mllmq 4 mae_balanced mse
    run_cell 1 mllmq 8 mae_balanced mse
    run_cell 1 mllmq 3 mae_balanced mse
    echo "=== GPU1: all mllmq_mse done ==="
} 2>&1 | tee "$LOG_DIR/queue_janus_gpu1.log" &

PID1=$!
echo "GPU1 queue PID: $PID1"

echo ""
echo "=== Queue started ==="
echo "  GPU0 (mllmq_mae): PID $PID0, W4→W8→W3"
echo "  GPU1 (mllmq_mse): PID $PID1, W4→W8→W3"
echo ""
echo "Monitor:"
echo "  watch -n 5 nvidia-smi"
echo "  tail -f $LOG_DIR/queue_janus_gpu0.log"
echo "  tail -f $LOG_DIR/queue_janus_gpu1.log"
echo ""
echo "Wait for completion:"
echo "  wait $PID0 $PID1"