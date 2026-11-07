#!/usr/bin/env bash
# Janus-Pro MLLMQ scale search — W4→W8→W3 sequential queue
# Usage:
#   GPU0 (mae, shared-denom):     bash scripts/run_janus_w4_w8_w3.sh mae 0
#   GPU1 (mse, per-modality):     bash scripts/run_janus_w4_w8_w3.sh mse 1
set -euo pipefail

TAG=${1:?Usage: $0 <mae|mse> <gpu_id>}
GPU=${2:?Usage: $0 <mae|mse> <gpu_id>}

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
DATA=/nas3/kangjia/mllmq/data/coco/janus_mixed_calib_256.json
IMG=/nas3/kangjia/mllmq/data/coco/train2017
NSAMPLES=256
LOG_DIR=/nas3/kangjia/mllmq/output/logs
CACHE_DIR=/nas3/kangjia/mllmq/MLLMQ/scale_cache/mllmq
OUT_DIR=/nas3/kangjia/mllmq/output/janus-pro-7b/models

mkdir -p "$LOG_DIR" "$CACHE_DIR" "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=$GPU
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

case "$TAG" in
    mae) LOSS=mae ;;
    mse) LOSS=mae_balanced ;;
    *)   echo "Unknown tag: $TAG (use mae or mse)"; exit 1 ;;
esac

for WBIT in 4 8 3; do
    SCALE="$CACHE_DIR/janus_${TAG}_w${WBIT}a16.pt"
    OUT="$OUT_DIR/janus_${TAG}_w${WBIT}a16"
    LOG="$LOG_DIR/scale_janus_${TAG}_w${WBIT}.log"

    if [ -f "$SCALE" ]; then
        echo "[$(date +%H:%M:%S)] SKIP ${TAG}_w${WBIT}: already cached at $SCALE"
        continue
    fi

    echo "============================================================"
    echo "[$(date +%H:%M:%S)] GPU${GPU} | START ${TAG}_w${WBIT}"
    echo "  scale: $SCALE"
    echo "  log:   $LOG"
    echo "============================================================"

    python main_quant.py \
        --model_path "$MODEL" --device cuda:0 --calib_data coco \
        --data_path "$DATA" --image_folder "$IMG" \
        --n_samples $NSAMPLES --method mllmq \
        --w_bit "$WBIT" --a_bit 16 --w_group 128 \
        --run_process --loss_mode "$LOSS" \
        --scale_path "$SCALE" --output_path "$OUT" \
        >> "$LOG" 2>&1

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] FAIL ${TAG}_w${WBIT} (rc=$RC)"
        exit $RC
    fi
    echo "[$(date +%H:%M:%S)] DONE  ${TAG}_w${WBIT}"
done

echo "[$(date +%H:%M:%S)] GPU${GPU} ALL DONE ($TAG)"