#!/usr/bin/env bash
# Janus-Pro MLLMQ full queue: scale search (W4→W8→W3) + understanding eval
#
# GPU0: mae (shared-denom)
# GPU1: mse (per-modality)
#
# Scale search outputs: scale_cache/mllmq/janus_{mae,mse}_w{4,8,3}a16.pt
# Eval outputs:         output/janus-pro-7b/eval/<benchmark>/<tag>/
#
# Usage:
#   bash scripts/run_janus_full_queue.sh
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
DATA=/nas3/kangjia/mllmq/data/coco/janus_mixed_calib_256.json
IMG=/nas3/kangjia/mllmq/data/coco/train2017
NSAMPLES=256
LOG_DIR=/nas3/kangjia/mllmq/output/logs
CACHE_DIR=/nas3/kangjia/mllmq/MLLMQ/scale_cache/mllmq
EVAL_OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/eval

mkdir -p "$LOG_DIR" "$CACHE_DIR" "$EVAL_OUT"

export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

# Understanding benchmarks (same 6 as InternVL2)
TASKS="mme_local,textvqa_val_local,mmmu_val,mathvista_testmini_local,pope_random_local,pope_popular_local,pope_adversarial_local,seedbench_2_local"

# ---------- helper ----------
run_scale() {
    local gpu=$1 tag=$2 wbit=$3 loss=$4
    local scale="$CACHE_DIR/janus_${tag}_w${wbit}a16.pt"
    local log="$LOG_DIR/scale_janus_${tag}_w${wbit}.log"

    if [ -f "$scale" ]; then
        echo "[$(date +%H:%M:%S)] SKIP scale ${tag}_w${wbit} (cached)"
        return 0
    fi

    echo "============================================================"
    echo "[$(date +%H:%M:%S)] GPU${gpu} SCALE ${tag}_w${wbit}A16"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$gpu python main_quant.py \
        --model_path "$MODEL" --device cuda:0 --calib_data coco \
        --data_path "$DATA" --image_folder "$IMG" \
        --n_samples $NSAMPLES --method mllmq \
        --w_bit "$wbit" --a_bit 16 --w_group 128 \
        --run_process --loss_mode "$loss" \
        --scale_path "$scale" \
        --output_path "/nas3/kangjia/mllmq/output/janus-pro-7b/models/janus_${tag}_w${wbit}a16" \
        > "$log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE scale ${tag}_w${wbit}A16 (rc=$?)"
}

run_eval() {
    local gpu=$1 tag=$2 wbit=$3
    local scale="$CACHE_DIR/janus_${tag}_w${wbit}a16.pt"
    local log="$LOG_DIR/eval_janus_${tag}_w${wbit}.log"

    echo "============================================================"
    echo "[$(date +%H:%M:%S)] GPU${gpu} EVAL ${tag}_w${wbit}A16"
    echo "  tasks: $TASKS"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$gpu python eval_benchmarks.py \
        --model_type janus \
        --model_path "$MODEL" \
        --method mllmq --w_bit "$wbit" --w_group 128 \
        --scale_path "$scale" \
        --tasks "$TASKS" \
        --task_include_path tasks_local \
        --output_base "$EVAL_OUT" \
        --device "cuda:0" \
        --log_samples \
        > "$log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE eval ${tag}_w${wbit}A16 (rc=$?)"
}

# ---------- GPU0: mae ----------
run_gpu0() {
    for w in 4 8 3; do
        run_scale 0 mae "$w" mae
        run_eval 0 mae "$w"
    done
    echo "[$(date +%H:%M:%S)] GPU0 ALL DONE (mae)"
}

# ---------- GPU1: mse ----------
run_gpu1() {
    for w in 4 8 3; do
        run_scale 1 mse "$w" mae_balanced
        run_eval 1 mse "$w"
    done
    echo "[$(date +%H:%M:%S)] GPU1 ALL DONE (mse)"
}

echo "============================================================"
echo "JANUS FULL QUEUE STARTED $(date)"
echo "  GPU0: mae  W4→eval→W8→eval→W3→eval"
echo "  GPU1: mse  W4→eval→W8→eval→W3→eval"
echo "============================================================"

# Launch both in parallel
run_gpu0 > "$LOG_DIR/queue_janus_gpu0.log" 2>&1 &
PID0=$!
echo "GPU0 PID: $PID0"

sleep 60  # stagger

run_gpu1 > "$LOG_DIR/queue_janus_gpu1.log" 2>&1 &
PID1=$!
echo "GPU1 PID: $PID1"

echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/queue_janus_gpu0.log"
echo "  tail -f $LOG_DIR/queue_janus_gpu1.log"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Wait:  wait $PID0 $PID1"

wait $PID0 $PID1
echo "[$(date +%H:%M:%S)] FULL QUEUE COMPLETE"