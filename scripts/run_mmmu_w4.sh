#!/usr/bin/env bash
# ============================================================================
# MMMU W4A16 4-way parallel — 1 cell per worker
# Idempotent: skips cells with *_results.json.
# Total: 4 cells (RTN/AWQ/MBQ/MLLMQ × W4A16), ~1h wall-clock.
# ============================================================================
set -uo pipefail

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || exit 1

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
LOG_DIR=/nas3/kangjia/mllmq/output/logs
SCALE_DIR="$PROJ_ROOT/MLLMQ/scale_cache"
mkdir -p "$LOG_DIR"

export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

TASK=mmmu_val

is_done() {
    local tag=$1
    local d="$OUTPUT_BASE/$TASK/$tag"
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}

run_cell() {
    local method=$1 w_bit=$2 scale=$3
    local tag="${method}_w${w_bit}a16"
    if is_done "$tag"; then
        echo "  SKIP $TASK/$tag (done)"
        return 0
    fi
    local log="$LOG_DIR/mmmu_w4_${tag}_$(date +%Y%m%d_%H%M%S).log"
    echo "  START [gpu${CUDA_VISIBLE_DEVICES}] $TASK/$tag $(date +%H:%M) -> $(basename "$log")"
    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" --method "$method" --w_bit "$w_bit" --a_bit 16 \
        --tasks "$TASK" --device cuda:0 --batch_size 1 \
        --output_base "$OUTPUT_BASE" \
        ${scale:+--scale_path "$scale"} \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  OK   [gpu${CUDA_VISIBLE_DEVICES}] $TASK/$tag $(date +%H:%M)"
    else
        echo "  FAIL [gpu${CUDA_VISIBLE_DEVICES}] $TASK/$tag rc=$rc"
    fi
    return $rc
}

echo "════════════════════════════════════════════════════════"
echo "MMMU W4A16 4-Way — $(date)"
echo "GPU0: RTN + AWQ   GPU1: MBQ + MLLMQ"
echo "════════════════════════════════════════════════════════"

# GPU0
worker_gpu0_rtn() {
    export CUDA_VISIBLE_DEVICES=0
    run_cell rtn 4 ""
}
worker_gpu0_awq() {
    export CUDA_VISIBLE_DEVICES=0
    run_cell awq 4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"
}
# GPU1
worker_gpu1_mbq() {
    export CUDA_VISIBLE_DEVICES=1
    run_cell mbq 4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"
}
worker_gpu1_mllmq() {
    export CUDA_VISIBLE_DEVICES=1
    run_cell mllmq 4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"
}

worker_gpu0_rtn > "$LOG_DIR/mmmu_w4_rtn.log" 2>&1 &
echo "[launcher] RTN GPU0 started (PID $!)"
sleep 60
worker_gpu0_awq > "$LOG_DIR/mmmu_w4_awq.log" 2>&1 &
echo "[launcher] AWQ GPU0 started (PID $!)"
sleep 60
worker_gpu1_mbq > "$LOG_DIR/mmmu_w4_mbq.log" 2>&1 &
echo "[launcher] MBQ GPU1 started (PID $!)"
sleep 60
worker_gpu1_mllmq > "$LOG_DIR/mmmu_w4_mllmq.log" 2>&1 &
echo "[launcher] MLLMQ GPU1 started (PID $!)"

echo "[launcher] All 4 workers launched."

while true; do
    sleep 300
    REMAINING=$(ps aux | grep -c "eval_benchmarks.py" 2>/dev/null || echo 0)
    GPU0_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | grep "^0," | cut -d',' -f2 | tr -d ' ')
    GPU1_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | grep "^1," | cut -d',' -f2 | tr -d ' ')
    echo "[monitor $(date +%H:%M)] $REMAINING eval | GPU0: ${GPU0_MEM}MiB  GPU1: ${GPU1_MEM}MiB"
    if [ "$REMAINING" -eq 0 ]; then
        break
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "MMMU W4A16 — ALL DONE — $(date)"
echo "════════════════════════════════════════════════════════"

echo ""
echo "=== Results ==="
for method in rtn awq mbq mllmq; do
    tag="${method}_w4a16"
    rf=$(ls "$OUTPUT_BASE/$TASK/$tag"/*_results.json 2>/dev/null | head -1)
    if [ -n "$rf" ]; then
        acc=$(python3 -c "
import json
d=json.load(open('$rf'))
r=d.get('results',{}).get('$TASK',{})
for k in ['mmmu_acc,none','acc,none','accuracy,none']:
    if k in r: print(r[k]); break
" 2>/dev/null)
        echo "  $tag: $acc"
    else
        echo "  $tag: MISSING"
    fi
done