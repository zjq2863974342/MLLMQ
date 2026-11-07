#!/usr/bin/env bash
# ============================================================================
# Parallel repair queue — 4 workers (2 per GPU), CUDA_VISIBLE_DEVICES isolation.
# Idempotent: skips cells that already have *_results.json.
# Total: 16 cells, ~10h wall-clock (each worker ~10h).
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

is_done() {
    local task=$1 tag=$2
    local d="$OUTPUT_BASE/$task/$tag"
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}

# Usage: run_cell task method w_bit scale_path
run_cell() {
    local task=$1 method=$2 w_bit=$3 scale=$4
    local tag
    if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi
    if is_done "$task" "$tag"; then
        echo "  SKIP $task/$tag (done)"
        return 0
    fi
    local log="$LOG_DIR/parallel_${tag}_${task}_$(date +%Y%m%d_%H%M%S).log"
    echo "  START [$CUDA_VISIBLE_DEVICES] $task/$tag $(date +%H:%M) -> $(basename "$log")"
    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" --method "$method" --w_bit "$w_bit" --a_bit 16 \
        --tasks "$task" --device cuda:0 --batch_size 1 \
        --output_base "$OUTPUT_BASE" \
        ${scale:+--scale_path "$scale"} \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  OK   [$CUDA_VISIBLE_DEVICES] $task/$tag $(date +%H:%M)"
    else
        echo "  FAIL [$CUDA_VISIBLE_DEVICES] $task/$tag rc=$rc"
    fi
    return $rc
}

# ──────────────────────────────────────────────────────────────────────────
# Worker 0: GPU0 slot0 — MME W8 (3 cells) + TextVQA fp16 (1 cell) = 4 cells ~10h
# ──────────────────────────────────────────────────────────────────────────
worker_0() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[W0 GPU0:A] started $(date +%H:%M)"
    run_cell mme_local awq   8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"
    run_cell mme_local mbq   8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"
    run_cell mme_local mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"
    run_cell textvqa_val_local fp16 16 ""
    echo "[W0 GPU0:A] DONE $(date +%H:%M)"
}

# ──────────────────────────────────────────────────────────────────────────
# Worker 1: GPU0 slot1 — TextVQA W4 (4 cells) = 4 cells ~10h
# ──────────────────────────────────────────────────────────────────────────
worker_1() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[W1 GPU0:B] started $(date +%H:%M)"
    run_cell textvqa_val_local rtn   4 ""
    run_cell textvqa_val_local awq   4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"
    run_cell textvqa_val_local mbq   4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"
    run_cell textvqa_val_local mllmq 4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"
    echo "[W1 GPU0:B] DONE $(date +%H:%M)"
}

# ──────────────────────────────────────────────────────────────────────────
# Worker 2: GPU1 slot0 — TextVQA W8 (4 cells) = 4 cells ~10h
# ──────────────────────────────────────────────────────────────────────────
worker_2() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[W2 GPU1:A] started $(date +%H:%M)"
    run_cell textvqa_val_local rtn   8 ""
    run_cell textvqa_val_local awq   8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"
    run_cell textvqa_val_local mbq   8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"
    run_cell textvqa_val_local mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"
    echo "[W2 GPU1:A] DONE $(date +%H:%M)"
}

# ──────────────────────────────────────────────────────────────────────────
# Worker 3: GPU1 slot1 — TextVQA W3 (4 cells) = 4 cells ~10h
# ──────────────────────────────────────────────────────────────────────────
worker_3() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[W3 GPU1:B] started $(date +%H:%M)"
    run_cell textvqa_val_local rtn   3 ""
    run_cell textvqa_val_local awq   3 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w3a16.pt"
    run_cell textvqa_val_local mbq   3 "$SCALE_DIR/mbq/internvl2_w3a16.pt"
    run_cell textvqa_val_local mllmq 3 "$SCALE_DIR/mllmq/internvl2_w3a16.pt"
    echo "[W3 GPU1:B] DONE $(date +%H:%M)"
}

echo "════════════════════════════════════════════════════════"
echo "Parallel Repair Queue — $(date)"
echo "4 workers: GPU0×2  GPU1×2  (CUDA_VISIBLE_DEVICES isolation)"
echo "════════════════════════════════════════════════════════"

# Launch all 4 workers in background
worker_0 > "$LOG_DIR/parallel_w0_gpu0a.log" 2>&1 &
PID0=$!
worker_1 > "$LOG_DIR/parallel_w1_gpu0b.log" 2>&1 &
PID1=$!
worker_2 > "$LOG_DIR/parallel_w2_gpu1a.log" 2>&1 &
PID2=$!
worker_3 > "$LOG_DIR/parallel_w3_gpu1b.log" 2>&1 &
PID3=$!

echo "Workers: W0=$PID0  W1=$PID1  W2=$PID2  W3=$PID3"
echo "Master waiting..."

wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "════════════════════════════════════════════════════════"
echo "Parallel Repair Queue — ALL DONE — $(date)"
echo "════════════════════════════════════════════════════════"