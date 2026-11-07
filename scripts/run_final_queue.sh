#!/usr/bin/env bash
# ============================================================================
# Final queue: MMMU W4 retry (2 cells) → TextVQA all 13 cells
# Idempotent: skips cells with *_results.json. No set -e — fail>try next.
# Phase 1: MMMU retry — 1 proc/GPU (~10 min)
# Phase 2: TextVQA — 2 procs/GPU (~7-9h)
# Total: 15 cells to run (2 retry + 13 fresh), ~8-10h wall-clock.
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

is_done() {
    local task=$1 tag=$2
    local d="$OUTPUT_BASE/$task/$tag"
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}

run_cell() {
    local task=$1 method=$2 w_bit=$3 scale=$4
    local tag
    if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi
    if is_done "$task" "$tag"; then
        echo "  SKIP $task/$tag (done)"
        return 0
    fi
    local log="$LOG_DIR/final_${tag}_${task}_$(date +%Y%m%d_%H%M%S).log"
    echo "  START [gpu${CUDA_VISIBLE_DEVICES}] $task/$tag $(date +%H:%M) -> $(basename "$log")"
    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" --method "$method" --w_bit "$w_bit" --a_bit 16 \
        --tasks "$task" --device cuda:0 --batch_size 1 \
        --output_base "$OUTPUT_BASE" \
        ${scale:+--scale_path "$scale"} \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  OK   [gpu${CUDA_VISIBLE_DEVICES}] $task/$tag $(date +%H:%M)"
    else
        echo "  FAIL [gpu${CUDA_VISIBLE_DEVICES}] $task/$tag rc=$rc"
        tail -5 "$log" | grep -iE "error|traceback|cuda|OOM" 2>/dev/null || true
    fi
    return $rc
}

# ═══════════════════════════════════════════════════════
# Phase 1: MMMU W4 retry (1 process per GPU, safe)
# ═══════════════════════════════════════════════════════
worker_mmmu_gpu0() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[MMMU GPU0] MMMU RTN W4 retry  $(date +%H:%M)"
    run_cell mmmu_val rtn 4 ""
    echo "[MMMU GPU0] DONE $(date +%H:%M)"
}

worker_mmmu_gpu1() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[MMMU GPU1] MMMU MBQ W4 retry  $(date +%H:%M)"
    run_cell mmmu_val mbq 4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"
    echo "[MMMU GPU1] DONE $(date +%H:%M)"
}

# ═══════════════════════════════════════════════════════
# Phase 2: TextVQA 13 cells (2 workers/GPU, 60s stagger)
# ═══════════════════════════════════════════════════════
worker_tvqa_w0() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[W0 GPU0] T+VQA: fp16 → RTN W4/8/3  $(date +%H:%M)"
    run_cell textvqa_val_local fp16 16 ""
    run_cell textvqa_val_local rtn  4 ""
    run_cell textvqa_val_local rtn  8 ""
    run_cell textvqa_val_local rtn  3 ""
    echo "[W0 GPU0] DONE $(date +%H:%M)"
}

worker_tvqa_w1() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[W1 GPU0] T+VQA: AWQ W4/8/3  $(date +%H:%M)"
    run_cell textvqa_val_local awq 4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"
    run_cell textvqa_val_local awq 8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"
    run_cell textvqa_val_local awq 3 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w3a16.pt"
    echo "[W1 GPU0] DONE $(date +%H:%M)"
}

worker_tvqa_w2() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[W2 GPU1] T+VQA: MBQ W4/8/3  $(date +%H:%M)"
    run_cell textvqa_val_local mbq 4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"
    run_cell textvqa_val_local mbq 8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"
    run_cell textvqa_val_local mbq 3 "$SCALE_DIR/mbq/internvl2_w3a16.pt"
    echo "[W2 GPU1] DONE $(date +%H:%M)"
}

worker_tvqa_w3() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[W3 GPU1] T+VQA: MLLMQ W4/8/3  $(date +%H:%M)"
    run_cell textvqa_val_local mllmq 4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"
    run_cell textvqa_val_local mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"
    run_cell textvqa_val_local mllmq 3 "$SCALE_DIR/mllmq/internvl2_w3a16.pt"
    echo "[W3 GPU1] DONE $(date +%H:%M)"
}

echo "════════════════════════════════════════════════════════"
echo "FINAL QUEUE — $(date)"
echo "Phase 1: MMMU W4 retry (1x/GPU, ~10min)"
echo "Phase 2: TextVQA 13 cells (2x/GPU, ~7-9h)"
echo "════════════════════════════════════════════════════════"

# ═══ Phase 1: MMMU W4 retry (parallel, 1/GPU) ═══
echo ""
echo "=== Phase 1: MMMU W4 retry ==="
worker_mmmu_gpu0 > "$LOG_DIR/final_mmmu_gpu0.log" 2>&1 &
MMMU_GPU0_PID=$!
echo "[launcher] MMMU GPU0 (RTN W4) started PID=$MMMU_GPU0_PID"
sleep 30
worker_mmmu_gpu1 > "$LOG_DIR/final_mmmu_gpu1.log" 2>&1 &
MMMU_GPU1_PID=$!
echo "[launcher] MMMU GPU1 (MBQ W4) started PID=$MMMU_GPU1_PID"

wait $MMMU_GPU0_PID $MMMU_GPU1_PID 2>/dev/null || true
sleep 5
echo "[launcher] Phase 1 (MMMU) DONE $(date +%H:%M)"

# Quick check: did MMMU retries succeed?
echo "  MMMU RTN W4: $(is_done mmmu_val rtn_w4a16 && echo '✅ DONE' || echo '❌ FAILED')"
echo "  MMMU MBQ W4: $(is_done mmmu_val block_w4a16 && echo '✅ DONE' || echo '❌ FAILED')"

# ═══ Phase 2: TextVQA 13 cells (staggered launch) ═══
echo ""
echo "=== Phase 2: TextVQA 13 cells ==="
echo "[launcher] Launching with 60s stagger..."

worker_tvqa_w0 > "$LOG_DIR/final_tvqa_w0.log" 2>&1 &
echo "[launcher] W0 GPU0 started (PID $!)"
sleep 60

worker_tvqa_w1 > "$LOG_DIR/final_tvqa_w1.log" 2>&1 &
echo "[launcher] W1 GPU0 started (PID $!)"
sleep 60

worker_tvqa_w2 > "$LOG_DIR/final_tvqa_w2.log" 2>&1 &
echo "[launcher] W2 GPU1 started (PID $!)"
sleep 60

worker_tvqa_w3 > "$LOG_DIR/final_tvqa_w3.log" 2>&1 &
echo "[launcher] W3 GPU1 started (PID $!)"

echo "[launcher] All 4 TextVQA workers launched."

# ── Monitor loop ──
while true; do
    sleep 300
    REMAINING=$(ps aux | grep "eval_benchmarks.py" | grep -v grep | wc -l)
    GPU0_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | grep "^0," | cut -d',' -f2 | tr -d ' ')
    GPU1_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | grep "^1," | cut -d',' -f2 | tr -d ' ')
    echo "[monitor $(date +%H:%M)] $REMAINING eval | GPU0: ${GPU0_MEM}MiB  GPU1: ${GPU1_MEM}MiB"
    if [ "$REMAINING" -eq 0 ]; then
        break
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "FINAL QUEUE — ALL DONE — $(date)"
echo "════════════════════════════════════════════════════════"

echo ""
echo "=== Results Summary ==="
for task in mmmu_val textvqa_val_local; do
    echo "--- $task ---"
    for d in "$OUTPUT_BASE/$task"/*/; do
        [ -d "$d" ] || continue
        tag=$(basename "$d")
        if ls "$d"/*_results.json >/dev/null 2>&1; then
            echo "  $tag: DONE"
        else
            echo "  $tag: MISSING"
        fi
    done
done