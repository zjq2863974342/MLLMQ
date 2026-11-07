#!/usr/bin/env bash
# ============================================================================
# 4-way parallel TextVQA queue — 2 workers per GPU
# Idempotent: skips cells with *_results.json already present.
# Total: 13 cells (fp16 + 4 methods × 3 bit widths), ~7.5-10h wall-clock.
#
# MMMU and MME are already complete (2026-09-11 / 2026-09-15).
#
# VRAM: 2 workers/GPU ≈ 42-46 GB (safe with 80 GB A800 + takeup)
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

TASK=textvqa_val_local

is_done() {
    local tag=$1
    local d="$OUTPUT_BASE/$TASK/$tag"
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}

run_cell() {
    local method=$1 w_bit=$2 scale=$3
    local tag
    if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi
    if is_done "$tag"; then
        echo "  SKIP $TASK/$tag (done)"
        return 0
    fi
    local log="$LOG_DIR/quad_tvqa_${tag}_$(date +%Y%m%d_%H%M%S).log"
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

# ═══════════════════════════════════════════════════════════════
# Worker definitions (4 workers × 2 GPUs)
# TextVQA ~2.5h/cell. 4 workers → ~7.5-10h wall-clock.
# W0 has fp16 baseline (no scale cache) + RTN (no scale search).
# ═══════════════════════════════════════════════════════════════

worker_gpu0_w0() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[W0 GPU0] TextVQA: fp16 baseline → RTN W4 → RTN W8 → RTN W3  $(date +%H:%M)"
    run_cell fp16 16 ""
    run_cell rtn   4 ""
    run_cell rtn   8 ""
    run_cell rtn   3 ""
    echo "[W0 GPU0] DONE $(date +%H:%M)"
}

worker_gpu0_w1() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[W1 GPU0] TextVQA: AWQ W4 → AWQ W8 → AWQ W3  $(date +%H:%M)"
    run_cell awq 4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"
    run_cell awq 8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"
    run_cell awq 3 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w3a16.pt"
    echo "[W1 GPU0] DONE $(date +%H:%M)"
}

worker_gpu1_w2() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[W2 GPU1] TextVQA: MBQ W4 → MBQ W8 → MBQ W3  $(date +%H:%M)"
    run_cell mbq 4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"
    run_cell mbq 8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"
    run_cell mbq 3 "$SCALE_DIR/mbq/internvl2_w3a16.pt"
    echo "[W2 GPU1] DONE $(date +%H:%M)"
}

worker_gpu1_w3() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[W3 GPU1] TextVQA: MLLMQ W4 → MLLMQ W8 → MLLMQ W3  $(date +%H:%M)"
    run_cell mllmq 4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"
    run_cell mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"
    run_cell mllmq 3 "$SCALE_DIR/mllmq/internvl2_w3a16.pt"
    echo "[W3 GPU1] DONE $(date +%H:%M)"
}

echo "════════════════════════════════════════════════════════"
echo "4-Way TextVQA Parallel Queue — $(date)"
echo "GPU0: W0(fp16+RTN) + W1(AWQ)   GPU1: W2(MBQ) + W3(MLLMQ)"
echo "2 workers/GPU, 3-4 cells/worker, ~7.5-10h wall-clock"
echo "13 cells total: fp16 + 4 methods × 3 bit widths"
echo "════════════════════════════════════════════════════════"

# ── Staggered launch (60s gap for safe VRAM ramp) ──
worker_gpu0_w0 > "$LOG_DIR/quad_tvqa_w0_gpu0.log" 2>&1 &
echo "[launcher] W0 GPU0 started (PID $!)"
sleep 60
worker_gpu0_w1 > "$LOG_DIR/quad_tvqa_w1_gpu0.log" 2>&1 &
echo "[launcher] W1 GPU0 started (PID $!)"
sleep 60
worker_gpu1_w2 > "$LOG_DIR/quad_tvqa_w2_gpu1.log" 2>&1 &
echo "[launcher] W2 GPU1 started (PID $!)"
sleep 60
worker_gpu1_w3 > "$LOG_DIR/quad_tvqa_w3_gpu1.log" 2>&1 &
echo "[launcher] W3 GPU1 started (PID $!)"

echo "[launcher] All 4 workers launched."

# Monitor loop
while true; do
    sleep 300
    REMAINING=$(pgrep -fc "eval_benchmarks.py" 2>/dev/null || echo 0)
    GPU0_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | grep "^0," | cut -d',' -f2 | tr -d ' ')
    GPU1_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | grep "^1," | cut -d',' -f2 | tr -d ' ')
    echo "[monitor $(date +%H:%M)] $REMAINING eval | GPU0: ${GPU0_MEM}MiB  GPU1: ${GPU1_MEM}MiB"
    if [ "$REMAINING" -eq 0 ]; then
        break
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "4-Way TextVQA — ALL DONE — $(date)"
echo "════════════════════════════════════════════════════════"

echo ""
echo "=== TextVQA Results Summary ==="
for d in "$OUTPUT_BASE/$TASK"/*/; do
    [ -d "$d" ] || continue
    tag=$(basename "$d")
    if ls "$d"/*_results.json >/dev/null 2>&1; then
        echo "  $tag: DONE"
    else
        echo "  $tag: MISSING"
    fi
done

echo ""
echo "=== Full Coverage Check ==="
for task in mmmu_val mme_local textvqa_val_local; do
    done_count=0 miss_count=0
    for d in "$OUTPUT_BASE/$task"/*/; do
        [ -d "$d" ] || continue
        if ls "$d"/*_results.json >/dev/null 2>&1; then
            ((done_count++))
        else
            ((miss_count++))
        fi
    done
    echo "  $task: $done_count done, $miss_count missing"
done