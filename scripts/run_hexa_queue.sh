#!/usr/bin/env bash
# ============================================================================
# 6-way parallel queue — 3 workers per GPU (CUDA_VISIBLE_DEVICES isolation)
# Idempotent: skips cells with *_results.json already present.
# Total: 19 cells (4 MMMU W4 + 2 MME W8 + 13 TextVQA), ~6-10h wall-clock.
#
# CPU/GPU optimizations:
#   - OMP_NUM_THREADS=4 per process (avoid 768-thread storm on 128 cores)
#   - expandable_segments to reduce CUDA memory fragmentation
#   - Staggered worker launch (30s gap) to avoid loading-phase VRAM overlap
# ============================================================================
set -uo pipefail

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || exit 1

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
LOG_DIR=/nas3/kangjia/mllmq/output/logs
SCALE_DIR="$PROJ_ROOT/MLLMQ/scale_cache"
mkdir -p "$LOG_DIR"

# ── CPU/GPU optimization env vars ──
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
    local log="$LOG_DIR/hexa_${tag}_${task}_$(date +%Y%m%d_%H%M%S).log"
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
    fi
    return $rc
}

# ═══════════════════════════════════════════════════════════════
# Worker definitions (balanced across 6 workers × 2 GPUs)
# MMMU ~0.75h/cell,  MME ~2.5h/cell,  TextVQA ~2.5h/cell
# ═══════════════════════════════════════════════════════════════

worker_gpu0_w0() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[W0 GPU0] MMMU RTN W4 → MME MBQ W8 → TextVQA MBQ W8  $(date +%H:%M)"
    run_cell mmmu_val rtn    4 ""                                                 # 0.75h
    run_cell mme_local mbq   8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"               # 2.5h
    run_cell textvqa_val_local mbq 8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"         # 2.5h
    echo "[W0 GPU0] DONE $(date +%H:%M)"
}

worker_gpu0_w1() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[W1 GPU0] MMMU AWQ W4 → MME MLLMQ W8 → TextVQA MLLMQ W8  $(date +%H:%M)"
    run_cell mmmu_val awq    4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"   # 0.75h
    run_cell mme_local mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"              # 2.5h
    run_cell textvqa_val_local mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"      # 2.5h
    echo "[W1 GPU0] DONE $(date +%H:%M)"
}

worker_gpu0_w2() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[W2 GPU0] MMMU MBQ W4 → TextVQA fp16 → TextVQA RTN W4  $(date +%H:%M)"
    run_cell mmmu_val mbq    4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"                # 0.75h
    run_cell textvqa_val_local fp16 16 ""                                         # 2.5h
    run_cell textvqa_val_local rtn   4 ""                                         # 2.5h
    echo "[W2 GPU0] DONE $(date +%H:%M)"
}

worker_gpu1_w3() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[W3 GPU1] MMMU MLLMQ W4 → TextVQA AWQ W4 → TextVQA MBQ W4  $(date +%H:%M)"
    run_cell mmmu_val mllmq  4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"              # 0.75h
    run_cell textvqa_val_local awq   4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"  # 2.5h
    run_cell textvqa_val_local mbq   4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"        # 2.5h
    echo "[W3 GPU1] DONE $(date +%H:%M)"
}

worker_gpu1_w4() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[W4 GPU1] TextVQA MLLMQ W4 → TextVQA RTN W8 → TextVQA AWQ W8  $(date +%H:%M)"
    run_cell textvqa_val_local mllmq 4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"      # 2.5h
    run_cell textvqa_val_local rtn   8 ""                                         # 2.5h
    run_cell textvqa_val_local awq   8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"  # 2.5h
    echo "[W4 GPU1] DONE $(date +%H:%M)"
}

worker_gpu1_w5() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[W5 GPU1] TextVQA W3 column (4 cells)  $(date +%H:%M)"
    run_cell textvqa_val_local rtn   3 ""                                         # 2.5h
    run_cell textvqa_val_local awq   3 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w3a16.pt"   # 2.5h
    run_cell textvqa_val_local mbq   3 "$SCALE_DIR/mbq/internvl2_w3a16.pt"        # 2.5h
    run_cell textvqa_val_local mllmq 3 "$SCALE_DIR/mllmq/internvl2_w3a16.pt"      # 2.5h
    echo "[W5 GPU1] DONE $(date +%H:%M)"
}

echo "════════════════════════════════════════════════════════"
echo "6-Way Parallel Queue — $(date)"
echo "GPU0: 3 workers   GPU1: 3 workers"
echo "CPU/Mem: OMP_NUM_THREADS=4 expandable_segments=True"
echo "Total: 19 cells (4 MMMU + 2 MME + 13 TextVQA)"
echo "════════════════════════════════════════════════════════"

# ── Staggered launch (30s gap) to avoid loading-phase VRAM overlap ──
echo "[launcher] Starting GPU0 workers..."
worker_gpu0_w0 > "$LOG_DIR/hexa_w0_gpu0.log" 2>&1 &
sleep 30
worker_gpu0_w1 > "$LOG_DIR/hexa_w1_gpu0.log" 2>&1 &
sleep 30
worker_gpu0_w2 > "$LOG_DIR/hexa_w2_gpu0.log" 2>&1 &

echo "[launcher] Starting GPU1 workers..."
worker_gpu1_w3 > "$LOG_DIR/hexa_w3_gpu1.log" 2>&1 &
sleep 30
worker_gpu1_w4 > "$LOG_DIR/hexa_w4_gpu1.log" 2>&1 &
sleep 30
worker_gpu1_w5 > "$LOG_DIR/hexa_w5_gpu1.log" 2>&1 &

echo "[launcher] All 6 workers launched. Main loop monitoring..."
echo "Logs: $LOG_DIR/hexa_w*.log"

# ── Monitor: log GPU status every 5 min ──
while true; do
    sleep 300
    REMAINING=$(pgrep -fc "eval_benchmarks.py" 2>/dev/null || echo 0)
    echo "[monitor $(date +%H:%M)] $REMAINING eval process(es) running"
    if [ "$REMAINING" -eq 0 ]; then
        break
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "6-Way Queue — ALL DONE — $(date)"
echo "════════════════════════════════════════════════════════"

# Quick summary
echo ""
echo "=== Results Summary ==="
for task in mmmu_val mme_local textvqa_val_local; do
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