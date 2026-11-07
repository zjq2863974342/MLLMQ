#!/usr/bin/env bash
# ============================================================================
# Dual Staggered Queue: SEED-Bench-2 + MathVista
#   1 GPU (A800-80GB), 2 parallel eval processes
#   Staggered launch: Worker 1 starts 5 min after Worker 0
#   -> Model+apply_quant overlap avoided, only inference-phase overlap
#   -> VRAM: ~24GB (W0 inference) + ~24GB (W1 inference) = ~48GB << 80GB
#
# CRITICAL: ~300s stagger ensures W0 has finished model loading + apply_quant
#   before W1 starts loading. apply_quant is where VRAM peaks (scale cache
#   layers逐层.cuda()), so avoiding that overlap prevents OOM.
#
# Idempotent: skips cells with *_results.json already present
# ============================================================================
set -u

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "FATAL: Cannot cd to MLLQ"; exit 1; }

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
LOG_DIR=/nas3/kangjia/mllmq/output/logs
SCALE_DIR=/nas3/kangjia/mllmq/MLLMQ/scale_cache
mkdir -p "$LOG_DIR"

# CPU thread control - prevent per-process thread trampolines
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

is_done() {
    local d="$OUTPUT_BASE/$1/$2"
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}

run_cell() {
    local task="$1"
    local method="$2"
    local w_bit="$3"
    local scale="$4"
    local tag
    if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi

    if is_done "$task" "$tag"; then
        echo "  [SKIP] $task/$tag (already has results)"
        return 0
    fi

    local log="$LOG_DIR/${task}_${tag}_$(date +%Y%m%d_%H%M%S).log"
    local t0
    t0=$(date +%s)
    echo "  [START] $task/$tag  $(date +%H:%M:%S)  log: $(basename $log)"
    local vram_before
    vram_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')

    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" --method "$method" --w_bit "$w_bit" --a_bit 16 \
        --tasks "$task" --device cuda:0 --batch_size 1 \
        --output_base "$OUTPUT_BASE" \
        ${scale:+--scale_path "$scale"} \
        > "$log" 2>&1

    local rc=$?
    local t1
    t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    local vram_after
    vram_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ')

    if [ $rc -eq 0 ]; then
        echo "  [DONE]  $task/$tag  elapsed=${elapsed}s  VRAM: ${vram_before}MiB -> ${vram_after}MiB  $(date +%H:%M:%S)"
    else
        echo "  [FAIL]  $task/$tag  rc=$rc  elapsed=${elapsed}s  VRAM: ${vram_before}MiB -> ${vram_after}MiB  $(date +%H:%M:%S)"
        echo "  [FAIL]  Check log: $log"
    fi
    return $rc
}

process_item() {
    local item="$1"
    IFS='|' read -r task_name method w_bit scale_file <<< "$item"
    local scale=""
    if [ -n "$scale_file" ] && [ -f "$SCALE_DIR/$scale_file" ]; then
        scale="$SCALE_DIR/$scale_file"
    fi
    run_cell "$task_name" "$method" "$w_bit" "$scale"
}

# ================================================================
# QUEUE: Shared order between both workers.
# is_done() ensures no duplicates - whichever finishes first
# gets credit, the second worker just skips completed cells.
# This naturally load-balances.
# ================================================================

MATHVISTA_ITEMS=(
    "mathvista_testmini_local|fp16|16|"
    "mathvista_testmini_local|rtn|3|"
    "mathvista_testmini_local|rtn|4|"
    "mathvista_testmini_local|rtn|8|"
    "mathvista_testmini_local|mbq|3|mbq/internvl2_w3a16.pt"
    "mathvista_testmini_local|mbq|4|mbq/internvl2_w4a16.pt"
    "mathvista_testmini_local|mbq|8|mbq/internvl2_w8a16.pt"
    "mathvista_testmini_local|mllmq|3|mllmq/internvl2_w3a16.pt"
    "mathvista_testmini_local|mllmq|4|mllmq/internvl2_w4a16.pt"
    "mathvista_testmini_local|mllmq|8|mllmq/internvl2_w8a16.pt"
    "mathvista_testmini_local|awq|3|awq/internvl2_InternVL2-8B_w3a16.pt"
    "mathvista_testmini_local|awq|4|awq/internvl2_InternVL2-8B_w4a16.pt"
    "mathvista_testmini_local|awq|8|awq/internvl2_InternVL2-8B_w8a16.pt"
)

SEED_ITEMS=(
    "seedbench_2_local|fp16|16|"
    "seedbench_2_local|rtn|3|"
    "seedbench_2_local|rtn|4|"
    "seedbench_2_local|rtn|8|"
    "seedbench_2_local|mbq|3|mbq/internvl2_w3a16.pt"
    "seedbench_2_local|mbq|4|mbq/internvl2_w4a16.pt"
    "seedbench_2_local|mbq|8|mbq/internvl2_w8a16.pt"
    "seedbench_2_local|mllmq|3|mllmq/internvl2_w3a16.pt"
    "seedbench_2_local|mllmq|4|mllmq/internvl2_w4a16.pt"
    "seedbench_2_local|mllmq|8|mllmq/internvl2_w8a16.pt"
    "seedbench_2_local|awq|3|awq/internvl2_InternVL2-8B_w3a16.pt"
    "seedbench_2_local|awq|4|awq/internvl2_InternVL2-8B_w4a16.pt"
    "seedbench_2_local|awq|8|awq/internvl2_InternVL2-8B_w8a16.pt"
)

TOTAL_MV=${#MATHVISTA_ITEMS[@]}
TOTAL_SB=${#SEED_ITEMS[@]}
TOTAL=$((TOTAL_MV + TOTAL_SB))

ALL_ITEMS=("${MATHVISTA_ITEMS[@]}" "${SEED_ITEMS[@]}")

echo "============================================================"
echo " Dual Staggered Queue: SEED-Bench-2 + MathVista"
echo " Mode: 2 processes/GPU, staggered 300s apart"
echo " Stagger reason: avoid apply_quant VRAM peak overlap"
echo " W0 model load (~17000MiB) + W1 model load overlapped -> OOM"
echo " With 300s stagger: W0 in inference (~24000MiB) when W1 starts -> ~48000MiB OK"
echo " GPU:  $(nvidia-smi -L 2>/dev/null || echo 'n/a')"
echo " VRAM: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " Model: $MODEL_PATH"
echo " Start: $(date)"
echo "============================================================"
echo ""
echo "MathVista=$TOTAL_MV (~1h/cell) -> SEED-Bench-2=$TOTAL_SB (~7h/cell)"
echo "Total=$TOTAL cells, est. wall-clock: ~$((TOTAL / 2))h (2 workers)"
echo ""

# ── Worker function ──
run_worker() {
    local wid="$1"
    echo ""
    echo "########################################"
    echo " Worker $wid Started: $(date)"
    echo "########################################"

    local ok=0 fail=0 skip=0

    # Each worker processes ALL items. is_done() prevents duplicates.
    for item in "${ALL_ITEMS[@]}"; do
        process_item "$item"
        local rc=$?
        if [ $rc -eq 0 ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    done

    echo ""
    echo "Worker $wid Final: ok=$ok fail=$fail skip=$skip ($((ok+fail+skip)) total) $(date)"
}

# ── Launch Worker 0 immediately ──
run_worker 0 &
PID0=$!
echo ""
echo "Worker 0 launched (PID=$PID0) at $(date +%H:%M:%S)"
echo ""

# Wait until W0's model is fully loaded into VRAM and stabilized
# Before that, W0 is at peak VRAM (model load on CPU->GPU + apply_quant)
echo "Waiting 300s before launching Worker 1 (let W0 stabilize to inference ~24GB)..."
echo "VRAM before stagger: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ') MiB"
sleep 300
echo ""
echo "300s elapsed. Launching Worker 1 at $(date +%H:%M:%S)..."
echo "VRAM before W1: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' ') MiB"

run_worker 1 &
PID1=$!
echo "Worker 1 launched (PID=$PID1)."
echo ""
echo "Both workers running:"
echo "  W0 PID=$PID0  W1 PID=$PID1"
echo ""
echo "Monitor:"
echo "  watch -n 30 'nvidia-smi -q -d MEMORY,UTILIZATION'"
echo "  tail -f $LOG_DIR/*.log | head -50"
echo ""

# Wait for both
wait $PID0; EC0=$?
wait $PID1; EC1=$?

echo ""
echo "============================================================"
echo "ALL DONE! $(date)"
echo "Worker 0: exit=$EC0  Worker 1: exit=$EC1"
echo "============================================================"

# Check remaining missing
echo ""
echo "--- Remaining missing cells ---"
need_any=0
for item in "${ALL_ITEMS[@]}"; do
    IFS='|' read -r task_name method w_bit scale_file <<< "$item"
    tag="fp16"
    if [ "$method" != "fp16" ]; then tag="${method}_w${w_bit}a16"; fi
    if ! is_done "$task_name" "$tag"; then
        echo "  MISSING: $task_name/$tag"
        need_any=1
    fi
done

if [ $need_any -eq 0 ]; then
    echo "  (none - all cells completed!)"
fi
echo "============================================================"
