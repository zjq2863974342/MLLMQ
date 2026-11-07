#!/usr/bin/env bash
# ============================================================================
# Sequential Queue: SEED-Bench-2 + MathVista
#   1 GPU (A800-80GB), 1 parallel eval process (max VRAM safety)
#   Idempotent: skips cells with *_results.json already present
#
# Strategy: MathVista first (fast ~1h/cell) → early signals + GPU util check
#           Then SEED-Bench-2 (slow ~6-8h/cell) → long haul
#
# VRAM safety: 1 process × ~24GB = well within 80GB A800 limit
#   apply_quant scale cache .cuda() peak is well within headroom
# ============================================================================
set -u

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "FATAL: Cannot cd to MLLMQ"; exit 1; }

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
LOG_DIR=/nas3/kangjia/mllmq/output/logs
SCALE_DIR=/nas3/kangjia/mllmq/MLLMQ/scale_cache
mkdir -p "$LOG_DIR"

# CPU thread control — prevent per-process thread trampolines
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

    # Log VRAM before
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

process_items() {
    local label="$1"
    shift
    local items=("$@")
    local i=0
    local n=${#items[@]}

    echo ""
    echo "============================================"
    echo "PHASE: $label ($n cells)"
    echo "============================================"

    for item in "${items[@]}"; do
        i=$((i+1))
        IFS='|' read -r task_name method w_bit scale_file <<< "$item"
        scale=""
        if [ -n "$scale_file" ] && [ -f "$SCALE_DIR/$scale_file" ]; then
            scale="$SCALE_DIR/$scale_file"
        fi

        local remaining=$(( n - i ))
        echo ""
        echo "--- [$i/$n] remaining=$remaining ---"

        run_cell "$task_name" "$method" "$w_bit" "$scale"
    done
}

# ═══════════════════════════════════════════════════════════════
# QUEUE DEFINITION
# MathVista first (fast ~1h/cell), then SEED-Bench-2 (slow ~6-8h/cell)
# ═══════════════════════════════════════════════════════════════

# MathVista cells
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

# SEED-Bench-2 cells
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

echo "============================================================"
echo " Sequential Queue: SEED-Bench-2 + MathVista"
echo " Mode: 1 process/GPU (max VRAM safety)"
echo " GPU:  $(nvidia-smi -L 2>/dev/null || echo 'n/a')"
echo " VRAM: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " Model: $MODEL_PATH"
echo " Start: $(date)"
echo "============================================================"
echo ""
echo "Queue: MathVista=$TOTAL_MV cells (~1h each) → SEED-Bench-2=$TOTAL_SB cells (~6-8h each)"
echo "Total: $TOTAL cells, est. wall-clock: ~$((TOTAL_MV * 1 + TOTAL_SB * 7))h"
echo ""

# ── Phase 1: MathVista (fast cells) ──
process_items "MathVista (quick results + GPU util check)" "${MATHVISTA_ITEMS[@]}"

# ── Phase 2: SEED-Bench-2 (slow cells) ──
process_items "SEED-Bench-2 (long runs, ~7h each)" "${SEED_ITEMS[@]}"

# ── Final summary ──
echo ""
echo "============================================================"
echo "ALL DONE! $(date)"
echo "  Total cells processed (incl. skips)"
echo "  MathVista:   $TOTAL_MV"
echo "  SEED-Bench:  $TOTAL_SB"
echo "============================================================"

# Check remaining missing
echo ""
echo "--- Remaining missing cells ---"
need_any=0
all_items=("${MATHVISTA_ITEMS[@]}" "${SEED_ITEMS[@]}")
for item in "${all_items[@]}"; do
    IFS='|' read -r task_name method w_bit scale_file <<< "$item"
    tag="fp16"
    if [ "$method" != "fp16" ]; then tag="${method}_w${w_bit}a16"; fi
    if ! is_done "$task_name" "$tag"; then
        echo "  MISSING: $task_name/$tag"
        need_any=1
    fi
done

if [ $need_any -eq 0 ]; then
    echo "  (none — all cells completed!)"
fi
echo "============================================================"