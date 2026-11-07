#!/usr/bin/env bash
# ============================================================================
# Run MME + TextVQA on InternVL2-8B for the 5 W4 cells (fp16/rtn/awq/mbq/mllmq).
# Prereq: W4 scale caches exist (awq/mbq) — see run_multi_benchmark.sh Phase 0.
# Usage:  bash scripts/run_core_bench_w4.sh
# Output: output/internvl2-8b/eval/{mme,textvqa_val}/<method>_w4a16/*_results.json
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJ_ROOT=$(pwd)
MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
SCALE_DIR="$PROJ_ROOT/scale_cache"
LOG_DIR=/nas3/kangjia/mllmq/output/logs
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/core_bench_w4_${TIMESTAMP}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

TASKS="mme,textvqa_val"

run_cell() {
    local method=$1 scale=$2
    local args="--method $method --w_bit 4 --a_bit 16"
    if [ -n "$scale" ]; then
        args="$args --scale_path $scale"
    fi
    echo ">>> [$(date +%H:%M)] $method W4A16  tasks=$TASKS"
    python eval_benchmarks.py \
        --model_path "$MODEL_PATH" \
        $args \
        --tasks "$TASKS" \
        --device cuda:0 --batch_size 1 \
        --output_base "$OUTPUT_BASE" 2>&1 | sed "s/^/  [$method] /"
}

echo "=============================================="
echo "Core W4 Benchmarks — $(date)   tasks=$TASKS"
echo "=============================================="

run_cell "fp16"  ""
run_cell "rtn"   ""
run_cell "awq"   "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"
run_cell "mbq"   "$SCALE_DIR/mbq/internvl2_w4a16.pt"
run_cell "mllmq" "$SCALE_DIR/mllmq/internvl2_w4a16.pt"

echo "=============================================="
echo "Core W4 Benchmarks DONE — $(date)"
echo "Log: $MAIN_LOG"
echo "=============================================="