#!/bin/bash
# ============================================================================
# Multi-benchmark evaluation for InternVL2-8B quantized models.
#
# Benchmarks: MME, TextVQA, SEED-Bench 2, MathVista (+ MMMU for completeness)
# Methods: fp16, RTN, AWQ, MBQ, MLLMQ
# Bit widths: W4 (core), W3/W8 (supplementary)
#
# Usage:  bash scripts/run_multi_benchmark.sh
#
# Output: output/internvl2-8b/eval/<benchmark>/<method>_w<n>a16/
# ============================================================================

set -o pipefail

cd "$(dirname "$0")/.." || exit 1
PROJ_ROOT=$(pwd)
MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
SCALE_DIR="$PROJ_ROOT/scale_cache"
LOG_DIR=/nas3/kangjia/mllmq/output/logs
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/multi_bench_${TIMESTAMP}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=============================================="
echo "Multi-Benchmark Eval — started at $(date)"
echo "Output base: $OUTPUT_BASE"
echo "Log:         $MAIN_LOG"
echo "=============================================="

# ── Helpers ──
run_eval() {
    local method=$1 w_bit=$2 scale_path=$3 tasks=$4
    local tag
    if [ "$method" = "fp16" ]; then
        tag="fp16"
    else
        tag="${method}_w${w_bit}"
    fi

    echo ">>> [$(date +%H:%M)] $tag  tasks=$tasks"

    python eval_benchmarks.py \
        --model_path "$MODEL_PATH" \
        --method "$method" \
        --w_bit "$w_bit" \
        ${scale_path:+--scale_path "$scale_path"} \
        --tasks "$tasks" \
        --output_base "$OUTPUT_BASE" \
        --device cuda:0 \
        --batch_size 1 \
        2>&1 | sed "s/^/  [$tag] /"

    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "  [!] $tag FAILED (exit=$rc) — continuing"
        return 1
    fi
    echo "  [OK] $tag completed at $(date +%H:%M)"
    return 0
}

generate_scale() {
    local method=$1 w_bit=$2 scale_path=$3
    if [ -f "$scale_path" ]; then
        echo "[scale] $scale_path already exists, skipping"
        return 0
    fi
    echo "[scale] Generating $method W${w_bit}A16 ..."
    python main_quant_vl.py \
        --model_type internvl2 \
        --model_path "$MODEL_PATH" \
        --device cuda:0 \
        --calib_data coco \
        --data_path /nas3/kangjia/mllmq/data/coco/sharegpt4v_coco_calib.json \
        --image_folder /nas3/kangjia/mllmq/data/coco/train2017 \
        --n_samples 128 \
        --method "$method" \
        --w_bit "$w_bit" --a_bit 16 --w_group 128 \
        --run_process \
        --scale_path "$scale_path" \
        --output_path /tmp/mllmq_gen_scale_${method}_w${w_bit} \
        2>&1 | sed "s/^/  [gen:$method:w${w_bit}] /"
    return ${PIPESTATUS[0]}
}

# ── Task list ──
CORE_TASKS="mmmu_val,mme,textvqa_val,seedbench_2,mathvista_testmini"
W4_TASKS="mme,textvqa_val,seedbench_2,mathvista_testmini"  # MMMU already done for W4

# ════════════════════════════════════════════════════════════
# Phase 0: Generate missing W4 scale caches
# ════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 0: Generate missing scale caches ==="

generate_scale awq 4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"
generate_scale mbq 4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"

# ════════════════════════════════════════════════════════════
# Phase 1: fp16 baseline (all 5 benchmarks)
# ════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 1: fp16 baseline ==="
run_eval fp16 16 "" "$CORE_TASKS"

# ════════════════════════════════════════════════════════════
# Phase 2: W4A16 — the core comparison (4 methods)
# ════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 2: W4A16 (core comparison) ==="

run_eval rtn   4 ""                                                         "$W4_TASKS"
run_eval awq   4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"          "$W4_TASKS"
run_eval mbq   4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"                        "$W4_TASKS"
run_eval mllmq 4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"                      "$W4_TASKS"

# ════════════════════════════════════════════════════════════
# Phase 3: W8A16 — upper bound (methods with existing caches)
# ════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 3: W8A16 (upper bound) ==="

run_eval rtn   8 ""                                                         "$W4_TASKS"
run_eval awq   8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"          "$W4_TASKS"
run_eval mbq   8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"                        "$W4_TASKS"
run_eval mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"                      "$W4_TASKS"

# ════════════════════════════════════════════════════════════
# Phase 4: W3A16 — lower bound (methods with existing caches)
# ════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 4: W3A16 (lower bound) ==="

run_eval rtn   3 ""                                                         "$W4_TASKS"
run_eval awq   3 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w3a16.pt"          "$W4_TASKS"
run_eval mbq   3 "$SCALE_DIR/mbq/internvl2_w3a16.pt"                        "$W4_TASKS"
run_eval mllmq 3 "$SCALE_DIR/mllmq/internvl2_w3a16.pt"                      "$W4_TASKS"

echo ""
echo "=============================================="
echo "Batch completed at $(date)"
echo "Log: $MAIN_LOG"
echo "=============================================="

# Quick summary
echo ""
echo "=== Quick Summary ==="
for benchmark in mmmu_val mme textvqa_val seedbench_2 mathvista_testmini; do
    echo "  $benchmark:"
    for d in "$OUTPUT_BASE/$benchmark"/*/; do
        [ -d "$d" ] || continue
        tag=$(basename "$d")
        latest=$(ls -t "$d"/*_results.json 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            score=$(python3 -c "
import json
with open('$latest') as f:
    d = json.load(f)
results = d.get('results',{}).get('$benchmark',{})
# Find the accuracy metric
for k in sorted(results.keys()):
    if 'acc' in k.lower() or 'perc' in k.lower():
        print(f'{results[k]:.4f}')
        break
" 2>/dev/null)
            echo "    $tag: $score"
        fi
    done
done