#!/usr/bin/env bash
# ============================================================================
# Repair queue — fill ALL missing MME + TextVQA W4/W8/W3 cells.
# GPU gate: at most 2 eval processes concurrently (hard limit to avoid OOM).
# Idempotent: skips cells that already have *_results.json.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJ_ROOT=$(pwd)
cd "$PROJ_ROOT/MLLMQ" || exit 1

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
LOG_DIR=/nas3/kangjia/mllmq/output/logs
mkdir -p "$LOG_DIR"
export HF_DATASETS_OFFLINE=1
SCALE_DIR="$PROJ_ROOT/MLLMQ/scale_cache"

eval_count() { pgrep -f "eval_benchmarks.py.*cuda:0" 2>/dev/null | wc -l; }
is_done() {
    local task=$1 tag=$2
    local d="$OUTPUT_BASE/$task/$tag"
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}
gpu_wait() {
    while [ "$(eval_count)" -ge 2 ]; do
        sleep 60
    done
}

run_cell() {
    local task=$1 method=$2 w_bit=$3 scale=$4
    local tag
    if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi
    if is_done "$task" "$tag"; then
        echo ">>> SKIP $task/$tag — already done"
        return 0
    fi
    gpu_wait
    local log="$LOG_DIR/repair_${tag}_${task}_$(date +%Y%m%d_%H%M%S).log"
    echo ">>> [$task/$tag] start $(date +%H:%M) — log $log"
    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" --method "$method" --w_bit "$w_bit" --a_bit 16 \
        --tasks "$task" --device cuda:0 --batch_size 1 \
        --output_base "$OUTPUT_BASE" \
        ${scale:+--scale_path "$scale"} \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo ">>> [$task/$tag] OK $(date +%H:%M)"
    else
        echo ">>> [$task/$tag] FAILED rc=$rc — see $log"
    fi
    return $rc
}

echo "=============================================="
echo "Repair Queue — $(date)"
echo "Running: $(eval_count) eval process(es)"
echo "=============================================="

# ── 1. MME W8 missing (awq/mbq/mllmq) ──
run_cell mme_local awq   8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"
run_cell mme_local mbq   8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"
run_cell mme_local mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"

# ── 2. TextVQA W4 (fp16 + all 4 methods; rtn W4 之前被 kill 无结果，需重跑) ──
run_cell textvqa_val_local fp16  16 ""
run_cell textvqa_val_local rtn   4 ""
run_cell textvqa_val_local awq   4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"
run_cell textvqa_val_local mbq   4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"
run_cell textvqa_val_local mllmq 4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"

echo "=============================================="
echo "Repair Queue DONE — $(date)"
echo "=============================================="