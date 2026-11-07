#!/usr/bin/env bash
# ============================================================================
# W8 / W3 MME + TextVQA queue — run in PARALLEL with W4 queue on cuda:0.
# Each cell ~2.5h, total ~10 cells × 2.5h = ~25h.
# Idempotent: skips cells with *_results.json already present.
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

is_done() {
    local task=$1 tag=$2
    local d="$OUTPUT_BASE/$task/$tag"
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}

run_cell() {
    local task=$1 method=$2 w_bit=$3 scale=$4
    local tag="${method}_w${w_bit}a16"
    if is_done "$task" "$tag"; then
        echo ">>> SKIP $task/$tag — already done"
        return 0
    fi
    local log="$LOG_DIR/${tag}_${task}_$(date +%Y%m%d_%H%M%S).log"
    echo ">>> [$task/$tag] start $(date +%H:%M) — log $log"
    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" \
        --method "$method" --w_bit "$w_bit" --a_bit 16 \
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
    # brief cool-down to let parallel W4 queue breathe
    sleep 30
    return $rc
}

echo "=============================================="
echo "W8/W3 Queue — $(date)"
echo "=============================================="

# ── MME W8 (RTN / AWQ / MBQ / MLLMQ) ──
run_cell mme_local rtn   8 ""
run_cell mme_local awq   8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"
run_cell mme_local mbq   8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"
run_cell mme_local mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"

# ── TextVQA W8 (RTN / AWQ / MBQ / MLLMQ) ──
run_cell textvqa_val_local rtn   8 ""
run_cell textvqa_val_local awq   8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"
run_cell textvqa_val_local mbq   8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"
run_cell textvqa_val_local mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"

# ── MME W3 (RTN / AWQ / MBQ / MLLMQ) ──
run_cell mme_local rtn   3 ""
run_cell mme_local awq   3 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w3a16.pt"
run_cell mme_local mbq   3 "$SCALE_DIR/mbq/internvl2_w3a16.pt"
run_cell mme_local mllmq 3 "$SCALE_DIR/mllmq/internvl2_w3a16.pt"

# ── TextVQA W3 (RTN / AWQ / MBQ / MLLMQ) ──
run_cell textvqa_val_local rtn   3 ""
run_cell textvqa_val_local awq   3 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w3a16.pt"
run_cell textvqa_val_local mbq   3 "$SCALE_DIR/mbq/internvl2_w3a16.pt"
run_cell textvqa_val_local mllmq 3 "$SCALE_DIR/mllmq/internvl2_w3a16.pt"

echo "=============================================="
echo "W8/W3 Queue DONE — $(date)"
echo "=============================================="