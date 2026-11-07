#!/usr/bin/env bash
# ============================================================================
# SEED-Bench-2 Full Serial Queue (A100-80GB × 2, 200GB RAM)
#
# Runs remaining 12 SEED-Bench-2 cells using the FULL dataset (24,371 samples).
# Serial execution to guarantee RAM safety (~87GB per process on 200GB machine).
#
# Cells: 12 remaining (rtn_w4 already done)
# Estimated: ~3-4h per cell → ~36-48h total
# ============================================================================
set -u

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "[FATAL] Cannot cd to MLLMQ"; exit 1; }

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval/seedbench_2_local
LOG_DIR=/nas3/kangjia/mllmq/output/logs
SCALE_DIR=/nas3/kangjia/mllmq/MLLMQ/scale_cache
mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

TASK="seedbench_2_local"
DEVICE="cuda:0"

# ── helper ──────────────────────────────────────────────────
is_done() {
    local tag="$1"
    local d="$OUTPUT_BASE/$tag"
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}

run_cell() {
    local method="$1" w_bit="$2" scale="$3"
    local tag
    if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi

    echo ""
    echo "============================================================"
    echo " CELL: $tag  (method=$method w_bit=$w_bit)"
    echo "============================================================"

    if is_done "$tag"; then
        echo "  [SKIP] $tag (done)"
        return 0
    fi

    local log="$LOG_DIR/seed_${tag}_$(date +%Y%m%d_%H%M%S).log"
    echo "  [START] $tag $(date '+%F %T')"

    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" --method "$method" --w_bit "$w_bit" --a_bit 16 \
        --tasks "$TASK" --device "$DEVICE" --batch_size 1 \
        --output_base "$OUTPUT_BASE" \
        ${scale:+--scale_path "$scale"} \
        > "$log" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "  [DONE]  $tag $(date '+%F %T')"
    else
        echo "  [FAIL]  $tag rc=$rc $(date '+%F %T')"
        # Log tail on failure for quick diagnosis
        echo "  --- log tail ---"
        tail -30 "$log"
        echo "  --- end tail ---"
    fi

    # Brief cooldown between cells
    echo "  [COOLDOWN] 30s …"
    sleep 30
    return $rc
}

# ── SEED-Bench-2 remaining cells (12 of 13; rtn_w4 done) ────
echo "============================================================"
echo " SEED-Bench-2 Full Serial Queue"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " RAM: $(free -h | grep Mem | awk '{print $2}')"
echo " Remaining cells: 12 (rtn_w4 already done)"
echo " Start: $(date '+%F %T')"
echo "============================================================"

# 1. fp16 baseline
run_cell fp16 16 ""

# 2. RTN
run_cell rtn 3 ""
run_cell rtn 8 ""

# 3. AWQ
run_cell awq 3 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w3a16.pt"
run_cell awq 4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"
run_cell awq 8 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"

# 4. MBQ
run_cell mbq 3 "$SCALE_DIR/mbq/internvl2_w3a16.pt"
run_cell mbq 4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"
run_cell mbq 8 "$SCALE_DIR/mbq/internvl2_w8a16.pt"

# 5. MLLMQ
run_cell mllmq 3 "$SCALE_DIR/mllmq/internvl2_w3a16.pt"
run_cell mllmq 4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"
run_cell mllmq 8 "$SCALE_DIR/mllmq/internvl2_w8a16.pt"

echo ""
echo "============================================================"
echo " SEED-Bench-2 Queue COMPLETE"
echo " End: $(date '+%F %T')"
echo "============================================================"

# ── summary ─────────────────────────────────────────────────
echo ""
echo "=== Final results count ==="
find "$OUTPUT_BASE" -name '*_results.json' -type f 2>/dev/null | wc -l
echo "results files found"