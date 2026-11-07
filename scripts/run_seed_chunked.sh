#!/usr/bin/env bash
# ============================================================================
# SEED-Bench-2 Chunked Queue
#
# Problem: Full SEED-Bench-2 (24,371 samples) loads ~70 GB into RAM.
#          Model (~17 GB) + dataset (~70 GB) = ~87 GB peak per process.
#          With 200 GB total RAM, 1 process fits but 2 processes OOM.
#
# Solution: Split dataset into 5 chunks (~4,874 samples each).
#           Each chunk: ~15 GB dataset + ~17 GB model = ~32 GB peak.
#           5 chunks × 12 cells = 60 sub-tasks, sequential, no risk.
#
# Total: 12 cells × 5 chunks = 60 runs, ~1.3h/run including model load
#        Estimated total wall-clock: ~78 hours (~3.3 days) on 1 GPU
# ============================================================================
set -u

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "Cannot cd to MLLMQ"; exit 1; }

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval/seedbench_2_local
LOG_DIR=/nas3/kangjia/mllmq/output/logs
SCALE_DIR=/nas3/kangjia/mllmq/MLLMQ/scale_cache
AGGREGATE_SCRIPT=/nas3/kangjia/mllmq/scripts/aggregate_seed_results.py
mkdir -p "$LOG_DIR"

export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

TASK_PREFIX="seedbench_2_local_chunk"
NUM_CHUNKS=5
DEVICE="${CUDA_DEVICE:-cuda:0}"
LOG_TAG="${LOG_TAG:-}"

# Cooldown between cells (seconds) — set via env to stagger GPU load
COOLDOWN="${COOLDOWN:-5}"

is_done() {
    local task="$1" tag="$2"
    local d="$OUTPUT_BASE/${task}/$tag"
    if [ ! -d "$d" ]; then return 1; fi
    local f
    for f in "$d"/*_results.json; do
        [ -f "$f" ] || continue
        # Reject smoke-test results (limit < full dataset)
        if python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
limit = d.get('config', {}).get('limit')
if limit is not None and limit < 100:
    exit(1)  # smoke test, not full
" 2>/dev/null; then
            return 0  # full result found
        fi
    done
    return 1  # no valid full results
}

run_cell() {
    local task="$1" method="$2" w_bit="$3" scale="$4"
    local tag
    if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi

    if is_done "$task" "$tag"; then
        echo "  [SKIP] $task/$tag (done)"
        return 0
    fi

    local log="$LOG_DIR/${task}_${tag}_$(date +%Y%m%d_%H%M%S).log"
    echo "  [START] $task/$tag $(date +%H:%M:%S)"

    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" --method "$method" --w_bit "$w_bit" --a_bit 16 \
        --tasks "$task" --device "$DEVICE" --batch_size 1 \
        --output_base "$OUTPUT_BASE" \
        ${scale:+--scale_path "$scale"} \
        > "$log" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "  [DONE]  $task/$tag $(date +%H:%M:%S)"
    else
        echo "  [FAIL]  $task/$tag rc=$rc $(date +%H:%M:%S)"
    fi
    return $rc
}

#### CELL_GROUP: "q1" = GPU0 half, "q2" = GPU1 half ####
ALL_CELLS=(
    # Q1 (GPU0): fp16 + AWQ variants
    "fp16|16|"
    "rtn|3|"
    "rtn|8|"
    "awq|3|awq/internvl2_InternVL2-8B_w3a16.pt"
    "awq|4|awq/internvl2_InternVL2-8B_w4a16.pt"
    "awq|8|awq/internvl2_InternVL2-8B_w8a16.pt"
    # Q2 (GPU1): MBQ + MLLMQ variants
    "mbq|3|mbq/internvl2_w3a16.pt"
    "mbq|4|mbq/internvl2_w4a16.pt"
    "mbq|8|mbq/internvl2_w8a16.pt"
    "mllmq|3|mllmq/internvl2_w3a16.pt"
    "mllmq|4|mllmq/internvl2_w4a16.pt"
    "mllmq|8|mllmq/internvl2_w8a16.pt"
)

# Split: CELL_SUBSET=q1 → first 6, CELL_SUBSET=q2 → last 6
CELLS=()
if [ "${CELL_SUBSET:-}" = "q2" ]; then
    for ((i=6; i<12; i++)); do CELLS+=("${ALL_CELLS[$i]}"); done
else
    for ((i=0; i<6; i++)); do CELLS+=("${ALL_CELLS[$i]}"); done
fi

echo "============================================================"
echo " SEED-Bench-2 Chunked Queue (${CELL_SUBSET:-q1})"
echo " Cells: ${#CELLS[@]} × $NUM_CHUNKS chunks = $((${#CELLS[@]} * NUM_CHUNKS)) runs"
echo " Device: $DEVICE"
echo " GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo " Start: $(date)"
echo "============================================================"

total_ok=0 total_fail=0 total_skip=0

for cell in "${CELLS[@]}"; do
    method="" w_bit="" scale_file="" tag="" scale=""
    IFS='|' read -r method w_bit scale_file <<< "$cell"
    if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi
    scale=""
    if [ -n "$scale_file" ] && [ -f "$SCALE_DIR/$scale_file" ]; then
        scale="$SCALE_DIR/$scale_file"
    fi

    echo ""
    echo "============================================================"
    echo " CELL: $tag  (method=$method w_bit=$w_bit)"
    echo "============================================================"

    cell_ok=0 cell_fail=0 cell_skip=0

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        task="${TASK_PREFIX}${chunk}"

        if is_done "$task" "$tag"; then
            echo "  [SKIP] $task/$tag (done)"
            cell_skip=$((cell_skip + 1))
            continue
        fi

        run_cell "$task" "$method" "$w_bit" "$scale"
        rc=$?

        if [ $rc -eq 0 ]; then
            cell_ok=$((cell_ok + 1))
        else
            cell_fail=$((cell_fail + 1))
            echo "  ⚠️  Chunk $chunk failed. Continuing to next chunk..."
        fi

        # Brief pause between chunks
        sleep "$COOLDOWN"
    done

    echo "  Cell $tag: ok=$cell_ok fail=$cell_fail skip=$cell_skip"

    # Aggregate chunk results into single combined results.json
    if [ $cell_ok -gt 0 ] || [ $cell_skip -gt 0 ]; then
        echo "  [AGGREGATE] Combining chunk results for $tag ..."
        python3 "$AGGREGATE_SCRIPT" \
            --input_dir "$OUTPUT_BASE/$tag" \
            --output "$OUTPUT_BASE/$tag/aggregated_results.json" \
            > "$LOG_DIR/aggregate_${tag}_$(date +%Y%m%d_%H%M%S).log" 2>&1
        if [ $? -eq 0 ]; then
            echo "  [AGGREGATE] $tag: combined results saved"
        else
            echo "  [AGGREGATE] $tag: FAILED (see log)"
        fi
    fi

    total_ok=$((total_ok + cell_ok))
    total_fail=$((total_fail + cell_fail))
    total_skip=$((total_skip + cell_skip))

    echo "  [COOLDOWN] ${COOLDOWN}s …"
    sleep "$COOLDOWN"
done

echo ""
echo "============================================================"
echo " ALL DONE! $(date)"
echo " Total: ok=$total_ok fail=$total_fail skip=$total_skip"
echo "============================================================"

# Final aggregate check
echo ""
echo "=== Missing chunks check ==="
for cell in "${CELLS[@]}"; do
    IFS='|' read -r method w_bit scale_file <<< "$cell"
    if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi
    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        task="${TASK_PREFIX}${chunk}"
        if ! is_done "$task" "$tag"; then
            echo "  MISSING: $task/$tag"
        fi
    done
done
echo "=== Check complete ==="