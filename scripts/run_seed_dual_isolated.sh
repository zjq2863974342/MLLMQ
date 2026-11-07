#!/usr/bin/env bash
# ============================================================================
# SEED-Bench-2 Dual Isolated Queue (2× A800-80GB, 200GB RAM)
#
# Fixes the CUDA OOM issue by:
#   1. CUDA_VISIBLE_DEVICES isolation — each queue sees only its own GPU
#   2. Staggered launch — Q2 waits for Q1's first chunk to FINISH before
#      starting, ensuring Q1 model load+apply_quant is complete and VRAM
#      has stabilized before Q2 begins its model load.
#
# Cells: 12 cells × 5 chunks = 60 sub-tasks
# Already done: fp16 chunk0-1, block_w3a16 chunk0-1 (4/60)
# Remaining: 56 sub-tasks
#
# Est. wall-clock: ~56 × 1.5h/chunk ÷ 2 queues ≈ 42 hours
# ============================================================================
set -u

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "[FATAL] Cannot cd to MLLMQ"; exit 1; }

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval/seedbench_2_local
LOG_DIR=/nas3/kangjia/mllmq/output/logs
SCALE_DIR=/nas3/kangjia/mllmq/MLLMQ/scale_cache
AGGREGATE_SCRIPT=/nas3/kangjia/mllmq/scripts/aggregate_seed_results.py
mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# ── Environment ──────────────────────────────────────────────
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ── Config ───────────────────────────────────────────────────
NUM_CHUNKS=5
TASK_PREFIX="seedbench_2_local_chunk"
COOLDOWN=5          # seconds between chunks
STAGGER_INTERVAL=30 # seconds between stagger polls
Q1_READY_FILE="/tmp/seed_q1_model_loaded.$$"  # handshake marker
rm -f "$Q1_READY_FILE"  # clean stale marker

# ── Queue Definitions ────────────────────────────────────────
# Format: "method|w_bit|scale_file_relpath"
Q1_CELLS=(           # GPU 0
    "fp16|16|"
    "rtn|3|"
    "rtn|8|"
    "awq|3|awq/internvl2_InternVL2-8B_w3a16.pt"
    "awq|4|awq/internvl2_InternVL2-8B_w4a16.pt"
    "awq|8|awq/internvl2_InternVL2-8B_w8a16.pt"
)

Q2_CELLS=(           # GPU 1
    "mbq|3|block/internvl2_w3a16.pt"
    "mbq|4|block/internvl2_w4a16.pt"
    "mbq|8|block/internvl2_w8a16.pt"
    "mllmq|3|mllmq/internvl2_w3a16.pt"
    "mllmq|4|mllmq/internvl2_w4a16.pt"
    "mllmq|8|mllmq/internvl2_w8a16.pt"
)

# ── Helpers ──────────────────────────────────────────────────
get_tag() {
    local method="$1" w_bit="$2"
    [ "$method" = "fp16" ] && echo "fp16" || echo "${method}_w${w_bit}a16"
}

is_done() {
    local task="$1" tag="$2"
    local d="$OUTPUT_BASE/${task}/$tag"
    if [ ! -d "$d" ]; then return 1; fi
    # Check for valid results.json (not smoke test)
    local f
    for f in "$d"/*_results.json; do
        [ -f "$f" ] || continue
        python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
limit = d.get('config', {}).get('limit')
if limit is not None and limit < 100:
    exit(1)
" 2>/dev/null && return 0
    done
    return 1
}

run_chunk() {
    local task="$1" method="$2" w_bit="$3" scale="$4" tag="$5" gpu="$6"

    if is_done "$task" "$tag"; then
        echo "  [$gpu][SKIP] $task/$tag (done)"
        return 0
    fi

    local log="$LOG_DIR/${task}_${tag}_$(date +%Y%m%d_%H%M%S).log"
    echo "  [$gpu][START] $task/$tag $(date +%H:%M:%S)"

    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" --method "$method" --w_bit "$w_bit" --a_bit 16 \
        --tasks "$task" --device cuda:0 --batch_size 1 \
        --output_base "$OUTPUT_BASE" \
        ${scale:+--scale_path "$scale"} \
        > "$log" 2>&1 &
    local py_pid=$!

    # ── Handshake: once Q1's first actual chunk shows inference progress,
    #     signal Q2 that the model is loaded and VRAM is stable ──
    if [ "$gpu" = "GPU0" ] && [ ! -f "$Q1_READY_FILE" ]; then
        echo "  [$gpu] Waiting for model load + first 10 inferences before signaling Q2..."
        local poll_count=0
        while [ $poll_count -lt 60 ]; do  # 10 min max wait
            if grep -q "Model Responding:.*10/" "$log" 2>/dev/null; then
                touch "$Q1_READY_FILE"
                echo "  [$gpu][SIGNAL] Q1 model loaded and inferring. Q2 cleared to launch!"
                break
            fi
            if ! kill -0 $py_pid 2>/dev/null; then
                # Process died early — signal anyway so Q2 doesn't wait forever
                touch "$Q1_READY_FILE"
                echo "  [$gpu][SIGNAL] Q1 process ended early. Signaling Q2 anyway."
                break
            fi
            sleep 10
            poll_count=$((poll_count + 1))
        done
        if [ ! -f "$Q1_READY_FILE" ]; then
            touch "$Q1_READY_FILE"
            echo "  [$gpu][SIGNAL] Timeout. Signaling Q2."
        fi
    fi

    wait $py_pid
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "  [$gpu][DONE]  $task/$tag $(date +%H:%M:%S)"
    else
        echo "  [$gpu][FAIL]  $task/$tag rc=$rc $(date +%H:%M:%S)"
        echo "  [$gpu]  log: $log"
        tail -5 "$log" | sed 's/^/  [log-tail] /'
    fi
    return $rc
}

run_cell() {
    local method="$1" w_bit="$2" scale_file="$3" gpu="$4"
    local tag
    tag=$(get_tag "$method" "$w_bit")

    local scale=""
    if [ -n "$scale_file" ] && [ -f "$SCALE_DIR/$scale_file" ]; then
        scale="$SCALE_DIR/$scale_file"
    fi

    echo ""
    echo "=== [$gpu] CELL: $tag (method=$method w_bit=$w_bit) ==="

    local ok=0 fail=0 skip=0

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local task="${TASK_PREFIX}${chunk}"
        if is_done "$task" "$tag"; then
            echo "  [$gpu][SKIP] $task/$tag (done)"
            skip=$((skip + 1))
            continue
        fi
        run_chunk "$task" "$method" "$w_bit" "$scale" "$tag" "$gpu"
        local rc=$?
        case $rc in
            0) ok=$((ok + 1)) ;;
            *) fail=$((fail + 1))
               echo "  [$gpu] ⚠️  Chunk $chunk failed, continuing..."
               ;;
        esac
        sleep "$COOLDOWN"
    done

    echo "  [$gpu] Cell $tag: ok=$ok fail=$fail skip=$skip"

    # Aggregate if any chunks succeeded
    if [ $ok -gt 0 ] || [ $skip -gt 0 ]; then
        echo "  [$gpu][AGGREGATE] $tag ..."
        python3 "$AGGREGATE_SCRIPT" \
            --input_dir "$OUTPUT_BASE/$tag" \
            --output "$OUTPUT_BASE/$tag/aggregated_results.json" \
            > "$LOG_DIR/aggregate_${tag}_$(date +%Y%m%d_%H%M%S).log" 2>&1
        if [ $? -eq 0 ]; then
            echo "  [$gpu][AGGREGATE] $tag: OK"
        else
            echo "  [$gpu][AGGREGATE] $tag: FAILED"
        fi
    fi

    sleep "$COOLDOWN"
}

# ── Queue Runner ─────────────────────────────────────────────
run_queue() {
    local gpu_label="$1" gpu_id="$2"
    shift 2
    local cells=("$@")

    export CUDA_VISIBLE_DEVICES="$gpu_id"
    echo ""
    echo "############################################################"
    echo " QUEUE $gpu_label (CUDA_VISIBLE_DEVICES=$gpu_id)"
    echo " Cells: ${#cells[@]} × $NUM_CHUNKS chunks"
    echo " Start: $(date '+%F %T')"
    echo " GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
    echo "############################################################"

    for cell in "${cells[@]}"; do
        IFS='|' read -r method w_bit scale_file <<< "$cell"
        run_cell "$method" "$w_bit" "$scale_file" "$gpu_label"
    done

    echo ""
    echo "QUEUE $gpu_label DONE! $(date '+%F %T')"
}

# ── Stagger Logic ────────────────────────────────────────────
wait_for_q1_ready() {
    # Wait until Q1 has loaded its model, run apply_quant (if any),
    # and completed 10+ inferences — VRAM is now stable.
    # Q1 signals readiness by creating $Q1_READY_FILE.
    echo ""
    echo ">>> Waiting for Q1 to signal model-loaded (polling $Q1_READY_FILE) ..."
    echo ">>> Q1 will signal after model load + 10 inferences complete."
    echo ">>> This prevents VRAM peak overlap during model loading."

    local max_wait=1800  # 30 min timeout
    local waited=0

    while [ $waited -lt $max_wait ]; do
        if [ -f "$Q1_READY_FILE" ]; then
            echo ">>> Q1 signaled READY after ${waited}s. Launching Q2 now."
            rm -f "$Q1_READY_FILE"  # clean up
            return 0
        fi
        # Check if Q1 died
        if ! kill -0 "$Q1_PID" 2>/dev/null; then
            echo ">>> Q1 process died unexpectedly. Launching Q2 anyway."
            return 0
        fi
        sleep "$STAGGER_INTERVAL"
        waited=$((waited + STAGGER_INTERVAL))
        if [ $((waited % 120)) -eq 0 ]; then
            echo ">>> Still waiting for Q1 signal (${waited}s elapsed)..."
            echo "    GPU0 VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 2>/dev/null || echo '?')"
        fi
    done

    echo ">>> WARNING: Timeout (${max_wait}s). Launching Q2 anyway."
}

# ================================================================
# MAIN
# ================================================================
echo "============================================================"
echo " SEED-Bench-2 Dual Isolated Queue"
echo " Machine: $(hostname)"
echo " RAM: $(free -h | grep Mem | awk '{print $2}')"
echo " GPU0: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || echo '?')"
echo " GPU1: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 1 2>/dev/null || echo '?')"
echo " VRAM0: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 0 2>/dev/null)"
echo " VRAM1: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 1 2>/dev/null)"
echo " Strategy: Q1 on GPU0 (CUDA_VISIBLE_DEVICES=0), Q2 on GPU1 (=1)"
echo " Stagger: Q2 waits for Q1 first chunk to finish"
echo " Cells: Q1=${#Q1_CELLS[@]} Q2=${#Q2_CELLS[@]} × $NUM_CHUNKS chunks each"
echo " Start: $(date '+%F %T')"
echo "============================================================"

# ── Pre-flight check ────────────────────────────────────────
echo ""
echo "=== Pre-flight: Checking completed chunks ==="
done_count=0 pending_count=0
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    for cell in "${Q1_CELLS[@]}" "${Q2_CELLS[@]}"; do
        IFS='|' read -r method w_bit scale_file <<< "$cell"
        tag=$(get_tag "$method" "$w_bit")
        task="${TASK_PREFIX}${chunk}"
        if is_done "$task" "$tag"; then
            done_count=$((done_count + 1))
        else
            pending_count=$((pending_count + 1))
            echo "  ❌ $task/$tag"
        fi
    done
done
echo "  Done: $done_count  Pending: $pending_count  Total: $((done_count + pending_count))"
echo "=== Pre-flight complete ==="

# ── Launch Q1 (GPU 0) immediately ───────────────────────────
run_queue "GPU0" "0" "${Q1_CELLS[@]}" &
Q1_PID=$!
echo ""
echo ">>> Q1 launched (PID=$Q1_PID) at $(date +%H:%M:%S)"

# ── Wait for Q1 model-loaded signal, then launch Q2 (GPU 1) ─
wait_for_q1_ready

echo ""
echo ">>> VRAM before Q2 launch:"
echo "    GPU0: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 2>/dev/null)"
echo "    GPU1: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1 2>/dev/null)"

run_queue "GPU1" "1" "${Q2_CELLS[@]}" &
Q2_PID=$!
echo ""
echo ">>> Q2 launched (PID=$Q2_PID) at $(date +%H:%M:%S)"
echo ""
echo ">>> Both queues running:"
echo "    Q1 PID=$Q1_PID (GPU 0, CUDA_VISIBLE_DEVICES=0)"
echo "    Q2 PID=$Q2_PID (GPU 1, CUDA_VISIBLE_DEVICES=1)"
echo ""
echo ">>> Monitor commands:"
echo "    watch -n 30 nvidia-smi"
echo "    tail -f $LOG_DIR/seedbench_2_local_chunk*.log"
echo ""

# ── Wait for both ───────────────────────────────────────────
wait $Q1_PID; EC1=$?
wait $Q2_PID; EC2=$?

echo ""
echo "============================================================"
echo " ALL DONE! $(date '+%F %T')"
echo " Q1 exit=$EC1  Q2 exit=$EC2"
echo "============================================================"

# ── Final status report ────────────────────────────────────
echo ""
echo "=== Final Status Report ==="
ok=0 fail=0
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    task="${TASK_PREFIX}${chunk}"
    for cell in "${Q1_CELLS[@]}" "${Q2_CELLS[@]}"; do
        IFS='|' read -r method w_bit scale_file <<< "$cell"
        tag=$(get_tag "$method" "$w_bit")
        if is_done "$task" "$tag"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
            echo "  MISSING: $task/$tag"
        fi
    done
done
echo "  Done: $ok  Missing: $fail  Total: $((ok + fail))"

# ── Trigger aggregation for any cells not yet aggregated ───
echo ""
echo "=== Aggregation check ==="
for cell in "${Q1_CELLS[@]}" "${Q2_CELLS[@]}"; do
    IFS='|' read -r method w_bit scale_file <<< "$cell"
    tag=$(get_tag "$method" "$w_bit")
    agg="$OUTPUT_BASE/$tag/aggregated_results.json"
    if [ -f "$agg" ]; then
        echo "  ✅ $tag already aggregated"
    else
        echo "  🔄 Aggregating $tag ..."
        python3 "$AGGREGATE_SCRIPT" \
            --input_dir "$OUTPUT_BASE/$tag" \
            --output "$agg" \
            > "$LOG_DIR/aggregate_final_${tag}_$(date +%Y%m%d_%H%M%S).log" 2>&1
        if [ $? -eq 0 ]; then
            echo "  ✅ $tag aggregation OK"
        else
            echo "  ❌ $tag aggregation FAILED"
        fi
    fi
done

echo ""
echo "=== Script Complete ==="