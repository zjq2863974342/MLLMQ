#!/usr/bin/env bash
# ============================================================================
# Full Matrix Queue: SEED-Bench-2 + MathVista (the remaining experiments)
#   1 GPU (A800-80GB), 2 parallel workers, auto-stagger + is_done skip
#   Idempotent: each cell runs at most once.
# ============================================================================
set -u

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "Cannot cd to MLLMQ"; exit 1; }

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
LOG_DIR=/nas3/kangjia/mllmq/output/logs
SCALE_DIR=/nas3/kangjia/mllmq/MLLMQ/scale_cache
mkdir -p "$LOG_DIR"

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
        --tasks "$task" --device cuda:0 --batch_size 1 \
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

# Shared task list: each item is "task_name|method|w_bit|scale_file_or_empty"
SHARED_TASKS=(
    # SEED-Bench-2 (13 cells, 24,371 samples each)
    "seedbench_2_local|fp16|16|"
    "seedbench_2_local|rtn|3|"
    "seedbench_2_local|rtn|4|"
    "seedbench_2_local|rtn|8|"
    "seedbench_2_local|block|3|block/internvl2_w3a16.pt"
    "seedbench_2_local|block|4|block/internvl2_w4a16.pt"
    "seedbench_2_local|block|8|block/internvl2_w8a16.pt"
    "seedbench_2_local|mllmq|3|mllmq/internvl2_w3a16.pt"
    "seedbench_2_local|mllmq|4|mllmq/internvl2_w4a16.pt"
    "seedbench_2_local|mllmq|8|mllmq/internvl2_w8a16.pt"
    "seedbench_2_local|awq|3|awq/internvl2_InternVL2-8B_w3a16.pt"
    "seedbench_2_local|awq|4|awq/internvl2_InternVL2-8B_w4a16.pt"
    "seedbench_2_local|awq|8|awq/internvl2_InternVL2-8B_w8a16.pt"
    # MathVista (13 cells, 1,000 samples each)
    "mathvista_testmini_local|fp16|16|"
    "mathvista_testmini_local|rtn|3|"
    "mathvista_testmini_local|rtn|4|"
    "mathvista_testmini_local|rtn|8|"
    "mathvista_testmini_local|block|3|block/internvl2_w3a16.pt"
    "mathvista_testmini_local|block|4|block/internvl2_w4a16.pt"
    "mathvista_testmini_local|block|8|block/internvl2_w8a16.pt"
    "mathvista_testmini_local|mllmq|3|mllmq/internvl2_w3a16.pt"
    "mathvista_testmini_local|mllmq|4|mllmq/internvl2_w4a16.pt"
    "mathvista_testmini_local|mllmq|8|mllmq/internvl2_w8a16.pt"
    "mathvista_testmini_local|awq|3|awq/internvl2_InternVL2-8B_w3a16.pt"
    "mathvista_testmini_local|awq|4|awq/internvl2_InternVL2-8B_w4a16.pt"
    "mathvista_testmini_local|awq|8|awq/internvl2_InternVL2-8B_w8a16.pt"
)

run_worker() {
    local wid="$1"
    local ok=0 fail=0 skip=0

    echo ""
    echo "########################################"
    echo " Worker $wid Started: $(date)"
    echo "########################################"

    for item in "${SHARED_TASKS[@]}"; do
        local task_name method w_bit scale_file tag scale
        IFS='|' read -r task_name method w_bit scale_file <<< "$item"
        if [ "$method" = "fp16" ]; then tag="fp16"; else tag="${method}_w${w_bit}a16"; fi
        scale=""
        if [ -n "$scale_file" ] && [ -f "$SCALE_DIR/$scale_file" ]; then
            scale="$SCALE_DIR/$scale_file"
        fi

        if is_done "$task_name" "$tag"; then
            echo "  [SKIP] $task_name/$tag (done)"
            skip=$((skip+1))
            continue
        fi

        echo "  [START] $task_name/$tag $(date +%H:%M:%S)"
        python -u eval_benchmarks.py \
            --model_path "$MODEL_PATH" --method "$method" --w_bit "$w_bit" --a_bit 16 \
            --tasks "$task_name" --device cuda:0 --batch_size 1 \
            --output_base "$OUTPUT_BASE" \
            ${scale:+--scale_path "$scale"} \
            > "$LOG_DIR/${task_name}_${tag}_$(date +%Y%m%d_%H%M%S).log" 2>&1
        local rc=$?

        if [ $rc -eq 0 ]; then
            echo "  [DONE]  $task_name/$tag $(date +%H:%M:%S)"
            ok=$((ok+1))
        else
            echo "  [FAIL]  $task_name/$tag rc=$rc $(date +%H:%M:%S)"
            fail=$((fail+1))
        fi
    done

    echo ""
    echo "Worker $wid Final: ok=$ok fail=$fail skip=$skip ($(date))"
}

echo "============================================================"
echo " Full Matrix Queue: SEED-Bench-2 + MathVista"
echo " GPU: $(nvidia-smi -L 2>/dev/null || echo 'n/a')"
echo " VRAM used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo " Start: $(date)"
echo "============================================================"
echo ""
echo "SHARED_TASKS count: ${#SHARED_TASKS[@]} cells"
echo "  SEED-Bench-2: 13 cells (24,371 samples each, ~6-8h/cell)"
echo "  MathVista:    13 cells (1,000 samples each, ~1h/cell)"
echo ""

# Launch Worker 0 immediately
run_worker 0 &
PID0=$!

echo "Worker 0 launched (PID=$PID0). Staggering Worker 1 by 60s..."
sleep 60

# Launch Worker 1 (60s later to avoid simultaneous model loading)
run_worker 1 &
PID1=$!

echo "Worker 1 launched (PID=$PID1)."
echo ""
echo "Monitor: watch -n 30 'nvidia-smi -q -d MEMORY,UTILIZATION'"
echo "Logs:    $LOG_DIR/"
echo ""

# Wait for both workers
wait $PID0; EC0=$?
wait $PID1; EC1=$?

echo ""
echo "============================================================"
echo "ALL DONE! $(date)"
echo "Worker 0: exit=$EC0  Worker 1: exit=$EC1"
echo ""

# Final check: any missing cells?
need_any=0
for item in "${SHARED_TASKS[@]}"; do
    IFS='|' read -r task_name method w_bit scale_file <<< "$item"
    tag="fp16"
    if [ "$method" != "fp16" ]; then tag="${method}_w${w_bit}a16"; fi
    if ! is_done "$task_name" "$tag"; then
        echo "  MISSING: $task_name/$tag"
        need_any=1
    fi
done

if [ $need_any -eq 0 ]; then
    echo "All cells completed successfully!"
fi
echo "============================================================"