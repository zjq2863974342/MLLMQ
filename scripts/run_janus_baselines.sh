#!/usr/bin/env bash
# ============================================================================
# Janus-Pro-7B Baseline Quantization Queue
# RTN / AWQ / MBQ_block / MBQ_scalar — scale search + eval
#
# Usage (4 GPUs):
#   nohup bash scripts/run_janus_baselines.sh 0 awq > output/logs/baseline_awq.log 2>&1 &
#   nohup bash scripts/run_janus_baselines.sh 1 mbq_block > output/logs/baseline_mbq_block.log 2>&1 &
#   nohup bash scripts/run_janus_baselines.sh 2 mbq_scalar > output/logs/baseline_mbq_scalar.log 2>&1 &
#   nohup bash scripts/run_janus_baselines.sh 3 rtn > output/logs/baseline_rtn.log 2>&1 &
#
# GPU0-2: scale search first, then eval
# GPU3: RTN eval only (no scale cache needed)
# ============================================================================
set -u

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "Cannot cd to MLLMQ"; exit 1; }

GPU="${1:?usage: $0 <gpu_index> <method_group>}"
GROUP="${2:?usage: $0 <gpu_index> <awq|mbq_block|mbq_scalar|rtn>}"

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
DATA=/nas3/kangjia/mllmq/data/coco/janus_mixed_calib_256.json
IMG=/nas3/kangjia/mllmq/data/coco/train2017
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/eval
LOGS=/nas3/kangjia/mllmq/output/logs
mkdir -p "$LOGS"

export CUDA_VISIBLE_DEVICES=$GPU
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

NON_SEED_TASKS="mme_local,textvqa_val_local,mmmu_val,mathvista_testmini_local,pope_random_local,pope_popular_local,pope_adversarial_local"

# ── Resolve method args ──
case "$GROUP" in
    awq)
        SCALE_DIR="$PROJ_ROOT/MLLMQ/scale_cache/awq"
        SCALE_METHOD="awq"
        SCALE_EXTRA=""
        EVAL_METHOD="awq"
        EVAL_TAG="awq"
        ;;
    mbq_block)
        SCALE_DIR="$PROJ_ROOT/MLLMQ/scale_cache/mbq"
        SCALE_METHOD="mbq"
        SCALE_EXTRA=""
        EVAL_METHOD="mbq"
        EVAL_TAG="mbq_block"
        ;;
    mbq_scalar)
        SCALE_DIR="$PROJ_ROOT/MLLMQ/scale_cache/mbq"
        SCALE_METHOD="mbq"
        SCALE_EXTRA="--reweight --reweight_mode scalar"
        EVAL_METHOD="mbq"
        EVAL_TAG="mbq_scalar"
        ;;
    rtn)
        SCALE_DIR=""
        EVAL_METHOD="rtn"
        EVAL_TAG="rtn"
        ;;
    *)
        echo "Unknown group: $GROUP (use awq|mbq_block|mbq_scalar|rtn)"
        exit 1
        ;;
esac

# ── helpers ──
is_done_eval() {
    local task="$1" tag="$2"
    ls "$OUT/$task/$tag"/*_results.json >/dev/null 2>&1
}

run_eval_cell() {
    local w="$1" tag="$2" scale="$3"
    local dir_tag="${tag}_w${w}"

    local missing=0
    IFS=',' read -ra TASK_ARR <<< "$NON_SEED_TASKS"
    for task in "${TASK_ARR[@]}"; do
        if ! is_done_eval "$task" "$dir_tag"; then
            ((missing++))
        fi
    done
    if [ "$missing" -eq 0 ]; then
        echo "  [SKIP] $dir_tag — all 7 tasks done"
        return 0
    fi

    echo ""
    echo "============================================================"
    echo " EVAL: $dir_tag ($missing tasks remaining)  $(date +%H:%M:%S)"
    echo "============================================================"

    local log="$LOGS/eval_${dir_tag}_$(date +%Y%m%d_%H%M%S).log"
    local scale_arg=""
    [ -n "$scale" ] && scale_arg="--scale_path $scale"

    python -u eval_benchmarks.py \
        --model_type janus --model_path "$MODEL" \
        --method "$EVAL_METHOD" --method_tag "$EVAL_TAG" --w_bit "$w" --w_group 128 \
        $scale_arg \
        --tasks "$NON_SEED_TASKS" \
        --task_include_path tasks_local --output_base "$OUT" \
        --device cuda:0 --log_samples --batch_size 1 \
        >> "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  [DONE]  $dir_tag  $(date +%H:%M:%S)"
    else
        echo "  [FAIL]  $dir_tag rc=$rc  $(date +%H:%M:%S)"
    fi
    return $rc
}

# ======================================================================
# Phase 1: Scale Search (skip if group=rtn)
# ======================================================================
if [ "$GROUP" != "rtn" ]; then
    mkdir -p "$SCALE_DIR"

    for w in 4 8 3; do
        SCALE_PATH="$SCALE_DIR/janus_${GROUP}_w${w}a16.pt"

        if [ -f "$SCALE_PATH" ]; then
            echo "[$(date +%H:%M:%S)] SKIP scale cache: $SCALE_PATH (exists)"
            continue
        fi

        echo ""
        echo "============================================================"
        echo " SCALE SEARCH: $GROUP w${w}  GPU=$GPU  $(date)"
        echo "============================================================"
        t0=$(date +%s)

        python main_quant.py \
            --model_path "$MODEL" --device cuda:0 --calib_data coco \
            --data_path "$DATA" --image_folder "$IMG" \
            --n_samples 256 --method "$SCALE_METHOD" \
            --w_bit "$w" --a_bit 16 --w_group 128 \
            --run_process $SCALE_EXTRA \
            --scale_path "$SCALE_PATH" \
            --loss_mode mae \
            >> "$LOGS/scale_${GROUP}_w${w}_$(date +%Y%m%d_%H%M%S).log" 2>&1

        rc=$?
        elapsed=$(($(date +%s) - t0))
        if [ $rc -eq 0 ]; then
            echo "  [DONE] $SCALE_PATH (${elapsed}s)"
        else
            echo "  [FAIL] $SCALE_PATH rc=$rc"
        fi
    done
fi

# ======================================================================
# Phase 2: Eval
# ======================================================================
echo ""
echo "============================================================"
echo " EVAL PHASE: $GROUP  GPU=$GPU  $(date)"
echo "============================================================"

for w in 4 8 3; do
    SCALE_PATH=""
    if [ "$GROUP" != "rtn" ]; then
        SCALE_PATH="$SCALE_DIR/janus_${GROUP}_w${w}a16.pt"
        if [ ! -f "$SCALE_PATH" ]; then
            echo "  [SKIP] $GROUP w${w} — scale cache missing: $SCALE_PATH"
            continue
        fi
    fi
    run_eval_cell "$w" "$EVAL_TAG" "$SCALE_PATH"
done

echo ""
echo "============================================================"
echo " ALL DONE! $GROUP GPU=$GPU $(date)"
echo "============================================================"