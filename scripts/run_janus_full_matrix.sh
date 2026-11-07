#!/usr/bin/env bash
# ============================================================================
# Janus-Pro-7B MLLMQ full eval matrix (post decode-fix re-run)
#
# - 6 variants: {mae, mse} x {w4, w8, w3}
# - 7 non-SEED benchmarks (single process each)
# - SEED-Bench-2 in 5 chunks (RAM-bound, must run serially per variant)
# - Dual GPU: launch one instance per GPU (GPU0=mae, GPU1=mse)
#
# Usage:
#   nohup bash scripts/run_janus_full_matrix.sh 0 mae > output/logs/janus_gpu0_mae.log 2>&1 &
#   nohup bash scripts/run_janus_full_matrix.sh 1 mse > output/logs/janus_gpu1_mse.log 2>&1 &
# ============================================================================
set -u

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "Cannot cd to MLLMQ"; exit 1; }

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/eval
CACHE=/nas3/kangjia/mllmq/MLLMQ/scale_cache/mllmq
LOGS=/nas3/kangjia/mllmq/output/logs
mkdir -p "$LOGS"

export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

NON_SEED_TASKS=(mme_local textvqa_val_local mmmu_val mathvista_testmini_local pope_random_local pope_popular_local pope_adversarial_local)
SEED_TASKS=(seedbench_2_local_chunk0 seedbench_2_local_chunk1 seedbench_2_local_chunk2 seedbench_2_local_chunk3 seedbench_2_local_chunk4)

is_done() {
    # $1 = task, $2 = dir tag (e.g. mllmq_mae_w4a16)
    local task="$1" tag="$2"
    ls "$OUT/$task/$tag"/*_results.json >/dev/null 2>&1
}

run_eval() {
    local gpu="$1" mode="$2" w="$3" task="$4"
    local method_tag="mllmq_${mode}"
    local dir_tag="mllmq_${mode}_w${w}a16"
    local scale="$CACHE/janus_${mode}_w${w}a16.pt"

    if is_done "$task" "$dir_tag"; then
        echo "  [SKIP] $task/$dir_tag (already done)"
        return 0
    fi

    local log="$LOGS/eval_${dir_tag}_${task}_$(date +%Y%m%d_%H%M%S).log"
    echo "  [START] $task/$dir_tag  $(date +%H:%M:%S)  log=$log"
    CUDA_VISIBLE_DEVICES=$gpu python -u eval_benchmarks.py \
        --model_type janus --model_path "$MODEL" \
        --method mllmq --method_tag "$method_tag" --w_bit "$w" --w_group 128 \
        --scale_path "$scale" --tasks "$task" \
        --task_include_path tasks_local --output_base "$OUT" \
        --device cuda:0 --log_samples --batch_size 1 \
        >> "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  [DONE]  $task/$dir_tag  $(date +%H:%M:%S)"
    else
        echo "  [FAIL]  $task/$dir_tag rc=$rc  $(date +%H:%M:%S)"
    fi
    return $rc
}

aggregate_seed() {
    local mode="$1" w="$2"
    local dir_tag="mllmq_${mode}_w${w}a16"
    local outdir="$OUT/seedbench_2_local/$dir_tag"
    mkdir -p "$outdir"
    echo "  [AGGREGATE] $dir_tag"
    python3 - "$OUT" "$dir_tag" "$outdir" <<'PY'
import sys, glob, json, os
out, tag, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, '/nas3/kangjia/mllmq/scripts')
from aggregate_seed_results import aggregate_results
files = sorted(glob.glob(os.path.join(out, 'seedbench_2_local_chunk*', tag, '*_results.json')))
if len(files) < 5:
    print(f"  [AGGREGATE] only {len(files)}/5 chunk results found, skipping")
    sys.exit(0)
combined = aggregate_results(files)
with open(os.path.join(outdir, 'aggregated_results.json'), 'w') as f:
    json.dump(combined, f, indent=2, default=str)
print(f"  [AGGREGATE] saved {len(files)} chunks -> {os.path.join(outdir, 'aggregated_results.json')}")
for t, ms in combined['results'].items():
    for m, v in ms.items():
        print(f"    {t}/{m} = {v:.4f}")
PY
}

run_variant() {
    local gpu="$1" mode="$2" w="$3"

    echo ""
    echo "============================================================"
    echo " VARIANT: mllmq_${mode}_w${w}a16  (GPU$gpu)  $(date)"
    echo "============================================================"

    for task in "${NON_SEED_TASKS[@]}"; do
        run_eval "$gpu" "$mode" "$w" "$task"
    done

    for task in "${SEED_TASKS[@]}"; do
        run_eval "$gpu" "$mode" "$w" "$task"
        sleep 5
    done

    aggregate_seed "$mode" "$w"

    echo ""
    echo " COOLDOWN 60s before next variant..."
    sleep 60
}

# ── main ──
GPU="${1:?usage: $0 <gpu_index> <mae|mse>}"
MODE="${2:?usage: $0 <gpu_index> <mae|mse>}"
if [ "$MODE" != "mae" ] && [ "$MODE" != "mse" ]; then
    echo "MODE must be 'mae' or 'mse'"; exit 1
fi

echo "============================================================"
echo " Janus full matrix: GPU=$GPU MODE=$MODE  Start=$(date)"
echo "============================================================"

for w in 8 3 4; do
    run_variant "$GPU" "$MODE" "$w"
done

echo ""
echo "============================================================"
echo " ALL DONE! GPU=$GPU MODE=$MODE  $(date)"
echo "============================================================"