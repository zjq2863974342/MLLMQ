#!/usr/bin/env bash
# ============================================================================
# Janus-Pro-7B fp16 baseline (no quantization)
#
# Usage:
#   bash scripts/run_janus_fp16_baseline.sh <gpu_index>
# ============================================================================
set -u

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "Cannot cd to MLLMQ"; exit 1; }

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/eval
LOGS=/nas3/kangjia/mllmq/output/logs
mkdir -p "$LOGS"

export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

NON_SEED_TASKS=(mme_local textvqa_val_local mmmu_val mathvista_testmini_local pope_random_local pope_popular_local pope_adversarial_local)
SEED_TASKS=(seedbench_2_local_chunk0 seedbench_2_local_chunk1 seedbench_2_local_chunk2 seedbench_2_local_chunk3 seedbench_2_local_chunk4)

DIR_TAG="fp16"

is_done() {
    local task="$1"
    ls "$OUT/$task/$DIR_TAG"/*_results.json >/dev/null 2>&1
}

run_eval() {
    local gpu="$1" task="$2"

    if is_done "$task"; then
        echo "  [SKIP] $task/$DIR_TAG (already done)"
        return 0
    fi

    local log="$LOGS/eval_${DIR_TAG}_${task}_$(date +%Y%m%d_%H%M%S).log"
    echo "  [START] $task/$DIR_TAG  $(date +%H:%M:%S)  log=$log"
    CUDA_VISIBLE_DEVICES=$gpu python -u eval_benchmarks.py \
        --model_type janus --model_path "$MODEL" \
        --method fp16 --tasks "$task" \
        --task_include_path tasks_local --output_base "$OUT" \
        --device cuda:0 --log_samples --batch_size 1 \
        >> "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  [DONE]  $task/$DIR_TAG  $(date +%H:%M:%S)"
    else
        echo "  [FAIL]  $task/$DIR_TAG rc=$rc  $(date +%H:%M:%S)"
    fi
    return $rc
}

aggregate_seed() {
    local outdir="$OUT/seedbench_2_local/$DIR_TAG"
    mkdir -p "$outdir"
    echo "  [AGGREGATE] $DIR_TAG"
    python3 - "$OUT" "$DIR_TAG" "$outdir" <<'PY'
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

# ── main ──
GPU="${1:?usage: $0 <gpu_index>}"

echo "============================================================"
echo " Janus fp16 baseline: GPU=$GPU  Start=$(date)"
echo "============================================================"

for task in "${NON_SEED_TASKS[@]}"; do
    run_eval "$GPU" "$task"
done

for task in "${SEED_TASKS[@]}"; do
    run_eval "$GPU" "$task"
    sleep 5
done

aggregate_seed

echo ""
echo "============================================================"
echo " fp16 baseline DONE! GPU=$GPU  $(date)"
echo "============================================================"