#!/usr/bin/env bash
# ============================================================================
# Janus w8 补跑脚本（Pope 标点 + MathVista 句点修复后）
#
# 补跑 4 个 benchmark 的 mae_w8 / mse_w8（共 8 cell）：
#   pope_random_local, pope_popular_local, pope_adversarial_local,
#   mathvista_testmini_local
#
# Usage:
#   nohup bash scripts/run_janus_redo_w8.sh 0 mae > output/logs/redo_gpu0_mae.log 2>&1 &
#   nohup bash scripts/run_janus_redo_w8.sh 1 mse > output/logs/redo_gpu1_mse.log 2>&1 &
#
# 由 scripts/auto_redo_when_done.sh 在主队列全部结束后自动启动。
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

REDO_TASKS=(pope_random_local pope_popular_local pope_adversarial_local mathvista_testmini_local)
W=8

GPU="${1:?usage: $0 <gpu_index> <mae|mse>}"
MODE="${2:?usage: $0 <gpu_index> <mae|mse>}"
if [ "$MODE" != "mae" ] && [ "$MODE" != "mse" ]; then
    echo "MODE must be 'mae' or 'mse'"; exit 1
fi

METHOD_TAG="mllmq_${MODE}"
DIR_TAG="mllmq_${MODE}_w${W}a16"
SCALE="$CACHE/janus_${MODE}_w${W}a16.pt"

if [ ! -f "$SCALE" ]; then
    echo "Scale cache not found: $SCALE"; exit 1
fi

echo "============================================================"
echo " Janus w8 redo: GPU=$GPU MODE=$MODE  Start=$(date)"
echo " Tasks: ${REDO_TASKS[*]}"
echo "============================================================"

for task in "${REDO_TASKS[@]}"; do
    if ls "$OUT/$task/$DIR_TAG"/*_results.json >/dev/null 2>&1; then
        echo "  [SKIP] $task/$DIR_TAG (already done)"
        continue
    fi
    log="$LOGS/redo_${DIR_TAG}_${task}_$(date +%Y%m%d_%H%M%S).log"
    echo "  [START] $task/$DIR_TAG  $(date +%H:%M:%S)  log=$log"
    CUDA_VISIBLE_DEVICES=$GPU python -u eval_benchmarks.py \
        --model_type janus --model_path "$MODEL" \
        --method mllmq --method_tag "$METHOD_TAG" --w_bit "$W" --w_group 128 \
        --scale_path "$SCALE" --tasks "$task" \
        --task_include_path tasks_local --output_base "$OUT" \
        --device cuda:0 --log_samples --batch_size 1 \
        >> "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "  [DONE]  $task/$DIR_TAG  $(date +%H:%M:%S)"
    else
        echo "  [FAIL]  $task/$DIR_TAG rc=$rc  $(date +%H:%M:%S)"
    fi
done

echo ""
echo "============================================================"
echo " Redo done! GPU=$GPU MODE=$MODE  $(date)"
echo "============================================================"