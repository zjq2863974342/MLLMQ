#!/usr/bin/env bash
# ============================================================================
# 单 config 的理解侧 7 项 / SEED chunks（串行，skip-existing）
# TAG/METHOD/SCALE/GPU/W_BIT/A_BIT/TASKS 由调用方传入；OUTBASE 默认 5h 理解目录。
# 用法：
#   TAG=awq METHOD=awq SCALE=.../unified_understanding_128/awq_w3a16.pt GPU=1 \
#     bash scripts/run_understanding_7.sh
#   TAG=mllmq_both METHOD=mllmq SCALE=.../w4a8/mix384/mllmq_both_w4a8.pt GPU=1 \
#     W_BIT=4 A_BIT=8 TASKS="seedbench_2_local_chunk0 ... chunk4" \
#     OUTBASE=.../w4a8_supplement/seed LOG=.../w4a8_supplement/seed/logs \
#     bash scripts/run_understanding_7.sh
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
OUTBASE=${OUTBASE:-$ROOT/output/fixed_ablation_20261103_understanding_5h}
LOG=${LOG:-$ROOT/output/logs/understanding_5h}
GPU=${GPU:-0}
TAG=${TAG:?TAG required}
METHOD=${METHOD:?METHOD required}
SCALE=${SCALE:-}
W_BIT=${W_BIT:-3}
A_BIT=${A_BIT:-16}
LIMIT=${LIMIT:-}
TASKS=${TASKS:-"mme_local textvqa_val_local mmmu_val mathvista_testmini_local pope_random_local pope_popular_local pope_adversarial_local"}
mkdir -p "$OUTBASE" "$LOG"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$ROOT/data/benchmarks_gen/geneval/evaluation/.torch_home"
export MODELSCOPE_CACHE="$ROOT/data/benchmarks_gen/geneval/evaluation/.ms_cache"
fail=0
for t in $TASKS; do
  out="$OUTBASE/$TAG/$t/${TAG}_w${W_BIT}a${A_BIT}"
  if ls "$out"/*_results.json >/dev/null 2>&1; then echo "[SKIP] $TAG $t"; continue; fi
  echo "[RUN] $TAG $t GPU$GPU $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=$GPU python eval_benchmarks.py \
    --model_type janus --model_path "$MODEL" \
    --method "$METHOD" --method_tag "$TAG" \
    --w_bit "$W_BIT" --a_bit "$A_BIT" --w_group 128 ${SCALE:+--scale_path "$SCALE"} \
    --tasks "$t" --output_base "$OUTBASE/$TAG" --device cuda:0 --log_samples ${LIMIT:+--limit "$LIMIT"} \
    > "$LOG/${TAG}_${t}.log" 2>&1
  if ls "$out"/*_results.json >/dev/null 2>&1; then
    echo "[DONE] $TAG $t"
  else
    echo "[FAIL] $TAG $t (see $LOG/${TAG}_${t}.log)"; fail=$((fail+1))
  fi
done
echo "[CONFIG DONE] $TAG fail=$fail $(date '+%F %T')"
exit 0