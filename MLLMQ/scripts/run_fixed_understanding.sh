#!/usr/bin/env bash
# ============================================================================
# 单个 fixed_ablation 配置的理解侧 7 项（fixed wrapper, W3A16, n=256 tri calib）
#   MMMU / MME / TextVQA / MathVista / POPE(random,popular,adversarial)
#   串行跑在一个 GPU 上，skip-existing；产物写入
#     output/fixed_ablation_20261103/<TAG>/understanding_7/<task>/<TAG>_w3a16/
# Usage:
#   TAG=mllmq_channel_only METHOD=mllmq GPU=0 \
#     SCALE=/nas3/.../scale_cache/fixed_ablation/mllmq_channel_only.pt \
#     bash scripts/run_fixed_understanding.sh
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
LOG=$ROOT/output/logs
GPU=${GPU:-0}
TAG=${TAG:?TAG required}
METHOD=${METHOD:-mllmq}
SCALE=${SCALE:-}
OUTBASE=$ROOT/output/fixed_ablation_20261103/$TAG/understanding_7
TASKS=${TASKS:-"mmmu_val mme_local textvqa_val_local mathvista_testmini_local pope_random_local pope_popular_local pope_adversarial_local"}

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"

log(){ echo "[$(date '+%F %T')] $1"; }

if [ "$METHOD" != "rtn" ] && [ "$METHOD" != "fp16" ]; then
  [ -s "$SCALE" ] || { log "❌ 缺 scale: $SCALE"; exit 2; }
fi
mkdir -p "$OUTBASE"

log "===== fixed_und TAG=$TAG METHOD=$METHOD GPU=$GPU ====="
fail=0
for t in $TASKS; do
  out="$OUTBASE/$t/${TAG}_w3a16"
  if ls "$out"/*_results.json >/dev/null 2>&1; then
    log "[SKIP] $t"
    continue
  fi
  log "[RUN ] $t"
  CUDA_VISIBLE_DEVICES=$GPU python eval_benchmarks.py \
    --model_type janus --model_path "$MODEL" \
    --method "$METHOD" --method_tag "$TAG" \
    --w_bit 3 --a_bit 16 --w_group 128 \
    ${SCALE:+--scale_path "$SCALE"} \
    --tasks "$t" --output_base "$OUTBASE" --device cuda:0 \
    > "$LOG/fixed_und_${TAG}_${t}.log" 2>&1
  if ls "$out"/*_results.json >/dev/null 2>&1; then
    log "[DONE] $t :: $(ls "$out"/*_results.json | head -1)"
  else
    log "[FAIL] $t :: 见 $LOG/fixed_und_${TAG}_${t}.log"
    fail=$((fail+1))
  fi
done

[ "$fail" -eq 0 ] && log "✅ ALL DONE $TAG" || log "⚠️ $TAG 有 $fail 项失败"
exit 0