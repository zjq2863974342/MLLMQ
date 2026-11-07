#!/usr/bin/env bash
# ============================================================================
# UNIFIED understanding-side calibration (2026-11-03).
#
# Motivation: AWQ's reconstruction loss is masked by caption_mask, so it only
# uses answer-text tokens; vision/generation tokens never enter its objective.
# For a fair understanding-side comparison, every method is rebuilt on the SAME
# 128 understanding-only samples (caption-masked objective for all).
#
# Pool   : data/coco/janus_understanding_128.json  (128 understanding samples)
# Standard: W3A16, group 128, zero_point True, LM 30 layers only, fixed wrapper
# Output : MLLMQ/scale_cache/unified_understanding_128/*.pt
#
# Waits for the tri-modal build (run_unified_scales.sh) to finish first.
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
C_UND=$ROOT/data/coco/janus_understanding_128.json
OUT=$ROOT/MLLMQ/scale_cache/unified_understanding_128
LOG=$ROOT/output/logs
mkdir -p "$OUT" "$LOG"

grep -q "attention_mask.bool()" qmllm/models/janus/janus.py || {
  echo "GATE FAIL: janus.py is not fixed"; exit 2; }
[ -s "$C_UND" ] || { echo "GATE FAIL: missing $C_UND"; exit 2; }
echo "[gate] fixed wrapper + $C_UND"

# wait for the tri-modal 7-scale build (precise argv match, excludes self)
while [ -n "$(ps -eo pid,args | awk '$2=="bash" && $3=="scripts/run_unified_scales.sh"{print $1}')" ]; do
  echo "[wait] tri-modal build still running $(date '+%F %T')"; sleep 60
done
echo "[start] tri-modal build finished $(date '+%F %T')"

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

build(){ # name gpu envs extra...
  local name=$1 gpu=$2 envs=$3; shift 3
  local out=$OUT/$name.pt
  if [ -s "$out" ]; then echo "[SKIP] $name"; return 0; fi
  echo "[START] $name GPU$gpu $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=$gpu env $envs python main_quant.py \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$C_UND" --image_folder "$IMG" --n_samples 128 \
    --w_bit 3 --a_bit 16 --w_group 128 --run_process \
    --scale_path "$out" "$@" > "$LOG/unified_und_scale_${name}.log" 2>&1
  if [ -s "$out" ]; then echo "[OK]    $name $(date '+%F %T')"; else echo "[FAIL]  $name"; tail -8 "$LOG/unified_und_scale_${name}.log"; fi
}

build awq_w3a16                0 ""                                         --method awq &
build mbq_block_masked_w3a16   0 "MLLMQ_MASK_BLOCK=1"                       --method mbq --loss_mode mae &
build mbq_scalar_w3a16         1 ""                                         --method mbq --reweight --reweight_mode scalar --loss_mode mae &
build mllmq_channel_only_w3a16 1 "MLLMQ_DISABLE_T2I=1 MLLMQ_DISABLE_GEN_IMG=1" --method mllmq --reweight_mode channel --loss_mode mae &
wait

echo "===== understanding-only scale status ====="
missing=0
for n in awq_w3a16 mbq_block_masked_w3a16 mbq_scalar_w3a16 mllmq_channel_only_w3a16; do
  if [ -s "$OUT/$n.pt" ]; then echo "OK   $n"; else echo "MISS $n"; missing=$((missing+1)); fi
done
[ "$missing" -eq 0 ] && echo "ALL 4 UNDERSTANDING SCALES READY" || { echo "$missing scales missing"; exit 1; }