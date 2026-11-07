#!/usr/bin/env bash
# ============================================================================
# UNIFIED calibration rebuild for the 8-config ablation (2026-11-03).
#
# Standard (all learned methods):
#   calib : data/coco/janus_mixed_tri_256.json  (128 und + 64 T2I text + 64 gen_tf)
#   n     : 256            w_bit : 3      a_bit : 16
#   group : 128            zero_point : True
#   scope : LM 30 layers only (vision encoder / gen_head stay fp16)
#   wrapper: fixed janus.py (attention_mask.bool())
# RTN has no scale by definition; it is evaluated with the same model / wrapper /
# eval protocol and is listed as calibration-free.
#
# Output: MLLMQ/scale_cache/fixed_ablation_unified/*.pt
# Logs  : output/logs/unified_scale_<name>.log
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
C_TRI=$ROOT/data/coco/janus_mixed_tri_256.json
OUT=$ROOT/MLLMQ/scale_cache/fixed_ablation_unified
LOG=$ROOT/output/logs
mkdir -p "$OUT" "$LOG"

grep -q "attention_mask.bool()" qmllm/models/janus/janus.py || {
  echo "GATE FAIL: janus.py is not fixed"; exit 2; }
[ -s "$C_TRI" ] || { echo "GATE FAIL: missing $C_TRI"; exit 2; }
echo "[gate] fixed wrapper + $C_TRI"

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

build(){ # name gpu envs extra...
  local name=$1 gpu=$2 envs=$3; shift 3
  local out=$OUT/$name.pt
  if [ -s "$out" ]; then echo "[SKIP] $name"; return 0; fi
  echo "[START] $name GPU$gpu $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=$gpu env $envs python main_quant.py \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$C_TRI" --image_folder "$IMG" --n_samples 256 \
    --w_bit 3 --a_bit 16 --w_group 128 --run_process \
    --scale_path "$out" "$@" > "$LOG/unified_scale_${name}.log" 2>&1
  if [ -s "$out" ]; then echo "[OK]    $name $(date '+%F %T')"; else echo "[FAIL]  $name"; tail -8 "$LOG/unified_scale_${name}.log"; fi
}

# 4 concurrent (2 per GPU), then the remaining 3
build awq_w3a16                0 ""                                         --method awq &
build mbq_block_w3a16          0 ""                                         --method mbq --loss_mode mae &
build mbq_scalar_w3a16         1 ""                                         --method mbq --reweight --reweight_mode scalar --loss_mode mae &
build mllmq_channel_only_w3a16 1 "MLLMQ_DISABLE_T2I=1 MLLMQ_DISABLE_GEN_IMG=1" --method mllmq --reweight_mode channel --loss_mode mae &
wait

build mllmq_t2i_text_w3a16     0 "MLLMQ_DISABLE_GEN_IMG=1"                  --method mllmq --reweight_mode channel --loss_mode mae &
build mllmq_t2i_img_w3a16      1 "MLLMQ_DISABLE_T2I=1"                      --method mllmq --reweight_mode channel --loss_mode mae &
wait

build mllmq_both_w3a16         0 ""                                         --method mllmq --reweight_mode channel --loss_mode mae &
wait

echo "===== unified scale status ====="
missing=0
for n in awq_w3a16 mbq_block_w3a16 mbq_scalar_w3a16 mllmq_channel_only_w3a16 mllmq_t2i_text_w3a16 mllmq_t2i_img_w3a16 mllmq_both_w3a16; do
  if [ -s "$OUT/$n.pt" ]; then echo "OK   $n"; else echo "MISS $n"; missing=$((missing+1)); fi
done
[ "$missing" -eq 0 ] && echo "ALL 7 UNIFIED SCALES READY" || { echo "$missing scales missing"; exit 1; }