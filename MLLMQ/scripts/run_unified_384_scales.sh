#!/usr/bin/env bash
# ============================================================================
# UNIFIED 384-pool scales (2026-11-03).
#
# Pool: data/coco/janus_mixed_tri_384.json
#       128 understanding + 128 T2I text + 128 gen_tf = 384 samples.
# Why 384: one shared pool for all methods (fair) *and* full n=128 per
# modality (the original per-term budget), instead of the 64/64 in tri-256.
#
# Standard: W3A16, group 128, zero_point True, LM 30 layers, fixed wrapper.
# Output: MLLMQ/scale_cache/unified_384/*.pt
# Waits for the tri-256 / understanding-128 / gen-only builds first.
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
C_384=$ROOT/data/coco/janus_mixed_tri_384.json
OUT=$ROOT/MLLMQ/scale_cache/unified_384
LOG=$ROOT/output/logs
mkdir -p "$OUT" "$LOG"

grep -q "attention_mask.bool()" qmllm/models/janus/janus.py || {
  echo "GATE FAIL: janus.py is not fixed"; exit 2; }
[ -s "$C_384" ] || { echo "GATE FAIL: missing $C_384"; exit 2; }
echo "[gate] fixed wrapper + $C_384"

# wait for earlier build drivers (precise argv match, excludes self)
while [ -n "$(ps -eo pid,args | awk '($2=="bash" && $3=="scripts/run_unified_scales.sh") || ($2=="bash" && $3=="scripts/run_unified_understanding_scales.sh") || ($2=="bash" && $3=="scripts/run_unified_gen_only_scale.sh") {print $1}')" ]; do
  echo "[wait] earlier scale builds still running $(date '+%F %T')"; sleep 60
done

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

build(){ # name gpu envs extra...
  local name=$1 gpu=$2 envs=$3; shift 3
  local out=$OUT/$name.pt
  if [ -s "$out" ]; then echo "[SKIP] $name"; return 0; fi
  echo "[START] $name GPU$gpu $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=$gpu env $envs python main_quant.py \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$C_384" --image_folder "$IMG" --n_samples 384 \
    --w_bit 3 --a_bit 16 --w_group 128 --run_process \
    --scale_path "$out" "$@" > "$LOG/unified384_scale_${name}.log" 2>&1
  if [ -s "$out" ]; then echo "[OK]    $name $(date '+%F %T')"; else echo "[FAIL]  $name"; tail -8 "$LOG/unified384_scale_${name}.log"; fi
}

# n=384 下单个 build 峰值 ~32–36GB；80G 卡上 2 个并发会 OOM。
# 因此统一 1 build / GPU（两卡各一个），7 个 build 分 4 波。
build awq_w3a16                0 ""                                         --method awq &
build mbq_block_masked_w3a16   1 "MLLMQ_MASK_BLOCK=1"                       --method mbq --loss_mode mae &
wait

build mbq_scalar_w3a16         0 ""                                         --method mbq --reweight --reweight_mode scalar --loss_mode mae &
build mllmq_channel_only_w3a16 1 "MLLMQ_DISABLE_T2I=1 MLLMQ_DISABLE_GEN_IMG=1" --method mllmq --reweight_mode channel --loss_mode mae &
wait

build mllmq_t2i_text_w3a16     0 "MLLMQ_DISABLE_GEN_IMG=1"                  --method mllmq --reweight_mode channel --loss_mode mae &
build mllmq_t2i_img_w3a16      1 "MLLMQ_DISABLE_T2I=1"                      --method mllmq --reweight_mode channel --loss_mode mae &
wait

build mllmq_both_w3a16         0 ""                                         --method mllmq --reweight_mode channel --loss_mode mae &
wait

echo "===== unified 384 scale status ====="
missing=0
for n in awq_w3a16 mbq_block_masked_w3a16 mbq_scalar_w3a16 mllmq_channel_only_w3a16 mllmq_t2i_text_w3a16 mllmq_t2i_img_w3a16 mllmq_both_w3a16; do
  if [ -s "$OUT/$n.pt" ]; then echo "OK   $n"; else echo "MISS $n"; missing=$((missing+1)); fi
done
[ "$missing" -eq 0 ] && echo "ALL 7 UNIFIED-384 SCALES READY" || { echo "$missing scales missing"; exit 1; }