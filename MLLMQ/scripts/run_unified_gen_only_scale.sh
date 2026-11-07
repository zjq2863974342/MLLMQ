#!/usr/bin/env bash
# ============================================================================
# MLLMQ generation-only calibration scale (2026-11-03).
#
# Config: no understanding-side (vision/caption) reweighting; only generation-
# side vis/cap calibration, where
#     vis = generated image tokens (gen_img_mask)
#     cap = T2I prompt text tokens (t2i_prompt_mask)
# Calibration pool: data/coco/janus_gen_tf_128.json (128 gen_tf samples only)
# Standard: W3A16, group 128, zero_point True, LM 30 layers, fixed wrapper.
# Requires the env-gated MLLMQ_GEN_ONLY code path in pre_quant.py.
#
# Output: MLLMQ/scale_cache/unified_generation_only/mllmq_gen_only_w3a16.pt
# Waits for the tri-modal and understanding-only builds to finish first.
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
C_GT=$ROOT/data/coco/janus_gen_tf_128.json
OUT=$ROOT/MLLMQ/scale_cache/unified_generation_only
LOG=$ROOT/output/logs
mkdir -p "$OUT" "$LOG"

grep -q "MLLMQ_GEN_ONLY" qmllm/methods/mbq/quantize/pre_quant.py || {
  echo "GATE FAIL: MLLMQ_GEN_ONLY code path missing"; exit 2; }
grep -q "attention_mask.bool()" qmllm/models/janus/janus.py || {
  echo "GATE FAIL: janus.py is not fixed"; exit 2; }
[ -s "$C_GT" ] || { echo "GATE FAIL: missing $C_GT"; exit 2; }
echo "[gate] fixed wrapper + GEN_ONLY path + $C_GT"

# wait for the two earlier build drivers (precise argv match, excludes self)
while [ -n "$(ps -eo pid,args | awk '($2=="bash" && $3=="scripts/run_unified_scales.sh") || ($2=="bash" && $3=="scripts/run_unified_understanding_scales.sh") {print $1}')" ]; do
  echo "[wait] earlier scale builds still running $(date '+%F %T')"; sleep 60
done

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

OUTP=$OUT/mllmq_gen_only_w3a16.pt
if [ -s "$OUTP" ]; then
  echo "[SKIP] mllmq_gen_only_w3a16"
else
  echo "[START] mllmq_gen_only_w3a16 $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=0 env MLLMQ_GEN_ONLY=1 python main_quant.py \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$C_GT" --image_folder "$IMG" --n_samples 128 \
    --w_bit 3 --a_bit 16 --w_group 128 --run_process \
    --scale_path "$OUTP" \
    --method mllmq --reweight_mode channel --loss_mode mae \
    > "$LOG/unified_gen_only_scale.log" 2>&1
  [ -s "$OUTP" ] && echo "[OK] mllmq_gen_only_w3a16 $(date '+%F %T')" || { echo "[FAIL] mllmq_gen_only_w3a16"; tail -12 "$LOG/unified_gen_only_scale.log"; exit 1; }
fi

echo "===== generation-only scale status ====="
[ -s "$OUTP" ] && echo "OK   mllmq_gen_only_w3a16" || echo "MISS mllmq_gen_only_w3a16"
[ -s "$OUTP" ] && echo "GENERATION-ONLY SCALE READY"