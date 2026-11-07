#!/usr/bin/env bash
# ============================================================================
# Fixed-wrapper 6-scale build for the 8-config ablation.
# Run on the remote machine AFTER pausing RTN/AWQ (skip-existing is safe).
#
# Gates:
#   - janus.py must contain the fixed masks (attention_mask.bool()).
#   - data/coco/janus_mixed_tri_256.json must exist.
# Output: MLLMQ/scale_cache/fixed_ablation/*.pt
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
C_MIX=$ROOT/data/coco/janus_mixed_calib_256.json
C_GT=$ROOT/data/coco/janus_mixed_gen_tf_256.json
C_TRI=$ROOT/data/coco/janus_mixed_tri_256.json
OUT=$ROOT/MLLMQ/scale_cache/fixed_ablation
LOG=$ROOT/output/logs
mkdir -p "$OUT"

# ---- gates ----
grep -q "attention_mask.bool()" qmllm/models/janus/janus.py || {
  echo "GATE FAIL: janus.py is not fixed"; exit 2; }
[ -s "$C_TRI" ] || { echo "GATE FAIL: missing $C_TRI"; exit 2; }
[ -s "$C_MIX" ] || { echo "GATE FAIL: missing $C_MIX"; exit 2; }
[ -s "$C_GT" ]  || { echo "GATE FAIL: missing $C_GT"; exit 2; }
echo "[gates] janus.py fixed + calib files present"

build(){ # name gpu calib envs extra...
  local name=$1 gpu=$2 calib=$3 envs=$4; shift 4
  local out=$OUT/$name.pt
  if [ -s "$out" ]; then echo "[SKIP] $name"; return 0; fi
  echo "[START] $name GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu env $envs python main_quant.py \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$calib" --image_folder "$IMG" --n_samples 256 \
    --w_bit 3 --a_bit 16 --w_group 128 --run_process \
    --scale_path "$out" "$@" > "$LOG/fixed_scale_${name}.log" 2>&1
  if [ -s "$out" ]; then echo "[OK] $name"; else echo "[FAIL] $name"; tail -8 "$LOG/fixed_scale_${name}.log"; fi
}

# 6 builds: 2 per GPU on GPU0/1, 1 each on GPU2/3 (all 4 cards busy)
build mbq_block        0 "$C_MIX" ""                    --method mbq --loss_mode mae &
build mbq_scalar       0 "$C_MIX" ""                    --method mbq --reweight --reweight_mode scalar --loss_mode mae &
build mllmq_channel_only 1 "$C_MIX" "MLLMQ_DISABLE_T2I=1" --method mllmq --reweight_mode channel --loss_mode mae &
build mllmq_t2i_text   1 "$C_MIX" ""                    --method mllmq --reweight_mode channel --loss_mode mae &
build mllmq_t2i_img    2 "$C_GT"  "MLLMQ_DISABLE_T2I=1" --method mllmq --reweight_mode channel --loss_mode mae &
build mllmq_both       3 "$C_TRI" ""                    --method mllmq --reweight_mode channel --loss_mode mae &
wait

echo "===== scale status ====="
missing=0
for n in mbq_block mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img mllmq_both; do
  if [ -s "$OUT/$n.pt" ]; then echo "OK   $n"; else echo "MISS $n"; missing=$((missing+1)); fi
done
[ "$missing" -eq 0 ] && echo "ALL 6 SCALES READY" || { echo "$missing scales missing"; exit 1; }