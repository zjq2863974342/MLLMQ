#!/usr/bin/env bash
# ============================================================================
# Generate ONE W4A8 T2I-CompBench tag on ONE GPU (resume via skip-existing).
#
# Usage:
#   TAG=rtn GPU=0 bash queue/t2i_w4a8_gen_one.sh
#   TAG=rtn GPU=0 LOG_DIR=/path bash queue/t2i_w4a8_gen_one.sh
#
# Output: output/janus-pro-7b/t2i_compbench_current/w4a8/<tag>/t2i_compbench
# Log:    ${LOG_DIR:-output/logs/t2i_compbench_current/remote_w4a8}/<tag>.log
# ============================================================================
set -o pipefail

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
SC8=$PROJ/scale_cache/w4a8/mix384
OUT=$ROOT/output/janus-pro-7b/t2i_compbench_current/w4a8
LOG=${LOG_DIR:-$ROOT/output/logs/t2i_compbench_current/remote_w4a8}
TAG=${TAG:?set TAG}
GPU=${GPU:?set GPU}
TARGET=${TARGET:-2400}

mkdir -p "$OUT" "$LOG"
cd "$PROJ" || exit 2

export PYTHONPATH="$PROJ:$ROOT/checkpoints/Janus:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME="$ROOT/checkpoints/torch_home"

log(){ echo "[$(date '+%F %T')] [gen $TAG GPU$GPU] $*"; }
count(){ find "$OUT/$TAG/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }

case "$TAG" in
  rtn)
    ARGS=(--quant_method rtn --w_bit 4 --a_bit 8); ENVS=() ;;
  mbq_scalar)
    ARGS=(--quant_method mbq_scalar --w_bit 4 --a_bit 8 --scale_path "$SC8/mbq_scalar_w4a8.pt"); ENVS=() ;;
  mllmq_channel_only)
    ARGS=(--quant_method mllmq_mae --w_bit 4 --a_bit 8 --scale_path "$SC8/mllmq_channel_only_w4a8.pt")
    ENVS=(MLLMQ_DISABLE_T2I=1 MLLMQ_DISABLE_GEN_IMG=1) ;;
  mllmq_t2i_text)
    ARGS=(--quant_method mllmq_mae --w_bit 4 --a_bit 8 --scale_path "$SC8/mllmq_t2i_text_w4a8.pt")
    ENVS=(MLLMQ_DISABLE_GEN_IMG=1) ;;
  mllmq_t2i_img)
    ARGS=(--quant_method mllmq_mae --w_bit 4 --a_bit 8 --scale_path "$SC8/mllmq_t2i_img_w4a8.pt"); ENVS=() ;;
  mllmq_both)
    ARGS=(--quant_method mllmq_mae --w_bit 4 --a_bit 8 --scale_path "$SC8/mllmq_both_w4a8.pt"); ENVS=() ;;
  awq)
    ARGS=(--quant_method awq --w_bit 4 --a_bit 8 --scale_path "$SC8/awq_w4a8.pt"); ENVS=() ;;
  mllmq_gen_only)
    ARGS=(--quant_method mllmq_mae --w_bit 4 --a_bit 8 --scale_path "$SC8/mllmq_gen_only_w4a8.pt"); ENVS=() ;;
  *)
    log "unknown TAG=$TAG"; exit 2 ;;
esac

n=$(count)
if [ "$n" -ge "$TARGET" ]; then log "SKIP ($n/$TARGET)"; exit 0; fi
if pgrep -f "[e]val_generate_quant.py.*--output_dir $OUT/$TAG" >/dev/null 2>&1; then
  log "already running"; exit 0
fi

log "START ($n/$TARGET)"
CUDA_VISIBLE_DEVICES="$GPU" env "${ENVS[@]}" python -u "$PROJ/eval_generate_quant.py" \
  --model_path "$MODEL" --benchmark t2i_compbench \
  --output_dir "$OUT/$TAG" --num_samples 1 "${ARGS[@]}" >> "$LOG/$TAG.log" 2>&1
rc=$?
n2=$(count)
log "EXIT rc=$rc ($n2/$TARGET)"
[ "$n2" -ge "$TARGET" ]