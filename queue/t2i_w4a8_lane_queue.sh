#!/usr/bin/env bash
# ============================================================================
# W4A8 T2I-CompBench++ P1 — per-GPU lanes (2 T2I procs per card).
#
# Instead of waiting for ALL SEED tags to finish, each GPU lane starts its
# two T2I tags as soon as that GPU's own SEED tag is done. SEED stays strictly
# 1 proc/GPU; T2I never shares a card with the SEED process it replaces.
#
#   GPU0: wait rtn             -> T2I rtn + mbq_scalar
#   GPU1: wait mllmq_both      -> T2I mllmq_channel_only + mllmq_t2i_text
#   GPU2: wait awq             -> T2I mllmq_t2i_img + mllmq_both
#   GPU3: wait mllmq_gen_only  -> T2I awq + mllmq_gen_only
#
# Output: output/janus-pro-7b/t2i_compbench_current/w4a8/<tag>/t2i_compbench
# Logs:   output/logs/local4_supplement/t2i_w4a8/<tag>.log
# Idempotent: skip when 2400 PNGs, skip when the same tag is already running.
# Does NOT score.
# ============================================================================
set -u

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
SC8=$PROJ/scale_cache/w4a8
OUT=$ROOT/output/janus-pro-7b/t2i_compbench_current/w4a8
LOG=$ROOT/output/logs/local4_supplement/t2i_w4a8
SEED=$ROOT/output/janus-pro-7b/w4a8_supplement/seed
LOCK=$ROOT/queue/t2i_w4a8_lane.lock

mkdir -p "$OUT" "$LOG"
exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] another T2I lane queue running"; exit 0; }

log(){ echo "[$(date '+%F %T')] $1"; }
png(){ find "$1/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }
chunks(){ find "$1" -name '*_results.json' 2>/dev/null \
            | sed 's/.*chunk\([0-9]\).*/\1/' | sort -u | wc -l; }
seed_running(){ pgrep -f "eval_benchmarks\.py.*--output_base $SEED/$1" >/dev/null 2>&1; }
t2i_running(){ pgrep -f "eval_generate_quant\.py.*--output_dir $OUT/$1" >/dev/null 2>&1; }

wait_seed(){ # tag
  local tag=$1 n
  while seed_running "$tag"; do sleep 60; done
  n=$(chunks "$SEED/$tag")
  log "SEED $tag released (${n}/5)"
}

run_one(){ # tag gpu method wb ab scale envs
  local tag=$1 gpu=$2 method=$3 wb=$4 ab=$5 scale=$6 envs=$7
  local out="$OUT/$tag" n n2 rc
  n=$(png "$out")
  [ "$n" -ge 2400 ] && { log "SKIP $tag ($n/2400)"; return 0; }
  if t2i_running "$tag"; then log "SKIP $tag (already running)"; return 0; fi
  log "START $tag GPU$gpu (${n}/2400)${scale:+ scale=$(basename "$scale")}"
  CUDA_VISIBLE_DEVICES=$gpu env $envs python -u "$PROJ/eval_generate_quant.py" \
    --model_path "$MODEL" --benchmark t2i_compbench \
    --output_dir "$out" --num_samples 1 \
    --quant_method "$method" --w_bit "$wb" --a_bit "$ab" \
    ${scale:+--scale_path "$scale"} >> "$LOG/$tag.log" 2>&1
  rc=$?
  n2=$(png "$out")
  if [ "$rc" -ne 0 ] && [ "$n2" -le "$n" ]; then
    log "FAIL $tag rc=$rc no-new ($n2/2400); see $LOG/$tag.log"
    touch "$LOG/$tag.FAILED"
  else
    log "EXIT $tag rc=$rc ($n2/2400)"
  fi
}

export PYTHONPATH="$PROJ:$ROOT/checkpoints/Janus:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME="$ROOT/checkpoints/torch_home"

log "T2I W4A8 lane queue armed (2 procs/card, start as each GPU's SEED ends)"

lane0(){ wait_seed rtn;            run_one rtn 0 rtn 4 8 "" "" & \
                                   run_one mbq_scalar 0 mbq_scalar 4 8 "$SC8/mix384/mbq_scalar_w4a8.pt" "" & wait; }
lane1(){ wait_seed mllmq_both;     run_one mllmq_channel_only 1 mllmq_mae 4 8 "$SC8/mix384/mllmq_channel_only_w4a8.pt" "MLLMQ_DISABLE_T2I=1 MLLMQ_DISABLE_GEN_IMG=1" & \
                                   run_one mllmq_t2i_text 1 mllmq_mae 4 8 "$SC8/mix384/mllmq_t2i_text_w4a8.pt" "MLLMQ_DISABLE_GEN_IMG=1" & wait; }
lane2(){ wait_seed awq;            run_one mllmq_t2i_img 2 mllmq_mae 4 8 "$SC8/mix384/mllmq_t2i_img_w4a8.pt" "" & \
                                   run_one mllmq_both 2 mllmq_mae 4 8 "$SC8/mix384/mllmq_both_w4a8.pt" "" & wait; }
lane3(){ wait_seed mllmq_gen_only; run_one awq 3 awq 4 8 "$SC8/mix384/awq_w4a8.pt" "" & \
                                   run_one mllmq_gen_only 3 mllmq_mae 4 8 "$SC8/mix384/mllmq_gen_only_w4a8.pt" "" & wait; }

lane0 & lane1 & lane2 & lane3 &
wait

for t in rtn mbq_scalar mllmq_channel_only mllmq_t2i_text \
         mllmq_t2i_img mllmq_both awq mllmq_gen_only; do
  log "FINAL $t: $(png "$OUT/$t")/2400"
done
log "T2I W4A8 LANES DONE"