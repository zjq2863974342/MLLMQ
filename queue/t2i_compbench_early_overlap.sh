#!/usr/bin/env bash
# ============================================================================
# Early P1 overlap: start 4 of the 8 T2I-CompBench W3A16 configs (1 per GPU)
# while the W4A8 SEED phase runs.
#
# Rationale (user 2026-11-06): the remaining §8 work is ~7-8h; use the idle
# GPU capacity instead of waiting for ALL DONE. SEED is RAM-bound (~56GB
# RSS/proc) and must stay at 1 SEED proc/GPU; it does NOT forbid a second
# non-SEED process per card, so we run 1 P1 + 1 SEED per GPU.
#
# The 4 early tags are the FIRST tag of each GPU's pair in the tail queue's
# fixed mapping:
#   GPU0 rtn, GPU1 mllmq_channel_only, GPU2 mllmq_t2i_img, GPU3 awq
# The tail queue later launches the remaining 4; its in-progress check
# prevents duplicate processes for the early tags, giving 2 P1/GPU after SEED.
#
# Wait for phase2 t2i_text scoring to finish, then launch. Does NOT wait for
# the jobs (setsid detached). Does NOT score.
# ============================================================================
set -u

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
OUT=$ROOT/output/janus-pro-7b/t2i_compbench_current
LOG=$ROOT/output/logs/t2i_compbench_current
SC=$PROJ/scale_cache
LOCK=$ROOT/queue/t2i_compbench_early.lock

mkdir -p "$OUT" "$LOG"

exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] another early-overlap launcher running"; exit 0; }

log(){ echo "[$(date '+%F %T')] $1"; }
count_png(){ find "$1/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }
tag_running(){ pgrep -f "eval_generate_quant.py.*--output_dir $OUT/$1" >/dev/null 2>&1; }

log "early overlap armed; waiting phase2 t2i_text scoring"
while pgrep -f 'score_dpg_coco.py.*mllmq_t2i_text' >/dev/null 2>&1 \
   || pgrep -f 'compute_dpg_bench.py.*mllmq_t2i_text' >/dev/null 2>&1; do
  sleep 60
done
sleep 30
log "phase2 scoring done; launching 4 early P1 tags (1/GPU)"

cd "$PROJ" || { log "ABORT: cannot cd $PROJ"; exit 2; }

export PYTHONPATH="$PROJ:$ROOT/checkpoints/Janus:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_HOME=$ROOT/checkpoints/torch_home

run_one(){
  local tag=$1 gpu=$2; shift 2
  local n avail
  n=$(count_png "$OUT/$tag")
  if [ "$n" -ge 2400 ]; then log "SKIP $tag ($n/2400)"; return 0; fi
  if tag_running "$tag"; then log "SKIP $tag (already running)"; return 0; fi
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if [ "${avail:-0}" -lt 25 ]; then
    log "SKIP $tag: available RAM ${avail}GB < 25GB (SEED protection)"
    return 0
  fi
  log "GPU$gpu START $tag ($n/2400)"
  setsid nohup env CUDA_VISIBLE_DEVICES=$gpu python -u eval_generate_quant.py \
    --model_path "$MODEL" --benchmark t2i_compbench \
    --output_dir "$OUT/$tag" --num_samples 1 "$@" \
    >> "$LOG/$tag.log" 2>&1 < /dev/null &
  log "GPU$gpu pid=$! tag=$tag"
}

run_one rtn                0 --quant_method rtn       --w_bit 3
run_one mllmq_channel_only 1 --quant_method mllmq_mae --w_bit 3 --scale_path "$SC/unified_384/mllmq_channel_only_w3a16.pt"
run_one mllmq_t2i_img      2 --quant_method mllmq_mae --w_bit 3 --scale_path "$SC/unified_384/mllmq_t2i_img_w3a16.pt"
run_one awq                3 --quant_method awq       --w_bit 3 --scale_path "$SC/unified_384/awq_w3a16.pt"

log "early overlap launched (4 tags, 1/GPU)"