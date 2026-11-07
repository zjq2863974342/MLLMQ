#!/usr/bin/env bash
# ============================================================================
# Tail queue: T2I-CompBench++ P1 for the 副机 (W3A16, 8 configs).
#
# Starts ONLY after the current §8 queue is fully done:
#   - no w4a8_remote_queue_v2 / w4a8_backfill_scheduler / backfill driver
#   - W3A16 DPG 8 tags: 1065 PNGs + non-empty score each
#   - W4A8 DPG 4 tags: 1065 PNGs + non-empty score each
#   - W4A8 SEED 4 tags: 5/5 unique chunks each
#
# Then runs 8 configs with 8 concurrent processes = 2 per GPU
# (user request: 充分利用显卡, 尽量 2 任务/GPU).
#
# Outputs:
#   OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/t2i_compbench_current
#   LOG=/nas3/kangjia/mllmq/output/logs/t2i_compbench_current
# Does NOT score, does NOT touch the legacy t2i_compbench/ directory,
# does NOT run W4A8 T2I, does NOT run mbq_block_masked.
# ============================================================================
set -u

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
OUT=$ROOT/output/janus-pro-7b/t2i_compbench_current
LOG=$ROOT/output/logs/t2i_compbench_current
SC=$PROJ/scale_cache
SUPP=$ROOT/output/janus-pro-7b/w4a8_supplement
W3OUT=$ROOT/output/fixed_ablation_20261103_unified
LOCK=$ROOT/queue/t2i_compbench_tail.lock

mkdir -p "$OUT" "$LOG"

exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] another t2i_compbench tail queue running"; exit 0; }

log(){ echo "[$(date '+%F %T')] $1"; }

count_png(){ find "$1/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }
tag_running(){ pgrep -f "eval_generate_quant.py.*--output_dir $OUT/$1" >/dev/null 2>&1; }
seed_chunks(){
  find "$SUPP/seed/$1" -name '*_results.json' 2>/dev/null \
    | sed 's/.*chunk\([0-9]\).*/\1/' | sort -u | wc -l
}

section8_done(){
  # 1) no queue process alive (bracket avoids pgrep self-match)
  if pgrep -f '[w]4a8_remote_queue_v2|[w]4a8_backfill_scheduler|[r]un_w4a8_backfill_remote|[r]un_w4a8_dpg_shard|[r]un_dpg_unified_remote|[e]val_generate_quant.py' >/dev/null 2>&1; then
    return 1
  fi
  # 2) W4A8 SEED 4 tags: 5/5 chunks
  local t
  for t in mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img; do
    [ "$(seed_chunks "$t")" -ge 5 ] || return 1
  done
  # 3) W4A8 DPG 4 tags: 1065 + score
  for t in rtn mbq_scalar mllmq_channel_only mllmq_t2i_text; do
    [ "$(find "$SUPP/dpg/$t/dpg_bench" -name '*.png' 2>/dev/null | wc -l)" -ge 1065 ] || return 1
    [ -s "$SUPP/dpg_scores/$t.txt" ] || return 1
  done
  # 4) W3A16 DPG 8 tags: 1065 + score
  for t in rtn mbq_block_masked mbq_scalar mllmq_channel_only \
           mllmq_t2i_text mllmq_t2i_img mllmq_both mllmq_gen_only; do
    [ "$(find "$W3OUT/$t/dpg_bench" -name '*.png' 2>/dev/null | wc -l)" -ge 1065 ] || return 1
    [ -s "$W3OUT/dpg_scores/$t.txt" ] || return 1
  done
  return 0
}

progress_line(){
  local s="" t n
  for t in mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img; do
    s="$s $t:$(seed_chunks "$t")/5"
  done
  for t in rtn mbq_scalar mllmq_channel_only mllmq_t2i_text; do
    n=$(find "$SUPP/dpg/$t/dpg_bench" -name '*.png' 2>/dev/null | wc -l)
    s="$s W4dpg/$t:$n"
  done
  echo "$s"
}

# ---------------------------------------------------------------------------
# 0. Wait for §8 ALL DONE (do not jump the queue)
# ---------------------------------------------------------------------------
log "tail queue armed; waiting for §8 ALL DONE"
while ! section8_done; do
  log "wait §8:$(progress_line)"
  sleep 300
done
log "§8 ALL DONE; preparing T2I-CompBench++ P1"

# ---------------------------------------------------------------------------
# 1. GPU idle guard (ignore ~830MiB container-reclaim leftovers)
# ---------------------------------------------------------------------------
busy_checks=0
while true; do
  over=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk '$1>10000{c++} END{print c+0}')
  [ -z "$over" ] && over=0
  if [ "$over" -eq 0 ]; then
    break
  fi
  busy_checks=$((busy_checks + 1))
  log "GPU busy on $over card(s); check $busy_checks/6"
  if [ "$busy_checks" -ge 6 ]; then
    log "ABORT: GPUs still occupied after ~30min; not starting (report to user)"
    touch "$LOG/ABORT_GPU_BUSY"
    exit 3
  fi
  sleep 300
done

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
  local n n2 rc
  n=$(count_png "$OUT/$tag")
  if [ "$n" -ge 2400 ]; then
    log "SKIP $tag ($n/2400)"
    return 0
  fi
  if tag_running "$tag"; then
    log "SKIP $tag (already running; early-overlap job owns it)"
    return 0
  fi
  log "GPU$gpu START $tag ($n/2400)"
  CUDA_VISIBLE_DEVICES=$gpu python -u eval_generate_quant.py \
    --model_path "$MODEL" --benchmark t2i_compbench \
    --output_dir "$OUT/$tag" --num_samples 1 "$@" \
    >> "$LOG/$tag.log" 2>&1
  rc=$?
  n2=$(count_png "$OUT/$tag")
  if [ "$rc" -ne 0 ] && [ "$n2" -le "$n" ]; then
    log "FAIL $tag rc=$rc no-new ($n2/2400); see $LOG/$tag.log"
    touch "$LOG/$tag.FAILED"
  else
    log "EXIT $tag rc=$rc ($n2/2400)"
  fi
}

# ---------------------------------------------------------------------------
# 2. Wave: 8 configs concurrently, 2 per GPU
# ---------------------------------------------------------------------------
log "launch 8 configs, 2 per GPU (GPU0-GPU3)"

run_one rtn                0 --quant_method rtn       --w_bit 3 &
run_one mbq_scalar         0 --quant_method mbq_scalar --w_bit 3 --scale_path "$SC/unified_384/mbq_scalar_w3a16.pt" &
run_one mllmq_channel_only 1 --quant_method mllmq_mae --w_bit 3 --scale_path "$SC/unified_384/mllmq_channel_only_w3a16.pt" &
run_one mllmq_t2i_text     1 --quant_method mllmq_mae --w_bit 3 --scale_path "$SC/unified_384/mllmq_t2i_text_w3a16.pt" &
run_one mllmq_t2i_img      2 --quant_method mllmq_mae --w_bit 3 --scale_path "$SC/unified_384/mllmq_t2i_img_w3a16.pt" &
run_one mllmq_both         2 --quant_method mllmq_mae --w_bit 3 --scale_path "$SC/unified_384/mllmq_both_w3a16.pt" &
run_one awq                3 --quant_method awq       --w_bit 3 --scale_path "$SC/unified_384/awq_w3a16.pt" &
run_one mllmq_gen_only     3 --quant_method mllmq_mae --w_bit 3 --scale_path "$SC/unified_generation_only/mllmq_gen_only_w3a16.pt" &

wait

# ---------------------------------------------------------------------------
# 3. Final counts (no scoring)
# ---------------------------------------------------------------------------
for t in rtn mbq_scalar mllmq_channel_only mllmq_t2i_text \
         mllmq_t2i_img mllmq_both awq mllmq_gen_only; do
  log "FINAL $t: $(count_png "$OUT/$t")/2400"
done
log "TAIL QUEUE DONE"