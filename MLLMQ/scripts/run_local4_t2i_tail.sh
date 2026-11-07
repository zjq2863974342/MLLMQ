#!/usr/bin/env bash
# ============================================================================
# Host local4 tail queue: T2I-CompBench++ P1 (W4A8 8 configs)
#
# Waits for the running local4 supplement scheduler/driver to finish AND all
# local4 artifacts to be complete, then generates the legacy/Table-F subset:
#   8 categories x 300 prompts = 2400 images/config, num_samples=1, 384x384.
#
# Scope (host only; remote handles W3A16 8):
#   rtn, mbq_scalar, mllmq_channel_only, mllmq_t2i_text,
#   mllmq_t2i_img, mllmq_both, awq, mllmq_gen_only
#
# Output:
#   /nas3/kangjia/mllmq/output/janus-pro-7b/t2i_compbench_current/w4a8/<tag>/
#   /nas3/kangjia/mllmq/output/logs/t2i_compbench_current/<tag>.log
#
# Rules: 2 processes/GPU (8 concurrent, GPU = index % 4); never overwrite legacy
#   output/janus-pro-7b/t2i_compbench/; do not kill other tasks.
# ============================================================================
set -o pipefail

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
SC8=$PROJ/scale_cache/w4a8/mix384
OUT8=$ROOT/output/janus-pro-7b/w4a8_supplement
OUT16=$ROOT/output/fixed_ablation_20261103_unified
UND16=$ROOT/output/fixed_ablation_20261103_understanding_5h
EOUT=$ROOT/output/fixed_ablation_20261103_understanding_5h_seed
OUT=$ROOT/output/janus-pro-7b/t2i_compbench_current/w4a8
LOG=$ROOT/output/logs/t2i_compbench_current
TARGET=2400
LOCK=/tmp/local4_t2i_tail.lock

mkdir -p "$OUT" "$LOG"
exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] another local4 T2I tail is running"; exit 0; }

cd "$PROJ" || exit 2
export PYTHONPATH="$PROJ:$ROOT/checkpoints/Janus:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME="$ROOT/checkpoints/torch_home"

log(){ echo "[$(date '+%F %T')] $1"; }
count_png(){ find "$1" -name '*.png' 2>/dev/null | wc -l; }
seed_chunks(){ find "$1" -name '*_results.json' 2>/dev/null | sed 's/.*chunk\([0-9]\).*/\1/' | sort -u | wc -l; }

check_local4_artifacts(){
  local miss="" t d n
  for t in mllmq_t2i_img mllmq_both awq mllmq_gen_only; do
    [ -s "$OUT8/dpg_scores/$t.txt" ] || miss="$miss score:$t"
  done
  [ -s "$OUT16/dpg_scores/awq.txt" ] || miss="$miss score:w3awq"
  for d in "$OUT8/dpg/mllmq_t2i_img/dpg_bench" "$OUT8/dpg/mllmq_both/dpg_bench" \
           "$OUT8/dpg/awq/dpg_bench" "$OUT8/dpg/mllmq_gen_only/dpg_bench" \
           "$OUT16/awq/dpg_bench"; do
    n=$(count_png "$d"); [ "$n" -ge 1065 ] || miss="$miss dpg:$d($n)"
  done
  for t in rtn mllmq_both awq mllmq_gen_only; do
    n=$(seed_chunks "$OUT8/seed/$t"); [ "$n" -ge 5 ] || miss="$miss seed:$t($n)"
  done
  for t in awq mllmq_gen_only; do
    n=$(seed_chunks "$EOUT/$t"); [ "$n" -ge 5 ] || miss="$miss seed:$t($n)"
  done
  for d in "$OUT8/und/awq" "$OUT8/und/mllmq_gen_only"; do
    n=$(find "$d" -name '*_results.json' 2>/dev/null | wc -l)
    [ "$n" -ge 4 ] || miss="$miss und:$d($n)"
  done
  n=$(find "$UND16/mllmq_gen_only" -name '*_results.json' 2>/dev/null | wc -l)
  [ "$n" -ge 7 ] || miss="$miss und:w3gen($n)"
  for d in "$OUT8/geneval/awq/geneval" "$OUT8/geneval/mllmq_gen_only/geneval"; do
    n=$(count_png "$d"); [ "$n" -ge 600 ] || miss="$miss geneval:$d($n)"
    [ -s "$d/results.jsonl" ] || miss="$miss geneval-results:$d"
  done
  echo "$miss"
}

wait_local4_done(){
  local waited=0 miss
  while pgrep -f '[l]ocal4_supplement_scheduler|[r]un_local4_supplement' >/dev/null; do
    log "waiting for local4 scheduler/driver ..."
    sleep 300
  done
  log "local4 scheduler/driver gone; verifying artifacts"
  while true; do
    miss=$(check_local4_artifacts)
    if [ -z "$miss" ]; then log "local4 artifacts complete"; return 0; fi
    log "local4 incomplete:$miss — wait 300s (monitor may restart local4)"
    sleep 300
    waited=$((waited+1))
    if [ "$waited" -ge 288 ]; then log "❌ still incomplete after 24h; abort tail"; return 1; fi
  done
}

run_one(){ # tag gpu extra args...
  local tag=$1 gpu=$2; shift 2
  local n rc
  n=$(count_png "$OUT/$tag/t2i_compbench")
  if [ "$n" -ge "$TARGET" ]; then log "SKIP $tag ($n/$TARGET)"; return 0; fi
  log "GPU$gpu START $tag ($n/$TARGET)"
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/eval_generate_quant.py" \
    --model_path "$MODEL" --benchmark t2i_compbench \
    --output_dir "$OUT/$tag" --num_samples 1 "$@" >> "$LOG/$tag.log" 2>&1
  rc=$?
  n=$(count_png "$OUT/$tag/t2i_compbench")
  log "GPU$gpu EXIT $tag rc=$rc ($n/$TARGET)"
  [ "$n" -ge "$TARGET" ]
}

launch_tag(){ # tag gpu
  local tag=$1 gpu=$2
  case "$tag" in
    rtn)
      run_one rtn "$gpu" --quant_method rtn --w_bit 4 --a_bit 8 ;;
    mbq_scalar)
      run_one mbq_scalar "$gpu" --quant_method mbq_scalar --w_bit 4 --a_bit 8 \
        --scale_path "$SC8/mbq_scalar_w4a8.pt" ;;
    mllmq_channel_only)
      run_one mllmq_channel_only "$gpu" --quant_method mllmq_mae --w_bit 4 --a_bit 8 \
        --scale_path "$SC8/mllmq_channel_only_w4a8.pt" ;;
    mllmq_t2i_text)
      run_one mllmq_t2i_text "$gpu" --quant_method mllmq_mae --w_bit 4 --a_bit 8 \
        --scale_path "$SC8/mllmq_t2i_text_w4a8.pt" ;;
    mllmq_t2i_img)
      run_one mllmq_t2i_img "$gpu" --quant_method mllmq_mae --w_bit 4 --a_bit 8 \
        --scale_path "$SC8/mllmq_t2i_img_w4a8.pt" ;;
    mllmq_both)
      run_one mllmq_both "$gpu" --quant_method mllmq_mae --w_bit 4 --a_bit 8 \
        --scale_path "$SC8/mllmq_both_w4a8.pt" ;;
    awq)
      run_one awq "$gpu" --quant_method awq --w_bit 4 --a_bit 8 \
        --scale_path "$SC8/awq_w4a8.pt" ;;
    mllmq_gen_only)
      run_one mllmq_gen_only "$gpu" --quant_method mllmq_mae --w_bit 4 --a_bit 8 \
        --scale_path "$SC8/mllmq_gen_only_w4a8.pt" ;;
    *) log "❌ unknown tag $tag"; return 2 ;;
  esac
}

main(){
  log "local4 T2I tail start (W4A8 8 configs, 2400/config)"
  wait_local4_done || return 1

  TAGS=(rtn mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img mllmq_both awq mllmq_gen_only)
  log "launching 8 configs, 2 processes/GPU (GPU = index % 4)"
  for i in "${!TAGS[@]}"; do
    launch_tag "${TAGS[$i]}" $((i % 4)) &
  done
  wait
  for attempt in 1 2; do
    local i t n all=1
    for i in "${!TAGS[@]}"; do
      t=${TAGS[$i]}; n=$(count_png "$OUT/$t/t2i_compbench")
      if [ "$n" -lt "$TARGET" ]; then
        all=0; log "retry attempt $attempt: $t ($n/$TARGET)"
        launch_tag "$t" $((i % 4))
      fi
    done
    [ "$all" -eq 1 ] && break
  done

  local t n ok=1
  for t in "${TAGS[@]}"; do
    n=$(count_png "$OUT/$t/t2i_compbench")
    log "STATUS $t $n/$TARGET"
    [ "$n" -ge "$TARGET" ] || ok=0
  done
  if [ "$ok" -eq 1 ]; then log "LOCAL4 T2I ALL DONE"; else log "❌ LOCAL4 T2I INCOMPLETE"; return 1; fi
}

main