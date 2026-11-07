#!/usr/bin/env bash
# ============================================================================
# W4A8 auto-fill (host only, und phase)
#
#   Keeps up to MAX_PER_GPU (default 2) W4A8 understanding eval processes per
#   GPU by launching ONE safe pending tag per invocation.
#
#   Safety:
#     - only `und` is touched; SEED / scales / geneval / DPG are never started;
#     - exits if SEED (run_understanding_7) is running (RAM-bound, 1/GPU);
#     - exits if scales < 10;
#     - if a main und driver is running, only last-wave tags
#       (mllmq_t2i_img / mllmq_both) are eligible, because the main queue will
#       not reach them for >= 2 waves; all other pending tags are left to it;
#     - one launch per invocation; cron runs every 30 min;
#     - flock prevents concurrent auto-fill runs.
#
#   Usage:
#     DRY_RUN=1 bash scripts/w4a8_autofill.sh   # print decision only
#     bash scripts/w4a8_autofill.sh             # real launch
# ============================================================================
set -u

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
OUT=$ROOT/output/janus-pro-7b/w4a8_supplement
LOG=$ROOT/output/logs/w4a8_supplement/autofill.log
LOCK=/tmp/w4a8_autofill.lock

TAGS=(rtn mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img mllmq_both)
LAST_WAVE=(mllmq_t2i_img mllmq_both)
# test hook: override the eligible last-wave tags without editing the file
if [ -n "${LAST_WAVE_OVERRIDE:-}" ]; then read -r -a LAST_WAVE <<< "$LAST_WAVE_OVERRIDE"; fi
GPUS=${GPUS:-"0 1"}
NGPU=$(echo $GPUS | wc -w)
MAX_PER_GPU=${MAX_PER_GPU:-2}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$(dirname "$LOG")"
exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] another autofill running" >> "$LOG"; exit 0; }
ts(){ date '+%F %T'; }
log(){ echo "[$(ts)] $1" >> "$LOG"; [ "$DRY_RUN" = "1" ] && echo "$1"; }

# 1) Hard gates
if pgrep -f 'run_understanding_7\.sh' >/dev/null 2>&1; then
  log "SEED runner active; skip"; exit 0
fi
nscales=$(find "$PROJ/scale_cache/w4a8" -name '*.pt' 2>/dev/null | wc -l)
if [ "$nscales" -lt 10 ]; then
  log "scales incomplete ($nscales/10); skip"; exit 0
fi
done_tags=0
for tag in "${TAGS[@]}"; do
  n=$(find "$OUT/und/$tag" -name '*_results.json' 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && done_tags=$((done_tags + 1))
done
if [ "$done_tags" -ge 6 ]; then
  log "und complete (6/6 tags); skip"; exit 0
fi

# 2) Running und processes: per-GPU count + running tags
declare -A gpu_count
running_tags=""
total=0
for pid in $(pgrep -f 'eval_benchmarks\.py.*w4a8_supplement/und' 2>/dev/null); do
  total=$((total + 1))
  gpu=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  [ -z "$gpu" ] && gpu="?"
  gpu_count[$gpu]=$(( ${gpu_count[$gpu]:-0} + 1 ))
  tag=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | sed -n 's/.*--method_tag \([^ ]*\).*/\1/p')
  [ -n "$tag" ] && running_tags="$running_tags $tag"
done

# 3) Pending tags = no 4/4 results and not currently running
pending=()
for tag in "${TAGS[@]}"; do
  n=$(find "$OUT/und/$tag" -name '*_results.json' 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && continue
  case " $running_tags " in *" $tag "*) continue;; esac
  pending+=("$tag")
done

log "total=$total gpu0=${gpu_count[0]:-0} gpu1=${gpu_count[1]:-0} running='${running_tags# }' pending='${pending[*]:-}'"

[ ${#pending[@]} -eq 0 ] && { log "no pending tag; skip"; exit 0; }

# 4) If the main und driver is alive, only fill the last wave safely.
main_running=$(pgrep -f 'bash scripts/run_w4a8_supplement\.sh$' 2>/dev/null | head -1)
if [ -n "$main_running" ]; then
  filtered=()
  for t in "${pending[@]}"; do
    for lw in "${LAST_WAVE[@]}"; do
      [ "$t" = "$lw" ] && filtered+=("$t")
    done
  done
  if [ ${#filtered[@]} -eq 0 ]; then
    log "main driver alive; no safe last-wave pending; skip"; exit 0
  fi
  pending=("${filtered[@]}")
  log "main driver alive; safe pending='${pending[*]}'"
fi

# 5) Pick the least-loaded GPU with a free slot
pick_gpu=""
for g in $GPUS; do
  c=${gpu_count[$g]:-0}
  if [ "$c" -lt "$MAX_PER_GPU" ]; then
    if [ -z "$pick_gpu" ] || [ "$c" -lt "${gpu_count[$pick_gpu]:-0}" ]; then
      pick_gpu=$g
    fi
  fi
done
[ -z "$pick_gpu" ] && { log "no GPU slot; skip"; exit 0; }

tag=${pending[0]}
if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN would start '$tag' on GPU$pick_gpu"
  exit 0
fi

lane_log="$ROOT/output/logs/w4a8_supplement/autofill_${tag}_$(date +%Y%m%d_%H%M%S).log"
setsid nohup env PHASES=und ONLY_TAGS="$tag" JOBS_PER_GPU=1 GPUS="$pick_gpu" WAIT_SEED=0 \
  bash "$PROJ/scripts/run_w4a8_supplement.sh" > "$lane_log" 2>&1 < /dev/null &
log "START '$tag' GPU$pick_gpu pid=$! log=$lane_log"