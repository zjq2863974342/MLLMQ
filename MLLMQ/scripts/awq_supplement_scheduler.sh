#!/usr/bin/env bash
# ============================================================================
# AWQ supplement scheduler — LOCAL host.
#
#   Every INTERVAL seconds:
#     - if run_awq_supplement_local.sh is alive: wait;
#     - else if any AWQ stage is missing: start the driver;
#     - else log ALL DONE and exit.
#
#   The driver is skip-existing, so restarting after reboot is safe.
#   Only one scheduler/driver at a time (flock).
# ============================================================================
set -u

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
SC8=$ROOT/MLLMQ/scale_cache/w4a8
OUT16=$ROOT/output/fixed_ablation_20261103_unified
OUT8=$ROOT/output/janus-pro-7b/w4a8_supplement
EOUT=$ROOT/output/fixed_ablation_20261103_understanding_5h_seed
LOG=$ROOT/output/logs/awq_supplement
MODE=${MODE:-local}
INTERVAL=${INTERVAL:-60}
DPG_TARGET=${DPG_TARGET:-1065}
GENEVAL_TARGET=${GENEVAL_TARGET:-600}
TASK_UND=(mmmu_val mme_local textvqa_val_local mathvista_testmini_local)
LOCK=/tmp/awq_supplement_scheduler_${MODE}.lock

mkdir -p "$LOG"
exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] another AWQ scheduler running" >> "$LOG/scheduler.log"; exit 0; }

ts(){ date '+%F %T'; }
log(){ echo "[$(ts)] $1" | tee -a "$LOG/scheduler.log"; }
count_png(){ find "$1" -name '*.png' 2>/dev/null | wc -l; }
seed_chunks(){ find "$1" -name '*_results.json' 2>/dev/null | sed 's/.*chunk\([0-9]\).*/\1/' | sort -u | wc -l; }
und_done(){ ls "$OUT8/und/awq/$1/awq_w4a8"/*_results.json >/dev/null 2>&1; }

missing(){
  local miss=""
  [ "$(count_png "$OUT16/awq/dpg_bench")" -ge "$DPG_TARGET" ] && [ -s "$OUT16/dpg_scores/awq.txt" ] || miss="$miss W3-DPG"
  [ "$(seed_chunks "$EOUT/awq")" -ge 5 ] || miss="$miss W3-SEED"
  [ -s "$SC8/und128/awq_w4a8.pt" ] && [ -s "$SC8/mix384/awq_w4a8.pt" ] || miss="$miss W4-SCALE"
  for t in "${TASK_UND[@]}"; do und_done "$t" || { miss="$miss W4-UND:$t"; break; }; done
  [ "$(count_png "$OUT8/geneval/awq/geneval")" -ge "$GENEVAL_TARGET" ] && [ -s "$OUT8/geneval/awq/geneval/results.jsonl" ] || miss="$miss W4-GENEVAL"
  [ "$(count_png "$OUT8/dpg/awq/dpg_bench")" -ge "$DPG_TARGET" ] && [ -s "$OUT8/dpg_scores/awq.txt" ] || miss="$miss W4-DPG"
  [ "$(seed_chunks "$OUT8/seed/awq")" -ge 5 ] || miss="$miss W4-SEED"
  echo "$miss"
}

log "AWQ scheduler start mode=$MODE"
while true; do
  if pgrep -f "run_awq_supplement_local\.sh" >/dev/null 2>&1; then
    log "driver running; wait"; sleep "$INTERVAL"; continue
  fi
  m=$(missing)
  if [ -z "$m" ]; then log "ALL DONE (AWQ W3A16 + W4A8)"; exit 0; fi
  log "pending:$m"
  outlog="$LOG/awq_backfill_$(date +%Y%m%d_%H%M%S).log"
  setsid nohup bash "$PROJ/scripts/run_awq_supplement_local.sh" > "$outlog" 2>&1 < /dev/null &
  log "started AWQ driver pid=$! log=$outlog"
  sleep "$INTERVAL"
done