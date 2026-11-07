#!/usr/bin/env bash
# ============================================================================
# W4A8 backfill scheduler (local or remote).
#
#   MODE=local  -> run_w4a8_backfill_local.sh  (2 GPUs; DPG 2/GPU cap, SEED 1/GPU)
#   MODE=remote -> run_w4a8_backfill_remote.sh (4 GPUs; same caps)
#
#   Every INTERVAL seconds (default 60):
#     - if the backfill driver is alive: wait;
#     - else if DPG/SEED tasks for this mode are missing: start the driver;
#     - else log ALL DONE and exit.
#
#   The driver itself retries missing tags up to 3 times; this outer loop
#   restarts the whole driver if it dies. skip-existing makes it safe.
#
#   Usage:
#     MODE=local INTERVAL=60 setsid nohup bash scripts/w4a8_backfill_scheduler.sh \
#       > ../output/logs/w4a8_supplement/local_scheduler_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -u
ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
OUT=$ROOT/output/janus-pro-7b/w4a8_supplement
LOG=$ROOT/output/logs/w4a8_supplement
MODE=${MODE:-local}
INTERVAL=${INTERVAL:-60}
DRY_RUN=${DRY_RUN:-0}
LOCK=/tmp/w4a8_backfill_scheduler_${MODE}.lock
mkdir -p "$LOG"

if [ "$MODE" = "remote" ]; then
  DRIVER=run_w4a8_backfill_remote.sh
  SEED_TAGS=${REMOTE_SEED_TAGS:-"mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img"}
  DPG_TAGS=${REMOTE_DPG_TAGS:-"rtn mbq_scalar mllmq_channel_only mllmq_t2i_text"}
else
  DRIVER=run_w4a8_backfill_local.sh
  SEED_TAGS=${LOCAL_SEED_TAGS:-"rtn mllmq_both"}
  DPG_TAGS=${LOCAL_DPG_TAGS:-"mllmq_t2i_img mllmq_both"}
fi
DPG_TARGET=${DPG_TARGET:-1065}

exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] another scheduler ($MODE) running" >> "$LOG/backfill_scheduler_${MODE}.log"; exit 0; }
ts(){ date '+%F %T'; }
log(){ echo "[$(ts)] $1" | tee -a "$LOG/backfill_scheduler_${MODE}.log"; }

dpg_missing(){
  local miss="" n
  for t in $DPG_TAGS; do
    n=$(find "$OUT/dpg/$t/dpg_bench" -name '*.png' 2>/dev/null | wc -l)
    if [ "$n" -lt "$DPG_TARGET" ] || [ ! -s "$OUT/dpg_scores/$t.txt" ]; then miss="$miss $t"; fi
  done
  echo "$miss"
}
seed_missing(){
  local miss="" u
  for t in $SEED_TAGS; do
    u=$(find "$OUT/seed/$t" -name '*_results.json' 2>/dev/null | sed 's/.*chunk\([0-9]\).*/\1/' | sort -u | wc -l)
    [ "$u" -lt 5 ] && miss="$miss $t"
  done
  echo "$miss"
}

log "scheduler start mode=$MODE driver=$DRIVER"
while true; do
  if pgrep -f "run_w4a8_backfill_${MODE}\.sh" >/dev/null 2>&1; then
    log "driver running; wait"
    sleep "$INTERVAL"; continue
  fi
  dm=$(dpg_missing); sm=$(seed_missing)
  if [ -z "$dm" ] && [ -z "$sm" ]; then
    log "ALL DONE (DPG + SEED for $MODE)"; exit 0
  fi
  log "pending: DPG='${dm:-none}' SEED='${sm:-none}'"
  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN would start $DRIVER"; sleep "$INTERVAL"; continue
  fi
  outlog="$LOG/${MODE}_backfill_$(date +%Y%m%d_%H%M%S).log"
  setsid nohup bash "$PROJ/scripts/$DRIVER" > "$outlog" 2>&1 < /dev/null &
  log "started $DRIVER pid=$! log=$outlog"
  sleep "$INTERVAL"
done