#!/usr/bin/env bash
# ============================================================================
# Persistent LOCAL scheduler loop (no LLM in the execution path).
#
#   Every INTERVAL seconds (default 60) it calls w4a8_autofill.sh, which does
#   the memory gating and may launch at most one safe und lane.  This loop is
#   intentionally independent of the 30-min LLM watchdog: the LLM only reads
#   logs and recommends rebalancing / aborts.
#
#   Single instance via flock.  Safe to restart.
# ============================================================================
set -u
ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
LOG=$ROOT/output/logs/w4a8_supplement/scheduler_loop.log
LOCK=/tmp/w4a8_scheduler_loop.lock
INTERVAL=${INTERVAL:-60}
mkdir -p "$(dirname "$LOG")"

exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] another scheduler loop running" >> "$LOG"; exit 0; }

while true; do
  echo "[$(date '+%F %T')] tick" >> "$LOG"
  bash "$PROJ/scripts/w4a8_autofill.sh" >> "$LOG" 2>&1
  # keep the heartbeat log bounded (~10MB)
  if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 10485760 ]; then
    mv -f "$LOG" "$LOG.1"
  fi
  sleep "$INTERVAL"
done