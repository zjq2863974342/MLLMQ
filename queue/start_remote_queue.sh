#!/usr/bin/env bash
# Idempotent launcher for the persistent remote queue orchestrator.
# The orchestrator itself holds an flock, so starting twice is safe.
set -u
ROOT=/nas3/kangjia/mllmq
QUEUE=$ROOT/queue
PROJ=$ROOT/MLLMQ
LOG=$ROOT/output/logs/w4a8_supplement
mkdir -p "$LOG"

cd "$PROJ" || exit 2
setsid nohup bash "$QUEUE/w4a8_remote_queue_v2.sh" \
  >> "$LOG/remote_queue_v2_launcher.log" 2>&1 < /dev/null &
echo "launcher pid=$!"