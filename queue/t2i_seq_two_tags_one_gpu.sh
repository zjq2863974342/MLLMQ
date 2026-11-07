#!/usr/bin/env bash
# Sequentially generate two W4A8 T2I-CompBench tags on ONE GPU.
# Keeps exactly 1 generation process per GPU: wait FIRST -> 2400, then SECOND.
#
# Usage:
#   FIRST=rtn SECOND=mbq_scalar GPU=0 bash queue/t2i_seq_two_tags_one_gpu.sh
# ============================================================================
set -o pipefail

ROOT=/nas3/kangjia/mllmq
OUT=$ROOT/output/janus-pro-7b/t2i_compbench_current/w4a8
FIRST=${FIRST:?set FIRST}
SECOND=${SECOND:?set SECOND}
GPU=${GPU:?set GPU}
TARGET=${TARGET:-2400}

count(){ find "$OUT/$1/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }
log(){ echo "[$(date '+%F %T')] [seq GPU$GPU] $*"; }

log "wait $FIRST ($(count "$FIRST")/$TARGET)"
while :; do
  n=$(count "$FIRST")
  [ "$n" -ge "$TARGET" ] && break
  sleep 60
done

log "$FIRST complete; start $SECOND ($(count "$SECOND")/$TARGET)"
TAG="$SECOND" GPU="$GPU" bash "$ROOT/queue/t2i_w4a8_gen_one.sh"
log "$SECOND done ($(count "$SECOND")/$TARGET)"