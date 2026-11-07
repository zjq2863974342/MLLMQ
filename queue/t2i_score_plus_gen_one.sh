#!/usr/bin/env bash
# ============================================================================
# One-GPU lane: 1 T2I score + 1 W4A8 generation (+ W4A8 score after generation).
#
# Layout per machine (4 GPUs):
#   GPU0..3: [W3A16 <tag> scoring] + [W4A8 <tag> generation]
#   after that GPU's generation reaches 2400 -> score the W4A8 <tag>.
#
# Usage:
#   TAG=rtn GPU=0 GATE_FILE= bash queue/t2i_score_plus_gen_one.sh
#
# Env:
#   TAG           same config name for W3A16 score and W4A8 generation
#   GPU           physical GPU index
#   GATE_FILE     if non-empty, wait for this file before starting generation
#                 (default: queue/REMOTE_GEN_GO; use GATE_FILE= on host)
#   SCORE_AFTER_GEN  1 (default) -> score W4A8 tag once generation is complete
#   WAIT_PNG      threshold for the W4A8 score phase (default 2400)
# ============================================================================
set -o pipefail

ROOT=/nas3/kangjia/mllmq
TAG=${TAG:?set TAG}
GPU=${GPU:?set GPU}
GATE_FILE=${GATE_FILE-$ROOT/queue/REMOTE_GEN_GO}
SCORE_AFTER_GEN=${SCORE_AFTER_GEN:-1}
SCORE_W3=${SCORE_W3:-1}
WAIT_PNG=${WAIT_PNG:-2400}

log(){ echo "[$(date '+%F %T')] [lane $TAG GPU$GPU] $*"; }

# 1) W3A16 scoring starts immediately (samples already complete).
SPID=""
if [ "$SCORE_W3" = 1 ]; then
  WIDTH=w3a16 TAG="$TAG" GPU="$GPU" WAIT_PNG=0 \
    bash "$ROOT/queue/t2i_current_score_one.sh" &
  SPID=$!
fi

# 2) Wait for the host to release this tag (avoid duplicate generation).
if [ -n "$GATE_FILE" ]; then
  while [ ! -f "$GATE_FILE" ]; do
    log "wait host release gate: $GATE_FILE"
    sleep 60
  done
  log "gate open: $GATE_FILE"
fi

# 3) W4A8 generation (skip-existing; exits 0 immediately if already complete
#    or already running on this host).
TAG="$TAG" GPU="$GPU" bash "$ROOT/queue/t2i_w4a8_gen_one.sh"
GRC=$?
SRC=0
if [ -n "$SPID" ]; then
  wait $SPID
  SRC=$?
fi
log "phase done: gen_rc=$GRC w3score_rc=$SRC"

# 4) W4A8 scoring once that tag is complete.
if [ "$SCORE_AFTER_GEN" = 1 ]; then
  WIDTH=w4a8 TAG="$TAG" GPU="$GPU" WAIT_PNG="$WAIT_PNG" \
    bash "$ROOT/queue/t2i_current_score_one.sh"
fi

log "lane finished"