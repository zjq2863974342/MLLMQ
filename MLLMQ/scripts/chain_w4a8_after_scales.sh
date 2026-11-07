#!/usr/bin/env bash
# ============================================================================
# Chain: wait for the currently running W4A8 scales driver, then start
# `PHASES="und geneval"` on the same host GPUs (skip-existing).
#
# Usage:
#   setsid nohup bash scripts/chain_w4a8_after_scales.sh <driver_pid> \
#     > ../output/logs/w4a8_supplement/chain_watcher_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#
# Safety:
#   - waits for the driver PID (and any lingering main_quant) to exit;
#   - checks the 7 scales actually required by und+geneval;
#   - if any is missing, logs and exits WITHOUT starting evaluation.
# ============================================================================
set -u

DRIVER_PID=${1:-}
ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
SC=$PROJ/scale_cache/w4a8
LOG=$ROOT/output/logs/w4a8_supplement
mkdir -p "$LOG"
ts(){ date '+%F %T'; }

echo "[$(ts)] [chain] watching driver pid=${DRIVER_PID:-none}"
if [ -n "$DRIVER_PID" ]; then
  while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 60; done
fi
echo "[$(ts)] [chain] scales driver exited"

# Defensive: wait for any lingering scale search on this host.
while pgrep -f 'main_quant.py .*scale_cache/w4a8' >/dev/null 2>&1; do
  sleep 60
done
echo "[$(ts)] [chain] no lingering main_quant"

required=(
  "$SC/und128/mbq_scalar_w4a8.pt"
  "$SC/und128/mllmq_channel_only_w4a8.pt"
  "$SC/mix384/mbq_scalar_w4a8.pt"
  "$SC/mix384/mllmq_channel_only_w4a8.pt"
  "$SC/mix384/mllmq_t2i_text_w4a8.pt"
  "$SC/mix384/mllmq_t2i_img_w4a8.pt"
  "$SC/mix384/mllmq_both_w4a8.pt"
)
missing=""
for f in "${required[@]}"; do
  [ -s "$f" ] || missing="$missing $(basename "$f")"
done
if [ -n "$missing" ]; then
  echo "[$(ts)] [chain] scales incomplete:$missing"
  echo "[$(ts)] [chain] und/geneval NOT started; restart scales with PHASES=scales"
  exit 1
fi

cd "$PROJ" || exit 1
# Coordination (2026-11-04): host owns `scales -> und`; remote owns `geneval`.
# After the host W4A8 und finishes, resume the two optional SEED configs
# (mllmq_channel_only / rtn, chunks 2-4; run_understanding_7.sh skips existing).
CHAIN_PHASES=${CHAIN_PHASES:-und}
CHAIN_GPUS=${CHAIN_GPUS:-"0 1"}
OUTLOG="$LOG/driver_${CHAIN_PHASES// /_}_$(date +%Y%m%d_%H%M%S).log"
echo "[$(ts)] [chain] running '$CHAIN_PHASES' on GPUs '$CHAIN_GPUS' (foreground) log=$OUTLOG"
PHASES="$CHAIN_PHASES" GPUS="$CHAIN_GPUS" WAIT_SEED=0 \
  bash scripts/run_w4a8_supplement.sh > "$OUTLOG" 2>&1 < /dev/null
echo "[$(ts)] [chain] W4A8 '$CHAIN_PHASES' finished rc=$?"

if [ "${RUN_SEED_AFTER:-1}" = "1" ]; then
  SEED_SC=$ROOT/MLLMQ/scale_cache/unified_understanding_128
  SEED_OUT=$ROOT/output/fixed_ablation_20261103_understanding_5h_seed
  SEED_LOG=$ROOT/output/logs/understanding_5h_seed
  mkdir -p "$SEED_OUT" "$SEED_LOG"
  export TASKS="seedbench_2_local_chunk0 seedbench_2_local_chunk1 seedbench_2_local_chunk2 seedbench_2_local_chunk3 seedbench_2_local_chunk4"
  export OUTBASE="$SEED_OUT" LOG="$SEED_LOG"
  STAMP=$(date +%Y%m%d_%H%M%S)
  TAG=mllmq_channel_only METHOD=mllmq SCALE="$SEED_SC/mllmq_channel_only_w3a16.pt" GPU=0 \
    setsid nohup bash scripts/run_understanding_7.sh \
    > "$SEED_LOG/full_mllmq_channel_resume_${STAMP}.log" 2>&1 < /dev/null &
  echo "[$(ts)] [chain] launched SEED mllmq_channel_only pid=$!"
  TAG=rtn METHOD=rtn SCALE="" GPU=1 \
    setsid nohup bash scripts/run_understanding_7.sh \
    > "$SEED_LOG/full_rtn_resume_${STAMP}.log" 2>&1 < /dev/null &
  echo "[$(ts)] [chain] launched SEED rtn pid=$!"
fi