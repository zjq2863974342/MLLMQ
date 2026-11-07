#!/usr/bin/env bash
# Remote queue v2: 2 DPG shards per GPU where jobs allow (never >2/GPU).
# Phase1: 3 tags x 2 shards on GPU1-3 while GPU0 scores W3A16.
# Phase2: 4th tag x 2 shards on GPU2-3 (1 each) after phase1; score on GPU2.
# Phase3: after all DPG scores + W3 scores, official remote scheduler -> SEED 1/GPU.
set -u
ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
OUT=$ROOT/output/janus-pro-7b/w4a8_supplement
LOGD=$ROOT/output/logs/w4a8_supplement
W3OUT=$ROOT/output/fixed_ablation_20261103_unified
W3LOG=$ROOT/output/logs/fixed_ablation_20261103_unified
mkdir -p "$LOGD"

LOCK=/nas3/kangjia/mllmq/queue/orchestrator.lock
exec 9>"$LOCK"
flock -n 9 || { echo "another orchestrator v2 running"; exit 0; }
LOG=$LOGD/remote_queue_v2_$(date +%Y%m%d_%H%M%S).log
exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] orchestrator v2 start pid=$$"
cd "$PROJ" || exit 2

W3_TAGS="rtn mbq_block_masked mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img mllmq_both mllmq_gen_only"
W4_TAGS="rtn mbq_scalar mllmq_channel_only mllmq_t2i_text"
TARGET=1065
count_png(){ find "$1/dpg_bench" -name '*.png' 2>/dev/null | wc -l; }
score_ok(){ [ -s "$1" ]; }

# ---------- phase0 ----------
while true; do
  inc=""
  for t in $W3_TAGS; do n=$(count_png "$W3OUT/$t"); [ "$n" -ge "$TARGET" ] || inc="$inc $t($n)"; done
  [ -z "$inc" ] && { echo "[$(date '+%F %T')] phase0 W3 gen complete"; break; }
  if ! pgrep -f 'run_dpg_unified_remote' >/dev/null; then
    echo "[$(date '+%F %T')] phase0 W3 driver absent; restart (skip-existing)"
    WAIT_GENEVAL=0 setsid nohup bash scripts/run_dpg_unified_remote.sh \
      > "$W3LOG/dpg_driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
    sleep 10
  fi
  echo "[$(date '+%F %T')] phase0 waiting W3 gen:$inc"
  sleep 60
done
while pgrep -f 'eval_generate_quant.py.*fixed_ablation_20261103_unified' >/dev/null; do sleep 20; done
echo "[$(date '+%F %T')] phase0 W3 gen procs exited (GPU0 scoring)"

# ---------- phase1: 3 tags x 2 shards, 2/GPU on GPU1-3 ----------
pids=()
launch(){ setsid nohup bash /nas3/kangjia/mllmq/queue/run_w4a8_dpg_shard.sh "$1" "$2" 2 "$3" >/dev/null 2>&1 & pids+=($!); }
launch rtn 0 1;                launch rtn 1 1
launch mbq_scalar 0 2;         launch mbq_scalar 1 2
launch mllmq_channel_only 0 3; launch mllmq_channel_only 1 3
echo "[$(date '+%F %T')] phase1 launched ${#pids[@]} shards (2/GPU on GPU1-3)"
for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date '+%F %T')] phase1 shards done"

# ---------- phase1b: score 3 tags on GPU1; t2i_text shards on GPU2/3 ----------
PHASES=dpg ONLY_TAGS="rtn mbq_scalar mllmq_channel_only" GPUS="1" WAIT_SEED=0 JOBS_PER_GPU=1 \
  bash scripts/run_w4a8_supplement.sh \
  > "$LOGD/remote_dpg_gpu1_score3_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
score3=$!
echo "[$(date '+%F %T')] phase1b score3 pid=$score3; launch t2i_text shards on GPU2/3"
pids2=()
launch2(){ setsid nohup bash /nas3/kangjia/mllmq/queue/run_w4a8_dpg_shard.sh "$1" "$2" 2 "$3" >/dev/null 2>&1 & pids2+=($!); }
launch2 mllmq_t2i_text 0 2
launch2 mllmq_t2i_text 1 3
for p in "${pids2[@]}"; do wait "$p"; done
echo "[$(date '+%F %T')] t2i_text shards done; waiting score3"
wait "$score3"
echo "[$(date '+%F %T')] phase1b done"

# ---------- phase2: score t2i_text; wait all W4 scores ----------
PHASES=dpg ONLY_TAGS="mllmq_t2i_text" GPUS="2" WAIT_SEED=0 JOBS_PER_GPU=1 \
  bash scripts/run_w4a8_supplement.sh \
  > "$LOGD/remote_dpg_gpu2_t2i_text_$(date +%Y%m%d_%H%M%S).log" 2>&1
echo "[$(date '+%F %T')] phase2 t2i_text scoring returned"

while true; do
  miss=""; for t in $W4_TAGS; do score_ok "$OUT/dpg_scores/$t.txt" || miss="$miss $t"; done
  [ -z "$miss" ] && break
  echo "[$(date '+%F %T')] phase2 waiting W4 scores:$miss"
  sleep 60
done
while pgrep -f 'scripts/run_w4a8_supplement.sh' >/dev/null; do sleep 20; done
echo "[$(date '+%F %T')] all W4A8 DPG scores ready"

# ---------- phase3: W3 scores + official scheduler (SEED 1/GPU) ----------
for attempt in 1 2 3; do
  miss=""; for t in $W3_TAGS; do score_ok "$W3OUT/dpg_scores/$t.txt" || miss="$miss $t"; done
  [ -z "$miss" ] && break
  if ! pgrep -f 'run_dpg_unified_remote' >/dev/null; then
    echo "[$(date '+%F %T')] phase3 W3 driver absent, restart (skip-existing)"
    WAIT_GENEVAL=0 setsid nohup bash scripts/run_dpg_unified_remote.sh \
      > "$W3LOG/dpg_driver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
    sleep 10
  fi
  echo "[$(date '+%F %T')] phase3 waiting W3 scores:$miss"
  sleep 60
done
while pgrep -f 'run_dpg_unified_remote' >/dev/null; do sleep 20; done
echo "[$(date '+%F %T')] W3 scores complete; starting remote scheduler"

if ! pgrep -f 'w4a8_backfill_scheduler.sh' >/dev/null; then
  MODE=remote INTERVAL=60 setsid nohup bash scripts/w4a8_backfill_scheduler.sh \
    > "$LOGD/remote_scheduler_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
  sleep 10
fi
for i in $(seq 1 1440); do
  pgrep -f 'w4a8_backfill_scheduler.sh' >/dev/null || { echo "[$(date '+%F %T')] scheduler exited"; break; }
  sleep 60
done
echo "[$(date '+%F %T')] orchestrator v2 done"