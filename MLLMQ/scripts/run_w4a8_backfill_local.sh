#!/usr/bin/env bash
# ============================================================================
# W4A8 backfill — LOCAL host (2 GPUs).
#
#   ORDER: DPG first (2 tasks/GPU cap), then SEED (strictly 1 proc/GPU).
#   Split with remote:
#     local  DPG : mllmq_t2i_img + mllmq_both
#     local  SEED: rtn + mllmq_both
#     remote DPG : rtn + mbq_scalar + channel_only + t2i_text
#     remote SEED: mbq_scalar + channel_only + t2i_text + t2i_img
#
#   skip-existing; retries missing tags up to 3 times; safe after reboot.
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
OUT=$ROOT/output/janus-pro-7b/w4a8_supplement
LOG=$ROOT/output/logs/w4a8_supplement
mkdir -p "$LOG"

GPUS=${GPUS:-"0 1"}
SEED_TAGS=${LOCAL_SEED_TAGS:-"rtn mllmq_both"}
DPG_TAGS=${LOCAL_DPG_TAGS:-"mllmq_t2i_img mllmq_both"}
DPG_TARGET=${DPG_TARGET:-1065}

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

echo "[$(date '+%F %T')] LOCAL backfill start: DPG='$DPG_TAGS' SEED='$SEED_TAGS' GPUS='$GPUS'"

# ── DPG first: cap 2 tasks/GPU ──────────────────────────────────────────────
for attempt in 1 2 3; do
  miss=$(dpg_missing)
  [ -z "$miss" ] && break
  echo "[$(date '+%F %T')] DPG attempt $attempt missing:$miss"
  PHASES=dpg ONLY_TAGS="$miss" GPUS="$GPUS" WAIT_SEED=0 JOBS_PER_GPU=2 \
    bash scripts/run_w4a8_supplement.sh \
    > "$LOG/local_dpg_$(date +%Y%m%d_%H%M%S).log" 2>&1
  sleep 30
done
if [ -n "$(dpg_missing)" ]; then
  echo "[$(date '+%F %T')] ❌ DPG incomplete:$(dpg_missing)"; exit 1
fi
echo "[$(date '+%F %T')] DPG complete"

# ── SEED second: strictly 1 proc/GPU ────────────────────────────────────────
for attempt in 1 2 3; do
  miss=$(seed_missing)
  [ -z "$miss" ] && break
  echo "[$(date '+%F %T')] SEED attempt $attempt missing:$miss"
  PHASES=seed ONLY_TAGS="$miss" GPUS="$GPUS" WAIT_SEED=0 JOBS_PER_GPU=1 \
    bash scripts/run_w4a8_supplement.sh \
    > "$LOG/local_seed_$(date +%Y%m%d_%H%M%S).log" 2>&1
  sleep 30
done
if [ -n "$(seed_missing)" ]; then
  echo "[$(date '+%F %T')] ❌ SEED incomplete:$(seed_missing)"; exit 1
fi
echo "[$(date '+%F %T')] SEED complete"
echo "[$(date '+%F %T')] LOCAL backfill ALL DONE"