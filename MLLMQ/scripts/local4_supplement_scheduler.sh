#!/usr/bin/env bash
# ============================================================================
# LOCAL 4-GPU supplement scheduler (post container restart, 2026-11-05)
#
#   Every INTERVAL seconds:
#     - if run_local4_supplement.sh is alive: wait;
#     - else if any local task is missing: start the driver;
#     - else log ALL DONE and exit.
#
#   Scope (local = complement of remote's 4 configs):
#     W4A8 DPG : t2i_img, both, awq, mllmq_gen_only
#     W4A8 SEED: rtn, both, awq, mllmq_gen_only
#     W4A8     : awq + gen_only full rows (und4/GenEval600/DPG/SEED)
#     W3A16    : awq (DPG resume + score + SEED),
#                gen_only (und7 + SEED; DPG on remote)
#     W3A16 t2i_img/both already complete -> skipped.
#
#   The driver is skip-existing, so restarting after reboot is safe.
# ============================================================================
set -u

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
SC16=$ROOT/MLLMQ/scale_cache
SC8=$ROOT/MLLMQ/scale_cache/w4a8
OUT16=$ROOT/output/fixed_ablation_20261103_unified
OUT8=$ROOT/output/janus-pro-7b/w4a8_supplement
UND16=$ROOT/output/fixed_ablation_20261103_understanding_5h
EOUT=$ROOT/output/fixed_ablation_20261103_understanding_5h_seed
LOG=$ROOT/output/logs/local4_supplement
MODE=${MODE:-local4}
INTERVAL=${INTERVAL:-60}
DPG_TARGET=${DPG_TARGET:-1065}
GENEVAL_TARGET=${GENEVAL_TARGET:-600}
TASK_UND4=(mmmu_val mme_local textvqa_val_local mathvista_testmini_local)
TASK_UND7=(mme_local textvqa_val_local mmmu_val mathvista_testmini_local pope_random_local pope_popular_local pope_adversarial_local)
LOCK=/tmp/local4_supplement_scheduler_${MODE}.lock

mkdir -p "$LOG"
# Remote takeover guard: set by queue/TAKEOVER_BY_REMOTE when another machine
# has taken over the remaining W4A8 T2I tags; do not relaunch locally.
if [ -f "$ROOT/queue/TAKEOVER_BY_REMOTE" ]; then
  echo "[$(date '+%F %T')] REMOTE TAKEOVER active; local4 scheduler exits" >> "$LOG/scheduler.log"
  exit 0
fi
exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] another local4 scheduler running" >> "$LOG/scheduler.log"; exit 0; }

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -lt 4 ]; then
  echo "[$(date '+%F %T')] need 4 GPUs, found $NGPU; refusing to start" | tee -a "$LOG/scheduler.log"
  exit 2
fi

ts(){ date '+%F %T'; }
log(){ echo "[$(ts)] $1" | tee -a "$LOG/scheduler.log"; }
count_png(){ find "$1" -name '*.png' 2>/dev/null | wc -l; }
seed_chunks(){ find "$1" -name '*_results.json' 2>/dev/null | sed 's/.*chunk\([0-9]\).*/\1/' | sort -u | wc -l; }
und_done(){ ls "$1/$2/$3"/*_results.json >/dev/null 2>&1; }

missing(){
  local miss="" t
  # scales
  [ -s "$SC8/und128/awq_w4a8.pt" ] && [ -s "$SC8/mix384/awq_w4a8.pt" ] || miss="$miss W4AWQ-SCALE"
  [ -s "$SC8/mix384/mllmq_gen_only_w4a8.pt" ] || miss="$miss W4GEN-SCALE"
  # W4A8 DPG + scores
  [ "$(count_png "$OUT8/dpg/mllmq_t2i_img/dpg_bench")" -ge "$DPG_TARGET" ] || miss="$miss W4-DPG:t2i_img"
  [ "$(count_png "$OUT8/dpg/mllmq_both/dpg_bench")" -ge "$DPG_TARGET" ] || miss="$miss W4-DPG:both"
  [ "$(count_png "$OUT8/dpg/awq/dpg_bench")" -ge "$DPG_TARGET" ] || miss="$miss W4-DPG:awq"
  [ "$(count_png "$OUT8/dpg/mllmq_gen_only/dpg_bench")" -ge "$DPG_TARGET" ] || miss="$miss W4-DPG:gen_only"
  [ -s "$OUT8/dpg_scores/mllmq_t2i_img.txt" ] || miss="$miss W4-SCORE:t2i_img"
  [ -s "$OUT8/dpg_scores/mllmq_both.txt" ] || miss="$miss W4-SCORE:both"
  [ -s "$OUT8/dpg_scores/awq.txt" ] || miss="$miss W4-SCORE:awq"
  [ -s "$OUT8/dpg_scores/mllmq_gen_only.txt" ] || miss="$miss W4-SCORE:gen_only"
  # W4A8 AWQ und/genel
  for t in "${TASK_UND4[@]}"; do und_done "$OUT8/und/awq" "$t" awq_w4a8 || { miss="$miss W4AWQ-UND:$t"; break; }; done
  [ "$(count_png "$OUT8/geneval/awq/geneval")" -ge "$GENEVAL_TARGET" ] && [ -s "$OUT8/geneval/awq/geneval/results.jsonl" ] || miss="$miss W4AWQ-GENEVAL"
  # W4A8 gen_only und/genel
  for t in "${TASK_UND4[@]}"; do und_done "$OUT8/und/mllmq_gen_only" "$t" mllmq_gen_only_w4a8 || { miss="$miss W4GEN-UND:$t"; break; }; done
  [ "$(count_png "$OUT8/geneval/mllmq_gen_only/geneval")" -ge "$GENEVAL_TARGET" ] && [ -s "$OUT8/geneval/mllmq_gen_only/geneval/results.jsonl" ] || miss="$miss W4GEN-GENEVAL"
  # W4A8 SEED
  [ "$(seed_chunks "$OUT8/seed/rtn")" -ge 5 ] || miss="$miss W4-SEED:rtn"
  [ "$(seed_chunks "$OUT8/seed/mllmq_both")" -ge 5 ] || miss="$miss W4-SEED:both"
  [ "$(seed_chunks "$OUT8/seed/awq")" -ge 5 ] || miss="$miss W4-SEED:awq"
  [ "$(seed_chunks "$OUT8/seed/mllmq_gen_only")" -ge 5 ] || miss="$miss W4-SEED:gen_only"
  # W3A16 AWQ
  [ -s "$SC16/unified_384/awq_w3a16.pt" ] && [ -s "$SC16/unified_understanding_128/awq_w3a16.pt" ] || miss="$miss W3AWQ-SCALE"
  [ "$(count_png "$OUT16/awq/dpg_bench")" -ge "$DPG_TARGET" ] || miss="$miss W3AWQ-DPG"
  [ -s "$OUT16/dpg_scores/awq.txt" ] || miss="$miss W3AWQ-SCORE"
  [ "$(seed_chunks "$EOUT/awq")" -ge 5 ] || miss="$miss W3AWQ-SEED"
  # W3A16 gen_only (DPG is remote; only und7 + SEED local)
  [ -s "$SC16/unified_generation_only/mllmq_gen_only_w3a16.pt" ] || miss="$miss W3GEN-SCALE"
  for t in "${TASK_UND7[@]}"; do und_done "$UND16/mllmq_gen_only" "$t" mllmq_gen_only_w3a16 || { miss="$miss W3GEN-UND:$t"; break; }; done
  [ "$(seed_chunks "$EOUT/mllmq_gen_only")" -ge 5 ] || miss="$miss W3GEN-SEED"
  # W4A8 T2I-CompBench++ P1 (8 tags x 2400)
  for t in rtn mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img mllmq_both awq mllmq_gen_only; do
    [ "$(find "$ROOT/output/janus-pro-7b/t2i_compbench_current/w4a8/$t/t2i_compbench" -name '*.png' 2>/dev/null | wc -l)" -ge 2400 ] || { miss="$miss T2I-W4:$t"; break; }
  done
  echo "$miss"
}

log "local4 scheduler start mode=$MODE GPUs=$NGPU"
while true; do
  if pgrep -f "run_local4_supplement\.sh" >/dev/null 2>&1; then
    log "driver running; wait"; sleep "$INTERVAL"; continue
  fi
  m=$(missing)
  if [ -z "$m" ]; then log "ALL DONE (local4 grid)"; exit 0; fi
  log "pending:$m"
  outlog="$LOG/driver_$(date +%Y%m%d_%H%M%S).log"
  setsid nohup bash "$PROJ/scripts/run_local4_supplement.sh" > "$outlog" 2>&1 < /dev/null &
  log "started local4 driver pid=$! log=$outlog"
  sleep "$INTERVAL"
done