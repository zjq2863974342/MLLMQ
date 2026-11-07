#!/usr/bin/env bash
# ============================================================================
# Exp3 理解侧验证：GAMQ mixed k12（3.40 bit）在 7 个理解 benchmark 上
#   对照：已有的 mllmq_mae W3A16 / W4A16（current calib）理解侧结果
#   4 卡按 benchmark 并行（每个进程加载一次模型）
# 产出：output/janus-pro-7b/eval_mixed/<task>/gamt_k12_w3a16/*_results.json
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && GPUS="0 1 2 3" setsid nohup bash scripts/run_exp3_understanding_mixed.sh \
#     > ../output/logs/exp3_und_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
SCALE=$ROOT/MLLMQ/scale_cache/mllmq/janus_mixed_w34_spread_k12.pt
OUTBASE=$ROOT/output/janus-pro-7b/eval_mixed
LOG=$ROOT/output/logs
GPUS=${GPUS:-"0 1 2 3"}
TAG=gamt_k12
TASKS=${TASKS:-"mme_local textvqa_val_local mmmu_val mathvista_testmini_local pope_random_local pope_popular_local pope_adversarial_local"}

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== GAMQ 理解侧：7 benchmarks（4 卡并行）====="
[ -s "$SCALE" ] || { fail "缺 scale $SCALE"; exit 1; }
mkdir -p "$OUTBASE"

run_task(){ # task gpu
  local task=$1 gpu=$2
  local out="$OUTBASE/$task/${TAG}_w3a16"
  [ -d "$out" ] && ls "$out"/*_results.json >/dev/null 2>&1 && { good "[GPU$gpu] SKIP $task"; return 0; }
  info "[GPU$gpu] RUN $task"
  CUDA_VISIBLE_DEVICES=$gpu python eval_benchmarks.py \
    --model_type janus --model_path "$MODEL" --method mllmq_mixed --method_tag "$TAG" \
    --w_bit 3 --a_bit 16 --w_group 128 --scale_path "$SCALE" \
    --tasks "$task" --output_base "$OUTBASE" --device cuda:0 \
    > "$LOG/exp3_und_${task}.log" 2>&1
  [ $? -eq 0 ] && good "[GPU$gpu] DONE $task" || fail "[GPU$gpu] FAIL $task（见 $LOG/exp3_und_${task}.log）"
}

gpus=($GPUS); n=${#gpus[@]}; idx=0
for t in $TASKS; do
  run_task "$t" "${gpus[$((idx % n))]}" &
  idx=$((idx+1))
  if [ $((idx % n)) -eq 0 ]; then wait; fi
done
wait

log "════ GAMQ 理解侧完成（结果在 $OUTBASE）════"
for t in $TASKS; do
  d="$OUTBASE/$t/${TAG}_w3a16"
  ls "$d"/*_results.json >/dev/null 2>&1 && good "$t :: $(ls "$d"/*_results.json | head -1)" || fail "$t ✗"
done
info "ALL DONE"