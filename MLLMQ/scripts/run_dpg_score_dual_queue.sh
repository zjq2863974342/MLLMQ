#!/bin/bash
# DPG-Bench 评分双队列（single 段1 · 4 个未评分 config）
# 用法：cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_dpg_score_dual_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/dpg_score_single_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

set -o pipefail
cd /nas3/kangjia/mllmq

# OFA audio-stack shim: mplug VQA needs modelscope's OFA modules importable
# without librosa/soundfile/fairseq (speech-only deps, absent on this host).
# See MLLMQ/scripts/pyshim/sitecustomize.py.
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}"

SCORE_SCRIPT="/nas3/kangjia/mllmq/MLLMQ/scripts/score_dpg_coco.py"
CSV=/nas3/kangjia/mllmq/data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv
OUT=/nas3/kangjia/mllmq/output/dpg_scores
LOG=/nas3/kangjia/mllmq/output/logs
COOLDOWN=30
MAX_RETRY=3

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date +%H:%M:%S)] $1"; }
good() { echo -e "[$(date +%H:%M:%S)] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date +%H:%M:%S)] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date +%H:%M:%S)] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date +%H:%M:%S)] ${CYAN}📌 $1${NC}"; }

result_exists() {
  local f="$1"
  [ -s "$f" ] && grep -q "DPG-Bench score:" "$f"
}

run_one() {
  local tag=$1 image_root=$2 pic_num=$3 gpu=$4
  local out_f="${OUT}/${tag}_results.txt"
  local logf="${LOG}/dpg_score_${tag}.log"

  result_exists "$out_f" && { good "SKIP ${tag} — result exists"; return 0; }

  local attempt=1
  while [ "$attempt" -le "$MAX_RETRY" ]; do
    result_exists "$out_f" && { good "DONE ${tag} — ${out_f}"; return 0; }

    info "[GPU${gpu}] LAUNCH ${tag} (attempt ${attempt}/${MAX_RETRY})"
    CUDA_VISIBLE_DEVICES=$gpu python "$SCORE_SCRIPT" \
        --image-root-path "$image_root" \
        --resolution 384 --pic-num "$pic_num" \
        --csv "$CSV" --tag "$tag" --out "$out_f" > "$logf" 2>&1
    local rc=$?

    if result_exists "$out_f"; then
      good "[GPU${gpu}] DONE ${tag}"
      return 0
    fi
    warn "[GPU${gpu}] INCOMPLETE ${tag} (rc=${rc})"
    attempt=$(( attempt + 1 ))
    sleep $COOLDOWN
  done
  fail "[GPU${gpu}] GAVE-UP ${tag}"
  return 1
}

# GPU0: mllmq_mae_w3 + mbq_block_w3  (2 configs)
# GPU1: awq_w3 + rtn_w3              (2 configs)
# 把可能的慢/重要放 GPU1 以和生成日志习惯一致
queue_gpu0() {
  run_one "mllmq_mae_w3_single" "/nas3/kangjia/mllmq/output/janus-pro-7b/dpg_bench/mllmq_mae_w3/dpg_bench" 1 0 || return 1
  run_one "mbq_block_w3_single" "/nas3/kangjia/mllmq/output/janus-pro-7b/dpg_bench/mbq_block_w3/dpg_bench" 1 0 || return 1
  good "GPU0 queue complete"
}
queue_gpu1() {
  run_one "awq_w3_single" "/nas3/kangjia/mllmq/output/janus-pro-7b/dpg_bench/awq_w3/dpg_bench" 1 1 || return 1
  run_one "rtn_w3_single" "/nas3/kangjia/mllmq/output/janus-pro-7b/dpg_bench/rtn_w3/dpg_bench" 1 1 || return 1
  good "GPU1 queue complete"
}

log "===== DPG-Bench 段1 single 评分 Dual-Queue Driver ====="
log "GPU0: mllmq_mae_w3 → mbq_block_w3"
log "GPU1: awq_w3 → rtn_w3"
log "huggingface_hub: $(python -c 'import huggingface_hub; print(huggingface_hub.__version__)')"

mkdir -p "$OUT" "$LOG"
python -c "import modelscope" 2>/dev/null || { fail "modelscope not installed"; exit 2; }

queue_gpu0 &
PID0=$!
queue_gpu1 &
PID1=$!
wait $PID0; RC0=$?
wait $PID1; RC1=$?

if [ "$RC0" -ne 0 ] || [ "$RC1" -ne 0 ]; then
  fail "评分队列出错 rc0=$RC0 rc1=$RC1"
  exit 1
fi
info "🎉 DPG-Bench 段1 single 评分 ALL DONE"
