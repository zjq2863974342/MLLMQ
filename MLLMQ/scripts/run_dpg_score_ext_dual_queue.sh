#!/bin/bash
# DPG-Bench 段3 single 评分双队列（8 config：W3 残 2 + W4 全 6）
#   生成在对端机器（20:11 起，落共享 NAS）→ 本机段2 grid4 评完两卡空了以后接力评分。
#   与段1 同协议：每 prompt 1 张 384×384，评分 --pic-num 1 --resolution 384。
#
# 🔴 起跑前门禁：本机不得有任何其它 CUDA 任务（含 grid4 评分）。
# 🔴 本脚本会先等各 tag 生成满 1065 张再评（对端还在产图，不能提前评半截）。
#
# 用法：cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_dpg_score_ext_dual_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/dpg_score_ext_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

set -o pipefail
cd /nas3/kangjia/mllmq

# OFA audio-stack shim: mplug VQA needs modelscope's OFA modules importable
# without librosa/soundfile/fairseq (speech-only deps, absent on this host).
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}"

SCORE_SCRIPT="/nas3/kangjia/mllmq/MLLMQ/scripts/score_dpg_coco.py"
CSV=/nas3/kangjia/mllmq/data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv
IMG_ROOT=/nas3/kangjia/mllmq/output/janus-pro-7b/dpg_bench
OUT=/nas3/kangjia/mllmq/output/dpg_scores
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=1065
COOLDOWN=30
MAX_RETRY=3
WAIT_POLL=300          # 等生成落盘轮询间隔（秒）
WAIT_MAX=57600         # 单 tag 最长等待 16h

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date +%H:%M:%S)] $1"; }
good() { echo -e "[$(date +%H:%M:%S)] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date +%H:%M:%S)] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date +%H:%M:%S)] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date +%H:%M:%S)] ${CYAN}📌 $1${NC}"; }

count_pngs() { find "${IMG_ROOT}/$1/dpg_bench" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l; }
result_exists() { local f="$1"; [ -s "$f" ] && grep -q "DPG-Bench score:" "$f"; }

# 等某个 config 生成满 1065 张（对端还在产图）
wait_generated() {
  local cfg=$1 waited=0 n
  while :; do
    n=$(count_pngs "$cfg")
    [ "$n" -ge "$TARGET" ] && { good "[$cfg] 生成齐 $n/$TARGET"; return 0; }
    if [ "$waited" -ge "$WAIT_MAX" ]; then
      fail "[$cfg] 等生成超时（$n/$TARGET，等了 $(( waited / 3600 ))h）→ 跳过"
      return 1
    fi
    info "[$cfg] 等生成中 $n/$TARGET …（已等 $(( waited / 60 ))min）"
    sleep $WAIT_POLL; waited=$(( waited + WAIT_POLL ))
  done
}

run_one() {
  local cfg=$1 gpu=$2 pic_num=$3 image_root=$4
  local tag="${cfg}_single"
  local out_f="${OUT}/${tag}_results.txt"
  local logf="${LOG}/dpg_score_${tag}.log"

  result_exists "$out_f" && { good "SKIP ${tag} — result exists"; return 0; }
  wait_generated "$cfg" || return 1
  result_exists "$out_f" && { good "SKIP ${tag} — result exists"; return 0; }

  local attempt=1
  while [ "$attempt" -le "$MAX_RETRY" ]; do
    info "[GPU${gpu}] LAUNCH ${tag} (attempt ${attempt}/${MAX_RETRY})"
    CUDA_VISIBLE_DEVICES=$gpu python "$SCORE_SCRIPT" \
        --image-root-path "$image_root" \
        --resolution 384 --pic-num "$pic_num" \
        --csv "$CSV" --tag "$tag" --out "$out_f" > "$logf" 2>&1
    local rc=$?

    result_exists "$out_f" && { good "[GPU${gpu}] DONE ${tag}"; return 0; }
    warn "[GPU${gpu}] INCOMPLETE ${tag} (rc=${rc})"
    attempt=$(( attempt + 1 ))
    sleep $COOLDOWN
  done
  fail "[GPU${gpu}] GAVE-UP ${tag}"
  return 1
}

# 队列排布与段3 生成一致（先完的先评）
queue_gpu0() {
  for cfg in mbq_scalar_w3 rtn_w4 mbq_block_w4 mbq_scalar_w4; do
    run_one "$cfg" 0 1 "${IMG_ROOT}/${cfg}/dpg_bench" || return 1
  done
  good "GPU0 ext queue complete"
}
queue_gpu1() {
  for cfg in mllmq_mse_w3 mllmq_mae_w4 awq_w4 mllmq_mse_w4; do
    run_one "$cfg" 1 1 "${IMG_ROOT}/${cfg}/dpg_bench" || return 1
  done
  good "GPU1 ext queue complete"
}

log "===== DPG-Bench 段3 single 评分 Dual-Queue Driver ====="
log "GPU0: mbq_scalar_w3 → rtn_w4 → mbq_block_w4 → mbq_scalar_w4"
log "GPU1: mllmq_mse_w3 → mllmq_mae_w4 → awq_w4 → mllmq_mse_w4"
log "huggingface_hub: $(python -c 'import huggingface_hub; print(huggingface_hub.__version__)')"

mkdir -p "$OUT" "$LOG"
python -c "import modelscope" 2>/dev/null || { fail "modelscope not installed"; exit 2; }

# ── 前置门禁：本机不得有其它 GPU 任务（含 grid4 评分 / 生成）──
GPU_BUSY_PAT='[e]val_generate_quant|[s]core_pref|[s]core_fid|[s]core_clip|[m]ain_quant|[e]val_benchmarks|[s]core_dpg_coco|[c]ompute_dpg_bench|[r]un_dpg_score_dual_queue|[r]un_dpg_score_grid4_dual_queue'
if pgrep -f "$GPU_BUSY_PAT" >/dev/null 2>&1; then
  fail "检测到其它 GPU 任务在跑 → 拒绝启动（等段2 grid4 评完再起）"
  pgrep -af "$GPU_BUSY_PAT" | head -5
  exit 3
fi
for g in 0 1; do
  if CUDA_VISIBLE_DEVICES=$g timeout -s 9 60 python -c \
       "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize()" >/dev/null 2>&1; then
    good "GPU${g} CUDA 上下文自检通过"
  else
    fail "GPU${g} 无法创建 CUDA 上下文 → 拒绝启动"; exit 2
  fi
done

queue_gpu0 &
PID0=$!
queue_gpu1 &
PID1=$!
wait $PID0; RC0=$?
wait $PID1; RC1=$?

echo ""
if [ "$RC0" -ne 0 ] || [ "$RC1" -ne 0 ]; then
  fail "评分队列出错 rc0=$RC0 rc1=$RC1"
else
  info "🎉 DPG-Bench 段3 single 评分 ALL DONE"
fi
for cfg in mbq_scalar_w3 mllmq_mse_w3 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4; do
  f="${OUT}/${cfg}_single_results.txt"
  if result_exists "$f"; then good "  ${cfg}_single: $(grep 'DPG-Bench score:' "$f")"; else fail "  ${cfg}_single: 未完成"; fi
done