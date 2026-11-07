#!/bin/bash
# DPG-Bench 段2 · 4-grid 校验双队列（对齐验证，2026-10-20 建）
#
# 2 config × 1065 prompt × 4 张 = 8520 张（每 prompt 4 张 384×384 拼 2×2 grid 768×768）
#   评分按官方 --pic-num 4 --resolution 384（对 768² 图裁 4 个 384² crop）
#   只跑 fp16 + mllmq_mae_w3（校验 single 口径相对官方 4-grid 的偏移方向+幅度）
#
# 输出：OUT/<tag>/dpg_bench/<item_id>.png（768×768 2×2 grid）
#
# 🔴 纪律：生成在跑时禁新 CUDA 进程；不要与段1 single 同卡并行（两段共用同机同 GPU 会抢卡）
#
# Usage: cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_dpg_grid4_dual_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/dpg_grid4_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=/nas3/kangjia/mllmq/checkpoints/torch_home

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
# 独立目录：段2 grid 图不覆盖段1 single 图（评分 key 同名，两段分开评）
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/dpg_grid4
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=1065
NUM_SAMPLES=4
COOLDOWN=30
MAX_RETRY=3

DRIVER_PID=$$
T0=$(date +%s)

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date +%H:%M:%S)] $1"; }
good() { echo -e "[$(date +%H:%M:%S)] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date +%H:%M:%S)] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date +%H:%M:%S)] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date +%H:%M:%S)] ${CYAN}📌 $1${NC}"; }

count_pngs() { find "${OUT}/$1" -name '*.png' 2>/dev/null | wc -l; }
# 段1 single 图是 num_samples=1 命名 <item_id>.png，段2 grid 也是 <item_id>.png（768²）
# 但两段输出目录不同（dpg_bench vs dpg_grid4），互不干扰 → 无需 legacy 门禁

is_running() {
  ps aux | grep "eval_generate_quant" | grep -v grep | grep -q "${1}"
}

cleanup() {
  warn "Interrupted — killing children..."
  pkill -P $DRIVER_PID 2>/dev/null
  log "Driver exiting."
  exit 130
}
trap cleanup SIGINT SIGTERM

run_one() {
  local tag=$1 method=$2 w=$3 gpu=$4 scale_path=${5:-}
  local outd="${OUT}/${tag}"
  local logf="${LOG}/dpg_grid4_${tag}.log"
  local n=$(count_pngs "$tag")

  if [ "$n" -ge "$TARGET" ]; then
    good "[GPU${gpu}] SKIP $tag — $n/$TARGET already done"
    return 0
  fi
  if is_running "$tag"; then
    warn "[GPU${gpu}] $tag 已有进程($n/$TARGET) → 等它"
    while is_running "$tag"; do sleep 30; done
    n=$(count_pngs "$tag")
    [ "$n" -ge "$TARGET" ] && { good "[GPU${gpu}] SKIP $tag — $n/$TARGET"; return 0; }
  fi

  local attempt=1
  while [ $attempt -le $MAX_RETRY ]; do
    n=$(count_pngs "$tag")
    [ "$n" -ge "$TARGET" ] && break
    sleep $COOLDOWN
    info "[GPU${gpu}] LAUNCH $tag ($method w${w}, $n/$TARGET existing) attempt ${attempt}/${MAX_RETRY}"
    local cmd=(
      python eval_generate_quant.py
      --model_path "$MODEL" --benchmark dpg_bench
      --output_dir "$outd" --num_samples "$NUM_SAMPLES"
      --quant_method "$method" --w_bit "$w"
    )
    [ -n "$scale_path" ] && cmd+=(--scale_path "$scale_path")
    CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" >> "$logf" 2>&1
    local rc=$?
    local after=$(count_pngs "$tag")
    local gain=$(( after - n ))
    if [ "$after" -ge "$TARGET" ]; then
      good "[GPU${gpu}] DONE $tag — $after/$TARGET"
      return 0
    fi
    if [ "$gain" -eq 0 ] && [ "$rc" -ne 0 ]; then
      fail "[GPU${gpu}] ABORT $tag — rc=$rc 且 0 张新增 → 疑环境损坏（见 $logf 末尾）"
      fail "Driver 主动终止，不谎报 ALL DONE。先复验 hub 是否仍 0.36.2 再重拉。"
      kill $DRIVER_PID 2>/dev/null
      exit 4
    fi
    warn "[GPU${gpu}] INCOMPLETE $tag — $after/$TARGET (本次 +$gain, rc=$rc)"
    attempt=$(( attempt + 1 ))
  done
  local final=$(count_pngs "$tag")
  [ "$final" -ge "$TARGET" ] && { good "[GPU${gpu}] DONE $tag — $final/$TARGET"; return 0; }
  fail "[GPU${gpu}] GAVE-UP $tag — $final/$TARGET after $MAX_RETRY attempts"
  return 1
}

gpu_queue() {
  local gpu=$1; shift
  local tasks=("$@")
  info "[GPU${gpu}] Queue started: ${#tasks[@]} tasks"
  for task in "${tasks[@]}"; do
    read -r tag method w scale <<< "$task"
    scale_path="${scale/#__NONE__/}"
    run_one "$tag" "$method" "$w" "$gpu" "$scale_path" || return 1
  done
  good "[GPU${gpu}] Queue complete!"
}

# ════════════════════════════════════════════════════════
#  format: "tag method w_bit scale_path"
#  ════════════════════════════════════════════════════════
GPU0_TASKS=(
  "fp16 fp16 0 __NONE__"
)
GPU1_TASKS=(
  "mllmq_mae_w3 mbq 3 ${SCALE}/mllmq/janus_mae_w3a16.pt"
)

log "===== DPG-Bench 段2 · 4-grid 校验 Dual-Queue Driver ====="
log "目标：2 config × ${TARGET} prompt × ${NUM_SAMPLES} 张 = $(( TARGET * NUM_SAMPLES * 2 )) 张（768×768 2×2 grid）"
log "GPU0: fp16"
log "GPU1: mllmq_mae_w3"
log "每 prompt 4 张 ≈ 48s/easy-4 → 单 config ~7h，双卡并行 ~7h"

# ── scale cache 齐备性 ──
for tag_task in "${GPU0_TASKS[@]}" "${GPU1_TASKS[@]}"; do
  read -r t m w sc <<< "$tag_task"
  sc="${sc/#__NONE__/}"
  if [ -n "$sc" ] && [ ! -f "$sc" ]; then fail "scale cache 缺失: $sc ($t)"; exit 5; fi
done
good "scale cache 齐备"

# ── 前置门禁：不得有其它 GPU 任务 ──
if pgrep -f '[e]val_generate_quant|[s]core_pref|[s]core_fid|[s]core_clip|[m]ain_quant|[e]val_benchmarks' >/dev/null 2>&1; then
  fail "检测到其它 GPU 任务在跑 → 拒绝启动（并发建 CUDA 上下文危险，见 dialog §0）"
  pgrep -af '[e]val_generate_quant|[s]core_pref|[m]ain_quant' | head -3
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

gpu_queue 0 "${GPU0_TASKS[@]}" &
PID0=$!
gpu_queue 1 "${GPU1_TASKS[@]}" &
PID1=$!
wait $PID0; RC0=$?
wait $PID1; RC1=$?

ELAPSED=$(( ($(date +%s) - T0) / 60 ))
echo ""
echo "========================================="
if [ $RC0 -ne 0 ] || [ $RC1 -ne 0 ]; then
  fail "Driver 退出但队列未全绿 (rc0=$RC0 rc1=$RC1) — 逐 config 核对下面矩阵"
else
  info "🎉 DPG-Bench 段2 4-grid — ALL DONE!"
fi
good "Elapsed: ~${ELAPSED} min ($(( ELAPSED / 60 ))h $(( ELAPSED % 60 ))min)"
echo "========================================="
COMPLETED=0
for tag in fp16 mllmq_mae_w3; do
  n=$(count_pngs "$tag")
  if [ "$n" -ge "$TARGET" ]; then good "  $tag: $n/$TARGET"; COMPLETED=$((COMPLETED+1)); else fail "  $tag: $n/$TARGET"; fi
done
info "Overall: $COMPLETED/2 configs complete"
echo "========================================="
log "下一步：段2 全出后评分 → compute_dpg_bench.py --image-root-path <tag>/dpg_bench --resolution 384 --pic-num 4"