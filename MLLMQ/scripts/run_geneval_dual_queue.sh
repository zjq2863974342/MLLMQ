#!/bin/bash
# GenEval 双独立队列 — 每 GPU 独立推进，一卡空闲即接下一个任务
# 不再等待双卡同步：GPU0 跑完一个立刻接下一个，GPU1 同理。
# Usage: nohup bash scripts/run_geneval_dual_queue.sh > output/logs/geneval_dual_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/geneval
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=2212
COOLDOWN=30  # GPU 释放后冷却秒数

DRIVER_PID=$$
T0=$(date +%s)

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date +%H:%M:%S)] $1"; }
good() { echo -e "[$(date +%H:%M:%S)] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date +%H:%M:%S)] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date +%H:%M:%S)] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date +%H:%M:%S)] ${CYAN}📌 $1${NC}"; }

count_pngs() { find "${OUT}/$1" -name '*.png' 2>/dev/null | wc -l; }

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

# ── 单任务执行（阻塞式） ──────────────────────────────

run_one() {
  local tag=$1 method=$2 w=$3 gpu=$4 scale_path=${5:-}
  local outd="${OUT}/${tag}"
  local logf="${LOG}/geneval_${tag}.log"
  local n=$(count_pngs "$tag")

  # Skip done
  if [ "$n" -ge "$TARGET" ]; then
    good "[GPU${gpu}] SKIP $tag — $n/$TARGET already done"
    return 0
  fi

  # Skip if another process is already running for this tag
  if is_running "$tag"; then
    warn "[GPU${gpu}] $tag has existing process ($n/$TARGET) — waiting..."
    while is_running "$tag"; do sleep 30; done
    n=$(count_pngs "$tag")
    if [ "$n" -ge "$TARGET" ]; then
      good "[GPU${gpu}] SKIP $tag — $n/$TARGET (completed by existing process)"
      return 0
    fi
    warn "[GPU${gpu}] $tag process exited but only $n/$TARGET png — relaunching"
  fi

  # Cooldown after previous GPU task
  sleep $COOLDOWN

  info "[GPU${gpu}] LAUNCH $tag ($method w${w}, $n/$TARGET existing)"
  local cmd=(
    python eval_generate_quant.py
    --model_path "$MODEL" --benchmark geneval
    --output_dir "$outd" --num_samples 4
    --quant_method "$method" --w_bit "$w"
  )
  [ -n "$scale_path" ] && cmd+=(--scale_path "$scale_path")

  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" >> "$logf" 2>&1
  local rc=$?
  local final=$(count_pngs "$tag")
  if [ "$final" -ge "$TARGET" ]; then
    good "[GPU${gpu}] DONE $tag — $final/$TARGET"
  else
    fail "[GPU${gpu}] FAIL $tag — $final/$TARGET (rc=$rc)"
  fi
  return $rc
}

# ── GPU 独立队列 ──────────────────────────────────────

gpu_queue() {
  local gpu=$1; shift
  local tasks=("$@")
  info "[GPU${gpu}] Queue started: ${#tasks[@]} tasks"
  for task in "${tasks[@]}"; do
    read -r tag method w scale <<< "$task"
    scale_path="${scale/#__NONE__/}"
    run_one "$tag" "$method" "$w" "$gpu" "$scale_path"
  done
  good "[GPU${gpu}] Queue complete!"
}

# ════════════════════════════════════════════════════════
#  任务分配（W8→W4→W3 交错到双 GPU）
#  format: "tag method w_bit scale_path" (scale_path=__NONE__ for rtn)
# ════════════════════════════════════════════════════════

GPU0_TASKS=(
  # W8
  "mbq_scalar_w8 mbq 8 ${SCALE}/mbq/janus_mbq_scalar_w8a16.pt"
  "mllmq_mse_w8 mbq 8 ${SCALE}/mllmq/janus_mse_w8a16.pt"
  # W4
  "rtn_w4 rtn 4 __NONE__"
  "mbq_block_w4 mbq 4 ${SCALE}/mbq/janus_mbq_block_w4a16.pt"
  "mllmq_mae_w4 mbq 4 ${SCALE}/mllmq/janus_mae_w4a16.pt"
  # W3
  "rtn_w3 rtn 3 __NONE__"
  "mbq_block_w3 mbq 3 ${SCALE}/mbq/janus_mbq_block_w3a16.pt"
  "mllmq_mae_w3 mbq 3 ${SCALE}/mllmq/janus_mae_w3a16.pt"
)

GPU1_TASKS=(
  # W8
  "mllmq_mae_w8 mbq 8 ${SCALE}/mllmq/janus_mae_w8a16.pt"
  # W4
  "awq_w4 awq 4 ${SCALE}/awq/janus_awq_w4a16.pt"
  "mbq_scalar_w4 mbq 4 ${SCALE}/mbq/janus_mbq_scalar_w4a16.pt"
  "mllmq_mse_w4 mbq 4 ${SCALE}/mllmq/janus_mse_w4a16.pt"
  # W3
  "awq_w3 awq 3 ${SCALE}/awq/janus_awq_w3a16.pt"
  "mbq_scalar_w3 mbq 3 ${SCALE}/mbq/janus_mbq_scalar_w3a16.pt"
  "mllmq_mse_w3 mbq 3 ${SCALE}/mllmq/janus_mse_w3a16.pt"
)

# ════════════════════════════════════════════════════════
#  启动
# ════════════════════════════════════════════════════════

log "===== GenEval Dual-Queue Driver ====="
log "GPU0: ${#GPU0_TASKS[@]} tasks | GPU1: ${#GPU1_TASKS[@]} tasks"
log "Strategy: independent per-GPU queues, ${COOLDOWN}s cooldown between tasks"

# ── 前置门禁（2026-10-09 driver-lock 事故纪律，dialog §0 / TODO Phase 3 Step 0b）──
# 教训: GPU0 曾 nvidia-smi 全绿但建不出 CUDA 上下文；脚本无 set -e 时 GPU0 队列会
#       8 个任务连环秒失败后仍打印 "Queue complete!"，静默丢一半产能。
# 因此启动前必须: ① 无其他 GPU 任务在跑（避免并发建上下文复发）② 双卡真实 CUDA 分配自检。
if pgrep -f 'eval_generate_quant|evaluate_images|main_quant|eval_benchmarks' >/dev/null 2>&1; then
  fail "检测到其他 GPU 任务在跑 → 按纪律 §0 拒绝启动（并发建上下文正是 11:48 挂死触发形态）"
  exit 3
fi
for g in 0 1; do
  if CUDA_VISIBLE_DEVICES=$g timeout -s 9 60 python -c \
       "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize()" >/dev/null 2>&1; then
    good "GPU${g} CUDA 上下文自检通过"
  else
    fail "GPU${g} 无法创建 CUDA 上下文（nvidia-smi 可能是假绿，见 dialog 问题 10）→ 拒绝启动"
    warn "不要单卡降级跑（用户 2026-10-09 14:49 决策）；等平台 host GPU reset 后再拉起本脚本"
    exit 2
  fi
done

# 两队列并行启动
gpu_queue 0 "${GPU0_TASKS[@]}" &
PID0=$!
gpu_queue 1 "${GPU1_TASKS[@]}" &
PID1=$!

wait $PID0; RC0=$?
wait $PID1; RC1=$?

ELAPSED=$(( ($(date +%s) - T0) / 60 ))
echo ""
echo "========================================="
info "🎉 GenEval Dual-Queue — ALL DONE!"
echo "========================================="
good "Elapsed: ~${ELAPSED} min ($(( ELAPSED / 60 ))h $(( ELAPSED % 60 ))min)"
echo ""
log "Final matrix:"
for tag in fp16 rtn_w8 rtn_w4 rtn_w3 \
           awq_w8 awq_w4 awq_w3 \
           mbq_block_w8 mbq_block_w4 mbq_block_w3 \
           mbq_scalar_w8 mbq_scalar_w4 mbq_scalar_w3 \
           mllmq_mae_w8 mllmq_mae_w4 mllmq_mae_w3 \
           mllmq_mse_w8 mllmq_mse_w4 mllmq_mse_w3; do
  n=$(count_pngs "$tag")
  [ "$n" -ge "$TARGET" ] && good "  $tag: $n/$TARGET" || fail "  $tag: $n/$TARGET"
done

COMPLETED=0
for tag in fp16 rtn_w8 rtn_w4 rtn_w3 \
  awq_w8 awq_w4 awq_w3 \
  mbq_block_w8 mbq_block_w4 mbq_block_w3 \
  mbq_scalar_w8 mbq_scalar_w4 mbq_scalar_w3 \
  mllmq_mae_w8 mllmq_mae_w4 mllmq_mae_w3 \
  mllmq_mse_w8 mllmq_mse_w4 mllmq_mse_w3; do
  [ "$(count_pngs "$tag")" -ge "$TARGET" ] && COMPLETED=$((COMPLETED + 1))
done
info "Overall: $COMPLETED/19 configs complete"
echo "========================================="