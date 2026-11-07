#!/bin/bash
# COCO-val 双队列 — 每 GPU 独立推进，生成 FID+CLIP+偏好分评测用图
# 25k captions (5k img × 5) × 4 samples = 100k 张/config
# Usage: setsid nohup bash MLLMQ/scripts/run_coco_val_dual_queue.sh \
#           > output/logs/coco_val_dual_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/coco_val
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=5000    # 5k prompts (1 caption/image) × 1 sample
COOLDOWN=30

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
  local logf="${LOG}/coco_val_${tag}.log"
  local n=$(count_pngs "$tag")

  if [ "$n" -ge "$TARGET" ]; then
    good "[GPU${gpu}] SKIP $tag — $n/$TARGET already done"
    return 0
  fi

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

  sleep $COOLDOWN

  info "[GPU${gpu}] LAUNCH $tag ($method w${w}, $n/$TARGET existing)"
  local cmd=(
    python eval_generate_quant.py
    --model_path "$MODEL" --benchmark coco_val
    --output_dir "$outd" --num_samples 1
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
#  GenEval 评测矩阵同 Layout（W8→W4→W3 交错到双 GPU）
#  format: "tag method w_bit scale_path" (scale_path=__NONE__ for rtn)
# ════════════════════════════════════════════════════════

GPU0_TASKS=(
  # fp16 baseline
  "fp16 fp16 0 __NONE__"
  # W4
  "rtn_w4 rtn 4 __NONE__"
  "mbq_block_w4 mbq 4 ${SCALE}/mbq/janus_mbq_block_w4a16.pt"
  "mllmq_mae_w4 mbq 4 ${SCALE}/mllmq/janus_mae_w4a16.pt"
)

GPU1_TASKS=(
  # W4
  "awq_w4 awq 4 ${SCALE}/awq/janus_awq_w4a16.pt"
  "mbq_scalar_w4 mbq 4 ${SCALE}/mbq/janus_mbq_scalar_w4a16.pt"
  "mllmq_mse_w4 mbq 4 ${SCALE}/mllmq/janus_mse_w4a16.pt"
)

# ════════════════════════════════════════════════════════
#  启动
# ════════════════════════════════════════════════════════

log "===== COCO-val Dual-Queue Driver ====="
log "Benchmark: coco_val (5k prompts × 1 sample = 5k/config)"
log "GPU0: 4 tasks (incl. fp16) | GPU1: 3 tasks"
log "Strategy: independent per-GPU queues, ${COOLDOWN}s cooldown between tasks"
log "ETA: ~17h/config, wall-clock ~68h (GPU0 bottleneck)"

# ── 前置门禁 ──
if pgrep -f '[e]val_generate_quant|[e]valuate_images|[m]ain_quant|[e]val_benchmarks' >/dev/null 2>&1; then
  fail "检测到其他 GPU 任务在跑 → 拒绝启动（并发建 CUDA 上下文危险）"
  exit 3
fi
for g in 0 1; do
  if CUDA_VISIBLE_DEVICES=$g timeout -s 9 60 python -c \
       "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize()" >/dev/null 2>&1; then
    good "GPU${g} CUDA 上下文自检通过"
  else
    fail "GPU${g} 无法创建 CUDA 上下文 → 拒绝启动"
    exit 2
  fi
done

# 双队列并行
gpu_queue 0 "${GPU0_TASKS[@]}" &
PID0=$!
gpu_queue 1 "${GPU1_TASKS[@]}" &
PID1=$!

wait $PID0; RC0=$?
wait $PID1; RC1=$?

ELAPSED=$(( ($(date +%s) - T0) / 60 ))
echo ""
echo "========================================="
info "🎉 COCO-val Dual-Queue — ALL DONE!"
echo "========================================="
good "Elapsed: ~${ELAPSED} min ($(( ELAPSED / 60 ))h $(( ELAPSED % 60 ))min)"
echo ""
log "Final matrix:"
for tag in fp16 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4; do
	  n=$(count_pngs "$tag")
	  [ "$n" -ge "$TARGET" ] && good "  $tag: $n/$TARGET" || fail "  $tag: $n/$TARGET"
done

COMPLETED=0
for tag in fp16 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4; do
	  [ "$(count_pngs "$tag")" -ge "$TARGET" ] && COMPLETED=$((COMPLETED + 1))
done
info "Overall: $COMPLETED/7 configs complete"
echo "========================================="
