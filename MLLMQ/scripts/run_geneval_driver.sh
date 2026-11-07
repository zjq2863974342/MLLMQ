#!/bin/bash
# GenEval 全量队列驱动 — 双 GPU 并行，支持断点续跑和 skip-done
# Usage:  nohup bash scripts/run_geneval_driver.sh > output/logs/geneval_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# 或前台: bash scripts/run_geneval_driver.sh
#
# 安全机制:
#   1. 启动前检查：如果已有同名输出目录的进程在跑，等待其完成（不重复启动）
#   2. Skip-done: png >= 2212 自动跳过
#   3. Resume: eval_generate_quant.py 内部检查 os.path.exists(save_path) → continue
#   4. 错峰: GPU0 先启动，90s 后 GPU1 启动，避免同时 apply_quant VRAM 尖峰
#   5. 每轮结束后等待两进程都退出，再启动下一轮

set -o pipefail

# ── 环境 ──────────────────────────────────────────────
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/geneval
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
STAGGER=90        # 错峰秒数
TARGET=2212       # 每 config 预期图片数

DRIVER_PID=$$
T0=$(date +%s)
ROUND=0
COMPLETED=0

# ── 颜色 ──────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "[$(date +%H:%M:%S)] $1"; }
good() { echo -e "[$(date +%H:%M:%S)] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date +%H:%M:%S)] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date +%H:%M:%S)] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date +%H:%M:%S)] ${CYAN}📌 $1${NC}"; }

# ── 工具函数 ──────────────────────────────────────────

count_pngs() {
  find "${OUT}/$1" -name '*.png' 2>/dev/null | wc -l
}

is_running() {
  # 检查是否有同名 tag 的 eval_generate_quant 进程正在运行
  local tag=$1
  ps aux | grep "eval_generate_quant" | grep -v grep | grep -q "${tag}"
}

wait_for_existing() {
  local tag=$1
  log "Waiting for existing $tag process to finish..."
  while is_running "$tag"; do
    sleep 30
    local n=$(count_pngs "$tag")
    echo -n "  ${tag}: ${n}/${TARGET} png  "
  done
  echo ""
}

# ── 单 cell 执行 ──────────────────────────────────────

run_cell() {
  local tag=$1 method=$2 w=$3 gpu=$4
  local scale_path=${5:-}
  local outd="${OUT}/${tag}"
  local logf="${LOG}/geneval_${tag}.log"

  ROUND=$((ROUND + 1))
  local n=$(count_pngs "$tag")

  # ─ Skip done ─
  if [ "$n" -ge "$TARGET" ]; then
    good "SKIP $tag — $n/$TARGET already done"
    COMPLETED=$((COMPLETED + 1))
    return 0
  fi

  # ─ 已有进程在跑 → 等它结束 ─
  if is_running "$tag"; then
    warn "$tag has existing process — waiting for it ($n/$TARGET png)"
    wait_for_existing "$tag"
    n=$(count_pngs "$tag")
    if [ "$n" -ge "$TARGET" ]; then
      good "SKIP $tag — $n/$TARGET (completed by existing process)"
      COMPLETED=$((COMPLETED + 1))
      return 0
    fi
  fi

  # ─ 启动 ─
  info "LAUNCH $tag on GPU${gpu} ($method w${w}, $n/$TARGET existing)"
  local cmd=(
    python eval_generate_quant.py
    --model_path "$MODEL"
    --benchmark geneval
    --output_dir "$outd"
    --num_samples 4
    --quant_method "$method"
    --w_bit "$w"
  )
  [ -n "$scale_path" ] && cmd+=(--scale_path "$scale_path")

  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "${cmd[@]}" >> "$logf" 2>&1
    local rc=$?
    local final=$(find "${outd}" -name '*.png' 2>/dev/null | wc -l)
    if [ "$final" -ge "$TARGET" ]; then
      good "DONE $tag — $final/$TARGET (rc=$rc)"
    else
      fail "FAIL $tag — $final/$TARGET (rc=$rc)"
    fi
    exit $rc
  ) &
  echo $!
}

# ── 清理 ──────────────────────────────────────────────

cleanup() {
  warn "Interrupted — killing child processes..."
  pkill -P $DRIVER_PID 2>/dev/null
  log "Driver exiting."
  exit 130
}
trap cleanup SIGINT SIGTERM

# ── 等待双进程完成 ────────────────────────────────────

wait_pair() {
  local pid0=$1 pid1=$2
  local failed=0
  wait $pid0 2>/dev/null || failed=$((failed + 1))
  wait $pid1 2>/dev/null || failed=$((failed + 1))
  sleep $STAGGER
  return $failed
}

# ════════════════════════════════════════════════════════
#  任务矩阵
# ════════════════════════════════════════════════════════

log "===== GenEval Driver Start ====="
log "W8 → W4 → W3, 2 GPU parallel, ${STAGGER}s stagger"
log "Target: ${TARGET} png/config = 553 prompts × 4 samples"

# ─── W8 Batch ─────────────────────────────────────────
info "=== W8 BATCH (rtn_w8 ✅ done) ==="

# Round W8-1: awq_w8, mbq_block_w8
PID0=$(run_cell awq_w8 awq 8 0 "${SCALE}/awq/janus_awq_w8a16.pt")
sleep $STAGGER
PID1=$(run_cell mbq_block_w8 mbq 8 1 "${SCALE}/mbq/janus_mbq_block_w8a16.pt")
wait_pair $PID0 $PID1

# Round W8-2: mbq_scalar_w8, mllmq_mae_w8
PID0=$(run_cell mbq_scalar_w8 mbq 8 0 "${SCALE}/mbq/janus_mbq_scalar_w8a16.pt")
sleep $STAGGER
PID1=$(run_cell mllmq_mae_w8 mbq 8 1 "${SCALE}/mllmq/janus_mae_w8a16.pt")
wait_pair $PID0 $PID1

# Round W8-3: mllmq_mse_w8 (solo)
PID0=$(run_cell mllmq_mse_w8 mbq 8 0 "${SCALE}/mllmq/janus_mse_w8a16.pt")
wait $PID0 2>/dev/null
sleep $STAGGER

info "=== W8 BATCH COMPLETE ==="

# ─── W4 Batch ─────────────────────────────────────────
info "=== W4 BATCH ==="

# Round W4-1: rtn_w4 (resume), awq_w4
PID0=$(run_cell rtn_w4 rtn 4 0 "")
sleep $STAGGER
PID1=$(run_cell awq_w4 awq 4 1 "${SCALE}/awq/janus_awq_w4a16.pt")
wait_pair $PID0 $PID1

# Round W4-2: mbq_block_w4, mbq_scalar_w4
PID0=$(run_cell mbq_block_w4 mbq 4 0 "${SCALE}/mbq/janus_mbq_block_w4a16.pt")
sleep $STAGGER
PID1=$(run_cell mbq_scalar_w4 mbq 4 1 "${SCALE}/mbq/janus_mbq_scalar_w4a16.pt")
wait_pair $PID0 $PID1

# Round W4-3: mllmq_mae_w4, mllmq_mse_w4
PID0=$(run_cell mllmq_mae_w4 mbq 4 0 "${SCALE}/mllmq/janus_mae_w4a16.pt")
sleep $STAGGER
PID1=$(run_cell mllmq_mse_w4 mbq 4 1 "${SCALE}/mllmq/janus_mse_w4a16.pt")
wait_pair $PID0 $PID1

info "=== W4 BATCH COMPLETE ==="

# ─── W3 Batch ─────────────────────────────────────────
info "=== W3 BATCH ==="

# Round W3-1: rtn_w3 (resume), awq_w3
PID0=$(run_cell rtn_w3 rtn 3 0 "")
sleep $STAGGER
PID1=$(run_cell awq_w3 awq 3 1 "${SCALE}/awq/janus_awq_w3a16.pt")
wait_pair $PID0 $PID1

# Round W3-2: mbq_block_w3, mbq_scalar_w3
PID0=$(run_cell mbq_block_w3 mbq 3 0 "${SCALE}/mbq/janus_mbq_block_w3a16.pt")
sleep $STAGGER
PID1=$(run_cell mbq_scalar_w3 mbq 3 1 "${SCALE}/mbq/janus_mbq_scalar_w3a16.pt")
wait_pair $PID0 $PID1

# Round W3-3: mllmq_mae_w3, mllmq_mse_w3
PID0=$(run_cell mllmq_mae_w3 mbq 3 0 "${SCALE}/mllmq/janus_mae_w3a16.pt")
sleep $STAGGER
PID1=$(run_cell mllmq_mse_w3 mbq 3 1 "${SCALE}/mllmq/janus_mse_w3a16.pt")
wait_pair $PID0 $PID1

info "=== W3 BATCH COMPLETE ==="

# ════════════════════════════════════════════════════════
#  总结
# ════════════════════════════════════════════════════════

ELAPSED=$(( ($(date +%s) - T0) / 60 ))
echo ""
echo "========================================="
info "🎉 GenEval Driver — ALL DONE!"
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

COMPLETED=$(for tag in fp16 rtn_w8 rtn_w4 rtn_w3 \
  awq_w8 awq_w4 awq_w3 \
  mbq_block_w8 mbq_block_w4 mbq_block_w3 \
  mbq_scalar_w8 mbq_scalar_w4 mbq_scalar_w3 \
  mllmq_mae_w8 mllmq_mae_w4 mllmq_mae_w3 \
  mllmq_mse_w8 mllmq_mse_w4 mllmq_mse_w3; do
  [ "$(count_pngs "$tag")" -ge "$TARGET" ] && echo 1
done | wc -l)
echo ""
info "Overall: $COMPLETED/19 configs complete"
echo "========================================="