#!/bin/bash
# COCO-val W4 续跑双队列 — 2026-10-15 17:50 容器重启(17:18 杀死 W4 队列)后新建
#
# 与 run_coco_val_dual_queue.sh 的差异（仅 3 处，其余逐行沿用已验证 50h+ 的版本）:
#  ① 队列重排: GPU0 直上旗舰 mllmq_mae_w4(0/5000，最长杆)，两个残档(mbq_block_w4 4910、
#     mllmq_mse_w4 4742)全压到 GPU1 → mae_w4 起跑时间不被短任务阻塞（W3 队列同款排序原则）
#  ② 每 config 最多 3 次重试（skip-existing 续跑语义 → 重试零浪费）
#  ③ 零进展熔断: 某次尝试 0 张新增 且 rc≠0 → 判定环境坏（10-13 hub==2.2.0 那种形态），
#     整个 driver 立刻大声退出，绝不走到末尾"谎报 ALL DONE"
#
# Usage: cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_coco_val_w4_resume_dual_queue.sh \
#           > /nas3/kangjia/mllmq/output/logs/coco_val_w4_resume_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

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

# ── 单任务执行（阻塞式，带重试） ──────────────────────

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

  local attempt=1
  while [ $attempt -le $MAX_RETRY ]; do
    n=$(count_pngs "$tag")
    [ "$n" -ge "$TARGET" ] && break
    sleep $COOLDOWN

    info "[GPU${gpu}] LAUNCH $tag ($method w${w}, $n/$TARGET existing) attempt ${attempt}/${MAX_RETRY}"
    local cmd=(
      python eval_generate_quant.py
      --model_path "$MODEL" --benchmark coco_val
      --output_dir "$outd" --num_samples 1
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
      # 3 秒即崩 + 一张图没出 = 环境级故障（10-13 hub==2.2.0 形态）→ 熔断，不静默往后走
      fail "[GPU${gpu}] ABORT $tag — rc=$rc 且 0 张新增 → 疑环境损坏（见 $logf 末尾）"
      fail "Driver 主动终止，不谎报 ALL DONE。请先跑 scripts/restore_eval_env.sh + import 校验，再重拉队列。"
      kill $DRIVER_PID 2>/dev/null
      exit 4
    fi

    warn "[GPU${gpu}] INCOMPLETE $tag — $after/$TARGET (本次 +$gain, rc=$rc)"
    attempt=$(( attempt + 1 ))
  done

  local final=$(count_pngs "$tag")
  if [ "$final" -ge "$TARGET" ]; then
    good "[GPU${gpu}] DONE $tag — $final/$TARGET"
    return 0
  fi
  fail "[GPU${gpu}] GAVE-UP $tag — $final/$TARGET after $MAX_RETRY attempts"
  return 1
}

# ── GPU 独立队列 ──────────────────────────────────────

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
#  续跑 Layout（10-15 17:50 快照: fp16 5236 · rtn_w4 5000 · awq_w4 5000 ·
#  mbq_block_w4 4910 · mbq_scalar_w4 5000 · mllmq_mae_w4 0 · mllmq_mse_w4 4742）
#  format: "tag method w_bit scale_path" (scale_path=__NONE__ for rtn/fp16)
#  已完成的 4 个写在最前，run_one 会 SKIP（保留原矩阵语义，也便于中途重启后自检）
# ════════════════════════════════════════════════════════

GPU0_TASKS=(
  # 已完成 → SKIP
  "fp16 fp16 0 __NONE__"
  "rtn_w4 rtn 4 __NONE__"
  # 长跑主力：旗舰 0/5000，最早起跑（17.4h，决定 W4 的完工时间）
  "mllmq_mae_w4 mbq 4 ${SCALE}/mllmq/janus_mae_w4a16.pt"
)

GPU1_TASKS=(
  # 已完成 → SKIP
  "awq_w4 awq 4 ${SCALE}/awq/janus_awq_w4a16.pt"
  "mbq_scalar_w4 mbq 4 ${SCALE}/mbq/janus_mbq_scalar_w4a16.pt"
  # 两个残档收尾（合计 ~348 张 ≈ 1.4h），压到 GPU1 以免阻塞 GPU0 的 mae_w4
  "mbq_block_w4 mbq 4 ${SCALE}/mbq/janus_mbq_block_w4a16.pt"
  "mllmq_mse_w4 mbq 4 ${SCALE}/mllmq/janus_mse_w4a16.pt"
)

# ════════════════════════════════════════════════════════
#  启动
# ════════════════════════════════════════════════════════

log "===== COCO-val W4 RESUME Dual-Queue Driver ====="
log "Benchmark: coco_val (5k prompts × 1 sample = 5k/config)"
log "GPU0: mae_w4(0/5000) 长跑 | GPU1: mbq_block_w4(?/5000) + mse_w4(?/5000) 残档收尾"
log "Strategy: independent per-GPU queues, ${COOLDOWN}s cooldown, max ${MAX_RETRY} attempts/config"

# ── 起跑快照（写进 driver 日志，判断"是否真在续跑"以 png 为准）──
for tag in fp16 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4; do
  log "  snapshot $tag: $(count_pngs "$tag")/$TARGET"
done

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
if [ $RC0 -ne 0 ] || [ $RC1 -ne 0 ]; then
  fail "Driver 退出但队列未全绿 (rc0=$RC0 rc1=$RC1) — 逐 config 核对下面矩阵，别看这行就以为完成了"
else
  info "🎉 COCO-val W4 RESUME — ALL DONE!"
fi
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
