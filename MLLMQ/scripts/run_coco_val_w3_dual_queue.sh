#!/bin/bash
# COCO-val W3 双队列 — 与 W4 队列(另一容器/A800 双卡)并行，互不接触
# 派生自 run_coco_val_dual_queue.sh（已验证跑过 50h+ 的版本），改动仅 3 处，见下
#
# ── 与 W4 版的差异（刻意，勿"顺手改回"）─────────────────────────
#  1) 不含 fp16：fp16 与位宽无关，复用 W4 队列已产出的 coco_val/fp16 (5236 张)。
#     重复生成只会让 W3/W4 各自对着两份不同的 fp16 参考分布做 FID → 破坏可比性。
#  2) 全 W3：6 config × 5000 张 = 30k 张。scale cache 6/6 已就绪，零 scale-search。
#  3) 队列排序按"论文价值×重启风险"而非原版的 W8→W4→W3 顺序。本机 dialog.md 记录
#     ≥5 次重启；W4 队列把旗舰 mllmq_mae_w4 排到最后，导致它 14h 后仍是 0 张。
#     这里把两卡 slot1 都留给 GenEval-W3 的 story 三角中的头两名：
#       GPU0 slot1 = mllmq_mae_w3  (旗舰, GenEval W3 🏆0.5848)  → ~17h
#       GPU1 slot1 = mbq_block_w3  (头号对手, GenEval W3 0.5837) → ~17h
#       GPU0 slot2 = rtn_w3        (崩溃基线, GenEval W3 −27.8%) → ~34h  ← 三角补齐
#     剩余 3 个低区分度 config 排在 slot2/3。
#
# ── 与 W4 容器(另一物理节点, A800 双卡)的隔离性 ──────────────────
#  · 输出目录/日志名均按 tag 分文件，*_w3 与 *_w4 天然不重叠（启动前已确认无 *w3* 残留）
#  · scale_cache 只读；num_samples=1 硬编码与 W4 同构 → FID/CLIP 可比
#  · PID 命名空间互不可见：本容器的门禁/is_running 只作用于本容器的进程
#  · 唯一共享面是 NAS(/nas3, 4.1T free)。4 worker 并发写 PNG 会争 IO → 监控吞吐，
#    若较 solo 基线(~4.4 图/min/卡)掉 >20%，考虑降为单队列
#
# Usage: setsid nohup bash MLLMQ/scripts/run_coco_val_w3_dual_queue.sh \
#           > output/logs/coco_val_w3_dual_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/coco_val
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=5000    # 5k prompts (1 caption/image) × 1 sample — 与 W4 版一致
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
    warn "[GPU${gpu}] INCOMPLETE $tag — $final/$TARGET (rc=$rc)"
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
#  W3 任务表（6 config，无 fp16；排序见文件头说明）
#  format: "tag method w_bit scale_path" (scale_path=__NONE__ for rtn)
# ════════════════════════════════════════════════════════

GPU0_TASKS=(
  "mllmq_mae_w3 mbq 3 ${SCALE}/mllmq/janus_mae_w3a16.pt"        # 旗舰，slot1
  "rtn_w3 rtn 3 __NONE__"                                        # 崩溃基线，slot2 → 三角补齐
  "mllmq_mse_w3 mbq 3 ${SCALE}/mllmq/janus_mse_w3a16.pt"          # 低区分度
)

GPU1_TASKS=(
  "mbq_block_w3 mbq 3 ${SCALE}/mbq/janus_mbq_block_w3a16.pt"     # 头号对手，slot1
  "awq_w3 awq 3 ${SCALE}/awq/janus_awq_w3a16.pt"
  "mbq_scalar_w3 mbq 3 ${SCALE}/mbq/janus_mbq_scalar_w3a16.pt"
)

ALL_TAGS=(mllmq_mae_w3 rtn_w3 mllmq_mse_w3 mbq_block_w3 awq_w3 mbq_scalar_w3)

# ════════════════════════════════════════════════════════
#  启动
# ════════════════════════════════════════════════════════

log "===== COCO-val W3 Dual-Queue Driver ====="
log "Benchmark: coco_val (5k prompts × 1 sample = 5k/config) × 6 W3 configs"
log "fp16 baseline: 复用 W4 队列的 coco_val/fp16（本队列不生成）"
log "GPU0: ${#GPU0_TASKS[@]} tasks | GPU1: ${#GPU1_TASKS[@]} tasks"
log "Strategy: independent per-GPU queues, ${COOLDOWN}s cooldown between tasks"
log "ETA: ~17h/config, wall-clock ~51h（story 三角 mae+block+rtn @ ~34h）"

# ── scale cache 前置校验（缺任何一个都会秒崩，10-13 的教训：先验再跑）──
MISSING=0
for task in "${GPU0_TASKS[@]}" "${GPU1_TASKS[@]}"; do
  read -r tag method w scale <<< "$task"
  [ "$scale" = "__NONE__" ] && continue
  if [ ! -f "$scale" ]; then
    fail "缺少 scale cache: $tag → $scale"
    MISSING=1
  else
    good "scale cache OK: $tag → $(basename "$scale") ($(du -h "$scale" | cut -f1))"
  fi
done
[ "$MISSING" -eq 1 ] && exit 4

# ── 前置门禁 ──
if pgrep -f '[e]val_generate_quant|[e]valuate_images|[m]ain_quant|[e]val_benchmarks' >/dev/null 2>&1; then
  fail "检测到本容器内有其他 GPU 任务在跑 → 拒绝启动（并发建 CUDA 上下文危险）"
  fail "注: 另一容器的 W4 队列不在本容器 PID 空间内，不会也不应触发此门禁"
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
info "🎉 COCO-val W3 Dual-Queue — ALL DONE!"
echo "========================================="
good "Elapsed: ~${ELAPSED} min ($(( ELAPSED / 60 ))h $(( ELAPSED % 60 ))min)"
echo ""
log "Final matrix:"
for tag in "${ALL_TAGS[@]}"; do
	  n=$(count_pngs "$tag")
	  [ "$n" -ge "$TARGET" ] && good "  $tag: $n/$TARGET" || fail "  $tag: $n/$TARGET"
done

COMPLETED=0
for tag in "${ALL_TAGS[@]}"; do
	  [ "$(count_pngs "$tag")" -ge "$TARGET" ] && COMPLETED=$((COMPLETED + 1))
done
info "Overall: $COMPLETED/${#ALL_TAGS[@]} W3 configs complete"
info "fp16 (shared with W4 queue): $(count_pngs fp16)/$TARGET"
echo "========================================="
