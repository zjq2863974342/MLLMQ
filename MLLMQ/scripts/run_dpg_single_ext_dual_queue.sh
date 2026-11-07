#!/bin/bash
# DPG-Bench 段3 · single 补全跑双队列（主表扩档，2026-10-21 晚建）
#
# 8 config × 1065 张（每 prompt 1 张 384×384，与段1 同协议，评分口径 --pic-num 1 --resolution 384）
#   段3 范围（用户 2026-10-21 晚拍板，扩 10-20 的"砍掉"范围）：
#     W3 残留 2：mbq_scalar_w3 + mllmq_mse_w3   → 补全 W3 × 6 方法整行
#     W4 全 6 ：rtn_w4 + awq_w4 + mbq_block_w4 + mbq_scalar_w4 + mllmq_mae_w4 + mllmq_mse_w4 → 新增 W4 整行
#   → DPG 生成主表 = fp16 + 6 方法 × (W3+W4) = 16 config，与 GenEval / COCO-val 口径对齐
#
# 输出：OUT/<tag>/dpg_bench/<item_id>.png（single 协议无 _0000 后缀）
# 🔴 评分由对端容器负责（13:05 起对端队列在评段1/段2），本机不起任何评分进程
#    —— 固定输出路径 output/dpg_scores/<tag>_results.txt 不命名空间化，两台并跑同 tag 会互相覆写
#
# 🔴 纪律：本机生成在跑时禁新 CUDA 进程（评分等全退出再动；pgrep 用 [xyz] 防自匹配）
#
# Usage: cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_dpg_single_ext_dual_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/dpg_single_ext_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=/nas3/kangjia/mllmq/checkpoints/torch_home

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/dpg_bench
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=1065
NUM_SAMPLES=1
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

run_one() {
  local tag=$1 method=$2 w=$3 gpu=$4 scale_path=${5:-}
  local outd="${OUT}/${tag}"
  local logf="${LOG}/dpg_single_${tag}.log"
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
#  format: "tag method w_bit scale_path"（rtn 无 scale cache）
#  MLLMQ/MBQ 系一律 `--quant_method mbq` + 各自 scale cache（MLLMQ 复用 mbq 量化路径）
#  ════════════════════════════════════════════════════════
# 队列排序（论文价值优先，仿段1/W3 COCO 的"长杆先行"原则）：
#   slot1 两卡 = W3 残留 2 → 最快补全 W3 整行主表（~3.7h 内 W3 × 6 方法 DPG 全齐）；
#   slot2 = W4 故事头对: rtn_w4(崩溃基线, GPU0) vs mllmq_mae_w4(旗舰, GPU1)；
#   slot3 = mbq_block_w4 / awq_w4（W4 头部对比四元在 slot2-3 内聚齐）；
#   slot4 = 同族次优填空 mbq_scalar_w4 / mllmq_mse_w4。
GPU0_TASKS=(
  "mbq_scalar_w3 mbq 3 ${SCALE}/mbq/janus_mbq_scalar_w3a16.pt"
  "rtn_w4 rtn 4 __NONE__"
  "mbq_block_w4 mbq 4 ${SCALE}/mbq/janus_mbq_block_w4a16.pt"
  "mbq_scalar_w4 mbq 4 ${SCALE}/mbq/janus_mbq_scalar_w4a16.pt"
)
GPU1_TASKS=(
  "mllmq_mse_w3 mbq 3 ${SCALE}/mllmq/janus_mse_w3a16.pt"
  "mllmq_mae_w4 mbq 4 ${SCALE}/mllmq/janus_mae_w4a16.pt"
  "awq_w4 awq 4 ${SCALE}/awq/janus_awq_w4a16.pt"
  "mllmq_mse_w4 mbq 4 ${SCALE}/mllmq/janus_mse_w4a16.pt"
)

log "===== DPG-Bench 段3 · single 补全 Dual-Queue Driver ====="
log "目标：8 config × ${TARGET} 张（每 prompt 1 张 384×384）= W3 残留 2 + W4 全 6"
log "GPU0: 4 config (mbq_scalar_w3 → rtn_w4 → mbq_block_w4 → mbq_scalar_w4)"
log "GPU1: 4 config (mllmq_mse_w3 → mllmq_mae_w4 → awq_w4 → mllmq_mse_w4)"
log "实测速率 ~12.7 s/it → 单 config ~3.75h；每卡 4 config ≈ 15h"

# ── scale cache 齐备性 ──
for tag_task in "${GPU0_TASKS[@]}" "${GPU1_TASKS[@]}"; do
  read -r t m w sc <<< "$tag_task"
  sc="${sc/#__NONE__/}"
  if [ -n "$sc" ] && [ ! -f "$sc" ]; then fail "scale cache 缺失: $sc ($t)"; exit 5; fi
done
good "scale cache 齐备：8/8（rtn_w4 无 cache 需求）"

# ── 起跑快照 ──
for tag in mbq_scalar_w3 mllmq_mse_w3 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4; do
  log "  snapshot $tag: $(count_pngs "$tag")/$TARGET"
done

# ── 前置门禁：不得有其它 GPU 任务 ──
# 🔴 2026-10-21 修：原 pattern 漏了 DPG 评分进程（score_dpg_coco / compute_dpg_bench），
#    导致评分在跑时门禁放行、启动即与评分抢卡。DPG 评分是常驻 CUDA 任务，必须纳入。
GPU_BUSY_PAT='[e]val_generate_quant|[s]core_pref|[s]core_fid|[s]core_clip|[m]ain_quant|[e]val_benchmarks|[s]core_dpg_coco|[c]ompute_dpg_bench'
if pgrep -f "$GPU_BUSY_PAT" >/dev/null 2>&1; then
  fail "检测到其它 GPU 任务在跑 → 拒绝启动（并发建 CUDA 上下文危险，见 dialog §0）"
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
  info "🎉 DPG-Bench 段3 single 补全 — ALL DONE!"
fi
good "Elapsed: ~${ELAPSED} min ($(( ELAPSED / 60 ))h $(( ELAPSED % 60 ))min)"
echo "========================================="
COMPLETED=0
for tag in mbq_scalar_w3 mllmq_mse_w3 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4; do
  n=$(count_pngs "$tag")
  if [ "$n" -ge "$TARGET" ]; then good "  $tag: $n/$TARGET"; COMPLETED=$((COMPLETED+1)); else fail "  $tag: $n/$TARGET"; fi
done
info "Overall: $COMPLETED/8 configs complete"
echo "========================================="
log "下一步：段3 全出后交对端容器评分（本机不动评分）→ compute_dpg_bench.py --pic-num 1 --resolution 384，tag 命名 <config>_single"
