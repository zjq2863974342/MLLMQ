#!/bin/bash
# GenEval 任务池调度器 —— 单调度器 · 共享任务池 · W4 优先 · 支持 1..N GPU
#
# 背景 (2026-10-09):
#   · GPU0 (PCI 48:00.0) 平台重启后仍无法创建 CUDA 上下文 (dialog.md 问题 10)，GPU1 健康
#   · 用户决策 15:30: W8 已评 6 个 config 极差仅 1.1pp = 噪声，区分度全在 W4/W3
#     → 任务池按 W4(6 方法) → W3(6 方法) → W8 残尾(mse_w8 1727/2212) 排序
#   · 与 run_geneval_dual_queue.sh 的区别: 不再用静态 GPU 分工表，改为共享池 + 抢锁，
#     单卡也能覆盖全部 13 config；GPU0 恢复后加一个 worker 即可，无需重新分配任务
#
# Usage:
#   GENEVAL_GPUS="1"   nohup bash scripts/run_geneval_pool.sh > OUT 2>&1 &   # GPU0 未恢复 → 单卡 GPU1
#   GENEVAL_GPUS="0 1" nohup bash scripts/run_geneval_pool.sh > OUT 2>&1 &   # GPU0 已恢复 → 双卡
#   DRYRUN=1 GENEVAL_GPUS="1" bash scripts/run_geneval_pool.sh               # 只打印决策，不启动任何生成
#
# 🔴 纪律 (dialog.md 重点注意事项 §0):
#   1. 同一时刻只允许一个调度器进程 —— 重启本脚本前必须先 pkill 掉旧的。
#   2. 本脚本**不会**自动探测 GPU0 健康度：探测要建 CUDA 上下文，而"其他 CUDA 进程运行时
#      并发建上下文"正是 10-09 11:48 驱动锁挂死的触发形态。要加 GPU0 请：
#      生成空窗期跑 `bash /nas3/kangjia/mllmq/scripts/probe_gpu0_health.sh` 确认 exit 0，
#      再 pkill 本脚本并带 GENEVAL_GPUS="0 1" 重启（png 在盘，断点续跑，最多丢当前 1 张）。

set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/geneval
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
LOCKDIR=/nas3/kangjia/mllmq/output/geneval_pool_locks
TARGET=2212
COOLDOWN=30          # GPU 释放后冷却秒数（错峰，避免 apply_quant 显存峰值叠加）
IDLE_SLEEP=60        # 池内无可用任务时的重试间隔
DRYRUN=${DRYRUN:-0}

DRIVER_PID=$$
T0=$(date +%s)
GPUS=(${GENEVAL_GPUS:-"0 1"})

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date +%H:%M:%S)] $1"; }
good() { echo -e "[$(date +%H:%M:%S)] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date +%H:%M:%S)] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date +%H:%M:%S)] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date +%H:%M:%S)] ${CYAN}📌 $1${NC}"; }

# ── 共享任务池（按"何时能产出有区分度的数据"排序）──────────────
# format: "tag method w_bit scale_path"   (__NONE__ = rtn，无需 scale cache)
POOL=(
  # ── W4: 6 方法全跑齐才能做显著性对比（用户明确要求优先）──
  "awq_w4 awq 4 ${SCALE}/awq/janus_awq_w4a16.pt"                  # 已 1092/2212 → 只差 1120，最快出第一个完整 W4
  "rtn_w4 rtn 4 __NONE__"                                         # 已 578/2212  → W4 baseline
  "mllmq_mse_w4 mbq 4 ${SCALE}/mllmq/janus_mse_w4a16.pt"          # 本方法（逐模态归一化）
  "mllmq_mae_w4 mbq 4 ${SCALE}/mllmq/janus_mae_w4a16.pt"          # 本方法（共享分母）
  "mbq_block_w4 mbq 4 ${SCALE}/mbq/janus_mbq_block_w4a16.pt"      # 直接竞品
  "mbq_scalar_w4 mbq 4 ${SCALE}/mbq/janus_mbq_scalar_w4a16.pt"    # 直接竞品
  # ── W3: 极低位宽，区分度同样大 ──
  "rtn_w3 rtn 3 __NONE__"                                         # 已 593/2212
  "awq_w3 awq 3 ${SCALE}/awq/janus_awq_w3a16.pt"
  "mllmq_mse_w3 mbq 3 ${SCALE}/mllmq/janus_mse_w3a16.pt"
  "mllmq_mae_w3 mbq 3 ${SCALE}/mllmq/janus_mae_w3a16.pt"
  "mbq_block_w3 mbq 3 ${SCALE}/mbq/janus_mbq_block_w3a16.pt"
  "mbq_scalar_w3 mbq 3 ${SCALE}/mbq/janus_mbq_scalar_w3a16.pt"
  # ── W8 残尾: 只差 485 张，但 W8 已证明是噪声级差异 → 放最后 ──
  "mllmq_mse_w8 mbq 8 ${SCALE}/mllmq/janus_mse_w8a16.pt"          # 已 1727/2212
)

count_pngs() { find "${OUT}/$1" -name '*.png' 2>/dev/null | wc -l; }

is_running() { ps aux | grep 'eval_generate_quant' | grep -v grep | grep -q -- "$1"; }

# mkdir 是原子的 → 拿不到锁说明另一 worker 已被分到该 tag
lock_take() { mkdir "$LOCKDIR/$1" 2>/dev/null && echo "$2" > "$LOCKDIR/$1/pid" 2>/dev/null; }
lock_free() { rm -rf "$LOCKDIR/$1"; }

cleanup() {
  warn "Interrupted — killing children + 释放锁..."
  pkill -P $DRIVER_PID 2>/dev/null
  for d in "$LOCKDIR"/*/; do [ -d "$d" ] && rm -rf "$d"; done
  log "Pool driver exiting."
  exit 130
}
trap cleanup SIGINT SIGTERM

# ── 单任务执行（阻塞式）──────────────────────────────────────
run_one() {
  local tag=$1 method=$2 w=$3 gpu=$4 scale_path=${5:-}
  local outd="${OUT}/${tag}" logf="${LOG}/geneval_${tag}.log"
  local n=$(count_pngs "$tag")

  sleep "$COOLDOWN"
  info "[GPU${gpu}] LAUNCH $tag ($method w${w}, 续 $n/$TARGET)"
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
    fail "[GPU${gpu}] FAIL $tag — $final/$TARGET (rc=$rc, 见 $logf)"
  fi
  return $rc
}

# ── worker: 不断从共享池抢第一个「未完成 + 未在跑 + 抢到锁」的任务 ──
worker() {
  local gpu=$1
  info "[GPU${gpu}] worker started"
  while :; do
    local picked="" ptag="" pm="" pw="" ps_=""

    for task in "${POOL[@]}"; do
      read -r tag method w scale <<< "$task"
      local n=$(count_pngs "$tag")
      [ "$n" -ge "$TARGET" ] && continue
      is_running "$tag" && continue
      if lock_take "$tag" "$$:$gpu"; then
        picked="$task"; ptag="$tag"; pm="$method"; pw="$w"
        ps_="${scale/#__NONE__/}"
        break
      fi
    done

    if [ -z "$picked" ]; then
      # 池内是否还有活干？区分「全完成」与「都被别的 worker 占着」
      local remain=0
      for task in "${POOL[@]}"; do
        read -r tag _ _ _ <<< "$task"
        [ "$(count_pngs "$tag")" -lt "$TARGET" ] && remain=$((remain+1))
      done
      if [ "$remain" -eq 0 ]; then
        good "[GPU${gpu}] 池内任务全部完成，worker 退出"
        return 0
      fi
      sleep "$IDLE_SLEEP"
      continue
    fi

    run_one "$ptag" "$pm" "$pw" "$gpu" "$ps_"
    lock_free "$ptag"
  done
}

# ── 启动 ──────────────────────────────────────────────────────
log "===== GenEval Pool Driver (W4-first) ====="
log "GPUs: [${GPUS[*]}] | 池内任务: ${#POOL[@]} | TARGET=$TARGET | DRYRUN=$DRYRUN"

# ── DRYRUN: 只打印计划后退出，绝不启动 worker（否则会无限重扫同一任务）──
if [ "$DRYRUN" = "1" ]; then
  RATE=4     # 实测 ~3.9 图/min/卡
  info "[DRYRUN] 任务池计划（按执行顺序）:"
  LEFT=0
  for task in "${POOL[@]}"; do
    read -r tag method w scale <<< "$task"
    n=$(count_pngs "$tag")
    if [ "$n" -ge "$TARGET" ]; then
      good "[DRYRUN]   SKIP  $tag — $n/$TARGET 已完成"
      continue
    fi
    if is_running "$tag"; then
      warn "[DRYRUN]   WAIT  $tag — $n/$TARGET 已有进程在跑"
      continue
    fi
    need=$((TARGET - n))
    hrs=$(( (need / RATE) / 60 )); mins=$(( ((need / RATE) % 60) ))
    LEFT=$((LEFT + need))
    info "[DRYRUN]   RUN   $tag — $n/$TARGET, 需 $need 张 ≈ ${hrs}h${mins}m, method=$method w=$w"
    case "$method" in
      rtn) [ "$scale" = "__NONE__" ] || fail "[DRYRUN]   ❌ rtn 不应带 scale_path: $tag";;
      *)   [ -f "$scale" ] && good "[DRYRUN]         scale ✅ $scale" || fail "[DRYRUN]         scale ❌ 缺文件: $scale";;
    esac
  done
  NG=${#GPUS[@]}
  TOT=$(( LEFT / RATE ))
  info "[DRYRUN] 合计需生成 $LEFT 张 ≈ $TOT GPU-min ≈ $((TOT/60))h；${NG} 卡并行 → wall-clock ≈ $((TOT/60/NG))h"
  log "[DRYRUN] 结束（未启动任何生成进程）"
  rm -rf "$LOCKDIR" 2>/dev/null
  exit 0
fi

# 残留锁清理：只有在本脚本独占启动时才安全（启动前已确认无 worker 在跑）
if [ -d "$LOCKDIR" ] && [ "$(ls -A "$LOCKDIR" 2>/dev/null)" ]; then
  warn "发现残留锁，清理: $(ls "$LOCKDIR" | tr '\n' ' ')"
  rm -rf "$LOCKDIR"
fi
mkdir -p "$LOCKDIR"

PIDS=()
for g in "${GPUS[@]}"; do
  # 启动前自检：该 GPU 必须能建 CUDA 上下文，否则任务会连环秒失败（10-09 GPU0 教训，
  # nvidia-smi 查询全绿也照样可能建不出上下文，必须真实分配探测）
  if ! CUDA_VISIBLE_DEVICES=$g timeout 60 python -c "import torch;torch.zeros(8,device='cuda');torch.cuda.synchronize()" >/dev/null 2>&1; then
    fail "GPU${g} 无法创建 CUDA 上下文 —— 跳过该卡（不启动会秒失败的 worker）"
    continue
  fi
  good "GPU${g} 健康自检通过"
  worker "$g" &
  PIDS+=($!)
  sleep 60   # worker 之间错峰启动，避免 apply_quant 显存峰值叠加
done

if [ ${#PIDS[@]} -eq 0 ]; then
  fail "没有可用 GPU，调度器退出。请先修复 GPU 或指定 GENEVAL_GPUS。"
  rm -rf "$LOCKDIR"
  exit 1
fi

for p in "${PIDS[@]}"; do wait "$p"; done

ELAPSED=$(( ($(date +%s) - T0) / 60 ))
rm -rf "$LOCKDIR"
echo ""
echo "========================================="
info "🎉 GenEval Pool — ALL DONE! (~${ELAPSED} min = $((ELAPSED/60))h $((ELAPSED%60))min)"
echo "========================================="
COMPLETED=0
for tag in fp16 rtn_w8 rtn_w4 rtn_w3 awq_w8 awq_w4 awq_w3 \
           mbq_block_w8 mbq_block_w4 mbq_block_w3 mbq_scalar_w8 mbq_scalar_w4 mbq_scalar_w3 \
           mllmq_mae_w8 mllmq_mae_w4 mllmq_mae_w3 mllmq_mse_w8 mllmq_mse_w4 mllmq_mse_w3; do
  n=$(count_pngs "$tag")
  if [ "$n" -ge "$TARGET" ]; then good "  $tag: $n/$TARGET"; COMPLETED=$((COMPLETED+1)); else fail "  $tag: $n/$TARGET"; fi
done
info "Overall: $COMPLETED/19 configs complete"
echo "========================================="
