#!/bin/bash
# COCO-val W8 单队列 — 只用 GPU1，填「W8 档无人跑」的空缺（dialog 遗留 #1）
# 2026-10-15 18:45 新建：GPU1 在 mllmq_mse_w4 收尾(~19:03)后有 ~15h 空窗
#
# 与 run_coco_val_w4_resume_dual_queue.sh 的差异:
#  ① 只有 GPU1 一条队列，**绝不碰 GPU0**（GPU0 上 mllmq_mae_w4 长跑 15h，不容打扰）
#  ② 门禁改成**按卡判忙**：原版 `pgrep -f eval_generate_quant` 会看到 GPU0 的 worker 而直接
#     exit 3 → 这里改成读每个 worker 的 /proc/<pid>/environ 里的 CUDA_VISIBLE_DEVICES，
#     只有 GPU1 空闲才起跑；不空则每 60s 轮询等待（最长 WAIT_MAX 分钟）
#  ③ 起跑前校验目标目录**不含 `*_0001.png`** —— mllmq_mae_w8 原来有 322 张 num_samples=4
#     老协议残档（只有 81 个唯一 prompt），若不隔离会被 skip-existing 顶占 5000 额度，
#     跑完静默缺 ~322 个 prompt。残档已于 18:44 mv 到
#     output/janus-pro-7b/coco_val_orphan_n4_20261013/（未删除，可 mv 回来）
#  ④ 保留 3 次重试 + 零进展熔断（同 W4 续跑版）
#
# Usage: setsid nohup bash scripts/run_coco_val_w8_gpu1_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/coco_val_w8_gpu1_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/coco_val
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=5000
COOLDOWN=30
MAX_RETRY=3
GPU=1
WAIT_MAX=240     # 等 GPU1 空闲的最长分钟数（mse_w4 预计 ~19:03 收，留足余量）

TAG=mllmq_mae_w8
METHOD=mbq
WBIT=8
SCALE_PATH="${SCALE}/mllmq/janus_mae_w8a16.pt"

DRIVER_PID=$$
T0=$(date +%s)

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date +%H:%M:%S)] $1"; }
good() { echo -e "[$(date +%H:%M:%S)] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date +%H:%M:%S)] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date +%H:%M:%S)] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date +%H:%M:%S)] ${CYAN}📌 $1${NC}"; }

count_pngs() { find "${OUT}/$1" -name '*.png' 2>/dev/null | wc -l; }

cleanup() {
  warn "Interrupted — killing children..."
  pkill -P $DRIVER_PID 2>/dev/null
  log "Driver exiting."
  exit 130
}
trap cleanup SIGINT SIGTERM

# ── 按卡判忙：只关心 GPU${GPU} 上有没有本项目的 worker ──
gpu_busy() {
  local want=$1 p dev
  for p in $(pgrep -f 'eval_generate_quant' 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    dev=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
    [ -z "$dev" ] && dev="0,1"          # 没设 = 两张卡都可能用
    case ",$dev," in *",$want,"*) echo "pid=$p CUDA_VISIBLE_DEVICES=$dev"; return 0 ;; esac
  done
  return 1
}

log "===== COCO-val W8 GPU${GPU}-only Queue ====="
log "config: $TAG ($METHOD w${WBIT}a16, scale=$SCALE_PATH)"
[ -f "$SCALE_PATH" ] || { fail "scale cache 不存在: $SCALE_PATH"; exit 5; }
good "scale cache 就位: $(ls -l "$SCALE_PATH" | awk '{print $5" bytes, mtime "$6" "$7" "$8}')"

# ── 协议纯净性校验：目标目录不得含老协议(*_0001.png)残档 ──
N_LEGACY=$(find "${OUT}/${TAG}" -name '*_000[1-9].png' 2>/dev/null | wc -l)
if [ "$N_LEGACY" -gt 0 ]; then
  fail "${OUT}/${TAG} 含 $N_LEGACY 张 *_0001+ 老协议图 → 拒绝起跑"
  fail "它们会被 skip-existing 顶占 $TARGET 额度，跑完静默缺 prompt。请整目录 mv 走后再来。"
  exit 6
fi
good "协议纯净性校验通过（0 张 *_0001+ 残档）；现有 $(count_pngs "$TAG")/$TARGET"

# ── 等 GPU${GPU} 空闲 ──
WAITED=0
while BUSY=$(gpu_busy $GPU); do
  if [ $WAITED -eq 0 ]; then
    info "GPU${GPU} 仍被占用（$BUSY）→ 每 60s 轮询等待，最长 ${WAIT_MAX}min"
    info "（GPU0 的 mllmq_mae_w4 不在等待条件里，本脚本不会碰它）"
  fi
  WAITED=$(( WAITED + 1 ))
  if [ $WAITED -ge $WAIT_MAX ]; then
    fail "等了 ${WAIT_MAX}min GPU${GPU} 仍忙（$BUSY）→ 放弃起跑，不抢卡"
    exit 7
  fi
  sleep 60
done
good "GPU${GPU} 已空闲（等待 ${WAITED}min）"

# ── 真实 CUDA 上下文自检（只探 GPU${GPU}，串行、不并发）──
if CUDA_VISIBLE_DEVICES=$GPU timeout -s 9 60 python -c \
     "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize()" >/dev/null 2>&1; then
  good "GPU${GPU} CUDA 上下文自检通过"
else
  fail "GPU${GPU} 无法创建 CUDA 上下文 → 拒绝启动"
  exit 2
fi

# ── 执行（带重试 + 零进展熔断）──
outd="${OUT}/${TAG}"
logf="${LOG}/coco_val_${TAG}.log"
attempt=1
FINAL_RC=1
while [ $attempt -le $MAX_RETRY ]; do
  n=$(count_pngs "$TAG")
  if [ "$n" -ge "$TARGET" ]; then
    good "DONE $TAG — $n/$TARGET（无需再跑）"
    FINAL_RC=0
    break
  fi
  if [ $attempt -eq 1 ]; then sleep $COOLDOWN; fi

  info "LAUNCH $TAG ($n/$TARGET existing) attempt ${attempt}/${MAX_RETRY}"
  CUDA_VISIBLE_DEVICES=$GPU python eval_generate_quant.py \
    --model_path "$MODEL" --benchmark coco_val \
    --output_dir "$outd" --num_samples 1 \
    --quant_method "$METHOD" --w_bit "$WBIT" \
    --scale_path "$SCALE_PATH" >> "$logf" 2>&1
  rc=$?
  after=$(count_pngs "$TAG")
  gain=$(( after - n ))

  if [ "$after" -ge "$TARGET" ]; then
    good "DONE $TAG — $after/$TARGET"
    FINAL_RC=0
    break
  fi
  if [ "$gain" -eq 0 ] && [ "$rc" -ne 0 ]; then
    fail "ABORT $TAG — rc=$rc 且 0 张新增 → 疑环境损坏（见 $logf 末尾）"
    fail "主动终止，不谎报完成。先跑环境复验（hub 必须是 0.36.2）再重拉。"
    exit 4
  fi
  warn "INCOMPLETE $TAG — $after/$TARGET (本次 +$gain, rc=$rc)"
  attempt=$(( attempt + 1 ))
done

ELAPSED=$(( ($(date +%s) - T0) / 60 ))
echo ""
echo "========================================="
if [ $FINAL_RC -eq 0 ]; then
  info "🎉 COCO-val $TAG — DONE ($(count_pngs "$TAG")/$TARGET)"
else
  fail "GAVE-UP $TAG — $(count_pngs "$TAG")/$TARGET after $MAX_RETRY attempts（别看上一条就以为完成了）"
fi
good "Elapsed: ~${ELAPSED} min ($(( ELAPSED / 60 ))h $(( ELAPSED % 60 ))min)"
echo "========================================="
exit $FINAL_RC
