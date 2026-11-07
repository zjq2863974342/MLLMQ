#!/bin/bash
# 评分-续跑串行控制器：等某个 tag 生成完成 → 暂停池调度器 → 评分 → 恢复池调度器
#
# 为什么需要它（dialog.md 重点注意事项 §0 纪律 #1）:
#   池调度器完成一个 task 后只 sleep COOLDOWN=30s 就接下一个，而 GenEval 评分单 config 需 ~8min
#   且必须建 CUDA 上下文 → 生成在跑时评分会与生成进程并发建上下文 = 10-09 11:48 驱动锁挂死的触发形态。
#   所以评分窗口必须**显式造出来**：先停干净、确认无任何 CUDA 进程，再评分，最后恢复。
#
# 🔴 硬顺序保证: 评分前必须确认 eval_generate_quant 进程数为 0，否则 ABORT（宁可少一个分数，不可再挂死一次）。
#
# Usage:
#   nohup bash /nas3/kangjia/mllmq/scripts/score_completed_and_resume.sh awq_w4 rtn_w4 ... \
#     > /nas3/kangjia/mllmq/output/logs/score_watcher.log 2>&1 &
#   GENEVAL_GPUS="0 1"  # GPU0 恢复后恢复池时用双卡（默认 "1"）

set -o pipefail

GE=/nas3/kangjia/mllmq/data/benchmarks_gen/geneval
EV=$GE/evaluation
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/geneval
LOGD=/nas3/kangjia/mllmq/output/logs
POOL=/nas3/kangjia/mllmq/MLLMQ/scripts/run_geneval_pool.sh
TARGET=2212
GPUS_FOR_RESUME=${GENEVAL_GPUS:-1}

export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"
# 评分侧线程限制（§10.3 线程风暴：不设则 273 线程占 ~29 核）
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
ts() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $1"; }

[ -z "$CLIP_BIN" ] && { say "❌ CLIP 权重未就绪，退出"; exit 1; }
[ -d "$EV/.torch_home/hub/checkpoints" ] || { say "❌ Faster R-CNN 权重未就绪，退出"; exit 1; }

# 评分用哪张卡：GPU0 已知坏了（问题 10），除非显式指定，否则钉 GPU1
SCORE_GPU=${SCORE_GPU:-1}

count_pngs() { find "$OUT/$1" -name '*.png' 2>/dev/null | wc -l; }
gen_procs() { pgrep -f 'eval_generate_quant' 2>/dev/null | wc -l; }
pool_procs() { pgrep -f 'run_geneval_pool.sh' 2>/dev/null | wc -l; }

# ── 停池：driver → python，逐级确认退出 ──────────────────────
stop_pool() {
  [ "$(pool_procs)" -gt 0 ] && { say "  停调度器 (pkill -f run_geneval_pool.sh)"; pkill -f 'run_geneval_pool.sh'; sleep 5; }
  # driver 的 trap 只 pkill -P DRIVER_PID，杀不到 worker 的孙子进程 python → 必须显式停生成进程
  if [ "$(gen_procs)" -gt 0 ]; then
    say "  停生成进程 (SIGTERM → 等待 → SIGKILL)"
    pkill -f 'eval_generate_quant'; sleep 20
    [ "$(gen_procs)" -gt 0 ] && { pkill -9 -f 'eval_generate_quant'; sleep 10; }
  fi
  # 🔴 硬门禁：仍有残留 = 绝不评分（并发 CUDA 上下文会重演驱动锁挂死）
  for i in $(seq 1 24); do
    [ "$(gen_procs)" -eq 0 ] && break
    say "  ⚠️ 仍有 $(gen_procs) 个生成进程未退出，等待 ${i}/24..."
    sleep 5
  done
  if [ "$(gen_procs)" -ne 0 ]; then
    say "🔴 ABORT: 生成进程杀不掉（可能已进 D 状态 = 驱动挂死复发），拒绝评分。需按 TODO Phase 3 请平台重启。"
    return 1
  fi
  say "  ✅ 已确认无任何 CUDA 生成进程，评分窗口打开"
  return 0
}

# ── 恢复池 ────────────────────────────────────────────────
start_pool() {
  local lf="$LOGD/geneval_pool_$(date +%Y%m%d_%H%M%S).log"
  say "  恢复池调度器 GENEVAL_GPUS='$GPUS_FOR_RESUME' → $lf"
  cd /nas3/kangjia/mllmq/MLLMQ || return 1
  GENEVAL_GPUS="$GPUS_FOR_RESUME" nohup bash "$POOL" > "$lf" 2>&1 &
  sleep 5
  [ "$(pool_procs)" -gt 0 ] && say "  ✅ 调度器已启动 (pid $(pgrep -f run_geneval_pool.sh | head -1))" \
                            || say "  ❌ 调度器启动失败，请人工检查 $lf"
}

# ── 截断 png 体检：被 kill 时正在写的那张可能损坏，而续跑只按路径存在性跳过 → 会永久坏图 ──
validate_newest_png() {
  local tag=$1
  local newest=$(find "$OUT/$tag" -name '*.png' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  [ -z "$newest" ] && return 0
  if ! python -c "from PIL import Image; im=Image.open('$newest'); im.load(); im.verify()" 2>/dev/null; then
    say "  ⚠️ 最新 png 损坏（kill 时正在写），删除以触发重新生成: $newest"
    rm -f "$newest"
  else
    say "  ✅ 最新 png 完整: $(basename "$newest")"
  fi
}

# ── 主循环 ────────────────────────────────────────────────
[ $# -eq 0 ] && { say "用法: $0 <tag> [tag...]"; exit 1; }

for TAG in "$@"; do
  say "=== [$TAG] 守护开始 ==="

  n=$(count_pngs "$TAG")
  if [ "$n" -ge "$TARGET" ]; then
    say "  [$TAG] 已 $n/$TARGET 完成，直接进入评分流程"
  else
    say "  [$TAG] 当前 $n/$TARGET，轮询等待（每 5min）..."
    while :; do
      n=$(count_pngs "$TAG")
      [ "$n" -ge "$TARGET" ] && { say "  [$TAG] ✅ 生成完成 $n/$TARGET"; break; }
      sleep 300
      say "  [$TAG] $n/$TARGET ($(( n*100/TARGET ))%)"
    done
  fi

  stop_pool || { say "[$TAG] 跳过评分（停池失败）"; continue; }
  validate_newest_png "$TAG"

  say "  [$TAG] 开始评分 on GPU${SCORE_GPU} ($(date +%H:%M:%S))"
  T0=$(date +%s)
  ( cd "$GE" && CUDA_VISIBLE_DEVICES=$SCORE_GPU python evaluation/evaluate_images.py \
      "$OUT/$TAG/geneval" --outfile "$OUT/$TAG/results.jsonl" --clip-model-path "$CLIP_BIN" \
      > "$LOGD/score_${TAG}.log" 2>&1 )
  rc=$?
  MIN=$(( ($(date +%s) - T0) / 60 ))
  if [ $rc -eq 0 ]; then
    say "  [$TAG] ✅ 评分完成 (rc=0, ${MIN}min):"
    ( cd "$GE" && python evaluation/summary_scores.py "$OUT/$TAG/results.jsonl" 2>/dev/null | tail -9 | sed 's/^/      /' )
  else
    say "  [$TAG] ❌ 评分失败 rc=$rc（见 $LOGD/score_${TAG}.log），results.jsonl 可能未被写入（崩在写文件前）"
  fi

  start_pool
  say "=== [$TAG] 守护结束 ==="
  echo ""
done
say "🏁 全部指定 tag 处理完毕"
