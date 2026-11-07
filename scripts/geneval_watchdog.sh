#!/bin/bash
# GenEval 看门狗 — 轮询到 19/19 生成完成 + 队列彻底停掉后，自动启动 Step 5 两路并行评分
#
# ── 为什么门禁要三重 + 防抖（dialog §0 纪律 #1 / TODO Step 4）──
# 生成在跑时新建任何 CUDA 上下文 = 10-09 11:48 驱动锁挂死的触发形态
# （「2 个进程把双卡跑满 + 第 3 个进程并发建上下文」，三者缺一不发生）。
# 评分器没有 CPU 退路：evaluate_images.py:45-46 是 DEVICE="cuda" + `assert DEVICE=="cuda"`，
# load_models():66 的 object_detector.to(DEVICE) 就是建上下文那一步；而它 GPU util≈0%
# （纯 CPU-bound）看起来"不占卡"——**危险在建上下文，不在占用率**，别被 util 骗了。
#
# 🔴 本脚本只做两件事：轮询（全程只读）+ 门禁通过后启动评分。
# 🔴 全文不含任何 kill / pkill —— 尤其绝不会碰平台保活 _takeup.py（dialog §0f，杀了平台可能回收 GPU）。
# 🔴 不启动任何生成任务，所以不会造成第二套调度器（纪律 #4）。
#
# Usage:
#   cd /nas3/kangjia/mllmq && setsid nohup bash scripts/geneval_watchdog.sh \
#     > output/logs/geneval_watchdog.log 2>&1 < /dev/null &
# Exit: 0=评分流程跑完  2=队列已退但没到 19/19（有 config 失败，需人工查）
#       3=评分环境/权重/自检不通过  4=超过最长等待仍未完成  5=已有另一个看门狗

set -o pipefail

ROOT=/nas3/kangjia/mllmq
GENEVAL=$ROOT/data/benchmarks_gen/geneval
EV=$GENEVAL/evaluation
OUT=$ROOT/output/janus-pro-7b/geneval
LOG=$ROOT/output/logs
TARGET=2212
NCONFIG=19

POLL=120          # 轮询间隔（秒）
CONFIRM=2         # 连续多少次全部通过才开火（防抖：避开 worker 换任务的瞬时空窗）
MAX_WAIT_H=72     # 最长等待小时数，超时报错退出，不永久挂着
LOCK=/tmp/geneval_watchdog.lock

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date '+%m-%d %H:%M:%S')] $1"; }
good() { echo -e "[$(date '+%m-%d %H:%M:%S')] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date '+%m-%d %H:%M:%S')] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date '+%m-%d %H:%M:%S')] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date '+%m-%d %H:%M:%S')] ${CYAN}📌 $1${NC}"; }

ALL_TAGS=(fp16 rtn_w8 rtn_w4 rtn_w3
          awq_w8 awq_w4 awq_w3
          mbq_block_w8 mbq_block_w4 mbq_block_w3
          mbq_scalar_w8 mbq_scalar_w4 mbq_scalar_w3
          mllmq_mae_w8 mllmq_mae_w4 mllmq_mae_w3
          mllmq_mse_w8 mllmq_mse_w4 mllmq_mse_w3)

# TODO Step 5 待评清单（13 个）；已有 >=2212 行 results.jsonl 的会自动跳过（可断点重跑）
TODO_TAGS=(mllmq_mse_w8 rtn_w4 rtn_w3 awq_w4 awq_w3 mbq_block_w4 mbq_block_w3
           mbq_scalar_w4 mbq_scalar_w3 mllmq_mae_w4 mllmq_mae_w3 mllmq_mse_w4 mllmq_mse_w3)

# ── 只读探测（全部 bracket 写法，避免自匹配 → 纪律 #9）──
# 注意：本脚本/harness 包装 shell 的 cmdline 里含 "geneval_watchdog"，
#       不含下列任何模式，所以这三个探测都不会匹配到自己。
n_gen()    { ps -eo cmd | grep -cE '[e]val_generate_quan[t]'; }
# n_queue_procs：匹配到的**全部**队列相关进程数，正常值是 4 而不是 1
#   （harness 包装 shell + driver `Ss` + 2 个 gpu_queue 子 shell，见 dialog 23:20 的 pid 拓扑）
#   这里只把它当"队列还活着吗"的布尔量用（>0 就等），队列真正跑完时这 4 个会一起消失。
#   ⚠️ 别把它读成"有 4 套调度器"（那是纪律 #4 的违规，判据是 `Ss` 会话首进程唯一）。
n_queue_procs() { ps -eo cmd | grep -cE 'geneva[l]_dual_queu[e]|geneva[l]_single_queu[e]'; }

count_done() {   # 19 config 里已达 2212 的个数（较贵：~42k 次 find，只在需要时才调）
  local n=0 t c
  for t in "${ALL_TAGS[@]}"; do
    c=$(find "$OUT/$t" -name '*.png' 2>/dev/null | wc -l)
    [ "$c" -ge "$TARGET" ] && n=$((n+1))
  done
  echo "$n"
}

results_lines() {                       # results.jsonl 行数（不存在→0，且不喷 stderr 噪声）
  local f="$OUT/$1/results.jsonl"
  if [ -f "$f" ]; then wc -l < "$f"; else echo 0; fi
}
tmp_lines() {                            # 同上，用于 .wdtmp 临时文件
  if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi
}

trap 'warn "watchdog 收到信号退出（已在跑的评分子进程不受影响，会继续到完）"; exit 0' SIGINT SIGTERM

# ══════════ 单实例保护（用 lockfile，不用 ps 匹配 cmdline）══════════
# 为什么不用 ps：harness 包装 shell 的 cmdline 里也含脚本名（跟 driver 24286 那个坑同型），
# 用 ps 计数会把自己的包装 shell 算成"另一个看门狗" → 永远 exit 5，脚本根本起不来。
old=$(cat "$LOCK" 2>/dev/null)
if [ -n "$old" ] && [ "$old" != "$$" ] && \
   ps -p "$old" -o cmd= 2>/dev/null | grep -q 'geneva[l]_watchdo[g]'; then
  fail "已有看门狗在跑 (pid=$old) → 本实例退出，避免重复开火"
  exit 5
fi
echo "$$" > "$LOCK"

log "===== GenEval Watchdog 启动 (pid=$$, lock=$LOCK) ====="
info "门禁: 19/19 完成 + 0 个生成进程 + 0 个调度器，且连续 $CONFIRM 次全通过才开火"
info "轮询 ${POLL}s · 最长等待 ${MAX_WAIT_H}h · 评分阶段 2 路并行（每卡 1 进程）"
info "本脚本不含任何 kill —— 不会碰平台保活 _takeup.py（§0f）"

# ══════════ 阶段 1：轮询等待 ══════════
MAX_ITER=$(( MAX_WAIT_H * 3600 / POLL ))
streak=0; fail_streak=0; iter=0

while :; do
  iter=$((iter+1))
  if [ "$iter" -gt "$MAX_ITER" ]; then
    fail "等待超过 ${MAX_WAIT_H}h 仍未 19/19 → 放弃（队列可能卡住，请人工查 driver 日志）"
    exit 4
  fi

  G=$(n_gen); D=$(n_queue_procs)

  # 便宜的检查先做：只要生成进程或调度器还在，就绝不去建 CUDA 上下文，
  # 也跳过昂贵的全量 find（省 NFS 元数据压力）
  if [ "$G" -gt 0 ] || [ "$D" -gt 0 ]; then
    streak=0; fail_streak=0
    if [ $((iter % 15)) -eq 1 ]; then   # 每 ~30min 报一次进度（顺带做一次 census）
      dn=$(count_done)
      log "生成中: worker=$G queue_procs=$D · $dn/$NCONFIG 完成 · 已等待 $(( iter * POLL / 60 ))min"
    fi
    sleep "$POLL"
    continue
  fi

  # 生成进程 + 调度器都不在了 → 才做昂贵的 census 决定开不开火
  dn=$(count_done)
  if [ "$dn" -lt "$NCONFIG" ]; then
    fail_streak=$((fail_streak+1)); streak=0
    warn "队列已退出但只有 $dn/$NCONFIG 完成（连续第 $fail_streak 次确认）"
    if [ "$fail_streak" -ge "$CONFIRM" ]; then
      fail "有 config 没跑完就退出了 → 拒绝评分（残缺图会给错分数），请人工查 driver 日志"
      exit 2
    fi
    sleep "$POLL"
    continue
  fi

  streak=$((streak+1)); fail_streak=0
  log "门禁通过 $streak/$CONFIRM：$dn/$NCONFIG · worker=0 · driver=0"
  if [ "$streak" -lt "$CONFIRM" ]; then
    sleep "$POLL"
    continue
  fi
  good "连续 $CONFIRM 次确认安全 → 进入评分阶段"
  break
done

# ══════════ 阶段 2：开火前自检（此时生成已全停，pip / CUDA 都安全）══════════

# 2a. 补回重启会丢的 3 个评分包（生成在跑时故意不装 → 见 TODO Step 5 前置说明）
if ! python -c "import open_clip, clip_benchmark, modelscope" >/dev/null 2>&1; then
  warn "评分包缺失（open_clip / clip_benchmark / modelscope）→ 生成已全停，现在装是安全的"
  bash "$ROOT/scripts/restore_eval_env.sh" 2>&1 | tail -3 | sed 's/^/     /'
  if ! python -c "import open_clip, clip_benchmark, modelscope" >/dev/null 2>&1; then
    fail "restore_eval_env.sh 之后仍 import 失败（pip 网络？）→ 退出，不评分"
    exit 3
  fi
fi
good "评分包 import OK"

# 2b. 权重预检（照抄 run_geneval_rescore.sh 的做法）
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
if [ -z "$CLIP_BIN" ]; then fail "CLIP 权重未就绪: $EV/.ms_cache"; exit 3; fi
if [ ! -f "$EV/.torch_home/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" ]; then
  fail "Faster R-CNN 权重未就绪: $EV/.torch_home"; exit 3
fi
info "CLIP_BIN=$CLIP_BIN"

# 2c. 双卡真实 CUDA 自检 —— 用项目既有脚本，不手写裸探测（纪律 #5）
bash "$ROOT/scripts/probe_gpu_health.sh" 2>&1 | tail -5 | sed 's/^/     /'
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  fail "probe_gpu_health.sh 未返回 0 → 有卡不可用，拒绝评分（否则会 rc=1 崩在 load_models）"
  exit 3
fi
good "双卡 CUDA 上下文自检通过"

# ══════════ 阶段 3：两路并行评分（每卡 1 进程，仍守「1 进程/GPU」）══════════

cd "$GENEVAL" || { fail "cd $GENEVAL 失败"; exit 3; }
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

# 动态认领：mkdir 在**本地** fs 上原子 → 两路不会抢同一个 config。
# 故意放 /tmp 而不是 /nas3：NFS 的 mkdir 原子性不保证，万一双写同一 results.jsonl 会损坏文件。
CLAIM=/tmp/geneval_wd_claims_$$
mkdir -p "$CLAIM"

score_track() {
  local gpu=$1 tag n rc tmp tlog="$LOG/geneval_step5_gpu${1}.log"
  echo "[$(date '+%H:%M:%S')] track GPU$gpu started (pid=$BASHPID)" >> "$tlog"
  for tag in "${TODO_TAGS[@]}"; do
    n=$(results_lines "$tag")
    if [ "$n" -ge "$TARGET" ]; then
      echo "[$(date '+%H:%M:%S')] [GPU$gpu] SKIP $tag — results.jsonl 已 $n 行" >> "$tlog"
      continue
    fi
    mkdir "$CLAIM/$tag" 2>/dev/null || continue    # 没抢到的交给另一路

    echo "[$(date '+%H:%M:%S')] [GPU$gpu] === scoring $tag ===" >> "$tlog"
    tmp="$OUT/$tag/results.jsonl.wdtmp"
    # 先写 .wdtmp 再原子 mv：中途被平台杀掉也不会留下半个 results.jsonl
    # （半个文件会被 results_lines 误判成"已评分"而永久跳过 → 静默丢分）
    CUDA_VISIBLE_DEVICES=$gpu python evaluation/evaluate_images.py "$OUT/$tag/geneval" \
      --outfile "$tmp" --clip-model-path "$CLIP_BIN" >> "$tlog" 2>&1
    rc=$?
    n=$(tmp_lines "$tmp")
    if [ "$rc" -eq 0 ] && [ "$n" -ge "$TARGET" ]; then
      mv "$tmp" "$OUT/$tag/results.jsonl"
      echo "[$(date '+%H:%M:%S')] [GPU$gpu] DONE $tag ($n 行)" >> "$tlog"
      python evaluation/summary_scores.py "$OUT/$tag/results.jsonl" 2>/dev/null | tail -11 >> "$tlog"
    else
      # 失败的 config 保留 claim（不让另一路重试同一个）+ 保留 .wdtmp 供事后查因
      echo "[$(date '+%H:%M:%S')] [GPU$gpu] FAIL $tag rc=$rc lines=$n — 保留 $tmp，继续下一个" >> "$tlog"
    fi
  done
  echo "[$(date '+%H:%M:%S')] track GPU$gpu finished" >> "$tlog"
}

good "启动两路评分（GPU0 / GPU1 各 1 进程；日志 geneval_step5_gpu{0,1}.log）"
score_track 0 &
T0=$!
score_track 1 &
T1=$!
wait $T0
wait $T1

# ══════════ 阶段 4：汇总 19 config 总分矩阵（Step 6 的原料）══════════
# 注：summary_scores.py 只读 jsonl 做聚合；且此刻评分已全部结束、GPU 空闲，
#     即使它 import 到 torch 也不存在并发达上下文的风险。
SUM="$LOG/geneval_final_scores.txt"
{
  echo "# GenEval Overall — 汇总于 $(date '+%F %T') by geneval_watchdog.sh"
  echo "# 评分器 = torchvision Faster R-CNN（非 Mask2Former，见 TODO Step 6 注：绝对值与论文不可直接对齐）"
  printf "%-16s %8s %10s\n" "config" "png" "Overall"
  for tag in "${ALL_TAGS[@]}"; do
    c=$(find "$OUT/$tag" -name '*.png' 2>/dev/null | wc -l)
    n=$(results_lines "$tag")
    if [ "$n" -ge "$TARGET" ]; then
      ov=$(python evaluation/summary_scores.py "$OUT/$tag/results.jsonl" 2>/dev/null \
             | grep -iE 'overall' | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
      printf "%-16s %8s %10s\n" "$tag" "$c" "${ov:-解析失败}"
    else
      printf "%-16s %8s %10s\n" "$tag" "$c" "未评分(${n}行)"
    fi
  done
} > "$SUM"

echo ""
cat "$SUM"
echo ""
good "Step 5 完成 — 分数矩阵已写入 $SUM"
info "下一步（Step 6）：把 Overall 整理成 方法×位宽 矩阵写入 TODO.md，对比 fp16=0.64563 算保持率"
log "===== Watchdog 正常结束 ====="
exit 0
