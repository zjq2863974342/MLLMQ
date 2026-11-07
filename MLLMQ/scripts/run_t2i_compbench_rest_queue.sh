#!/bin/bash
# ============================================================================
# T2I-CompBench「剩余 config 接跑队列」 —— GPU 粒度，哪张卡空哪张卡接
#
# 场景：对端生成容器已被回收，13 config 还差若干未跑满；本机 GPU0 可能在跑别的
#       （如 mllmq_mse_w4），GPU1 空 —— 本脚本让**空的卡立刻接下一个 tag**，
#       而不是干等两张卡都空（否则空闲卡白等数小时）。
#
# 语义：
#   · 启动时扫描全部 13 tag，只把「未满 2400 且本机无进程在写」的排进队列
#   · 两条 lane（GPU0 / GPU1）共享一个队列文件，flock 原子取号 → 绝不两卡同 tag
#     （🔴 同 tag 双写会互相截断，见 HANDOFF/dual_queue 硬约束）
#   · skip-existing：底层 eval_generate_quant.py 逐张 os.path.exists 跳过 → 断点续
#   · 跑完即退；任何时候重跑本脚本都会自动接着没跑完的继续
#
# 用法：
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_t2i_compbench_rest_queue.sh \
#     > /nas3/kangjia/mllmq/output/logs/t2i_rest_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   只看计划（不执行）：  PLAN=1 bash scripts/run_t2i_compbench_rest_queue.sh
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=/nas3/kangjia/mllmq/checkpoints/torch_home

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/t2i_compbench
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=2400
NUM_SAMPLES=1
COOLDOWN=30
MAX_RETRY=3
GPU_FREE_MIB=6000              # 平台保活 ~1471MiB；真实生成/评分任务 ≥7GB
POLL=60                        # 等卡间隙轮询秒数
PLAN=${PLAN:-0}
GPUS="0 1"

QFILE=/tmp/t2i_rest_queue.$$   # 本次运行的队列（tag 逐行）
QLOCK=/tmp/t2i_rest_queue.lock # 跨 lane 取号锁
DRIVER_PID=$$

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date '+%F %T')] $1"; }
good() { echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date '+%F %T')] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

cleanup() { warn "Interrupted — killing children..."; pkill -P $DRIVER_PID 2>/dev/null; log "Driver exiting."; exit 130; }
trap cleanup SIGINT SIGTERM

ALL_TAGS="fp16 rtn_w3 awq_w3 mbq_block_w3 mbq_scalar_w3 mllmq_mae_w3 mllmq_mse_w3 \
          rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4"

# tag -> "method|w_bit|scale_path"（与 run_t2i_compbench_dual_queue.sh 完全一致）
params() {
  case "$1" in
    fp16)          echo "fp16||";;
    rtn_w3)        echo "rtn|3|";;
    rtn_w4)        echo "rtn|4|";;
    awq_w3)        echo "awq|3|$SCALE/awq/janus_awq_w3a16.pt";;
    awq_w4)        echo "awq|4|$SCALE/awq/janus_awq_w4a16.pt";;
    mbq_block_w3)  echo "mbq_block|3|$SCALE/mbq/janus_mbq_block_w3a16.pt";;
    mbq_block_w4)  echo "mbq_block|4|$SCALE/mbq/janus_mbq_block_w4a16.pt";;
    mbq_scalar_w3) echo "mbq_scalar|3|$SCALE/mbq/janus_mbq_scalar_w3a16.pt";;
    mbq_scalar_w4) echo "mbq_scalar|4|$SCALE/mbq/janus_mbq_scalar_w4a16.pt";;
    mllmq_mae_w3)  echo "mllmq_mae|3|$SCALE/mllmq/janus_mae_w3a16.pt";;
    mllmq_mae_w4)  echo "mllmq_mae|4|$SCALE/mllmq/janus_mae_w4a16.pt";;
    mllmq_mse_w3)  echo "mllmq_mse|3|$SCALE/mllmq/janus_mse_w3a16.pt";;
    mllmq_mse_w4)  echo "mllmq_mse|4|$SCALE/mllmq/janus_mse_w4a16.pt";;
  esac
}

count_pngs() { find "${OUT}/$1/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }
# 注意：不加尾斜杠 —— cmdline 里是 ".../t2i_compbench/<tag> --num_samples ..."，带斜杠匹配不到
is_running() { pgrep -af '[e]val_generate_quant' | grep -q "t2i_compbench/$1"; }
gpu_used()   { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

build_queue() {   # 排入：未满 2400 且本机无进程在写
  : > "$QFILE"
  for t in $ALL_TAGS; do
    n=$(count_pngs "$t")
    [ "$n" -ge "$TARGET" ] && continue
    if is_running "$t"; then warn "不排队 $t — 已有进程在写($n/$TARGET)"; continue; fi
    echo "$t" >> "$QFILE"
  done
}

pop_tag() {       # flock 原子取号，跨 lane 不撞
  local t
  exec 9>"$QLOCK"; flock 9
  t=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$t" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp" 2>/dev/null; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-
  echo "$t"
}

run_one() {       # $1=tag  $2=gpu
  local tag=$1 gpu=$2 p method w sp n after rc attempt logf
  p=$(params "$tag"); method=${p%%|*}; w=$(echo "$p" | cut -d'|' -f2); sp=$(echo "$p" | cut -d'|' -f3)
  n=$(count_pngs "$tag")
  [ "$n" -ge "$TARGET" ] && { good "[GPU$gpu] SKIP $tag — $n/$TARGET"; return 0; }
  logf="${LOG}/t2i_${tag}.log"
  attempt=1
  while [ $attempt -le $MAX_RETRY ]; do
    n=$(count_pngs "$tag"); [ "$n" -ge "$TARGET" ] && break
    sleep $COOLDOWN
    info "[GPU$gpu] LAUNCH $tag ($method w${w:-fp16}, $n/$TARGET existing) attempt ${attempt}/${MAX_RETRY}"
    local cmd=(python eval_generate_quant.py
      --model_path "$MODEL" --benchmark t2i_compbench
      --output_dir "${OUT}/${tag}" --num_samples "$NUM_SAMPLES"
      --quant_method "$method")
    [ -n "$w" ]  && cmd+=(--w_bit "$w")
    [ -n "$sp" ] && cmd+=(--scale_path "$sp")
    CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" >> "$logf" 2>&1
    rc=$?
    after=$(count_pngs "$tag")
    [ "$after" -ge "$TARGET" ] && { good "[GPU$gpu] DONE $tag — $after/$TARGET"; return 0; }
    if [ "$rc" -ne 0 ] && [ "$after" -le "$n" ]; then
      fail "[GPU$gpu] ABORT $tag — rc=$rc 且 0 新增（非 ALL DONE，是启动失败）"; tail -5 "$logf"; return 1
    fi
    warn "[GPU$gpu] INCOMPLETE $tag rc=$rc ($after/$TARGET)"
    attempt=$(( attempt + 1 )); sleep $COOLDOWN
  done
  fail "[GPU$gpu] GAVE-UP $tag ($(count_pngs "$tag")/$TARGET)"; return 1
}

lane() {          # $1=gpu —— 先等本卡空，再取号（避免空卡干等、也避免占号等待）
  local gpu=$1 tag u
  while :; do
    while :; do
      u=$(gpu_used "$gpu"); [ "${u:-0}" -lt "$GPU_FREE_MIB" ] && break
      log "[GPU$gpu] 忙（used=${u}MiB）→ ${POLL}s 后再探"; sleep $POLL
    done
    tag=$(pop_tag)
    [ -z "$tag" ] && { log "[GPU$gpu] 队列已空 → lane 退出"; return 0; }
    is_running "$tag" && { warn "[GPU$gpu] $tag 正被别的进程写 → 跳过"; continue; }
    run_one "$tag" "$gpu"
  done
}

# ── main ──
log "===== T2I-CompBench 剩余 config 接跑队列（GPU 粒度）====="
echo ""; log "── 启动现状 ──"
for t in $ALL_TAGS; do printf "   %-18s %5s/%s\n" "$t" "$(count_pngs "$t")" "$TARGET"; done
echo ""
for g in 0 1; do
  if CUDA_VISIBLE_DEVICES=$g timeout -s 9 60 python -c \
       "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize()" >/dev/null 2>&1; then
    good "GPU${g} CUDA 上下文自检通过"
  else
    fail "GPU${g} 无法创建 CUDA 上下文 → 拒绝启动"; exit 2
  fi
done

build_queue
if [ ! -s "$QFILE" ]; then good "没有未跑满的 tag → 无需接跑，退出"; exit 0; fi
log "本次队列：$(tr '\n' ' ' < "$QFILE")"
if [ "$PLAN" = "1" ]; then echo ""; info "PLAN=1 → 仅打印计划，未执行。"; exit 0; fi

for g in $GPUS; do lane "$g" & done
wait

echo ""; log "════════ 最终清点 ════════"
allok=1
for t in $ALL_TAGS; do
  n=$(count_pngs "$t")
  if [ "$n" -ge "$TARGET" ]; then good "  $t  $n/$TARGET"; else fail "  $t  $n/$TARGET"; allok=0; fi
done
if [ "$allok" -eq 1 ]; then info "🎉 T2I-CompBench 13 config 全部生成完成"; else fail "仍有未满 tag —— 重跑本脚本即可续跑"; fi
rm -f "$QFILE"