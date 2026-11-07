#!/bin/bash
# COCO-val W8 全量双队列 —— 补齐 W8 档缺失的 4 个 config（2026-10-16 20:10 建，20:34 按用户指示撤 rtn_w8）
#
# 背景：COCO-val 的 W8 档此前只有 `mllmq_mae_w8`（10-16 11:31 完工），awq/mbq_block/mbq_scalar/mse
#       的 W8 全无 COCO 图（rtn_w8 用户指示不补）（GenEval 侧 19 config 是全的）。用户 10-16 20:0x 指示"补 W8，控制变量"。
#
# 🔴🔴 控制变量清单（逐条对应，勿破坏）：
#   ① **全部在本容器(A800)跑**：`fp16`、6×W4、`mllmq_mae_w8` 都是本机产的；对端 W3 在 A100 上。
#      W8 那 5 个 config 若在对端 A100 生成，就会让 W8 行内部混两种硬件 → 一律本机。
#   ② **同一份生成脚本、同一套 flag**：`eval_generate_quant.py --benchmark coco_val --num_samples 1`，
#      temperature=1.0 / cfg_weight=5.0 / 576 token / 384×384 / patch16 全部走脚本默认值，不改一行。
#   ③ **同一份 prompt 集**：`data/coco/annotations/captions_val2017.json`，按 annotation 文件序
#      每个 image_id 取首条 caption → 与 fp16/W4/W3 完全同构（脚本内 `seen` 去重逻辑，未动）。
#   ④ **同一份 scale cache**：与 GenEval W8 用的是同一批 `.pt`（10-01/10-02 产出，之后未重算）。
#   ⑤ **同一套评分口径**：FID real = `data/coco/val2017_real`、两侧统一 Resize(384)+CenterCrop、
#      `*_0000.png` 过滤、real 特征 npz 与已评的 10 个 config 共用同一份缓存。
#   ⑥ **起跑前纯净性门禁**：目标目录若含 `*_000[1-9].png`（num_samples=4 老协议残档）→ exit 6 拒跑。
#      mllmq_mae_w8 那 322 张残档（只覆盖 81 个 prompt）已隔离到 coco_val_orphan_n4_20261013/。
#
# 队列排序（论文价值优先，与 W3 队列同款原则）：slot1 放 GenEval-W8 的两个头号对手
#   （mbq_block_w8 0.6395 / awq_w8 0.6482）→ 这两个先出，"mae_w8 无损甚至略优"的对比就有了对手；
#   slot2 补 mbq_scalar_w8 + mllmq_mse_w8。
#   🔴 原 slot3 的 rtn_w8 于 10-16 20:34 按用户指示移除（详见 GPU0_TASKS 里的注释与理由）。
#
# Usage: cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_coco_val_w8_full_dual_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/coco_val_w8_full_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=/nas3/kangjia/mllmq/checkpoints/torch_home

MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/coco_val
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=5000
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
count_legacy() { find "${OUT}/$1" -name '*_000[1-9].png' 2>/dev/null | wc -l; }

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
  local logf="${LOG}/coco_val_${tag}.log"
  local n=$(count_pngs "$tag")

  # 控制变量 ⑥：纯净性门禁（老协议残档会顶占 TARGET 额度 → 跑完静默缺 prompt）
  local leg=$(count_legacy "$tag")
  if [ "$leg" -gt 0 ]; then
    fail "[GPU${gpu}] $tag 含 $leg 张 *_0001+ 老协议图 → 拒绝起跑（整目录 mv 走再来）"
    return 6
  fi

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
# ════════════════════════════════════════════════════════
GPU0_TASKS=(
  "mbq_block_w8 mbq 8 ${SCALE}/mbq/janus_mbq_block_w8a16.pt"
  "mbq_scalar_w8 mbq 8 ${SCALE}/mbq/janus_mbq_scalar_w8a16.pt"
  # 🔴 `rtn_w8` 于 10-16 20:34 按用户指示**移除**（不跑）。理由：GenEval W8 上 rtn=0.6482
  #    与 awq 并列第一（W8 档几乎不退化），COCO 上大概率也是"与 fp16 无差别"→ 16h 的信息量最低；
  #    省下这 16h 让双卡在 10-18 ~05:20 就全空，可以做多 seed 误差棒（当前更卡论文的短板）。
  #    后果：COCO 的 W8 行只有 5 个方法(fp16 + mbq_block/mbq_scalar/awq/mse/mae 中的量化四法+mae)，
  #    **缺 rtn_w8**；GenEval 侧 rtn_w8 是全的 → 论文里两张表的 W8 列方法集不同，必须注明。
)
GPU1_TASKS=(
  "awq_w8 awq 8 ${SCALE}/awq/janus_awq_w8a16.pt"
  "mllmq_mse_w8 mbq 8 ${SCALE}/mllmq/janus_mse_w8a16.pt"
)

log "===== COCO-val W8 FULL Dual-Queue Driver ====="
log "目标：补齐 W8 档 4 个 config × 5000 张（本机 A800，与 fp16/mae_w8 同机 → 控制变量①）"
log "GPU0: 2 config (mbq_block_w8 → mbq_scalar_w8)"
log "GPU1: 2 config (awq_w8 → mllmq_mse_w8)"
log "实测速率 ~12.0 s/it → 单 config ~16.7h；两卡各 2 个 ≈ 33.4h（rtn_w8 已按用户指示移除）"

# ── scale cache 齐备性 ──
for tag_task in "${GPU0_TASKS[@]}" "${GPU1_TASKS[@]}"; do
  read -r t m w sc <<< "$tag_task"
  sc="${sc/#__NONE__/}"
  if [ -n "$sc" ] && [ ! -f "$sc" ]; then fail "scale cache 缺失: $sc ($t)"; exit 5; fi
done
good "scale cache 齐备（4 份，rtn_w8 不在队列内）"

# ── 起跑快照 ──
for tag in awq_w8 mbq_block_w8 mbq_scalar_w8 mllmq_mse_w8 mllmq_mae_w8; do
  log "  snapshot $tag: $(count_pngs "$tag")/$TARGET (legacy $(count_legacy "$tag"))"
done

# ── 前置门禁：不得有其它 GPU 任务 ──
if pgrep -f '[e]val_generate_quant|[s]core_pref|[s]core_fid|[s]core_clip|[m]ain_quant|[e]val_benchmarks' >/dev/null 2>&1; then
  fail "检测到其它 GPU 任务在跑 → 拒绝启动（并发建 CUDA 上下文危险，见 dialog §0）"
  pgrep -af '[e]val_generate_quant|[s]core_pref|[m]ain_quant' | head -3
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
  info "🎉 COCO-val W8 FULL — ALL DONE!"
fi
good "Elapsed: ~${ELAPSED} min ($(( ELAPSED / 60 ))h $(( ELAPSED % 60 ))min)"
echo "========================================="
log "Final matrix (W8 共 5 config；rtn_w8 按用户指示不跑)":
COMPLETED=0
for tag in mllmq_mae_w8 awq_w8 mbq_block_w8 mbq_scalar_w8 mllmq_mse_w8; do
  n=$(count_pngs "$tag")
  if [ "$n" -ge "$TARGET" ]; then good "  $tag: $n/$TARGET"; COMPLETED=$((COMPLETED+1)); else fail "  $tag: $n/$TARGET"; fi
done
info "Overall: $COMPLETED/5 W8 configs complete"
echo "========================================="
log "下一步评分：GPU=0 bash scripts/run_coco_scoring_suite.sh（ONLY=\"awq_w8 mbq_block_w8 mbq_scalar_w8 mllmq_mse_w8\" 拆双卡）"
