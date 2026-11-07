#!/bin/bash
# ============================================================================
# T2I-CompBench++ · 生成双队列（8 类 × 300 val prompt = 2400 prompt，每 prompt 1 张 384²）
#
# 范围：fp16 + 6 方法 × (W3+W4) = 13 config —— 与 DPG-Bench / GenEval / COCO-val 口径对齐
# 规模：2400 × 13 = 31,200 张；实测 ~12.5 s/张 → 双卡 ≈ 54h
#
# 输出：OUT/<tag>/t2i_compbench/<category>/samples/<prompt原文>_<6位序号>.png
#   🔴 命名照官方约定（官方三个评分器都从文件名反解 prompt：file_name.split('_')[0]）
#      见 T2I-CompBench-main/UniDet_eval/2D_spatial_eval.py:167、BLIPvqa_eval/BLIP_vqa.py:34
#   🔴 已校验 2400 条 val prompt 零下划线、无斜杠 → 可直用原文作文件名
#
# 🔴 纪律：本机生成在跑时禁新 CUDA 进程（pgrep 一律用 [xyz] 防自匹配）
#
# Usage: cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_t2i_compbench_dual_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/t2i_compbench_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#        只看计划：  PLAN=1 bash scripts/run_t2i_compbench_dual_queue.sh
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
TARGET=2400            # 8 类 × 300
NUM_SAMPLES=1
COOLDOWN=30
MAX_RETRY=3
PLAN=${PLAN:-0}

DRIVER_PID=$$
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date '+%F %T')] $1"; }
good() { echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date '+%F %T')] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

count_pngs() { find "${OUT}/$1/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }
is_running() { ps aux | grep "[e]val_generate_quant" | grep -q "t2i_compbench/$1"; }

cleanup() { warn "Interrupted — killing children..."; pkill -P $DRIVER_PID 2>/dev/null; log "Driver exiting."; exit 130; }
trap cleanup SIGINT SIGTERM

run_one() {
  local tag=$1 method=$2 w=$3 gpu=$4 scale_path=${5:-}
  local outd="${OUT}/${tag}"
  local logf="${LOG}/t2i_${tag}.log"
  local n=$(count_pngs "$tag")

  if [ "$n" -ge "$TARGET" ]; then good "[GPU${gpu}] SKIP $tag — $n/$TARGET"; return 0; fi
  if is_running "$tag"; then
    warn "[GPU${gpu}] $tag 已有进程($n/$TARGET) → 等它"
    while is_running "$tag"; do sleep 30; done
    n=$(count_pngs "$tag"); [ "$n" -ge "$TARGET" ] && { good "[GPU${gpu}] SKIP $tag"; return 0; }
  fi

  local attempt=1
  while [ $attempt -le $MAX_RETRY ]; do
    n=$(count_pngs "$tag"); [ "$n" -ge "$TARGET" ] && break
    sleep $COOLDOWN
    info "[GPU${gpu}] LAUNCH $tag ($method w${w:-fp16}, $n/$TARGET existing) attempt ${attempt}/${MAX_RETRY}"
    local cmd=(python eval_generate_quant.py
      --model_path "$MODEL" --benchmark t2i_compbench
      --output_dir "$outd" --num_samples "$NUM_SAMPLES"
      --quant_method "$method")
    [ -n "$w" ] && cmd+=(--w_bit "$w")
    [ -n "$scale_path" ] && cmd+=(--scale_path "$scale_path")
    CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" >> "$logf" 2>&1
    local rc=$?
    local after=$(count_pngs "$tag"); local gain=$(( after - n ))
    if [ "$after" -ge "$TARGET" ]; then good "[GPU${gpu}] DONE $tag — $after/$TARGET"; return 0; fi
    # 🔴 防假 ALL DONE：rc≠0 且一张没新增 → 立即放弃，不要空转到 MAX_RETRY
    if [ "$rc" -ne 0 ] && [ "$gain" -le 0 ]; then
      fail "[GPU${gpu}] ABORT $tag — rc=$rc 且 0 新增（不是 ALL DONE，是启动失败）"
      tail -5 "$logf" 2>/dev/null
      return 1
    fi
    warn "[GPU${gpu}] INCOMPLETE $tag rc=$rc gain=$gain ($after/$TARGET)"
    attempt=$(( attempt + 1 )); sleep $COOLDOWN
  done
  fail "[GPU${gpu}] GAVE-UP $tag ($(count_pngs "$tag")/$TARGET)"; return 1
}

# 队列排布：13 config 均分两卡（7 / 6）
#   10-26 续跑调衡：剩余 mbq_block_w4(1531)/mllmq_mse_w4(1616)/mllmq_mae_w4(2400)，
#   末两 config 由 mae_w4↔mse_w4 对调 → GPU0 3147 it(~10.7h) / GPU1 2400 it(~8.1h)，
#   原排布 GPU0 3931 it(~13.3h) / GPU1 1616 it(~5.5h) 会让 GPU1 空转 ~7.8h。
queue_gpu0() {
  run_one fp16          fp16 ""  0 ""                                        || return 1
  run_one rtn_w3        rtn  3   0 ""                                        || return 1
  run_one mbq_block_w3  mbq_block 3 0 "$SCALE/mbq/janus_mbq_block_w3a16.pt"  || return 1
  run_one mllmq_mae_w3  mllmq_mae 3 0 "$SCALE/mllmq/janus_mae_w3a16.pt"      || return 1
  run_one rtn_w4        rtn  4   0 ""                                        || return 1
  run_one mbq_block_w4  mbq_block 4 0 "$SCALE/mbq/janus_mbq_block_w4a16.pt"  || return 1
  run_one mllmq_mse_w4  mllmq_mse 4 0 "$SCALE/mllmq/janus_mse_w4a16.pt"      || return 1
  good "GPU0 队列完成"
}
queue_gpu1() {
  run_one awq_w3        awq  3   1 "$SCALE/awq/janus_awq_w3a16.pt"            || return 1
  run_one mbq_scalar_w3 mbq_scalar 3 1 "$SCALE/mbq/janus_mbq_scalar_w3a16.pt" || return 1
  run_one mllmq_mse_w3  mllmq_mse 3 1 "$SCALE/mllmq/janus_mse_w3a16.pt"       || return 1
  run_one awq_w4        awq  4   1 "$SCALE/awq/janus_awq_w4a16.pt"            || return 1
  run_one mbq_scalar_w4 mbq_scalar 4 1 "$SCALE/mbq/janus_mbq_scalar_w4a16.pt" || return 1
  run_one mllmq_mae_w4  mllmq_mae 4 1 "$SCALE/mllmq/janus_mae_w4a16.pt"       || return 1
  good "GPU1 队列完成"
}

# ── main ──
log "===== T2I-CompBench 生成 · 双队列 ====="
log "GPU0: fp16 → rtn_w3 → mbq_block_w3 → mae_w3 → rtn_w4 → mbq_block_w4 → mse_w4 （7）"
log "GPU1: awq_w3 → mbq_scalar_w3 → mse_w3 → awq_w4 → mbq_scalar_w4 → mae_w4 （6）"
log "规模：2400 prompt × 13 config = 31,200 张；~12.5 s/张 → 双卡 ≈ 54h"

echo ""; log "── 现状 ──"
for tag in fp16 rtn_w3 awq_w3 mbq_block_w3 mbq_scalar_w3 mllmq_mae_w3 mllmq_mse_w3 \
           rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4; do
  printf "   %-18s %5s/%s\n" "$tag" "$(count_pngs "$tag")" "$TARGET"
done
echo ""

# 依赖自检
[ -d "/nas3/kangjia/mllmq/data/benchmarks_gen/T2I-CompBench-main/examples/dataset" ] \
  || { fail "T2I-CompBench prompt 目录不存在"; exit 2; }
miss=0
for p in "$SCALE/mbq/janus_mbq_block_w3a16.pt" "$SCALE/mbq/janus_mbq_block_w4a16.pt" \
         "$SCALE/mbq/janus_mbq_scalar_w3a16.pt" "$SCALE/mbq/janus_mbq_scalar_w4a16.pt" \
         "$SCALE/mllmq/janus_mae_w3a16.pt" "$SCALE/mllmq/janus_mae_w4a16.pt" \
         "$SCALE/mllmq/janus_mse_w3a16.pt" "$SCALE/mllmq/janus_mse_w4a16.pt" \
         "$SCALE/awq/janus_awq_w3a16.pt" "$SCALE/awq/janus_awq_w4a16.pt"; do
  [ -f "$p" ] || { fail "缺 scale cache: $p"; miss=1; }
done
[ "$miss" -eq 0 ] && good "scale cache 10/10 齐备（rtn 无需 cache）"

if [ "$PLAN" = "1" ]; then echo ""; info "PLAN=1 → 仅打印计划，未执行。"; exit 0; fi

# ── 门禁 ──
BUSY_PAT='[e]val_generate_quant|[e]val_benchmarks|[s]core_dpg_coco|[c]ompute_dpg_bench|[s]core_pref|[s]core_fid|[s]core_clip|[m]ain_quant|[r]un_dpg_score_dual_queue|[r]un_dpg_score_grid4_dual_queue|[r]un_janus_'
if pgrep -f "$BUSY_PAT" >/dev/null 2>&1; then
  fail "检测到其它 GPU 任务在跑 → 拒绝启动（见 dialog §0 并发建 CUDA 上下文危险）"
  pgrep -af "$BUSY_PAT" | head -5
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

queue_gpu0 & PID0=$!
queue_gpu1 & PID1=$!
wait $PID0; RC0=$?
wait $PID1; RC1=$?

echo ""; log "════════ 最终清点（目标 $TARGET）════════"
allok=1
for tag in fp16 rtn_w3 awq_w3 mbq_block_w3 mbq_scalar_w3 mllmq_mae_w3 mllmq_mse_w3 \
           rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4; do
  n=$(count_pngs "$tag")
  if [ "$n" -ge "$TARGET" ]; then good "  $tag  $n/$TARGET"; else fail "  $tag  $n/$TARGET"; allok=0; fi
done
if [ "$RC0" -eq 0 ] && [ "$RC1" -eq 0 ] && [ "$allok" -eq 1 ]; then
  info "🎉 T2I-CompBench 生成 ALL DONE"
else
  fail "未全部完成 rc0=$RC0 rc1=$RC1（逐 config 计数续跑，重跑本脚本即可）"
fi
echo ""
info "下一步：评分（需先装 BLIP / spaCy / CLIP-ViT-B-32 / UniDet 权重，见 Readme）"