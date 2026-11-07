#!/usr/bin/env bash
# ============================================================================
# P0-2 修复：干净口径对照（当前 calib + n=128，所有方法同源同量）
#
#   背景：主表里 mllmq 用 10-02 新 calib n=128，mbq/awq 用 10-01 旧 calib n=256
#         → 跨方法比较混入了「校准内容 + 校准量」两个变量。
#   本脚本：
#     1) 评分版本漂移复核：把主表 mllmq_mae_w3 的旧评分图重评一次，对比 Overall；
#     2) 补生成缺失的 awq W3 n=128 scale（其余 n128 scale 已存在且为当前 calib）；
#     3) 在 GenEval 25/tag 上跑 mbq_scalar / mbq_block / awq 的 W4/W3（各 600 图）；
#        mllmq(fixed) 的 W4/W3 由 run_mllmq_fixed_eval.sh 产出，最后一起汇总。
#
#   会等本机所有 GPU 队列（DPG → k10/k14 → mllmq_fixed）结束再开始。
# 产出：output/janus-pro-7b/ablation_clean25/<tag>/geneval/results.jsonl
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_clean_comparison.sh \
#     > ../output/logs/clean25_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/janus-pro-7b/ablation_clean25
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1"}
PER_TAG=${PER_TAG:-25}

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
[ -n "$CLIP_BIN" ] || { echo "CLIP 权重缺失"; exit 2; }
mkdir -p "$OUT"

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== 干净口径对照（current calib, n=128, GenEval 25/tag）====="
sleep 60
while pgrep -f "[r]un_exp3_dpg_mixed.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_exp3_mixed.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_mllmq_fixed_eval.sh" >/dev/null 2>&1; do
  log "等待本机 GPU 队列结束..."; sleep 120
done
log "本机队列结束 → 开始"

# ---------- 1) 评分版本漂移复核 ----------
if [ ! -f /tmp/audit_rescore_w3.jsonl ] || [ "$(wc -l < /tmp/audit_rescore_w3.jsonl 2>/dev/null || echo 0)" -lt 100 ]; then
  info "重评主表 mllmq_mae_w3（验证 open_clip 版本漂移）"
  CUDA_VISIBLE_DEVICES=${SCORE_GPU:-0} python "$EV/evaluate_images.py" \
    "$ROOT/output/janus-pro-7b/geneval/mllmq_mae_w3" \
    --outfile /tmp/audit_rescore_w3.jsonl --clip-model-path "$CLIP_BIN" \
    > "$LOG/audit_rescore_w3.log" 2>&1
  if [ -f /tmp/audit_rescore_w3.jsonl ]; then
    log "旧评分（主表记 0.5848）vs 新环境重评："
    python "$EV/summary_scores.py" /tmp/audit_rescore_w3.jsonl 2>/dev/null | grep -i '^Overall'
  else fail "重评失败"; fi
else good "评分漂移复核已存在"; fi

# ---------- 2) 补 AWQ W3 n=128 scale ----------
AWQ3=$CACHE/awq/janus_awq_w3a16_n128.pt
NEED_AWQ3=0
if [ -z "${ONLY:-}" ] || echo " ${ONLY:-} " | grep -q " awq_w3 "; then NEED_AWQ3=1; fi
if [ "$NEED_AWQ3" = "1" ] && [ ! -s "$AWQ3" ]; then
  info "生成 AWQ W3 n=128 scale（当前 calib）"
  CUDA_VISIBLE_DEVICES=0 python main_quant.py --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$CALIB" --image_folder "$IMG" --n_samples 128 \
    --w_bit 3 --a_bit 16 --w_group 128 --run_process --method awq --scale_path "$AWQ3" \
    > "$LOG/clean25_scale_awq_w3.log" 2>&1
  [ -s "$AWQ3" ] && good "AWQ W3 n128 scale OK" || fail "AWQ W3 scale 失败"
else good "AWQ W3 n128 scale 已存在"; fi

# ---------- 3) 对照评测 ----------
run_one(){ # tag qm w sp gpu
  local tag=$1 qm=$2 w=$3 sp=$4 gpu=$5
  local out="$OUT/$tag" gdir="$OUT/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag"; return 0; }
  [ -s "$sp" ] || { fail "[GPU$gpu] 缺 scale $sp"; return 1; }
  mkdir -p "$out"
  info "[GPU$gpu] $tag 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
    > "$LOG/clean25_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; return 1; }
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/clean25_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

TASKS=(
  "scalar_w4|mbq_scalar|4|$CACHE/mbq/janus_mbq_scalar_w4a16_n128.pt"
  "scalar_w3|mbq_scalar|3|$CACHE/mbq/janus_mbq_scalar_w3a16_n128.pt"
  "block_w4|mbq_block|4|$CACHE/mbq/janus_mbq_block_w4a16_n128.pt"
  "block_w3|mbq_block|3|$CACHE/mbq/janus_mbq_block_w3a16_n128.pt"
  "awq_w4|awq|4|$CACHE/awq/janus_awq_w4a16_n128.pt"
  "awq_w3|awq|3|$AWQ3"
)
# ONLY="scalar_w4 scalar_w3" 可只跑子集（用于两机分工，避免重复）
ONLY=${ONLY:-}
gpus=($GPUS); nlanes=${#gpus[@]}
idx=0
for t in "${TASKS[@]}"; do
  IFS='|' read -r tag qm w sp <<< "$t"
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $tag "; then
    log "SKIP $tag（不在 ONLY 列表）"; continue
  fi
  run_one "$tag" "$qm" "$w" "$sp" "${gpus[$((idx % nlanes))]}" &
  idx=$((idx+1))
  if [ $((idx % nlanes)) -eq 0 ]; then wait; fi
done
wait

log "════ 干净口径汇总（25/tag=600 图）════"
for t in ${ONLY:-scalar_w4 scalar_w3 block_w4 block_w3 awq_w4 awq_w3}; do
  f="$OUT/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
for t in w4 w3; do
  f="$ROOT/output/janus-pro-7b/ablation_mllmq_fixed/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "mllmq_fixed_$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || log "mllmq_fixed_$t 未就绪"
done
info "ALL DONE"