#!/usr/bin/env bash
# ============================================================================
# 校准消融（极速探路）：mllmq_mae W4，校准样本量 n ∈ {64,128,256}
#
# 目的：验证「半数校准样本达同等质量」——n=128 是否与 n=256 持平、n=64 是否已饱和
# 评测：GenEval 分层子集（每 tag 25 条 × 6 tag = 150 prompt × 4 = 600 张）+ 理解侧轻量组
#       （mmmu_val + pope_random_local）
#
# scale：n=64 → janus_mae_w4a16_n64.pt（新）
#        n=128 → janus_mae_w4a16.pt      （现有，= 所有已发布 mllmq 结果所用）
#        n=256 → janus_mae_w4a16_n256.pt（新）
#
# 用法：cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_calib_ablation_w4.sh \
#         > /nas3/kangjia/mllmq/output/logs/calibab_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache/mllmq
OUTBASE=$ROOT/output/janus-pro-7b/calib_ablation
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GENEVAL_PER_TAG=${GENEVAL_PER_TAG:-25}
UND_TASKS=${UND_TASKS:-"mmmu_val,pope_random_local"}
GPUS="0 1"

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "[$(date '+%F %T')] $1"; }
good() { echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail() { echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
FRCNN="$EV/.torch_home/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"

scale_of() { case "$1" in
  64|n64)   echo "$CACHE/janus_mae_w4a16_n64.pt" ;;
  128|n128) echo "$CACHE/janus_mae_w4a16.pt" ;;
  256|n256) echo "$CACHE/janus_mae_w4a16_n256.pt" ;;
esac; }

run_one() {   # $1=n  $2=gpu
  local n=$1 gpu=$2 scale out gdir
  scale=$(scale_of "$n"); out="$OUTBASE/mllmq_mae_w4_$n"; gdir="$out/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  info "===== [GPU$gpu] n=$n  scale=$(basename "$scale") ====="
  if [ ! -s "$scale" ]; then
    log "[GPU$gpu] n=$n 等 scale 就绪：$scale"
    local t=0; while [ ! -s "$scale" ] && [ $t -lt 1800 ]; do sleep 20; t=$((t+20)); done
  fi
  [ -s "$scale" ] || { fail "缺 scale $scale（等待超时）"; return 1; }
  mkdir -p "$out"

  # 1) GenEval 分层子集生成
  if [ ! -f "$gdir/results.jsonl" ]; then
    log "[GPU$gpu] n=$n GenEval 生成（${GENEVAL_PER_TAG}/tag）..."
    python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
      --output_dir "$out" --num_samples 4 --limit_per_tag "$GENEVAL_PER_TAG" \
      --quant_method mllmq_mae --w_bit 4 --scale_path "$scale" \
      > "$LOG/calibab_gen_n${n}_gpu${gpu}.log" 2>&1
    [ $? -ne 0 ] && { fail "n=$n GenEval 生成失败"; return 1; }
    good "[GPU$gpu] n=$n GenEval 生成完成"
  else good "[GPU$gpu] n=$n GenEval 已存在，跳过"; fi

  # 2) GenEval 评分（FRCNN + CLIP）
  if [ ! -f "$gdir/results.jsonl" ]; then
    log "[GPU$gpu] n=$n GenEval 评分..."
    python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
      --clip-model-path "$CLIP_BIN" > "$LOG/calibab_score_n${n}_gpu${gpu}.log" 2>&1
    [ $? -ne 0 ] && { fail "n=$n GenEval 评分失败"; return 1; }
  fi
  local overall
  overall=$(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -iE "^Overall|Overall:" | head -1)
  good "[GPU$gpu] n=$n GenEval 评分完成 :: $overall"

  # 3) 理解侧轻量组
  log "[GPU$gpu] n=$n 理解侧（$UND_TASKS）..."
  python eval_benchmarks.py --model_type janus --model_path "$MODEL" \
    --method mllmq --method_tag "mllmq_mae_w4_${n}" --w_bit 4 \
    --scale_path "$scale" --tasks "$UND_TASKS" \
    --output_base "$OUTBASE/understanding" --device cuda:0 \
    > "$LOG/calibab_und_n${n}_gpu${gpu}.log" 2>&1
  [ $? -ne 0 ] && { fail "n=$n 理解侧失败"; return 1; }
  good "[GPU$gpu] n=$n 理解侧完成"
  return 0
}

DRIVER_PID=$$
cleanup() { echo "[$(date '+%F %T')] Interrupted — killing children..."; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== 校准消融（极速探路）mllmq_mae W4 · n ∈ {64,128,256} ====="
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失"; exit 2; }
[ -f "$FRCNN" ] || { fail "FRCNN 权重缺失"; exit 2; }
mkdir -p "$OUTBASE/understanding"

# 队列：GPU0 = n64 → n128；GPU1 = n256
( for n in 64 128; do run_one "$n" 0; done; echo "[$(date '+%F %T')] GPU0 lane done" ) &
P0=$!
( run_one 256 1; echo "[$(date '+%F %T')] GPU1 lane done" ) &
P1=$!
wait $P0 $P1

log "════════ 最终清点 ════════"
for n in 64 128 256; do
  f="$OUTBASE/mllmq_mae_w4_$n/geneval/results.jsonl"
  if [ -f "$f" ]; then good "n=$n: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -iE '^Overall' | head -1)"; else fail "n=$n: 无 results.jsonl"; fi
done
info "ALL DONE"