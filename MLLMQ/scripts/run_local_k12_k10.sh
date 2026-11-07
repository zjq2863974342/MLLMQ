#!/usr/bin/env bash
# ============================================================================
# 本机：等 GAMQ 理解侧结束后，跑 GAMQ 的两个全量 GenEval
#   1) k12 全量 second seed  → ablation_mixed_full/w34_spread_k12_full_s2
#   2) k10 全量 first seed   → ablation_mixed_full/w34_spread_k10_full
#   标准采样口径（temperature=1.0, 4 张/prompt, 2212 图），与现有 k12/k10 结果可比。
#   2 卡 × 2 分片，串行两个 config，约 8h。
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_local_k12_k10.sh \
#     > ../output/logs/local_k12_k10_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
LOG=$ROOT/output/logs
CACHE=$ROOT/MLLMQ/scale_cache
K12=$CACHE/mllmq/janus_mixed_w34_spread_k12.pt
K10=$CACHE/mllmq/janus_mixed_w34_spread_k10.pt

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"
export PYTHONPATH="$ROOT/pydeps/geneval_eval${PYTHONPATH:+:$PYTHONPATH}"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
score_of(){ python "$EV/summary_scores.py" "$1" 2>/dev/null | awk '/Overall/{print $NF}' | tail -1; }

log "===== 等 GAMQ 理解侧结束 ====="
sleep 60
while pgrep -f "[r]un_exp3_understanding_mixed.sh" >/dev/null 2>&1; do
  log "等待理解侧..."; sleep 120
done
log "理解侧结束 → 开始 k12/k10 全量"

run_cfg(){ # tag scale
  local tag=$1 sp=$2
  local out="$ROOT/output/janus-pro-7b/$tag" gdir="$ROOT/output/janus-pro-7b/$tag/geneval"
  [ -s "$sp" ] || { fail "$tag 缺 scale $sp"; return 1; }
  if [ -f "$gdir/results.jsonl" ]; then good "SKIP $tag ($(score_of "$gdir/results.jsonl"))"; return 0; fi
  mkdir -p "$out"
  info "$tag：全量 2212 图（2 分片）"
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_geneval_shard.py \
    --model_path "$MODEL" --output_dir "$out" --num_samples 4 \
    --shard_id 0 --num_shards 2 --quant_method mllmq_mixed --w_bit 3 --scale_path "$sp" \
    > "$LOG/local_k12k10_gen_${tag}_s0.log" 2>&1 & local p0=$!
  CUDA_VISIBLE_DEVICES=1 python scripts/generate_geneval_shard.py \
    --model_path "$MODEL" --output_dir "$out" --num_samples 4 \
    --shard_id 1 --num_shards 2 --quant_method mllmq_mixed --w_bit 3 --scale_path "$sp" \
    > "$LOG/local_k12k10_gen_${tag}_s1.log" 2>&1 & local p1=$!
  local bad=0; wait $p0 || bad=$((bad+1)); wait $p1 || bad=$((bad+1))
  [ "$bad" -gt 0 ] && { fail "$tag 有分片失败"; return 1; }
  if [ -n "$CLIP_BIN" ]; then
    CUDA_VISIBLE_DEVICES=0 python "$EV/evaluate_images.py" "$gdir" \
      --outfile "$gdir/results.jsonl" --clip-model-path "$CLIP_BIN" \
      > "$LOG/local_k12k10_score_${tag}.log" 2>&1
    [ -f "$gdir/results.jsonl" ] && good "$tag :: $(score_of "$gdir/results.jsonl")" \
      || fail "$tag 评分失败"
  else
    fail "$tag 缺 CLIP 权重，仅保留图像"
  fi
}

run_cfg "ablation_mixed_full/w34_spread_k12_full_s2" "$K12"
run_cfg "ablation_mixed_full/w34_spread_k10_full"    "$K10"
info "ALL DONE"