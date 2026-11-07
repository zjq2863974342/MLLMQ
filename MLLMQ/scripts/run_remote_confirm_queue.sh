#!/usr/bin/env bash
# ============================================================================
# 副机确认队列（2026-11-02）：用独立 second seed 验证两个关键结论
#
#   Round 1（4 卡，~4h）
#     - GAMQ k12 全量 second seed     → ablation_mixed_full/w34_spread_k12_full_s2
#     - GAMQ k10 全量 first seed      → ablation_mixed_full/w34_spread_k10_full
#   Round 2（4 卡，~4h）
#     - MLLMQ W3 n128 全量 second seed → mllmq_vs_mbqscalar_full_w3/mllmq_mae_w3_n128_full_s2
#     - MBQ-scalar W3 n128 全量 second seed → mllmq_vs_mbqscalar_full_w3/mbq_scalar_w3_n128_full_s2
#   Round 3（4 卡，~4h）
#     - T2I 修复版 W4 全量（25/tag 无增益，需全量判定）→ ablation_mllmq_fixed_full/fixed_w4_full_s1
#     - T2I 修复版 W3 全量 → ablation_mllmq_fixed_full/fixed_w3_full_s1
#
#   目的：
#     1) k12 的 +4.30pp 是否可复现（当前只有单次）
#     2) k10 全量是否优于 k12（子集 k10 最好，但从未跑全量）
#     3) MLLMQ vs MBQ-scalar 的 W3 差异是否超过同配置重跑噪声（≈1.6pp）
#
#   全部为 2212 图（553 prompt × 4），每 config 2 分片并行；评分复用
#   evaluation/evaluate_images.py。已存在的 results.jsonl 自动跳过。
#   ⚠️ 只写下列新目录，不动主机正在使用的任何输出。
#
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_remote_confirm_queue.sh \
#     > ../output/logs/confirm_remote_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
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
MLLMQ_W3=$CACHE/mllmq/janus_mae_w3a16.pt          # canonical n=128
SCALAR_W3=$CACHE/mbq/janus_mbq_scalar_w3a16_n128.pt
FIXED_W4=$CACHE/mllmq/janus_mae_w4a16_n128_fixed.pt
FIXED_W3=$CACHE/mllmq/janus_mae_w3a16_n128_fixed.pt

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

score_cfg(){ # tag gdir gpu
  local tag=$1 gdir=$2 gpu=$3
  [ -f "$gdir/results.jsonl" ] && { good "SKIP score $tag"; return 0; }
  [ -n "$CLIP_BIN" ] || { fail "$tag 缺 CLIP 权重，跳过评分"; return 1; }
  CUDA_VISIBLE_DEVICES=$gpu python "$EV/evaluate_images.py" "$gdir" \
    --outfile "$gdir/results.jsonl" --clip-model-path "$CLIP_BIN" \
    > "$LOG/confirm_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] && good "$tag :: $(score_of "$gdir/results.jsonl")" \
    || fail "$tag 评分失败"
}

run_cfg(){ # tag qm w sp gpu_a gpu_b
  local tag=$1 qm=$2 w=$3 sp=$4 ga=$5 gb=$6
  local out="$ROOT/output/janus-pro-7b/$tag" gdir="$ROOT/output/janus-pro-7b/$tag/geneval"
  [ -s "$sp" ] || { fail "$tag 缺 scale $sp"; return 1; }
  if [ -f "$gdir/results.jsonl" ]; then good "SKIP gen $tag"; return 0; fi
  mkdir -p "$out"
  info "$tag 生成（2 分片，GPU $ga/$gb）"
  CUDA_VISIBLE_DEVICES=$ga python scripts/generate_geneval_shard.py \
    --model_path "$MODEL" --output_dir "$out" --num_samples 4 \
    --shard_id 0 --num_shards 2 --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
    > "$LOG/confirm_gen_${tag}_s0.log" 2>&1 & local p0=$!
  CUDA_VISIBLE_DEVICES=$gb python scripts/generate_geneval_shard.py \
    --model_path "$MODEL" --output_dir "$out" --num_samples 4 \
    --shard_id 1 --num_shards 2 --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
    > "$LOG/confirm_gen_${tag}_s1.log" 2>&1 & local p1=$!
  local bad=0; wait $p0 || bad=$((bad+1)); wait $p1 || bad=$((bad+1))
  [ "$bad" -gt 0 ] && { fail "$tag 有分片失败"; return 1; }
  good "$tag 生成完成 → 评分"
  score_cfg "$tag" "$gdir" "$ga"
}

log "==================== Round 1：GAMQ k12 seed2 + k10 全量 ===================="
run_cfg "ablation_mixed_full/w34_spread_k12_full_s2" mllmq_mixed 3 "$K12" 0 1 &
P1=$!
run_cfg "ablation_mixed_full/w34_spread_k10_full"    mllmq_mixed 3 "$K10" 2 3 &
P2=$!
wait $P1 $P2
log "Round 1 结束"

log "==================== Round 2：MLLMQ W3 seed2 + MBQ-scalar W3 seed2 ===================="
run_cfg "mllmq_vs_mbqscalar_full_w3/mllmq_mae_w3_n128_full_s2" mllmq_mae 3 "$MLLMQ_W3" 0 1 &
P3=$!
run_cfg "mllmq_vs_mbqscalar_full_w3/mbq_scalar_w3_n128_full_s2" mbq_scalar 3 "$SCALAR_W3" 2 3 &
P4=$!
wait $P3 $P4
log "Round 2 结束"

log "==================== Round 3：T2I 修复版 W4/W3 全量 ===================="
run_cfg "ablation_mllmq_fixed_full/fixed_w4_full_s1" mllmq_mae 4 "$FIXED_W4" 0 1 &
P5=$!
run_cfg "ablation_mllmq_fixed_full/fixed_w3_full_s1" mllmq_mae 3 "$FIXED_W3" 2 3 &
P6=$!
wait $P5 $P6
log "Round 3 结束"

log "════════ 汇总 ════════"
for tag in \
  ablation_mixed_full/w34_spread_k12_full_s2 \
  ablation_mixed_full/w34_spread_k10_full \
  mllmq_vs_mbqscalar_full_w3/mllmq_mae_w3_n128_full_s2 \
  mllmq_vs_mbqscalar_full_w3/mbq_scalar_w3_n128_full_s2 \
  ablation_mllmq_fixed_full/fixed_w4_full_s1 \
  ablation_mllmq_fixed_full/fixed_w3_full_s1; do
  f="$ROOT/output/janus-pro-7b/$tag/geneval/results.jsonl"
  [ -f "$f" ] && good "$tag :: $(score_of "$f")" || fail "$tag ✗"
done
info "ALL DONE"