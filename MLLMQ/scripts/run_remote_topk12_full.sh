#!/usr/bin/env bash
# ============================================================================
# 选层消融（副机）：k=12 的 spread vs generation-aware 选层，greedy 全量 553
#
#   目的：在**确定性解码**（temperature=0, seed=42）下，判断“选哪 12 层保 W4”
#         本身有没有可测差异，排除生成采样噪声。
#
#   三个 config（同一套 W4/W3 校准 scale，只改 w4_blocks）：
#     1) spread_k12    等间距选层 [0,2,5,8,10,12,15,18,20,22,25,28]
#     2) topratio_k12  按 t2i/caption 梯度比选 top-12（深度归一化，真正的
#                      generation-aware 候选；预计 [1..10,28,29]）
#     3) topraw_k12    按原始 mean|grad|.t2i 选 top-12（会被深度主导，预计 0..11，
#                      作为“early-layer”对照，不当作 generation-aware）
#
#   口径：GenEval 全量 553 prompt × 1 图/4 卡分片；greedy 下 1 张 == 4 张
#   产出：output/janus-pro-7b/ablation_mixed_greedy/<tag>/geneval/results.jsonl
#         scale_cache/mllmq/janus_mixed_w34_{spread,topratio,topraw}_k12.pt
#
#   会自动等 run_remote_greedy_compensation.sh 结束再开始（避免抢卡）。
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_remote_topk12_full.sh \
#     > ../output/logs/topk12_full_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache/mllmq
OUT=$ROOT/output/janus-pro-7b/ablation_mixed_greedy
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
LOG=$ROOT/output/logs
SEED=42

S_SPREAD=$CACHE/janus_mixed_w34_spread_k12.pt
S_TOPRATIO=$CACHE/janus_mixed_w34_topratio_k12.pt
S_TOPRAW=$CACHE/janus_mixed_w34_topraw_k12.pt
W4=$CACHE/janus_mae_w4a16.pt
W3=$CACHE/janus_mae_w3a16.pt

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
mkdir -p "$OUT"

log "===== 等待 greedy 补偿任务结束，避免抢卡 ====="
sleep 30
while pgrep -f "[r]un_remote_greedy_compensation.sh" >/dev/null 2>&1; do
  log "greedy 补偿仍在跑，等待..."; sleep 120
done
good "GPU 空闲，开始 k12 选层消融"

# ---------------------------------------------------------------- scales
build_scale(){ # tag mode score out
  local tag=$1 mode=$2 score=$3 out=$4
  [ -s "$out" ] && { good "SKIP scale $tag"; return 0; }
  python scripts/build_mixed_scale.py \
    --w4 "$W4" --w3 "$W3" --top_k 12 --mode "$mode" --score "$score" \
    --out "$out" > "$LOG/topk12_build_${tag}.log" 2>&1
  [ -s "$out" ] && good "scale $tag OK：$(grep -o 'selected W4 blocks.*' "$LOG/topk12_build_${tag}.log" | tail -1)" \
    || { fail "scale $tag 失败"; tail -5 "$LOG/topk12_build_${tag}.log"; }
}
build_scale spread   spread t2i   "$S_SPREAD"
build_scale topratio top    ratio "$S_TOPRATIO"
build_scale topraw   top    t2i   "$S_TOPRAW"

# ---------------------------------------------------------------- eval
run_cfg(){ # tag scale
  local tag=$1 sp=$2
  local out="$OUT/$tag" gdir="$OUT/$tag/geneval"
  [ -s "$sp" ] || { fail "$tag 缺 scale $sp"; return 1; }
  if [ -f "$gdir/results.jsonl" ]; then good "SKIP $tag ($(score_of "$gdir/results.jsonl"))"; return 0; fi
  mkdir -p "$out"
  info "$tag：553 prompt × 1（greedy, temp=0, seed=$SEED；4 卡分片）"
  local pids=()
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/generate_geneval_shard.py \
      --model_path "$MODEL" --output_dir "$out" --num_samples 1 \
      --shard_id "$gpu" --num_shards 4 \
      --quant_method mllmq_mixed --w_bit 3 --scale_path "$sp" \
      --temperature 0 --seed "$SEED" \
      > "$LOG/topk12_gen_${tag}_s${gpu}.log" 2>&1 &
    pids+=($!)
  done
  local bad=0
  for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
  [ "$bad" -gt 0 ] && { fail "$tag 有 $bad 个分片失败"; return 1; }
  if [ -n "$CLIP_BIN" ]; then
    CUDA_VISIBLE_DEVICES=0 python "$EV/evaluate_images.py" "$gdir" \
      --outfile "$gdir/results.jsonl" --clip-model-path "$CLIP_BIN" \
      > "$LOG/topk12_score_${tag}.log" 2>&1
    [ -f "$gdir/results.jsonl" ] && good "$tag :: $(score_of "$gdir/results.jsonl")" \
      || fail "$tag 评分失败"
  else
    fail "$tag 缺 CLIP 权重，仅保留图像"
  fi
}

run_cfg topratio_k12 "$S_TOPRATIO"
run_cfg spread_k12   "$S_SPREAD"
run_cfg topraw_k12   "$S_TOPRAW"

log "════════ k12 选层消融汇总（greedy 全量）════════"
for tag in topratio_k12 spread_k12 topraw_k12; do
  f="$OUT/$tag/geneval/results.jsonl"
  [ -f "$f" ] && good "$tag :: $(score_of "$f")" || fail "$tag ✗"
done
info "ALL DONE"