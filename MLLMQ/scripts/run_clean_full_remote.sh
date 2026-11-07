#!/usr/bin/env bash
# ============================================================================
# 副机夜间补充：干净口径 n=128 的【全量 GenEval】对照（W3 重点）
#   block_w3_full / awq_w3_full：各 2212 图，4 分片并行 ≈ 2h/配置
#   （awq_w3 的 n128 scale 由 run_clean_comparison.sh 生成）
# 产出：output/janus-pro-7b/ablation_clean25_full/<tag>/geneval/results.jsonl
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && GPUS="0 1 2 3" setsid nohup bash scripts/run_clean_full_remote.sh \
#     > ../output/logs/cleanfull_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/janus-pro-7b/ablation_clean25_full
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1 2 3"}
TARGET=2212

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

run_full(){ # tag qm w sp
  local tag=$1 qm=$2 w=$3 sp=$4
  local out="$OUT/$tag" gdir="$OUT/$tag/geneval"
  [ -f "$gdir/results.jsonl" ] && { good "SKIP $tag"; return 0; }
  [ -s "$sp" ] || { fail "缺 scale $sp"; return 1; }
  mkdir -p "$out"
  local gpus=($GPUS); n=${#gpus[@]}; i=0; pids=()
  for g in "${gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=$g python scripts/generate_geneval_shard.py \
      --model_path "$MODEL" --output_dir "$out" --num_samples 4 \
      --shard_id "$i" --num_shards "$n" \
      --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
      > "$LOG/cleanfull_gen_${tag}_s${i}.log" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  local bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
  [ "$bad" -gt 0 ] && { fail "$tag 有 $bad 个分片失败"; return 1; }
  log "$tag 生成完成：$(find "$gdir" -name '*.png' | wc -l)/$TARGET"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/cleanfull_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] && good "$tag :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')" || fail "$tag 评分失败"
}

log "===== 副机夜间：干净口径全量 GenEval（W3）====="
run_full block_w3_full mbq_block 3 "$CACHE/mbq/janus_mbq_block_w3a16_n128.pt"
run_full awq_w3_full awq 3 "$CACHE/awq/janus_awq_w3a16_n128.pt"
info "ALL DONE"