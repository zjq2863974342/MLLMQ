#!/usr/bin/env bash
# ============================================================================
# Exp3 全量复测 seed-3：GAMQ k12（spread, 3.40 bit）553 prompt × 4 图
#   与 s1（w34_spread_k12_full, 0.61423）/ s2（_s2）同协议：temp=1.0, no seed
#   2 分片并行，skip-existing，产物 output/.../w34_spread_k12_full_s3/
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_exp3_full_s3.sh \
#     > ../output/logs/exp3_full_s3_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/janus-pro-7b/ablation_mixed_full
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1"}
MODE=${MODE:-spread}
K=${K:-12}

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

SC=$CACHE/mllmq/janus_mixed_w34_${MODE}_k${K}.pt
TAG=w34_${MODE}_k${K}_full_s3
OUTD=$OUT/$TAG; GDIR=$OUT/$TAG/geneval
N_SHARDS=$(echo $GPUS | wc -w)
mkdir -p "$OUTD"

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== Exp3 全量复测 seed-3：$MODE k=$K（${N_SHARDS} 分片）====="
[ -s "$SC" ] || { fail "缺 scale $SC"; exit 1; }
if [ ! -f "$GDIR/results.jsonl" ]; then
  pids=(); i=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g python scripts/generate_geneval_shard.py \
      --model_path "$MODEL" --output_dir "$OUTD" --num_samples 4 \
      --shard_id "$i" --num_shards "$N_SHARDS" \
      --quant_method mllmq_mixed --w_bit 3 --scale_path "$SC" \
      > "$LOG/exp3full_s3_gen_s${i}.log" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
  [ "$bad" -gt 0 ] && { fail "$bad 个分片失败"; exit 1; }
  log "生成完成：$(find "$GDIR" -name '*.png' | wc -l) 张"
  python "$EV/evaluate_images.py" "$GDIR" --outfile "$GDIR/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/exp3full_score_s3.log" 2>&1
fi
[ -f "$GDIR/results.jsonl" ] && good "$TAG :: $(python "$EV/summary_scores.py" "$GDIR/results.jsonl" 2>/dev/null | grep -i '^Overall')" || fail "$TAG 未完成"
log "对照：s1=0.61423；s2 见 w34_spread_k12_full_s2"
info "ALL DONE"