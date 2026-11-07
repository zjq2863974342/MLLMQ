#!/usr/bin/env bash
# ============================================================================
# Exp 3：按层混合精度（生成敏感层保 W4，其余 W3）
#
#   选择依据：sensitive_v2 的逐 block T2I 梯度（mean_grad.t2i）排序，
#             top-k 层用 W4，其余 W3 → 平均权重位宽 3.27/3.40/3.53。
#   口径：GenEval 25/tag = 600 图；对照 W4A16=0.635、W3A16=副机同子集
#   产出：output/janus-pro-7b/ablation_mixed/w34_k{8,12,16}/geneval/results.jsonl
#         scale: scale_cache/mllmq/janus_mixed_w34_k{k}.pt
#
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_exp3_mixed.sh \
#     > ../output/logs/exp3_mixed_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/janus-pro-7b/ablation_mixed
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1"}
PER_TAG=${PER_TAG:-25}
KS=${KS:-"8 12 16"}
MODE=${MODE:-spread}

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

log "===== Exp3：按层混合精度（W4 保 top-k T2I 敏感层）====="
for k in $KS; do
  sc="$CACHE/mllmq/janus_mixed_w34_${MODE}_k${k}.pt"
  [ -s "$sc" ] || python scripts/build_mixed_scale.py \
    --w4 "$CACHE/mllmq/janus_mae_w4a16.pt" \
    --w3 "$CACHE/mllmq/janus_mae_w3a16.pt" \
    --top_k "$k" --mode "$MODE" --out "$sc" > "$LOG/exp3_build_${MODE}_k${k}.log" 2>&1
  [ -s "$sc" ] && good "scale $MODE k=$k OK" || { fail "scale k=$k 失败"; tail -3 "$LOG/exp3_build_${MODE}_k${k}.log"; exit 1; }
done

run_k(){ # k gpu
  local k=$1 gpu=$2 tag="w34_${MODE}_k${k}" out="$OUT/w34_${MODE}_k${k}" gdir="$OUT/w34_${MODE}_k${k}/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP k=$k"; return 0; }
  mkdir -p "$out"
  info "[GPU$gpu] $MODE k=$k 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --quant_method mllmq_mixed --w_bit 3 --scale_path "$CACHE/mllmq/janus_mixed_w34_${MODE}_k${k}.pt" \
    > "$LOG/exp3_gen_${MODE}_k${k}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] k=$k 生成失败"; tail -3 "$LOG/exp3_gen_${MODE}_k${k}.log"; return 1; }
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/exp3_score_${MODE}_k${k}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] k=$k 评分失败"; return 1; }
  good "[GPU$gpu] k=$k DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

idx=0
for k in $KS; do
  run_k "$k" $((idx % 2)) &
  idx=$((idx+1))
  if [ $((idx % 2)) -eq 0 ]; then wait; fi
done
wait

log "════ Exp3 结果 ════"
for k in $KS; do
  f="$OUT/w34_${MODE}_k${k}/geneval/results.jsonl"
  [ -f "$f" ] && good "$MODE k=$k :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$MODE k=$k ✗"
done
log "对照（同 25/tag）：W4A16=0.63500（k=30）；W3A16=见副机 mllmq_mae_w3a16_ctl（k=0）"
info "ALL DONE"