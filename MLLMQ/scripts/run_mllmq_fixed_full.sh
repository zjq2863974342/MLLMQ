#!/usr/bin/env bash
# ============================================================================
# 条件式补充：若修复版 MLLMQ 的 25/tag 结果明显优于旧 scale，则跑对应全量 GenEval
#   旧对照：W4=0.63500 / W3=0.56333
#   判定：提升 >= 1.0pp 才跑全量；W4/W3 只跑提升更大的那个（2 分片，~4h）
#   会等本机 run_mllmq_fixed_eval.sh / run_clean_comparison.sh 结束。
# 产出：output/janus-pro-7b/ablation_mllmq_fixed_full/<tag>/geneval/results.jsonl
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_mllmq_fixed_full.sh \
#     > ../output/logs/mllmq_fixed_full_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
OUT=$ROOT/output/janus-pro-7b/ablation_mllmq_fixed_full
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1"}

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

sleep 60
while pgrep -f "[r]un_exp3_mixed.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_mllmq_fixed_eval.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_clean_comparison.sh" >/dev/null 2>&1; do
  log "等待本机队列结束..."; sleep 120
done
log "本机队列结束，判定是否跑全量"

score_of(){ python "$EV/summary_scores.py" "$1" 2>/dev/null | awk '/Overall/{print $NF}' | tail -1; }
W4F=$ROOT/output/janus-pro-7b/ablation_mllmq_fixed/w4/geneval/results.jsonl
W3F=$ROOT/output/janus-pro-7b/ablation_mllmq_fixed/w3/geneval/results.jsonl
S4=$(score_of "$W4F"); S3=$(score_of "$W3F")
log "修复版 25/tag：W4=${S4:-NA}（旧 0.63500）, W3=${S3:-NA}（旧 0.56333）"

BEST=""; BEST_SCALE=""; BEST_TAG=""
python - "$S4" "$S3" <<'PY' > /tmp/_fixed_pick.txt
import sys
s4,s3=sys.argv[1],sys.argv[2]
try: s4=float(s4); s3=float(s3)
except: print(""); raise SystemExit
d4=s4-0.63500; d3=s3-0.56333
print(("w4" if d4>=d3 else "w3") if max(d4,d3)>=0.01 else "")
PY
BEST=$(cat /tmp/_fixed_pick.txt)
if [ -z "$BEST" ]; then
  log "修复版提升 <1.0pp 或无结果 → 不跑全量"
  exit 0
fi
if [ "$BEST" = "w4" ]; then
  BEST_TAG=mllmq_fixed_w4_full
  BEST_SCALE=$ROOT/MLLMQ/scale_cache/mllmq/janus_mae_w4a16_n128_fixed.pt
  BEST_W=4
else
  BEST_TAG=mllmq_fixed_w3_full
  BEST_SCALE=$ROOT/MLLMQ/scale_cache/mllmq/janus_mae_w3a16_n128_fixed.pt
  BEST_W=3
fi

# ---- 先做跨榜验证：DPG + T2I（用户要求：有增益时插入）----
log "修复版 $BEST 有增益 → 先跑 DPG + T2I 验证"
bash scripts/run_mllmq_fixed_validation.sh "$BEST_W" "$BEST_SCALE" "mllmq_fixed_${BEST}" || log "跨榜验证有失败，继续全量 GenEval"

gdir="$OUT/$BEST_TAG/geneval"; mkdir -p "$OUT/$BEST_TAG"
gpus=($GPUS); n=${#gpus[@]}; i=0; pids=()
for g in "${gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=$g python scripts/generate_geneval_shard.py \
    --model_path "$MODEL" --output_dir "$OUT/$BEST_TAG" --num_samples 4 \
    --shard_id "$i" --num_shards "$n" \
    --quant_method mllmq_mae --w_bit "$BEST_W" --scale_path "$BEST_SCALE" \
    > "$LOG/fixedfull_gen_s${i}.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
[ "$bad" -gt 0 ] && { fail "分片失败 $bad"; exit 1; }
python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
  --clip-model-path "$CLIP_BIN" > "$LOG/fixedfull_score.log" 2>&1
[ -f "$gdir/results.jsonl" ] && good "$BEST_TAG :: $(score_of "$gdir/results.jsonl")" || fail "评分失败"
info "ALL DONE"