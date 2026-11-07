#!/usr/bin/env bash
# ============================================================================
# 本机夜间补充：干净口径 n=128 的 W3 全量 GenEval 基线
#   scalar_w3_full（GPU0） + block_w3_full（GPU1）各 2212 图
#   两张卡各跑 1 个整跑（不用分片，避免 CPU 争抢），约 7.5h
#   无论 GAMQ/修复版结论如何，这两格都是论文主表必需的干净基线。
#   会等 run_mllmq_fixed_full.sh / run_clean_comparison.sh 结束再开始。
# 产出：output/janus-pro-7b/ablation_clean25_full/{scalar_w3_full,block_w3_full}/geneval/results.jsonl
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_clean_full_w3_local.sh \
#     > ../output/logs/cleanfull_w3_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/janus-pro-7b/ablation_clean25_full
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
while pgrep -f "[r]un_mllmq_fixed_full.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_mllmq_fixed_eval.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_clean_comparison.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_exp3_mixed.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_gamt_k10_dpg.sh" >/dev/null 2>&1; do
  log "等待本机现有队列结束..."; sleep 120
done

run_full(){ # tag qm w sp gpu
  local tag=$1 qm=$2 w=$3 sp=$4 gpu=$5
  local out="$OUT/$tag" gdir="$OUT/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag"; return 0; }
  [ -s "$sp" ] || { fail "[GPU$gpu] 缺 scale $sp"; return 1; }
  mkdir -p "$out"
  info "[GPU$gpu] $tag 全量 GenEval（2212 张）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
    > "$LOG/cleanfullw3_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; return 1; }
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/cleanfullw3_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] && good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')" || fail "[GPU$gpu] $tag 评分失败"
}

log "===== 干净口径 W3 全量基线 ====="
run_full scalar_w3_full mbq_scalar 3 "$CACHE/mbq/janus_mbq_scalar_w3a16_n128.pt" 0 & P0=$!
run_full block_w3_full  mbq_block  3 "$CACHE/mbq/janus_mbq_block_w3a16_n128.pt"  1 & P1=$!
wait $P0 $P1
info "ALL DONE"