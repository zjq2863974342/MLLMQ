#!/usr/bin/env bash
# ============================================================================
# Exp5：解码时补偿 —— CFG weight 重扫（最便宜的 G1 路线）
#
#   动机：量化会改变 cond/uncond 两支的相对误差，CFG = uncond + w*(cond-uncond)
#         会把误差放大；W3 上重扫 CFG 可能直接找回 1–3pp，无需重训/重校准。
#   口径：GenEval 10 prompt/tag ×6 ×4 = 240 图/配置（~50min），
#         cfg ∈ {4.0, 4.5, 5.0, 5.5}，temperature 固定 1.0
#         方法：fp16 / mllmq_mae W3 / mbq_scalar W3（三者都扫，best-vs-best 才公平）
#   产出：output/exp5_cfg/<method>_cfg<x>/geneval/results.jsonl
#
#   会等 Exp3 全量（run_exp3_full.sh）结束后自动开始。
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_exp5_cfg_sweep.sh \
#     > ../output/logs/exp5_cfg_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/exp5_cfg
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1"}
PER_TAG=${PER_TAG:-10}
CFGS=${CFGS:-"4.0 4.5 5.0 5.5"}
# 可在不同机器分工：METHODS="fp16 mllmq_w3" / METHODS="scalar_w3"
METHODS=${METHODS:-"fp16 mllmq_w3 scalar_w3"}

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

QFILE=/tmp/exp5_q.$$; QLOCK=/tmp/exp5_q.lock
build_queue(){
  : > "$QFILE"
  for c in $CFGS; do
    for m in $METHODS; do
      case "$m" in
        fp16)      echo "fp16|fp16|||$c" >> "$QFILE" ;;
        mllmq_w3)  echo "mllmq_w3|mllmq_mae|3|$CACHE/mllmq/janus_mae_w3a16.pt|$c" >> "$QFILE" ;;
        scalar_w3) echo "scalar_w3|mbq_scalar|3|$CACHE/mbq/janus_mbq_scalar_w3a16_n128.pt|$c" >> "$QFILE" ;;
      esac
    done
  done
}
pop_job(){ local j; exec 9>"$QLOCK"; flock 9; j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp"; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-; echo "$j"; }

run_job(){ # method qm w sp cfg gpu
  local m=$1 qm=$2 w=$3 sp=$4 cfg=$5 gpu=$6
  local tag="${m}_cfg${cfg}" out="$OUT/${m}_cfg${cfg}" gdir="$OUT/${m}_cfg${cfg}/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag"; return 0; }
  mkdir -p "$out"
  local extra=()
  [ -n "$sp" ] && extra=(--quant_method "$qm" --w_bit "$w" --scale_path "$sp")
  info "[GPU$gpu] $tag 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --cfg_weight "$cfg" --temperature 1.0 "${extra[@]}" \
    > "$LOG/exp5_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; tail -3 "$LOG/exp5_gen_${tag}.log"; return 1; }
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/exp5_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

lane(){ local gpu=$1 job m qm w sp cfg
  while :; do
    job=$(pop_job); [ -z "$job" ] && { log "[GPU$gpu] 队列空"; return; }
    IFS='|' read -r m qm w sp cfg <<< "$job"
    run_job "$m" "$qm" "$w" "$sp" "$cfg" "$gpu"
  done
}

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== Exp5：CFG weight 重扫（W3 解码时补偿）====="
if pgrep -f "run_exp3_full.sh" >/dev/null 2>&1; then
  log "等 Exp3 全量结束..."
  while pgrep -f "run_exp3_full.sh" >/dev/null 2>&1; do sleep 120; done
fi
log "开始 CFG 扫描"
build_queue
log "队列 $(wc -l < "$QFILE") 个任务；GPU=$GPUS"
for g in $GPUS; do lane "$g" & done
wait
rm -f "$QFILE"

log "════ Exp5 结果 ════"
for m in $METHODS; do
  for c in $CFGS; do
    f="$OUT/${m}_cfg${c}/geneval/results.jsonl"
    [ -f "$f" ] && good "${m} cfg=$c :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "${m} cfg=$c ✗"
  done
done
info "ALL DONE"