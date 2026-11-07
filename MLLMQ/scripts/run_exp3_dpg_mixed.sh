#!/usr/bin/env bash
# ============================================================================
# Exp3 跨 benchmark 验证 · DPG-Bench（single 口径，1065 prompts ×1 图）
#   配置：mllmq_mixed spread k12（平均 3.40 bit）
#   对照（已有）：fp16=83.412 / W3(mllmq)=82.348 / W4(mllmq)=82.648 / block_w3=82.738
#   2 卡分片并行（--shard_id/--num_shards），skip-existing，~2h + 评分 ~20min
# 产出：output/janus-pro-7b/dpg_bench/mllmq_mixed_w34_k12/dpg_bench/*.png
#       output/dpg_scores/mllmq_mixed_w34_k12_results.txt
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_exp3_dpg_mixed.sh \
#     > ../output/logs/exp3_dpg_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
SCALE=$ROOT/MLLMQ/scale_cache/mllmq/janus_mixed_w34_spread_k12.pt
TAG=mllmq_mixed_w34_k12
OUTD=$ROOT/output/janus-pro-7b/dpg_bench/$TAG
LOG=$ROOT/output/logs
TARGET=1065
GPUS=${GPUS:-"0 1"}

export HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=$ROOT/checkpoints/torch_home

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }
count(){ find "$OUTD/dpg_bench" -name '*.png' 2>/dev/null | wc -l; }

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== Exp3 DPG：mixed k12（$(echo $GPUS | wc -w) 分片） ====="
[ -s "$SCALE" ] || { fail "缺 scale $SCALE"; exit 1; }
mkdir -p "$OUTD/dpg_bench"

if [ "$(count)" -lt "$TARGET" ]; then
  n=$(echo $GPUS | wc -w); i=0; pids=()
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g python eval_generate_quant.py --model_path "$MODEL" \
      --benchmark dpg_bench --output_dir "$OUTD" --num_samples 1 \
      --quant_method mllmq_mixed --w_bit 3 --scale_path "$SCALE" \
      --shard_id "$i" --num_shards "$n" \
      > "$LOG/exp3_dpg_gen_s${i}.log" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
  [ "$bad" -gt 0 ] && { fail "$bad 个分片失败"; exit 1; }
  log "生成完成：$(count)/$TARGET"
fi

if [ ! -f "$ROOT/output/dpg_scores/${TAG}_results.txt" ]; then
  info "评分（mplug VQA，需 pyshim）"
  export PYTHONPATH="$ROOT/MLLMQ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}"
  CUDA_VISIBLE_DEVICES=${SCORE_GPU:-0} python scripts/score_dpg_coco.py \
    --image-root-path "$OUTD/dpg_bench" --resolution 384 --pic-num 1 \
    --csv "$ROOT/data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv" \
    --tag "$TAG" --out "$ROOT/output/dpg_scores/${TAG}_results.txt" \
    > "$LOG/exp3_dpg_score.log" 2>&1
fi
[ -s "$ROOT/output/dpg_scores/${TAG}_results.txt" ] && \
  good "$(grep -i 'DPG-Bench score' "$ROOT/output/dpg_scores/${TAG}_results.txt" | tail -1)" || \
  fail "评分未完成（见 $LOG/exp3_dpg_score.log）"
log "对照：fp16=83.412 / mllmq_w3=82.348 / mllmq_w4=82.648 / mbq_block_w3=82.738"
info "ALL DONE"