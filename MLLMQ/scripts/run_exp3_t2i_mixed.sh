#!/usr/bin/env bash
# ============================================================================
# Exp3 跨 benchmark 验证 · T2I-CompBench++（8 类 ×300，每 prompt 1 图）
#   配置：mllmq_mixed spread k12（平均 3.40 bit）
#   对照（已有）：fp16 / W3(mllmq) / W4(mllmq) 的 8 类分数（t2i_compbench_scores.md）
#   2 卡按 category 分片（--shard_id/--num_shards），skip-existing，~4h
# 产出：output/janus-pro-7b/t2i_compbench/mllmq_mixed_w34_k12/t2i_compbench/<cat>/samples/*.png
#       评分：先用 3-in-1 complex 聚合分（output/t2i_scores/mllmq_mixed_w34_k12_complex.txt）
#             若方向为正，再补 BLIP/UniDet 分类评分
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_exp3_t2i_mixed.sh \
#     > ../output/logs/exp3_t2i_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
SCALE=$ROOT/MLLMQ/scale_cache/mllmq/janus_mixed_w34_spread_k12.pt
TAG=mllmq_mixed_w34_k12
OUTD=$ROOT/output/janus-pro-7b/t2i_compbench/$TAG
LOG=$ROOT/output/logs
TARGET=2400
GPUS=${GPUS:-"0 1"}

export HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=$ROOT/checkpoints/torch_home

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }
count(){ find "$OUTD/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== Exp3 T2I-CompBench：mixed k12（$(echo $GPUS | wc -w) 分片） ====="
[ -s "$SCALE" ] || { fail "缺 scale $SCALE"; exit 1; }
mkdir -p "$OUTD/t2i_compbench"

if [ "$(count)" -lt "$TARGET" ]; then
  n=$(echo $GPUS | wc -w); i=0; pids=()
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g python eval_generate_quant.py --model_path "$MODEL" \
      --benchmark t2i_compbench --output_dir "$OUTD" --num_samples 1 \
      --quant_method mllmq_mixed --w_bit 3 --scale_path "$SCALE" \
      --shard_id "$i" --num_shards "$n" \
      > "$LOG/exp3_t2i_gen_s${i}.log" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
  [ "$bad" -gt 0 ] && { fail "$bad 个分片失败"; exit 1; }
  log "生成完成：$(count)/$TARGET"
fi

OUTF=$ROOT/output/t2i_scores/${TAG}_complex.txt
if [ ! -s "$OUTF" ]; then
  info "评分：3-in-1 complex 聚合"
  python "$ROOT/scripts/score_t2i_compbench_complex.py" \
    --tag-dir "$OUTD" --work "$ROOT/output/t2i_scores/_work_${TAG}" --num 1 \
    > "$LOG/exp3_t2i_score.log" 2>&1
  # 该脚本会把结果打印到 stdout；统一落盘
  tail -50 "$LOG/exp3_t2i_score.log" > "$OUTF"
fi
[ -s "$OUTF" ] && good "complex 评分完成 → $OUTF" || fail "评分未完成（见 $LOG/exp3_t2i_score.log）"
log "对照（complex，1 图/ prompt 口径）：fp16=0.3583 / mllmq_w3=0.3482 / mllmq_w4=0.3583 / scalar_w4=0.3622"
info "ALL DONE"