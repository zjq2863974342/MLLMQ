#!/usr/bin/env bash
# ============================================================================
# GAMQ 补充验证：用 k10（3.33 bit，子集最佳）重跑 DPG-Bench
#   动机：k12 DPG=81.99 低于均匀 W3（82.348），怀疑是 bit 分配而非 GAMQ 本身；
#         k10 在 GenEval 25/tag 子集上为 0.62000（优于 k12 0.61000）
#   配置：mllmq_mixed, w_bit=3, janus_mixed_w34_spread_k10.pt
#   DPG 1065 张，2 分片 + mplug 评分（与 k12 完全同口径）
#   产出：output/janus-pro-7b/dpg_bench/mllmq_mixed_w34_k10/
#         output/dpg_scores/gamt_k10_dpg_results.txt
#   会等本机固定队列（fixed_eval / fixed_full / clean_comparison / exp3）结束。
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_gamt_k10_dpg.sh \
#     > ../output/logs/gamt_k10_dpg_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
LOG=$ROOT/output/logs
SCALE=$ROOT/MLLMQ/scale_cache/mllmq/janus_mixed_w34_spread_k10.pt
OUT=$ROOT/output/janus-pro-7b/dpg_bench/mllmq_mixed_w34_k10
RES=$ROOT/output/dpg_scores/gamt_k10_dpg_results.txt
GPUS=${GPUS:-"0 1"}

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

[ -s "$SCALE" ] || { fail "缺 scale: $SCALE"; exit 2; }
mkdir -p "$OUT"

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

sleep 60
while pgrep -f "[r]un_mllmq_fixed_eval.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_mllmq_fixed_full.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_clean_comparison.sh" >/dev/null 2>&1 \
   || pgrep -f "[r]un_exp3_mixed.sh" >/dev/null 2>&1; do
  log "等待本机现有队列结束..."; sleep 120
done

if [ -s "$RES" ]; then good "k10 DPG 已完成，跳过"; exit 0; fi

info "k10 DPG：生成 1065 张（2 分片）"
gpus=($GPUS); n=${#gpus[@]}; i=0; pids=()
for g in "${gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=$g python eval_generate_quant.py --model_path "$MODEL" \
    --benchmark dpg_bench --output_dir "$OUT" --num_samples 1 \
    --quant_method mllmq_mixed --w_bit 3 --scale_path "$SCALE" \
    --shard_id "$i" --num_shards "$n" \
    > "$LOG/gamt_k10_dpg_gen_s${i}.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
[ "$bad" -gt 0 ] && { fail "生成有 $bad 分片失败"; exit 1; }
good "k10 DPG 生成完成"

info "k10 DPG：评分（mplug）"
export PYTHONPATH="$ROOT/MLLMQ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 python scripts/score_dpg_coco.py \
  --image-root-path "$OUT/dpg_bench" --resolution 384 --pic-num 1 \
  --csv "$ROOT/data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv" \
  --tag "gamt_k10_dpg" --out "$RES" > "$LOG/gamt_k10_dpg_score.log" 2>&1
[ -s "$RES" ] && good "k10 DPG DONE :: $(grep -i 'DPG-Bench score' "$RES" | tail -1)" || fail "k10 DPG 评分失败"
info "ALL DONE"