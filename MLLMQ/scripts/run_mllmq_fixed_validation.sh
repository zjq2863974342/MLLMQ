#!/usr/bin/env bash
# ============================================================================
# 修复版 MLLMQ 的跨榜验证：DPG-Bench + T2I-CompBench（25/tag 有增益时才跑）
#   args: <w_bit> <scale_path> <tag_prefix>
#   DPG: 1065 张，2 分片 + mplug 评分
#   T2I: 2400 张，按 category 2 分片 + complex 评分
#   串行使用两张卡；预计 DPG ~3.3h + T2I ~4.5h
# 产出：
#   output/janus-pro-7b/dpg_bench/<tag>_dpg/dpg_bench/*.png
#   output/dpg_scores/<tag>_dpg_results.txt
#   output/janus-pro-7b/t2i_compbench/<tag>_t2i/t2i_compbench/<cat>/samples/*.png
#   output/t2i_scores/<tag>_t2i_complex.txt
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1"}

W=${1:?usage: $0 <w_bit> <scale_path> <tag_prefix>}
SP=${2:?}
TAGP=${3:?}

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME=$ROOT/checkpoints/torch_home

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

run_sharded(){ # benchmark outdir tag
  local bench=$1 outd=$2 tag=$3
  local gpus=($GPUS); n=${#gpus[@]}; i=0; pids=()
  for g in "${gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=$g python eval_generate_quant.py --model_path "$MODEL" \
      --benchmark "$bench" --output_dir "$outd" --num_samples 1 \
      --quant_method mllmq_mae --w_bit "$W" --scale_path "$SP" \
      --shard_id "$i" --num_shards "$n" \
      > "$LOG/${tag}_gen_s${i}.log" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  local bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
  [ "$bad" -gt 0 ] && { fail "$tag 有 $bad 分片失败"; return 1; }
  good "$tag 生成完成"
}

# ---------------- DPG ----------------
DPG_OUT=$ROOT/output/janus-pro-7b/dpg_bench/${TAGP}_dpg
DPG_RES=$ROOT/output/dpg_scores/${TAGP}_dpg_results.txt
mkdir -p "$DPG_OUT"
if [ ! -s "$DPG_RES" ]; then
  info "DPG：生成 1065 张"
  run_sharded dpg_bench "$DPG_OUT" "${TAGP}_dpg" || exit 1
  info "DPG：评分（mplug）"
  export PYTHONPATH="$ROOT/MLLMQ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}"
  CUDA_VISIBLE_DEVICES=0 python scripts/score_dpg_coco.py \
    --image-root-path "$DPG_OUT/dpg_bench" --resolution 384 --pic-num 1 \
    --csv "$ROOT/data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv" \
    --tag "${TAGP}_dpg" --out "$DPG_RES" > "$LOG/${TAGP}_dpg_score.log" 2>&1
  [ -s "$DPG_RES" ] && good "DPG DONE :: $(grep -i 'DPG-Bench score' "$DPG_RES" | tail -1)" || fail "DPG 评分失败"
else good "DPG 已完成"; fi

# ---------------- T2I ----------------
T2I_OUT=$ROOT/output/janus-pro-7b/t2i_compbench/${TAGP}_t2i
T2I_RES=$ROOT/output/t2i_scores/${TAGP}_t2i_complex.txt
mkdir -p "$T2I_OUT"
if [ ! -s "$T2I_RES" ]; then
  info "T2I：生成 2400 张（按 category 分片）"
  run_sharded t2i_compbench "$T2I_OUT" "${TAGP}_t2i" || exit 1
  info "T2I：complex 评分"
  python "$ROOT/scripts/score_t2i_compbench_complex.py" \
    --tag-dir "$T2I_OUT" --work "$ROOT/output/t2i_scores/_work_${TAGP}" --num 1 \
    > "$LOG/${TAGP}_t2i_score.log" 2>&1
  tail -50 "$LOG/${TAGP}_t2i_score.log" > "$T2I_RES"
  [ -s "$T2I_RES" ] && good "T2I 评分完成 → $T2I_RES" || fail "T2I 评分失败"
else good "T2I 已完成"; fi
info "VALIDATION DONE ($TAGP)"