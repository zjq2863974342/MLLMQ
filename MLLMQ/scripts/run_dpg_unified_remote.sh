#!/usr/bin/env bash
# ============================================================================
# 副机 DPG-Bench 全量（备用驱动，默认不自动启动）
#   8 config × 1065 prompt，2 进程/卡（4 卡 = 8 进程）
#   使用 unified_384 的 6 个 scale + unified_generation_only 的 gen_only scale
#   RTN 无 scale；AWQ 由主机单独跑（不在本脚本）
#
# 产出：output/fixed_ablation_20261103_unified/<tag>/dpg_bench/*.png
# 评分：output/fixed_ablation_20261103_unified/dpg_scores/<tag>.txt
# 日志：output/logs/fixed_ablation_20261103_unified/dpg_*
#
# 用法（等 GenEval 全部评完再跑）：
#   cd /nas3/kangjia/mllmq/MLLMQ
#   WAIT_GENEVAL=1 setsid nohup bash scripts/run_dpg_unified_remote.sh \
#     > ../output/logs/fixed_ablation_20261103_unified/dpg_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#
# 预计：生成 ~5.5–6h；评分串行 ~4h（8×~31min），如需 4 并行可自行改 run_score 的并发（注意 mplug cache 竞态）。
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
SC=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/fixed_ablation_20261103_unified
LOG=$ROOT/output/logs/fixed_ablation_20261103_unified
DPG_CSV=$ROOT/data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv
mkdir -p "$OUT" "$LOG" "$OUT/dpg_scores"

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/data/benchmarks_gen/geneval/evaluation/.ms_cache}"

SCALES="\
$SC/unified_384/mbq_block_masked_w3a16.pt \
$SC/unified_384/mbq_scalar_w3a16.pt \
$SC/unified_384/mllmq_channel_only_w3a16.pt \
$SC/unified_384/mllmq_t2i_text_w3a16.pt \
$SC/unified_384/mllmq_t2i_img_w3a16.pt \
$SC/unified_384/mllmq_both_w3a16.pt \
$SC/unified_generation_only/mllmq_gen_only_w3a16.pt"

for f in $SCALES; do [ -s "$f" ] || { echo "GATE FAIL: missing $f"; exit 2; }; done
echo "[gate] all scales ready"

# 可选：等 GenEval 8 个 results.jsonl 齐了再跑（避免抢卡）
if [ "${WAIT_GENEVAL:-0}" = "1" ]; then
  for i in $(seq 1 720); do
    miss=""
    for t in rtn mbq_block_masked mbq_scalar mllmq_channel_only \
             mllmq_t2i_text mllmq_t2i_img mllmq_both mllmq_gen_only; do
      [ -f "$OUT/$t/geneval/results.jsonl" ] || miss="$miss $t"
    done
    [ -z "$miss" ] && { echo "[gate] GenEval done $(date '+%F %T')"; break; }
    echo "[gate $(date '+%H:%M')] waiting GenEval:$miss"
    sleep 60
  done
fi

run_gen(){ # tag method scale gpu
  local tag=$1 method=$2 scale=$3 gpu=$4
  local d="$OUT/$tag"
  if [ -f "$OUT/dpg_scores/$tag.txt" ]; then echo "[SKIP] $tag"; return 0; fi
  local extra=()
  [ -n "$scale" ] && extra+=(--scale_path "$scale")
  echo "[DPG-GEN] $tag GPU$gpu $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=$gpu python eval_generate_quant.py \
    --model_path "$MODEL" --benchmark dpg_bench \
    --output_dir "$d" --num_samples 1 \
    --shard_id 0 --num_shards 1 \
    --quant_method "$method" --w_bit 3 "${extra[@]}" \
    > "$LOG/dpg_gen_${tag}.log" 2>&1
  echo "[DPG-GEN-DONE] $tag rc=$? png=$(find "$d/dpg_bench" -name '*.png' 2>/dev/null | wc -l) $(date '+%F %T')"
}

run_score(){ # tag
  local tag=$1
  [ -f "$OUT/dpg_scores/$tag.txt" ] && { echo "[SCORE-SKIP] $tag"; return 0; }
  echo "[DPG-SCORE] $tag $(date '+%F %T')"
  PYTHONPATH="$ROOT/MLLMQ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES=0 python scripts/score_dpg_coco.py \
    --image-root-path "$OUT/$tag/dpg_bench" --resolution 384 --pic-num 1 \
    --csv "$DPG_CSV" --tag "unified_${tag}" --out "$OUT/dpg_scores/$tag.txt" \
    > "$LOG/dpg_score_${tag}.log" 2>&1
  if [ -s "$OUT/dpg_scores/$tag.txt" ]; then
    echo "[DPG-SCORE-DONE] $tag :: $(grep -i 'DPG-Bench score' "$OUT/dpg_scores/$tag.txt" | tail -1)"
  else
    echo "[DPG-SCORE-FAIL] $tag"
  fi
}

# 8 个生成进程：每卡 2 个
run_gen rtn                rtn         "" 0 &
run_gen mbq_block_masked   mbq_block   "$SC/unified_384/mbq_block_masked_w3a16.pt" 0 &
run_gen mbq_scalar         mbq_scalar  "$SC/unified_384/mbq_scalar_w3a16.pt" 1 &
run_gen mllmq_channel_only mllmq_mae   "$SC/unified_384/mllmq_channel_only_w3a16.pt" 1 &
run_gen mllmq_t2i_text     mllmq_mae   "$SC/unified_384/mllmq_t2i_text_w3a16.pt" 2 &
run_gen mllmq_t2i_img      mllmq_mae   "$SC/unified_384/mllmq_t2i_img_w3a16.pt" 2 &
run_gen mllmq_both         mllmq_mae   "$SC/unified_384/mllmq_both_w3a16.pt" 3 &
run_gen mllmq_gen_only     mllmq_mae   "$SC/unified_generation_only/mllmq_gen_only_w3a16.pt" 3 &
wait

# 评分串行（避免 mplug cache 并发竞态）
for tag in rtn mbq_block_masked mbq_scalar mllmq_channel_only \
           mllmq_t2i_text mllmq_t2i_img mllmq_both mllmq_gen_only; do
  run_score "$tag"
done
echo "[DPG ALL DONE] $(date '+%F %T')"