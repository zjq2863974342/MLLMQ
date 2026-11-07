#!/usr/bin/env bash
# 主机 AWQ 全量 GenEval（unified 384 pool，2 分片/2 卡）
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
SC=$ROOT/MLLMQ/scale_cache/unified_384/awq_w3a16.pt
OUT=$ROOT/output/fixed_ablation_20261103_unified/awq
LOG=$ROOT/output/logs/fixed_ablation_20261103_unified
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' | head -1)
[ -s "$SC" ] || { echo "missing $SC"; exit 2; }
[ -n "$CLIP_BIN" ] || { echo "missing CLIP bin"; exit 2; }
mkdir -p "$OUT" "$LOG"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"

if [ -f "$OUT/geneval/results.jsonl" ]; then
  echo "[SKIP] AWQ already scored"; exit 0
fi
echo "[GEN] awq 2 shards $(date '+%F %T')"
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i python scripts/generate_geneval_shard.py \
    --model_path "$MODEL" --output_dir "$OUT" --num_samples 4 \
    --shard_id $i --num_shards 2 --quant_method awq --w_bit 3 \
    --scale_path "$SC" --temperature 1.0 --seed 42 \
    > "$LOG/gen_awq_s${i}.log" 2>&1 &
done
wait
echo "[GEN-DONE] awq png=$(find "$OUT" -name '*.png' | wc -l) $(date '+%F %T')"
echo "[SCORE] awq $(date '+%F %T')"
python "$EV/evaluate_images.py" "$OUT/geneval" --outfile "$OUT/geneval/results.jsonl" \
  --clip-model-path "$CLIP_BIN" > "$LOG/score_awq.log" 2>&1
if [ -f "$OUT/geneval/results.jsonl" ]; then
  echo "[DONE] awq :: $(python "$EV/summary_scores.py" "$OUT/geneval/results.jsonl" 2>/dev/null | grep -i '^Overall')"
else
  echo "[FAIL] awq scoring"
fi