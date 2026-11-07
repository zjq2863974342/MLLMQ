#!/usr/bin/env bash
# One DPG shard for a W4A8 remote tag. Does not modify project scripts.
# usage: run_w4a8_dpg_shard.sh TAG SHARD_ID NUM_SHARDS GPU
set -u
ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
OUT=$ROOT/output/janus-pro-7b/w4a8_supplement
LOG=$ROOT/output/logs/w4a8_supplement
SC=$PROJ/scale_cache/w4a8

tag=$1; shard=$2; nshards=$3; gpu=$4
case "$tag" in
  rtn)                qm=rtn;        scale=""; envs="";;
  mbq_scalar)         qm=mbq_scalar; scale="$SC/mix384/mbq_scalar_w4a8.pt"; envs="";;
  mllmq_channel_only) qm=mllmq_mae;  scale="$SC/mix384/mllmq_channel_only_w4a8.pt"; envs="MLLMQ_DISABLE_T2I=1 MLLMQ_DISABLE_GEN_IMG=1";;
  mllmq_t2i_text)     qm=mllmq_mae;  scale="$SC/mix384/mllmq_t2i_text_w4a8.pt"; envs="MLLMQ_DISABLE_GEN_IMG=1";;
  *) echo "unknown tag $tag"; exit 2;;
esac

mkdir -p "$OUT/dpg/$tag" "$LOG"
cd "$PROJ" || exit 2
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
echo "[$(date '+%F %T')] START $tag shard $shard/$nshards GPU$gpu"
CUDA_VISIBLE_DEVICES=$gpu env $envs python "$PROJ/eval_generate_quant.py" \
  --model_path "$MODEL" --benchmark dpg_bench \
  --output_dir "$OUT/dpg/$tag" --num_samples 1 \
  --shard_id "$shard" --num_shards "$nshards" \
  --quant_method "$qm" --w_bit 4 --a_bit 8 ${scale:+--scale_path "$scale"} \
  > "$LOG/dpg_${tag}_shard${shard}_$(date +%Y%m%d_%H%M%S).log" 2>&1
rc=$?
echo "[$(date '+%F %T')] DONE $tag shard $shard rc=$rc png=$(find "$OUT/dpg/$tag/dpg_bench" -name '*.png' 2>/dev/null | wc -l)"
exit $rc