#!/usr/bin/env bash
# ============================================================================
# Exp2 补测：真实图像教师强制（gen_tf）在 W3 上的生成侧补偿
#
#   动机：Exp2 原本只测了 W4（headroom 仅 ~1pp，0.610 vs 0.635），
#         而 W3 距 W4 有 ~7pp 空间，才是能看出补偿效果的位置。
#   口径：与 Exp2 W4 完全一致
#     - calib: data/coco/janus_mixed_gen_tf_256.json（128 理解 + 128 真实图 gen_tf）
#     - n=128, W3A16, w_group=128, method=mllmq, loss_mode=mae
#     - 变体：off（MLLMQ_DISABLE_GEN_IMG=1）vs x1（MLLMQ_GEN_IMG_RW_SCALE=1）
#     - GenEval 全量 2212 图（553 prompt × 4），2 分片/config
#   产出：
#     scale_cache/mllmq/janus_gen_tf_w3a16_n128_gen_tf_{off,x1}.pt
#     output/janus-pro-7b/ablation_exp2_w3/{gen_tf_off_full,gen_tf_x1_full}/geneval/results.jsonl
#
#   会等 run_remote_confirm_queue.sh 结束后再开始（避免抢卡）。
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_remote_exp2_w3.sh \
#     > ../output/logs/exp2_w3_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_gen_tf_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache/mllmq
OUT=$ROOT/output/janus-pro-7b/ablation_exp2_w3
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
LOG=$ROOT/output/logs

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"
export PYTHONPATH="$ROOT/pydeps/geneval_eval${PYTHONPATH:+:$PYTHONPATH}"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
score_of(){ python "$EV/summary_scores.py" "$1" 2>/dev/null | awk '/Overall/{print $NF}' | tail -1; }

mkdir -p "$OUT"
[ -s "$CALIB" ] || { fail "缺 $CALIB"; exit 2; }

# 等 3 轮确认队列结束（如果没在跑就直接开始）
sleep 30
while pgrep -f "[r]un_remote_confirm_queue.sh" >/dev/null 2>&1; do
  log "等待 run_remote_confirm_queue.sh 结束..."; sleep 120
done

gen_scale(){ # variant gpu
  local v=$1 gpu=$2
  local sp=$CACHE/janus_gen_tf_w3a16_n128_gen_tf_${v}.pt
  [ -s "$sp" ] && { good "SKIP scale $v"; return 0; }
  info "[GPU$gpu] 生成 W3 scale（gen_tf ${v}）"
  if [ "$v" = "off" ]; then
    CUDA_VISIBLE_DEVICES=$gpu env MLLMQ_DISABLE_GEN_IMG=1 \
      python main_quant.py --model_path "$MODEL" --device cuda:0 \
      --calib_data coco --data_path "$CALIB" --image_folder "$IMG" --n_samples 128 \
      --w_bit 3 --a_bit 16 --w_group 128 --run_process --method mllmq --loss_mode mae \
      --scale_path "$sp" > "$LOG/exp2w3_scale_${v}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu env MLLMQ_GEN_IMG_RW_SCALE=1 \
      python main_quant.py --model_path "$MODEL" --device cuda:0 \
      --calib_data coco --data_path "$CALIB" --image_folder "$IMG" --n_samples 128 \
      --w_bit 3 --a_bit 16 --w_group 128 --run_process --method mllmq --loss_mode mae \
      --scale_path "$sp" > "$LOG/exp2w3_scale_${v}.log" 2>&1
  fi
  [ -s "$sp" ] && good "scale $v OK" || fail "scale $v FAIL"
}

run_full(){ # tag sp gpu_a gpu_b
  local tag=$1 sp=$2 ga=$3 gb=$4
  local out="$OUT/$tag" gdir="$OUT/$tag/geneval"
  [ -s "$sp" ] || { fail "$tag 缺 scale"; return 1; }
  if [ -f "$gdir/results.jsonl" ]; then good "SKIP $tag"; return 0; fi
  mkdir -p "$out"
  info "$tag 全量生成（2 分片）"
  CUDA_VISIBLE_DEVICES=$ga python scripts/generate_geneval_shard.py \
    --model_path "$MODEL" --output_dir "$out" --num_samples 4 \
    --shard_id 0 --num_shards 2 --quant_method mllmq_mae --w_bit 3 --scale_path "$sp" \
    > "$LOG/exp2w3_gen_${tag}_s0.log" 2>&1 & local p0=$!
  CUDA_VISIBLE_DEVICES=$gb python scripts/generate_geneval_shard.py \
    --model_path "$MODEL" --output_dir "$out" --num_samples 4 \
    --shard_id 1 --num_shards 2 --quant_method mllmq_mae --w_bit 3 --scale_path "$sp" \
    > "$LOG/exp2w3_gen_${tag}_s1.log" 2>&1 & local p1=$!
  local bad=0; wait $p0 || bad=$((bad+1)); wait $p1 || bad=$((bad+1))
  [ "$bad" -gt 0 ] && { fail "$tag 分片失败"; return 1; }
  if [ -n "$CLIP_BIN" ]; then
    CUDA_VISIBLE_DEVICES=$ga python "$EV/evaluate_images.py" "$gdir" \
      --outfile "$gdir/results.jsonl" --clip-model-path "$CLIP_BIN" \
      > "$LOG/exp2w3_score_${tag}.log" 2>&1
    [ -f "$gdir/results.jsonl" ] && good "$tag :: $(score_of "$gdir/results.jsonl")" \
      || fail "$tag 评分失败"
  else
    fail "$tag 缺 CLIP 权重，只保留图像"
  fi
}

log "===== Exp2-W3：真实图像补偿（off vs x1）====="
gen_scale off 0 & P0=$!
gen_scale x1  1 & P1=$!
wait $P0 $P1
# 两个 scale 都生成后再并行评测（避免 scale 未就绪）
run_full gen_tf_off_full "$CACHE/janus_gen_tf_w3a16_n128_gen_tf_off.pt" 0 1 & P2=$!
run_full gen_tf_x1_full  "$CACHE/janus_gen_tf_w3a16_n128_gen_tf_x1.pt"  2 3 & P3=$!
wait $P2 $P3

log "════════ Exp2-W3 汇总 ════════"
for tag in gen_tf_off_full gen_tf_x1_full; do
  f="$OUT/$tag/geneval/results.jsonl"
  [ -f "$f" ] && good "$tag :: $(score_of "$f")" || fail "$tag ✗"
done
info "ALL DONE"