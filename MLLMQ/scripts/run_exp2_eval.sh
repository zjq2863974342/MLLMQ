#!/usr/bin/env bash
# ============================================================================
# Exp2 评测（仅当 run_exp2_diagnostic.sh 显示 scale 有移动时才跑）
#   新校准集 = 128 理解 + 128 教师强制生成（gen_tf）
#   三种重加权：gen_img 权重 ×1 / ×2 / 关闭（MLLMQ_DISABLE_GEN_IMG=1）
#   口径：W4A16，n=128，GenEval 25/tag = 600 图
#   对照：mllmq W4=0.63500 / mbq_scalar W4=0.60667 / mbq_block W4=0.63333
# 产出：output/janus-pro-7b/ablation_exp2/<tag>/geneval/results.jsonl
#       scale: scale_cache/mllmq/janus_gen_tf_w4a16_n128{,_x2,_off}.pt
# Usage：
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_exp2_eval.sh \
#     > ../output/logs/exp2_eval_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_gen_tf_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache/mllmq
OUT=$ROOT/output/janus-pro-7b/ablation_exp2
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1"}
PER_TAG=${PER_TAG:-25}

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

log "===== Exp2：新校准集（gen_tf）评测 ====="
[ -s "$CALIB" ] || { fail "缺 $CALIB（先跑 build_exp2_mixed_calib.py）"; exit 1; }

run_variant(){ # tag weight gpu
  local tag=$1 weight=$2 gpu=$3
  local sp=$CACHE/janus_gen_tf_w4a16_n128_${tag}.pt
  local out="$OUT/${tag}" gdir="$OUT/${tag}/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag"; return 0; }

  if [ ! -s "$sp" ]; then
    info "[GPU$gpu] $tag 生成 scale（gen_img weight=$weight）"
    local extra=()
    [ "$weight" = "off" ] && extra=(MLLMQ_DISABLE_GEN_IMG=1)
    if [ "$weight" = "off" ]; then
      env MLLMQ_DISABLE_GEN_IMG=1 python main_quant.py --model_path "$MODEL" --device cuda:0 \
        --calib_data coco --data_path "$CALIB" --image_folder "$IMG" --n_samples 128 \
        --w_bit 4 --a_bit 16 --w_group 128 --run_process --method mllmq --loss_mode mae \
        --scale_path "$sp" > "$LOG/exp2_scale_${tag}.log" 2>&1
    else
      env MLLMQ_GEN_IMG_RW_SCALE=$weight python main_quant.py --model_path "$MODEL" --device cuda:0 \
        --calib_data coco --data_path "$CALIB" --image_folder "$IMG" --n_samples 128 \
        --w_bit 4 --a_bit 16 --w_group 128 --run_process --method mllmq --loss_mode mae \
        --scale_path "$sp" > "$LOG/exp2_scale_${tag}.log" 2>&1
    fi
    [ -s "$sp" ] || { fail "[GPU$gpu] $tag scale 失败"; return 1; }
  fi

  info "[GPU$gpu] $tag GenEval 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --quant_method mllmq_mae --w_bit 4 --scale_path "$sp" \
    > "$LOG/exp2_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; return 1; }
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/exp2_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

idx=0
for pair in "gen_tf_x1 1" "gen_tf_x2 2" "gen_tf_off off"; do
  set -- $pair
  run_variant "$1" "$2" $((idx % 2)) &
  idx=$((idx+1))
  if [ $((idx % 2)) -eq 0 ]; then wait; fi
done
wait

log "════ Exp2 结果 ════"
for t in gen_tf_x1 gen_tf_x2 gen_tf_off; do
  f="$OUT/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
log "对照（同 25/tag）：mllmq W4=0.63500 / mbq_scalar W4=0.60667 / mbq_block W4=0.63333"
info "ALL DONE"