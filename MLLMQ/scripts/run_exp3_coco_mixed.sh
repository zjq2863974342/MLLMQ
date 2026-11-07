#!/usr/bin/env bash
# ============================================================================
# Exp3 生成侧全 benchmark：COCO-val 5k（GAMQ mixed k12）
#   生成：4 分片 × 1250 图 ≈ 4.2h；评分：FID / CLIP / HPSv2+PickScore / ImageReward
#   对照：fp16 与 mllmq_mae_w4 的已有 COCO 评分（注意 W3 图来自对端 A100，跨档仅看趋势）
# 产出：output/janus-pro-7b/coco_val/gamt_k12/coco_val/*.png
#       output/janus-pro-7b/{fid_results,clip_scores,pref_scores}/*gamt_k12*
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && GPUS="0 1 2 3" setsid nohup bash scripts/run_exp3_coco_mixed.sh \
#     > ../output/logs/exp3_coco_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
SCALE=$ROOT/MLLMQ/scale_cache/mllmq/janus_mixed_w34_spread_k12.pt
TAG=gamt_k12
OUTD=$ROOT/output/janus-pro-7b/coco_val/$TAG
LOG=$ROOT/output/logs
GPUS=${GPUS:-"0 1 2 3"}
TARGET=5000

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }
count(){ find "$OUTD/coco_val" -name '*.png' 2>/dev/null | wc -l; }

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== GAMQ COCO-val 5k ====="
[ -s "$SCALE" ] || { fail "缺 scale $SCALE"; exit 1; }
mkdir -p "$OUTD/coco_val"

if [ "$(count)" -lt "$TARGET" ]; then
  gpus=($GPUS); n=${#gpus[@]}; i=0; pids=()
  for g in "${gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=$g python eval_generate_quant.py --model_path "$MODEL" \
      --benchmark coco_val --output_dir "$OUTD" --num_samples 1 \
      --quant_method mllmq_mixed --w_bit 3 --scale_path "$SCALE" \
      --shard_id "$i" --num_shards "$n" \
      > "$LOG/exp3_coco_gen_s${i}.log" 2>&1 &
    pids+=($!); i=$((i+1))
  done
  bad=0; for p in "${pids[@]}"; do wait "$p" || bad=$((bad+1)); done
  [ "$bad" -gt 0 ] && { fail "$bad 个分片失败"; exit 1; }
  log "生成完成：$(count)/$TARGET"
fi

# ---- 评分（4 个指标并行；各自默认输出目录与旧表一致）----
run_score(){ # name gpu cmd...
  local name=$1 gpu=$2; shift 2
  info "[GPU$gpu] score $name"
  CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/exp3_coco_score_${name}.log" 2>&1
  [ $? -eq 0 ] && good "score $name OK" || fail "score $name FAIL"
}

run_score fid 0 python "$ROOT/scripts/score_fid_coco.py" --config "$TAG" \
  --fake_dir "$OUTD/coco_val" &
run_score clip 1 python "$ROOT/scripts/score_clip_coco.py" \
  --images_dir "$OUTD/coco_val" --config "$TAG" &
run_score pref 2 python "$ROOT/MLLMQ/score_preferences.py" \
  --images_dir "$OUTD/coco_val" --config "$TAG" --models hpsv2,pickscore &
run_score ir 3 python "$ROOT/scripts/score_imagereward_coco.py" \
  --config "$TAG" --images_dir "$OUTD/coco_val" &
wait

log "════ GAMQ COCO 评分结果 ════"
for f in "$ROOT/output/janus-pro-7b/fid_results/${TAG}_fid.json" \
         "$ROOT/output/janus-pro-7b/clip_scores/${TAG}_clip.json" \
         "$ROOT/output/janus-pro-7b/pref_scores/${TAG}_pref.json" \
         "$ROOT/output/janus-pro-7b/pref_scores/${TAG}_imagereward.json"; do
  [ -f "$f" ] && good "found $f" || log "missing $f（检查对应日志）"
done
info "ALL DONE"