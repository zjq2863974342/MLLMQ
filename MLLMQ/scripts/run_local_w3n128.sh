#!/usr/bin/env bash
# ============================================================================
# 本机第二阶段：等 ★W4 正确对照结束后，接跑 W3 全量正确对照 n=128
#   GPU1：mllmq_mae_w3_n128（scale janus_mae_w3a16.pt 已存在，立即开跑）
#   GPU0：先生成 mbq_scalar_w3_n128 scale（~30min），再跑 mbq_scalar_w3_n128 全量
#
# 产出：output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3/{mllmq_mae_w3_n128_full,mbq_scalar_w3_n128_full}/geneval/results.jsonl
# 注意：与副机共用 NFS，但副机只写 n256/n64 tag 与 A2 scale，scale 名不重叠。
#
# 用法：
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_local_w3n128.sh \
#     > /nas3/kangjia/mllmq/output/logs/local_w3n128_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation

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

SP_M=$CACHE/mllmq/janus_mae_w3a16.pt
SP_S=$CACHE/mbq/janus_mbq_scalar_w3a16_n128.pt
mkdir -p "$OUT"

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== 本机 Phase 2：W3 全量正确对照 n=128 ====="
if pgrep -f "run_mllmq_correct_comparison.sh" >/dev/null 2>&1; then
  log "等 ★W4 正确对照结束..."
  while pgrep -f "run_mllmq_correct_comparison.sh" >/dev/null 2>&1; do sleep 120; done
fi
log "W4 正确对照已结束 → 启动 W3 n=128"

run_full(){ # tag qm sp gpu
  local tag=$1 qm=$2 sp=$3 gpu=$4 out="$OUT/$1" gdir="$OUT/$1/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag"; return 0; }
  if [ ! -s "$sp" ]; then
    log "[GPU$gpu] $tag 等 scale $(basename "$sp")"
    local t=0; while [ ! -s "$sp" ] && [ $t -lt 3600 ]; do sleep 20; t=$((t+20)); done
  fi
  [ -s "$sp" ] || { fail "[GPU$gpu] $tag 缺 scale"; return 1; }
  mkdir -p "$out"
  info "[GPU$gpu] RUN $tag 全量 GenEval（2212 张）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --quant_method "$qm" --w_bit 3 --scale_path "$sp" \
    > "$LOG/local_w3_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; tail -3 "$LOG/local_w3_gen_${tag}.log"; return 1; }
  info "[GPU$gpu] 评分 $tag"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/local_w3_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

# GPU0：先补 scalar w3 n=128 scale
(
  export CUDA_VISIBLE_DEVICES=0
  if [ ! -s "$SP_S" ]; then
    info "[GPU0] 生成 mbq_scalar w3 n=128 scale"
    python main_quant.py --model_path "$MODEL" --device cuda:0 --calib_data coco \
      --data_path "$CALIB" --image_folder "$IMG" --n_samples 128 \
      --w_bit 3 --a_bit 16 --w_group 128 --run_process \
      --method mbq --reweight --reweight_mode scalar --loss_mode mae \
      --scale_path "$SP_S" > "$LOG/local_w3_scale_scalar_n128.log" 2>&1
    [ -s "$SP_S" ] && good "scalar_w3_n128 scale OK" || { fail "scalar_w3_n128 scale FAIL"; exit 1; }
  else good "scalar_w3_n128 scale 已存在"; fi
  run_full mbq_scalar_w3_n128_full mbq_scalar "$SP_S" 0
) & P0=$!

# GPU1：mllmq 直接开跑
run_full mllmq_mae_w3_n128_full mllmq_mae "$SP_M" 1 & P1=$!

wait $P0 $P1

log "════ 本机 W3 n=128 结果 ════"
for t in mllmq_mae_w3_n128_full mbq_scalar_w3_n128_full; do
  f="$OUT/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
info "ALL DONE"