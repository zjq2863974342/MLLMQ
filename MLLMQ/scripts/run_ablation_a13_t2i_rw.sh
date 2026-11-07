#!/usr/bin/env bash
# ============================================================================
# A13：强制放大 T2I 重加权系数，检验「T2I 是否被低估」
#
# 背景：实测有效因子 rw_t2i = t2i/cap ≈ 0.97（≈1×），rw_vis ≈ 0.03（视觉被压 30×）。
#   因此 MLLMQ 相对 MBQ(scalar) 的增量 = 多加一个「与 caption 同权」的 T2I 重建项。
#   本实验把 rw_t2i 强制放大 k 倍，看生成侧是否变好。
#
# 设计：mllmq_mae W4, n=128；k ∈ {2, 3}（k=1 即现有 janus_mae_w4a16.pt=0.635）
#   每个 k：生成 scale（MLLMQ_T2I_RW_SCALE=k）→ GenEval 分层子集 eval（25/tag）
#   对照：k=1 (0.635) / mbq_scalar (0.607) / mbq_block (0.633)
# 产出：output/janus-pro-7b/ablation_a13/t2ix{k}/geneval/results.jsonl
#        scale: scale_cache/mllmq/janus_mae_w4a16_n128_t2ix{k}.pt
#
# 会先等 ★正确对照（run_mllmq_correct_comparison.sh）结束。
# Usage: setsid nohup bash scripts/run_ablation_a13_t2i_rw.sh > output/logs/a13_*.log 2>&1 &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache/mllmq
OUTBASE=$ROOT/output/janus-pro-7b/ablation_a13
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
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

run_k() {  # $1=k  $2=gpu
  local k=$1 gpu=$2 sc="$CACHE/janus_mae_w4a16_n128_t2ix${k}.pt"
  local tag="t2ix${k}" out="$OUTBASE/$tag" gdir="$OUTBASE/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  info "[GPU$gpu] A13 k=$k"
  if [ ! -s "$sc" ]; then
    info "[GPU$gpu] k=$k 生成 scale（MLLMQ_T2I_RW_SCALE=$k）"
    MLLMQ_T2I_RW_SCALE=$k python main_quant.py --model_path "$MODEL" --device cuda:0 \
      --calib_data coco --data_path "$CALIB" --image_folder "$IMG" \
      --n_samples 128 --w_bit 4 --a_bit 16 --w_group 128 --run_process \
      --method mllmq --loss_mode mae --scale_path "$sc" \
      > "$LOG/a13_scale_k${k}.log" 2>&1
    [ -s "$sc" ] || { fail "[GPU$gpu] k=$k scale 失败"; return 1; }
  else good "[GPU$gpu] k=$k scale 已存在"; fi

  if [ -f "$gdir/results.jsonl" ]; then good "[GPU$gpu] k=$k 已评测"; return 0; fi
  mkdir -p "$out"
  info "[GPU$gpu] k=$k GenEval 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --quant_method mllmq_mae --w_bit 4 --scale_path "$sc" \
    > "$LOG/a13_gen_k${k}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] k=$k 生成失败"; return 1; }
  info "[GPU$gpu] k=$k GenEval 评分"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/a13_score_k${k}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] k=$k 评分失败"; return 1; }
  good "[GPU$gpu] k=$k DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

DRIVER_PID=$$
cleanup(){ echo "[$(date '+%F %T')] Interrupted"; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== A13：强制放大 T2I 重加权（k=2,3），n=128 W4 ====="
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失"; exit 2; }
mkdir -p "$OUTBASE"
if pgrep -f "run_mllmq_correct_comparison.sh" >/dev/null 2>&1; then
  log "等 ★正确对照 结束..."
  while pgrep -f "run_mllmq_correct_comparison.sh" >/dev/null 2>&1; do sleep 60; done
  log "正确对照已结束 → 开始 A13"
fi

( run_k 2 0 ) & P0=$!
( run_k 3 1 ) & P1=$!
wait $P0 $P1

log "════════ A13 结果 ════════"
for k in 2 3; do
  f="$OUTBASE/t2ix${k}/geneval/results.jsonl"
  if [ -f "$f" ]; then good "t2ix$k :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')"; else fail "t2ix$k ✗"; fi
done
log "对照：k=1 mllmq_mae = 0.63500 ; mbq_scalar = 0.60667 ; mbq_block = 0.63333"
info "ALL DONE"