#!/usr/bin/env bash
# ============================================================================
# ★★★ 正确对照（论文核心）★★★
#   MLLMQ(channel, vision+caption+T2I 三模态)  vs  MBQ(scalar, 仅 vision+caption，丢 T2I)
#   评测口径：**生成侧** GenEval **全量 553 prompt × 4 = 2212 张**（非 150 子集，降噪）
#   理由：MLLMQ 相对 MBQ 的唯一增量 = 均衡生成侧输入文本 token(T2I) 模态，
#         其效果必须落在生成侧；且子集噪声带 ~2.3pp 会淹没 2-3pp 的真实差异。
#
# 对照矩阵（W4，n=64/128/256）：
#   mllmq_mae_w4   : janus_mae_w4a16_n64.pt / janus_mae_w4a16.pt / janus_mae_w4a16_n256.pt
#   mbq_scalar_w4  : janus_mbq_scalar_w4a16_n64.pt(需生成) / _n128.pt / janus_mbq_scalar_w4a16.pt
#
# 产出：output/janus-pro-7b/mllmq_vs_mbqscalar_full/<tag>/geneval/results.jsonl
#
# 用法：setout nohup ... &  （会先等当前综合队列结束）
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_mllmq_correct_comparison.sh \
#     > /nas3/kangjia/mllmq/output/logs/mllmq_correct_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUTBASE=$ROOT/output/janus-pro-7b/mllmq_vs_mbqscalar_full
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS="0 1"
PLAN=${PLAN:-0}

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "[$(date '+%F %T')] $1"; }
good() { echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail() { echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
FRCNN="$EV/.torch_home/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"

wait_for_prev() {
  if pgrep -f "run_calib_efficiency_queue.sh" >/dev/null 2>&1; then
    log "等「综合队列」结束再起正确对照..."
    while pgrep -f "run_calib_efficiency_queue.sh" >/dev/null 2>&1; do sleep 60; done
    log "综合队列已结束 → 开始正确对照"
  else
    log "未检测到综合队列，直接开始"
  fi
}

gen_scale() {  # $1=n  $2=scale_path
  local n=$1 sp=$2
  [ -s "$sp" ] && { good "SKIP scale $(basename "$sp")"; return 0; }
  info "RUN scale mbq_scalar w4 n=$n"
  python main_quant.py --model_path "$MODEL" --device cuda:0 --calib_data coco \
    --data_path "$CALIB" --image_folder "$IMG" --n_samples "$n" --w_bit 4 --a_bit 16 --w_group 128 \
    --run_process --method mbq --reweight --reweight_mode scalar --loss_mode mae \
    --scale_path "$sp" > "$LOG/mllmq_correct_scale_n${n}.log" 2>&1
  [ -s "$sp" ] && good "scale OK $(basename "$sp")" || { fail "scale FAIL n=$n"; return 1; }
}

run_full_geneval() {  # $1=tag $2=qmethod $3=wbit $4=scale $5=gpu
  local tag=$1 qm=$2 w=$3 sp=$4 gpu=$5
  local out="$OUTBASE/$tag" gdir="$OUTBASE/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  if [ -f "$gdir/results.jsonl" ]; then good "[GPU$gpu] SKIP $tag（已完成）"; return 0; fi
  if [ ! -s "$sp" ]; then
    log "[GPU$gpu] $tag 等 scale：$(basename "$sp")"
    local t=0; while [ ! -s "$sp" ] && [ $t -lt 2400 ]; do sleep 20; t=$((t+20)); done
  fi
  [ -s "$sp" ] || { fail "[GPU$gpu] $tag 缺 scale"; return 1; }
  mkdir -p "$out"
  info "[GPU$gpu] RUN $tag 全量 GenEval（553 prompt ×4 = 2212 张）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 \
    --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
    > "$LOG/mllmq_correct_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; tail -3 "$LOG/mllmq_correct_gen_${tag}.log"; return 1; }
  info "[GPU$gpu] RUN $tag 全量 GenEval 评分"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/mllmq_correct_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

QFILE=/tmp/mllmq_correct_queue.$$
QLOCK=/tmp/mllmq_correct_queue.lock

build_queue() {
  : > "$QFILE"
  echo "mbq_scalar_w4_n64|mbq_scalar|4|$CACHE/mbq/janus_mbq_scalar_w4a16_n64.pt" >> "$QFILE"
  echo "mbq_scalar_w4_n128|mbq_scalar|4|$CACHE/mbq/janus_mbq_scalar_w4a16_n128.pt" >> "$QFILE"
  echo "mbq_scalar_w4_n256|mbq_scalar|4|$CACHE/mbq/janus_mbq_scalar_w4a16.pt" >> "$QFILE"
  echo "mllmq_mae_w4_n64_full|mllmq_mae|4|$CACHE/mllmq/janus_mae_w4a16_n64.pt" >> "$QFILE"
  echo "mllmq_mae_w4_n128_full|mllmq_mae|4|$CACHE/mllmq/janus_mae_w4a16.pt" >> "$QFILE"
  echo "mllmq_mae_w4_n256_full|mllmq_mae|4|$CACHE/mllmq/janus_mae_w4a16_n256.pt" >> "$QFILE"
}

pop_job() {
  local j; exec 9>"$QLOCK"; flock 9
  j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp" 2>/dev/null; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-; echo "$j"
}

lane() {
  local gpu=$1 job tag qm w sp
  while :; do
    job=$(pop_job); [ -z "$job" ] && { log "[GPU$gpu] 队列已空"; return 0; }
    IFS='|' read -r tag qm w sp <<< "$job"
    run_full_geneval "$tag" "$qm" "$w" "$sp" "$gpu"
  done
}

DRIVER_PID=$$
cleanup() { echo "[$(date '+%F %T')] Interrupted"; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== ★ 正确对照：MLLMQ(channel/三模态) vs MBQ(scalar/丢T2I)，生成侧全量 GenEval ====="
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失"; exit 2; }
[ -f "$FRCNN" ] || { fail "FRCNN 权重缺失"; exit 2; }
mkdir -p "$OUTBASE"
build_queue
log "队列 $(wc -l < "$QFILE") 个全量 eval（每个 2212 张 ≈5h）"
if [ "$PLAN" = "1" ]; then cat "$QFILE"; exit 0; fi

wait_for_prev
# 先生成缺失的 mbq_scalar n64 scale（GPU0 上快速完成）
CUDA_VISIBLE_DEVICES=0 gen_scale 64 "$CACHE/mbq/janus_mbq_scalar_w4a16_n64.pt"

for g in $GPUS; do lane "$g" & done
wait

log "════════ 正确对照结果 ════════"
for tag in mbq_scalar_w4_n64 mbq_scalar_w4_n128 mbq_scalar_w4_n256 mllmq_mae_w4_n64_full mllmq_mae_w4_n128_full mllmq_mae_w4_n256_full; do
  f="$OUTBASE/$tag/geneval/results.jsonl"
  if [ -f "$f" ]; then good "$tag :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')"; else fail "$tag ✗"; fi
done
rm -f "$QFILE"
info "ALL DONE"