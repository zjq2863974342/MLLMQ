#!/usr/bin/env bash
# ============================================================================
# 另一台机（4 GPU）优先级队列 —— 与「本机 ★正确对照 / W3-n128」互补，互不冲突
#
#   任务 1：A1 全量最后一格 mbq_block_w4_n256（续跑 696/2212，skip-existing）
#   任务 2：W3 全量正确对照 n=256 —— mllmq_mae_w3_n256（scale 已存在，无需校准）
#   任务 3：W3 全量正确对照 n=256 —— mbq_scalar_w3（scale 已存在）
#   任务 4：A13 k=2（强制放大 T2I 重加权，n=128 W4，GenEval 25/tag 子集）
#   任务 5：A13 k=3
#   任务 6+（可选，WITH_A2=1）：A2 校准配比鲁棒性（自带队列，4 lane）
#
#   4 个 GPU 各一条 lane 从共享队列取任务；短任务先结束就先接下一档。
#   所有任务 skip-existing，可反复重启续跑。
#
# 产出：
#   A1  → output/janus-pro-7b/ablation_a1_full/mbq_block_w4_n256_full/geneval/results.jsonl
#   W3  → output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3/<tag>/geneval/results.jsonl
#   A13 → output/janus-pro-7b/ablation_a13/t2ix{2,3}/geneval/results.jsonl
#   A2  → output/janus-pro-7b/ablation_a2/<tag>/geneval/results.jsonl
#
# Usage（另一台机）：
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_remote_priority_queue.sh \
#     > /nas3/kangjia/mllmq/output/logs/remote_q_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   只预览队列：PLAN=1 bash scripts/run_remote_priority_queue.sh
#   不同卡数：GPUS="0 1 2 3"（默认）
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUT_A1=$ROOT/output/janus-pro-7b/ablation_a1_full
OUT_W3=$ROOT/output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3
OUT_A13=$ROOT/output/janus-pro-7b/ablation_a13
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1 2 3"}
PER_TAG=${PER_TAG:-25}
WITH_A2=${WITH_A2:-0}
export GPUS

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
FRCNN="$EV/.torch_home/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"

QFILE=/tmp/remote_q.$$; QLOCK=/tmp/remote_q.lock
build_queue(){
  : > "$QFILE"
  # 1) A1 全量最后一格
  echo "A1|mbq_block_w4_n256_full|mbq_block|4|$CACHE/mbq/janus_mbq_block_w4a16.pt" >> "$QFILE"
  # 2/3) W3 全量正确对照（两份 scale 均已存在，零校准成本）
  echo "W3|mllmq_mae_w3_n256_full|mllmq_mae|3|$CACHE/mllmq/janus_mae_w3a16_n256.pt" >> "$QFILE"
  echo "W3|mbq_scalar_w3_n256_full|mbq_scalar|3|$CACHE/mbq/janus_mbq_scalar_w3a16.pt" >> "$QFILE"
  # 4/5) A13 强制放大 T2I 重加权
  echo "A13|t2ix2|mllmq_mae|4|$CACHE/mllmq/janus_mae_w4a16_n128_t2ix2.pt" >> "$QFILE"
  echo "A13|t2ix3|mllmq_mae|4|$CACHE/mllmq/janus_mae_w4a16_n128_t2ix3.pt" >> "$QFILE"
}
pop_job(){ local j; exec 9>"$QLOCK"; flock 9; j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp"; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-; echo "$j"; }

# ---- 全量 GenEval 配置（生成 → 评分） ----
run_full(){ # outbase tag qm w sp gpu
  local outbase=$1 tag=$2 qm=$3 w=$4 sp=$5 gpu=$6
  local out="$outbase/$tag" gdir="$outbase/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag（已完成）"; return 0; }
  [ -s "$sp" ] || { fail "[GPU$gpu] 缺 scale $sp"; return 1; }
  mkdir -p "$out"
  info "[GPU$gpu] RUN $tag 全量 GenEval（2212 张，skip-existing 续跑）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
    > "$LOG/remote_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; tail -3 "$LOG/remote_gen_${tag}.log"; return 1; }
  info "[GPU$gpu] 评分 $tag"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/remote_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

# ---- A13: 放大 T2I 重加权 ----
run_a13(){ # tag qm w sp gpu
  local tag=$1 qm=$2 w=$3 sp=$4 gpu=$5
  local k=${tag#t2ix}
  local out="$OUT_A13/$tag" gdir="$OUT_A13/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  if [ ! -s "$sp" ]; then
    info "[GPU$gpu] A13 k=$k 生成 scale（MLLMQ_T2I_RW_SCALE=$k）"
    MLLMQ_T2I_RW_SCALE=$k python main_quant.py --model_path "$MODEL" --device cuda:0 \
      --calib_data coco --data_path "$CALIB" --image_folder "$IMG" \
      --n_samples 128 --w_bit 4 --a_bit 16 --w_group 128 --run_process \
      --method mllmq --loss_mode mae --scale_path "$sp" \
      > "$LOG/a13_scale_k${k}.log" 2>&1
    [ -s "$sp" ] || { fail "[GPU$gpu] A13 k=$k scale 失败"; return 1; }
  else good "[GPU$gpu] A13 k=$k scale 已存在"; fi
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] A13 k=$k 已评测"; return 0; }
  mkdir -p "$out"
  info "[GPU$gpu] A13 k=$k GenEval 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
    > "$LOG/a13_gen_k${k}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] A13 k=$k 生成失败"; return 1; }
  info "[GPU$gpu] A13 k=$k 评分"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/a13_score_k${k}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] A13 k=$k 评分失败"; return 1; }
  good "[GPU$gpu] A13 k=$k DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

lane(){ local gpu=$1 job kind tag qm w sp
  while :; do
    job=$(pop_job); [ -z "$job" ] && { log "[GPU$gpu] 队列空"; return 0; }
    IFS='|' read -r kind tag qm w sp <<< "$job"
    case "$kind" in
      A1)  run_full "$OUT_A1"  "$tag" "$qm" "$w" "$sp" "$gpu" ;;
      W3)  run_full "$OUT_W3"  "$tag" "$qm" "$w" "$sp" "$gpu" ;;
      A13) run_a13  "$tag" "$qm" "$w" "$sp" "$gpu" ;;
      *)   fail "[GPU$gpu] 未知任务 $job" ;;
    esac
  done
}

DRIVER_PID=$$
cleanup(){ echo "[$(date '+%F %T')] Interrupted"; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== 远端优先级队列：A1 n256 → W3 n256 ×2 → A13 k2/k3 ====="
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失（$EV/.ms_cache）"; exit 2; }
[ -f "$FRCNN" ] || { fail "FRCNN 权重缺失（$FRCNN）"; exit 2; }
mkdir -p "$OUT_A1" "$OUT_W3" "$OUT_A13"
build_queue
log "队列 $(wc -l < "$QFILE") 个任务；GPU = $GPUS"
if [ "$PLAN" = "1" ]; then cat "$QFILE"; rm -f "$QFILE"; exit 0; fi

for g in $GPUS; do lane "$g" & done
wait
rm -f "$QFILE"

log "════════ 远端队列结果 ════════"
for pair in \
  "$OUT_A1|mbq_block_w4_n256_full" \
  "$OUT_W3|mllmq_mae_w3_n256_full" \
  "$OUT_W3|mbq_scalar_w3_n256_full" \
  "$OUT_A13|t2ix2" \
  "$OUT_A13|t2ix3"; do
  base=${pair%%|*}; t=${pair##*|}; f="$base/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
log "对照（W4 全量）：mbq_scalar n64=0.63302 n128=0.63565 n256=0.63776；mbq_block n64=0.62164 n128=0.64142"
log "对照（A13, 150 子集）：k=1 mllmq_mae=0.63500；mbq_scalar=0.60667；mbq_block=0.63333"

if [ "$WITH_A2" = "1" ]; then
  log "===== 接跑 A2 校准配比鲁棒性（4 lane）====="
  bash scripts/run_ablation_a2_mix.sh
fi

info "ALL DONE"