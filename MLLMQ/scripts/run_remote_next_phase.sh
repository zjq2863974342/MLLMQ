#!/usr/bin/env bash
# ============================================================================
# 副机（4 GPU）第二阶段：等第一阶段队列跑完后接跑
#   Phase 2a：A2 校准配比鲁棒性（und/gen/bal × mllmq/mbq_scalar，n=256 W4，子集 25/tag）
#   Phase 2b：W3 全量正确对照 n=64（补 W3 的 n 趋势；mbq_scalar_w3_n64 scale 需新造）
#
# 用法（副机）：
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_remote_next_phase.sh \
#     > /nas3/kangjia/mllmq/output/logs/remote_p2_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   只预览：PLAN=1 bash scripts/run_remote_next_phase.sh
#   跳过 A2：SKIP_A2=1 ...  跳过 W3-n64：SKIP_W3N64=1 ...
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUT_W3=$ROOT/output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1 2 3"}
SKIP_A2=${SKIP_A2:-0}
SKIP_W3N64=${SKIP_W3N64:-0}
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

if [ "$PLAN" = "1" ]; then
  log "Phase 2a: A2（6 scale + 6 eval，4 lane）"
  log "Phase 2b: W3 n=64 全量（新造 scalar_w3_n64 scale + 2 个全量 eval）"
  exit 0
fi
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失"; exit 2; }

DRIVER_PID=$$; cleanup(){ echo "[$(date '+%F %T')] Interrupted"; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

# ---------- 等第一阶段 ----------
if pgrep -f "run_remote_priority_queue.sh" >/dev/null 2>&1; then
  log "第一阶段队列仍在跑，等它结束..."
  while pgrep -f "run_remote_priority_queue.sh" >/dev/null 2>&1; do sleep 120; done
  log "第一阶段结束 → 启动第二阶段"
else
  log "未检测到第一阶段队列，直接启动第二阶段"
fi

# ---------- Phase 2a: A2 ----------
if [ "$SKIP_A2" != "1" ]; then
  log "===== Phase 2a：A2 校准配比鲁棒性 ====="
  bash scripts/run_ablation_a2_mix.sh
fi

# ---------- Phase 2b: W3 n=64 全量 ----------
if [ "$SKIP_W3N64" != "1" ]; then
  log "===== Phase 2b：W3 全量 n=64 ====="
  SP_SCALAR_W3N64=$CACHE/mbq/janus_mbq_scalar_w3a16_n64.pt
  SP_MLLMQ_W3N64=$CACHE/mllmq/janus_mae_w3a16_n64.pt
  mkdir -p "$OUT_W3"

  if [ ! -s "$SP_SCALAR_W3N64" ]; then
    info "[GPU0] 生成 mbq_scalar w3 n=64 scale"
    CUDA_VISIBLE_DEVICES=0 python main_quant.py --model_path "$MODEL" --device cuda:0 \
      --calib_data coco --data_path "$CALIB" --image_folder "$IMG" \
      --n_samples 64 --w_bit 3 --a_bit 16 --w_group 128 --run_process \
      --method mbq --reweight --reweight_mode scalar --loss_mode mae \
      --scale_path "$SP_SCALAR_W3N64" > "$LOG/p2_scale_scalar_w3n64.log" 2>&1
    [ -s "$SP_SCALAR_W3N64" ] && good "scalar_w3_n64 scale OK" || { fail "scalar_w3_n64 scale FAIL"; exit 1; }
  else good "scalar_w3_n64 scale 已存在"; fi

  run_full(){ # tag qm sp gpu
    local tag=$1 qm=$2 sp=$3 gpu=$4 out="$OUT_W3/$1" gdir="$OUT_W3/$1/geneval"
    export CUDA_VISIBLE_DEVICES=$gpu
    [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag"; return 0; }
    [ -s "$sp" ] || { fail "[GPU$gpu] 缺 scale $sp"; return 1; }
    mkdir -p "$out"
    info "[GPU$gpu] RUN $tag 全量 GenEval"
    python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
      --output_dir "$out" --num_samples 4 --quant_method "$qm" --w_bit 3 --scale_path "$sp" \
      > "$LOG/p2_gen_${tag}.log" 2>&1
    [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; return 1; }
    python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
      --clip-model-path "$CLIP_BIN" > "$LOG/p2_score_${tag}.log" 2>&1
    [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
    good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
  }
  run_full mllmq_mae_w3_n64_full mllmq_mae "$SP_MLLMQ_W3N64" 0 &
  run_full mbq_scalar_w3_n64_full mbq_scalar "$SP_SCALAR_W3N64" 1 &
  wait
fi

log "════ 第二阶段结果 ════"
for t in mllmq_mae_w3_n64_full mbq_scalar_w3_n64_full; do
  f="$OUT_W3/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
for t in mllmq_und mllmq_gen mllmq_bal scalar_und scalar_gen scalar_bal; do
  f="$ROOT/output/janus-pro-7b/ablation_a2/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
info "ALL DONE"