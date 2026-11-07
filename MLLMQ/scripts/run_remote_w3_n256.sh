#!/usr/bin/env bash
# ============================================================================
# 副机（4 GPU）第二阶段·优先级最高：W3 全量正确对照 n=256
#   mllmq_mae_w3_n256（janus_mae_w3a16_n256.pt） vs mbq_scalar_w3（janus_mbq_scalar_w3a16.pt）
#   两份 scale 都已存在，零校准成本；用 4 分片并行（复用 scripts/generate_geneval_shard.py）
#
#   会等第一阶段的 4gpu 驱动（run_remote_priority_queue_4gpu.sh 或 run_remote_priority_queue.sh）结束。
#
# 产出：output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3/{mllmq_mae_w3_n256_full,mbq_scalar_w3_n256_full}/geneval/results.jsonl
#
# 用法（副机）：
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_remote_w3_n256.sh \
#     > /nas3/kangjia/mllmq/output/logs/remote_w3n256_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   PLAN=1 预览；WITH_A2=1 之后接 A2；WITH_W3N64=1 之后接 W3 n=64
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1 2 3"}
N_SHARDS=${N_SHARDS:-4}
WITH_A2=${WITH_A2:-0}
WITH_W3N64=${WITH_W3N64:-0}
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
  log "W3 n=256 分片：mllmq_mae_w3_n256 → mbq_scalar_w3_n256（每个 ${N_SHARDS} shard）"
  [ "$WITH_A2" = "1" ] && log "之后接 A2（4 lane）"
  [ "$WITH_W3N64" = "1" ] && log "之后接 W3 n=64"
  exit 0
fi
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失"; exit 2; }
mkdir -p "$OUT"

DRIVER_PID=$$; cleanup(){ echo "[$(date '+%F %T')] Interrupted"; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

# ---------- 等第一阶段 ----------
for pat in run_remote_priority_queue_4gpu.sh run_remote_priority_queue.sh; do
  if pgrep -f "$pat" >/dev/null 2>&1; then
    log "第一阶段（$pat）仍在跑，等它结束..."
    while pgrep -f "$pat" >/dev/null 2>&1; do sleep 120; done
    log "第一阶段结束"
  fi
done

# ---------- 一个配置：N_SHARDS 个分片并行 → 统一评分 ----------
run_config(){ # tag qm sp
  local tag=$1 qm=$2 sp=$3 out="$OUT/$1" gdir="$OUT/$1/geneval"
  [ -f "$gdir/results.jsonl" ] && { good "SKIP $tag（已完成）"; return 0; }
  [ -s "$sp" ] || { fail "$tag 缺 scale $sp"; return 1; }
  mkdir -p "$out"
  info "RUN $tag：${N_SHARDS} 分片全量生成"
  local pids=()
  local i=0
  for g in $GPUS; do
    [ "$i" -ge "$N_SHARDS" ] && break
    CUDA_VISIBLE_DEVICES=$g python scripts/generate_geneval_shard.py \
      --model_path "$MODEL" --output_dir "$out" --num_samples 4 \
      --shard_id "$i" --num_shards "$N_SHARDS" \
      --quant_method "$qm" --w_bit 3 --scale_path "$sp" \
      > "$LOG/w3n256_gen_${tag}_s${i}.log" 2>&1 &
    pids+=($!)
    i=$((i+1))
  done
  local failn=0
  for p in "${pids[@]}"; do wait "$p" || failn=$((failn+1)); done
  [ "$failn" -gt 0 ] && { fail "$tag 有 $failn 个分片异常（日志 w3n256_gen_${tag}_s*.log）"; return 1; }
  local n=$(find "$gdir" -name '*.png' | wc -l)
  log "$tag 生成完成：$n 张"
  [ "$n" -lt 2212 ] && info "⚠️ $tag 仅 $n/2212 张（若为 OOM 跳过，可重跑本脚本补齐）"
  info "评分 $tag"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/w3n256_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "$tag 评分失败"; return 1; }
  good "$tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

log "===== 副机 Phase 2：W3 全量 n=256（4 分片）====="
run_config mllmq_mae_w3_n256_full  mllmq_mae  "$CACHE/mllmq/janus_mae_w3a16_n256.pt"
run_config mbq_scalar_w3_n256_full mbq_scalar "$CACHE/mbq/janus_mbq_scalar_w3a16.pt"

log "════ W3 n=256 结果 ════"
for t in mllmq_mae_w3_n256_full mbq_scalar_w3_n256_full; do
  f="$OUT/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done

if [ "$WITH_A2" = "1" ]; then
  log "===== 接跑 A2（4 lane）====="
  bash scripts/run_ablation_a2_mix.sh
fi
if [ "$WITH_W3N64" = "1" ]; then
  log "===== 接跑 W3 n=64（复用 run_remote_next_phase.sh 的 W3 段）====="
  SKIP_A2=1 bash scripts/run_remote_next_phase.sh
fi

info "ALL DONE"