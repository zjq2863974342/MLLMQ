#!/usr/bin/env bash
# ============================================================================
# 副机 4-GPU 优先级队列（与 run_remote_priority_queue.sh 等价，但吃满 4 卡）
#
# 为什么要新脚本：原队列只有 3 个任务、最多 3 条 lane；本脚本把最重的
#   A1 全量生成（553 prompt × 4 = 2212 图）按 prompt 索引 idx%N 无损分片，
#   4 张卡各跑一个分片，A1 完成后空出的卡自动接 A13 k=2 / k=3。
#   分片目录名与全量运行完全一致（idx 就是 metadata 行号），skip-existing 保留。
#   不修改共享的 eval_generate_quant.py，只新增 scripts/generate_geneval_shard.py。
#
# 队列（优先级从高到低）：
#   1) A1GEN shard 0..N-1   （mbq_block_w4_n256 全量，续跑）
#   2) A13 k=2（scale→gen→score）
#   3) A13 k=3（scale→gen→score）
#   4) A1SCORE（等所有分片完成标记后评分）
#
# 产出：
#   A1  → output/janus-pro-7b/ablation_a1_full/mbq_block_w4_n256_full/geneval/
#   A13 → output/janus-pro-7b/ablation_a13/t2ix{2,3}/geneval/results.jsonl
#   scale → scale_cache/mllmq/janus_mae_w4a16_n128_t2ix{2,3}.pt
#
# ⚠️ 绝不触碰 output/janus-pro-7b/mllmq_vs_mbqscalar_full/（另一台机在用）
#
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup \
#     bash scripts/run_remote_priority_queue_4gpu.sh \
#     > /nas3/kangjia/mllmq/output/logs/remote_q4_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   只预览：PLAN=1 bash scripts/run_remote_priority_queue_4gpu.sh
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUT_A1=$ROOT/output/janus-pro-7b/ablation_a1_full
OUT_A13=$ROOT/output/janus-pro-7b/ablation_a13
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
META=$ROOT/data/benchmarks_gen/geneval/prompts/evaluation_metadata.jsonl

GPUS=${GPUS:-"0 1 2 3"}
N_SHARDS=${N_SHARDS:-4}
PER_TAG=${PER_TAG:-25}

TAG_A1=mbq_block_w4_n256_full
QM_A1=mbq_block
W=4
SP_A1=$CACHE/mbq/janus_mbq_block_w4a16.pt
EXPECTED=$(( $(wc -l < "$META") * 4 ))
MARKDIR=/tmp/a1_shards.$$

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

QFILE=/tmp/remote_q4.$$; QLOCK=/tmp/remote_q4.$$.lock
build_queue(){
  : > "$QFILE"
  for i in $(seq 0 $((N_SHARDS-1))); do echo "A1GEN|$i" >> "$QFILE"; done
  echo "A13|2" >> "$QFILE"
  echo "A13|3" >> "$QFILE"
  echo "A1SCORE|" >> "$QFILE"
}
pop_job(){ local j; exec 9>"$QLOCK"; flock 9; j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp"; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-; echo "$j"; }

# ---- A1: 一个分片的全量生成（skip-existing，可续跑） ----
run_a1gen(){ # shard gpu
  local shard=$1 gpu=$2
  export CUDA_VISIBLE_DEVICES=$gpu
  mkdir -p "$OUT_A1/$TAG_A1"
  info "[GPU$gpu] A1GEN shard $shard/$N_SHARDS（全量续跑，skip-existing）"
  python scripts/generate_geneval_shard.py --model_path "$MODEL" \
    --output_dir "$OUT_A1/$TAG_A1" --num_samples 4 \
    --shard_id "$shard" --num_shards "$N_SHARDS" \
    --quant_method "$QM_A1" --w_bit "$W" --scale_path "$SP_A1" \
    > "$LOG/a1_shard${shard}_of${N_SHARDS}_gpu${gpu}.log" 2>&1
  local rc=$?
  touch "$MARKDIR/shard_${shard}.done"
  if [ $rc -ne 0 ]; then
    fail "[GPU$gpu] A1GEN shard $shard 失败"; tail -5 "$LOG/a1_shard${shard}_of${N_SHARDS}_gpu${gpu}.log"; return 1
  fi
  good "[GPU$gpu] A1GEN shard $shard 完成"
}

# ---- A1：等所有分片完成后评分 ----
run_a1score(){ # gpu
  local gpu=$1 gdir="$OUT_A1/$TAG_A1/geneval" t=0
  while [ "$(ls "$MARKDIR"/shard_*.done 2>/dev/null | wc -l)" -lt "$N_SHARDS" ] && [ $t -lt 18000 ]; do
    sleep 30; t=$((t+30))
  done
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] A1 已评分，跳过"; return 0; }
  local cnt; cnt=$(find "$gdir" -name '*.png' 2>/dev/null | wc -l)
  info "[GPU$gpu] A1 评分（png=$cnt / 期望 $EXPECTED）"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/a1_score_r.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] A1 评分失败"; return 1; }
  good "[GPU$gpu] A1 DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

# ---- A13: 放大 T2I 重加权（scale → 生成 → 评分） ----
run_a13(){ # k gpu
  local k=$1 gpu=$2
  local tag=t2ix${k}
  local out="$OUT_A13/$tag" gdir="$OUT_A13/$tag/geneval"
  local sp="$CACHE/mllmq/janus_mae_w4a16_n128_t2ix${k}.pt"
  export CUDA_VISIBLE_DEVICES=$gpu
  if [ ! -s "$sp" ]; then
    info "[GPU$gpu] A13 k=$k 生成 scale（MLLMQ_T2I_RW_SCALE=$k）"
    MLLMQ_T2I_RW_SCALE=$k python main_quant.py --model_path "$MODEL" --device cuda:0 \
      --calib_data coco --data_path "$CALIB" --image_folder "$IMG" \
      --n_samples 128 --w_bit 4 --a_bit 16 --w_group 128 --run_process \
      --method mllmq --loss_mode mae --scale_path "$sp" \
      > "$LOG/a13_scale_k${k}_r.log" 2>&1
    [ -s "$sp" ] || { fail "[GPU$gpu] A13 k=$k scale 失败"; tail -5 "$LOG/a13_scale_k${k}_r.log"; return 1; }
  else good "[GPU$gpu] A13 k=$k scale 已存在"; fi
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] A13 k=$k 已评测，跳过"; return 0; }
  mkdir -p "$out"
  info "[GPU$gpu] A13 k=$k GenEval 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --quant_method mllmq_mae --w_bit 4 --scale_path "$sp" \
    > "$LOG/a13_gen_k${k}_r.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] A13 k=$k 生成失败"; tail -5 "$LOG/a13_gen_k${k}_r.log"; return 1; }
  info "[GPU$gpu] A13 k=$k 评分"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/a13_score_k${k}_r.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] A13 k=$k 评分失败"; return 1; }
  good "[GPU$gpu] A13 k=$k DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

lane(){ local gpu=$1 job kind a
  while :; do
    job=$(pop_job); [ -z "$job" ] && { log "[GPU$gpu] 队列空"; return 0; }
    IFS='|' read -r kind a <<< "$job"
    case "$kind" in
      A1GEN)   run_a1gen   "$a" "$gpu" ;;
      A13)     run_a13     "$a" "$gpu" ;;
      A1SCORE) run_a1score "$gpu" ;;
      *)       fail "[GPU$gpu] 未知任务 $job" ;;
    esac
  done
}

DRIVER_PID=$$
cleanup(){ echo "[$(date '+%F %T')] Interrupted"; pkill -P $DRIVER_PID 2>/dev/null; rm -rf "$MARKDIR" "$QFILE"; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== 远端 4-GPU 队列：A1 分片(${N_SHARDS}) → A13 k=2/k=3 ====="
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失（$EV/.ms_cache）"; exit 2; }
[ -f "$FRCNN" ] || { fail "FRCNN 权重缺失（$FRCNN）"; exit 2; }
[ -s "$SP_A1" ] || { fail "缺 A1 scale $SP_A1"; exit 2; }
mkdir -p "$OUT_A1" "$OUT_A13" "$MARKDIR"
build_queue
log "队列 $(wc -l < "$QFILE") 个任务；GPU = $GPUS；A1 分片 = $N_SHARDS；期望 ${EXPECTED} 图"
if [ "$PLAN" = "1" ]; then cat "$QFILE"; rm -f "$QFILE"; rm -rf "$MARKDIR"; exit 0; fi

for g in $GPUS; do lane "$g" & done
wait
rm -f "$QFILE"; rm -rf "$MARKDIR"

log "════════ 远端 4-GPU 队列结果 ════════"
f="$OUT_A1/$TAG_A1/geneval/results.jsonl"
[ -f "$f" ] && good "mbq_block_w4_n256 :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "mbq_block_w4_n256 ✗"
for k in 2 3; do
  f="$OUT_A13/t2ix${k}/geneval/results.jsonl"
  [ -f "$f" ] && good "t2ix${k} :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "t2ix${k} ✗"
done
log "对照：n64/block=0.62164、n128/block=0.64142（全量）；k=1=0.63500、mbq_scalar=0.60667、mbq_block=0.63333（150 子集）"
info "ALL DONE"