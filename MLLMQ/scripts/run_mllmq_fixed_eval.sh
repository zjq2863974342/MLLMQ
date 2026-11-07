#!/usr/bin/env bash
# ============================================================================
# 修复版 MLLMQ 重测（T2I 梯度首次真正生效）
#
#   背景：GradCacheHook 曾跳过纯文本生成样本 → 旧 mllmq scale 实际没有 T2I 项。
#         修复后旧 calib 的 T2I 项会让 76/120 层 scale 移动。
#   本脚本用「旧 calib（128 理解 + 128 纯文本生成）+ 修复后的 hook」重新生成
#   W4/W3 n=128 scale，并在 GenEval 25/tag 上与旧 scale 对照。
#
#   对照（同 25/tag 子集）：
#     旧 mllmq W4（无 T2I）= 0.63500；旧 mllmq W3 = 0.56333
#     mbq_scalar W4 = 0.60667；mbq_block W4 = 0.63333
#   产出：scale_cache/mllmq/janus_mae_w{4,3}a16_n128_fixed.pt
#         output/janus-pro-7b/ablation_mllmq_fixed/{w4,w3}/geneval/results.jsonl
#
#   会等本机 run_exp3_dpg_mixed.sh / run_exp3_mixed.sh 全部结束再开始。
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_mllmq_fixed_eval.sh \
#     > ../output/logs/mllmq_fixed_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache/mllmq
OUT=$ROOT/output/janus-pro-7b/ablation_mllmq_fixed
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

log "===== 修复版 MLLMQ 重测 ====="
# 等本机现有 GPU 队列（DPG → k10/k14）结束
sleep 60
while pgrep -f "[r]un_exp3_dpg_mixed.sh" >/dev/null 2>&1 || pgrep -f "[r]un_exp3_mixed.sh" >/dev/null 2>&1; do
  log "等待本机 GPU 队列结束..."
  sleep 120
done
log "本机队列已结束 → 开始修复版 MLLMQ"

gen_scale(){ # w sp
  local w=$1 sp=$2
  [ -s "$sp" ] && { good "SKIP scale w$w（已存在）"; return 0; }
  info "生成修复版 scale w$w n=128"
  CUDA_VISIBLE_DEVICES=0 python main_quant.py --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$CALIB" --image_folder "$IMG" --n_samples 128 \
    --w_bit "$w" --a_bit 16 --w_group 128 --run_process --method mllmq --loss_mode mae \
    --scale_path "$sp" > "$LOG/mllmq_fixed_scale_w${w}.log" 2>&1
  [ -s "$sp" ] && good "scale w$w OK" || { fail "scale w$w 失败"; tail -5 "$LOG/mllmq_fixed_scale_w${w}.log"; return 1; }
}

SP4=$CACHE/janus_mae_w4a16_n128_fixed.pt
SP3=$CACHE/janus_mae_w3a16_n128_fixed.pt
gen_scale 4 "$SP4" || exit 1
gen_scale 3 "$SP3" || exit 1

run_eval(){ # tag w sp gpu
  local tag=$1 w=$2 sp=$3 gpu=$4
  local out="$OUT/$tag" gdir="$OUT/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag"; return 0; }
  mkdir -p "$out"
  info "[GPU$gpu] $tag 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --quant_method mllmq_mae --w_bit "$w" --scale_path "$sp" \
    > "$LOG/mllmq_fixed_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; return 1; }
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/mllmq_fixed_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

run_eval w4 4 "$SP4" 0 & P0=$!
run_eval w3 3 "$SP3" 1 & P1=$!
wait $P0 $P1

log "════ 修复版 MLLMQ 结果（25/tag）════"
for t in w4 w3; do
  f="$OUT/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
log "对照（旧 scale，无 T2I 项）：W4=0.63500 / W3=0.56333；mbq_scalar W4=0.60667；mbq_block W4=0.63333"
info "ALL DONE"