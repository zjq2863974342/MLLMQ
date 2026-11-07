#!/usr/bin/env bash
# ============================================================================
# Exp 4（副机，4 GPU）：激活量化 A8 —— W{4,3}A8，mllmq vs mbq_scalar
#
#   动机：目前所有实验都是 W*A16（权重-only），prompt token 激活根本没被量化，
#         “token 级保护”无从谈起。切到 A8 后，prompt 激活才是一个真实变量，
#         三模态/逐通道重加权才有机会体现差异。
#
#   矩阵（n=128 校准，GenEval 25/tag 子集 = 600 图）：
#     W4A8: mllmq_mae / mbq_scalar
#     W3A8: mllmq_mae / mbq_scalar
#     + 补 W3A16 两个对照（同子集，便于直接比）
#   每个配置：main_quant --a_bit 8 生成 scale → eval --a_bit 8 → 评分
#
#   产出：output/janus-pro-7b/ablation_a6_a8/<tag>/geneval/results.jsonl
#         scale: scale_cache/{mllmq,mbq}/*_w{4,3}a8_n128.pt
#
# Usage（副机）：
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_exp4_act_quant.sh \
#     > ../output/logs/exp4_a8_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   预览：PLAN=1 ...；只跑 A8：CONTROLS=0 ...；只跑对照：ONLY_CONTROLS=1 ...
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/janus-pro-7b/ablation_a6_a8
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
GPUS=${GPUS:-"0 1 2 3"}
PER_TAG=${PER_TAG:-25}
CONTROLS=${CONTROLS:-1}
ONLY_CONTROLS=${ONLY_CONTROLS:-0}
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

QFILE=/tmp/exp4_q.$$; QLOCK=/tmp/exp4_q.lock
build_queue(){
  : > "$QFILE"
  if [ "$ONLY_CONTROLS" != "1" ]; then
    echo "mllmq_mae_w4a8|mllmq_mae|4|8|$CACHE/mllmq/janus_mae_w4a8_n128.pt" >> "$QFILE"
    echo "mbq_scalar_w4a8|mbq_scalar|4|8|$CACHE/mbq/janus_mbq_scalar_w4a8_n128.pt" >> "$QFILE"
    echo "mllmq_mae_w3a8|mllmq_mae|3|8|$CACHE/mllmq/janus_mae_w3a8_n128.pt" >> "$QFILE"
    echo "mbq_scalar_w3a8|mbq_scalar|3|8|$CACHE/mbq/janus_mbq_scalar_w3a8_n128.pt" >> "$QFILE"
  fi
  if [ "$CONTROLS" = "1" ]; then
    echo "mllmq_mae_w3a16_ctl|mllmq_mae|3|16|$CACHE/mllmq/janus_mae_w3a16.pt" >> "$QFILE"
    echo "mbq_scalar_w3a16_ctl|mbq_scalar|3|16|$CACHE/mbq/janus_mbq_scalar_w3a16_n128.pt" >> "$QFILE"
  fi
}
pop_job(){ local j; exec 9>"$QLOCK"; flock 9; j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp"; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-; echo "$j"; }

run_job(){ # tag qm w a sp gpu
  local tag=$1 qm=$2 w=$3 a=$4 sp=$5 gpu=$6
  local out="$OUT/$tag" gdir="$OUT/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag"; return 0; }

  if [ ! -s "$sp" ]; then
    info "[GPU$gpu] $tag 生成 scale（w${w}a${a}, n=128）"
    local extra=()
    if [ "$qm" = "mbq_scalar" ]; then
      extra=(--method mbq --reweight --reweight_mode scalar --loss_mode mae)
    else
      extra=(--method mllmq --loss_mode mae)
    fi
    python main_quant.py --model_path "$MODEL" --device cuda:0 --calib_data coco \
      --data_path "$CALIB" --image_folder "$IMG" --n_samples 128 \
      --w_bit "$w" --a_bit "$a" --w_group 128 --run_process "${extra[@]}" \
      --scale_path "$sp" > "$LOG/exp4_scale_${tag}.log" 2>&1
    [ -s "$sp" ] || { fail "[GPU$gpu] $tag scale 失败"; return 1; }
  fi

  mkdir -p "$out"
  info "[GPU$gpu] $tag 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --quant_method "$qm" --w_bit "$w" --a_bit "$a" --scale_path "$sp" \
    > "$LOG/exp4_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; tail -3 "$LOG/exp4_gen_${tag}.log"; return 1; }

  info "[GPU$gpu] $tag 评分"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/exp4_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

lane(){ local gpu=$1 job tag qm w a sp
  while :; do
    job=$(pop_job); [ -z "$job" ] && { log "[GPU$gpu] 队列空"; return; }
    IFS='|' read -r tag qm w a sp <<< "$job"
    run_job "$tag" "$qm" "$w" "$a" "$sp" "$gpu"
  done
}

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== Exp4：激活量化 A8（4 GPU）====="
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失（副机请设置 PYTHONPATH 指向 pydeps/geneval_eval）"; exit 2; }
mkdir -p "$OUT"; build_queue
log "队列 $(wc -l < "$QFILE") 个任务；GPU=$GPUS"
if [ "$PLAN" = "1" ]; then cat "$QFILE"; rm -f "$QFILE"; exit 0; fi

for g in $GPUS; do lane "$g" & done
wait
rm -f "$QFILE"

log "════ Exp4 结果 ════"
for t in mllmq_mae_w4a8 mbq_scalar_w4a8 mllmq_mae_w3a8 mbq_scalar_w3a8 mllmq_mae_w3a16_ctl mbq_scalar_w3a16_ctl; do
  f="$OUT/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
log "对照（同 25/tag 子集，A16）：mllmq_mae W4A16=0.63500 / mbq_scalar W4A16=0.60667"
info "ALL DONE"