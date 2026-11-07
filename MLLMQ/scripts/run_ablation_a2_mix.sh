#!/usr/bin/env bash
# ============================================================================
# A2（另一台机专用）：校准配比鲁棒性
#   问题：三模态归一化是否让 MLLMQ 对校准集配比更不敏感？（MBQ scalar 复制不了）
#   设计：配比 ∈ {undheavy 192/64, genheavy 64/192, balanced 128/128} × 方法 {mllmq, mbq_scalar}
#         n=256, W4；评测 GenEval 分层子集（25/tag = 600 张）
#   产出：output/janus-pro-7b/ablation_a2/<tag>/geneval/results.jsonl
#         scale: scale_cache/{mllmq,mbq}/..._<mix>.pt
#   ⚠️ 无冲突：新 calib json（data/coco/janus_mixed_calib_<mix>_256.json）、新 scale 名、独立输出目录
#
# Usage（另一台机）：
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_ablation_a2_mix.sh \
#     > /nas3/kangjia/mllmq/output/logs/a2_mix_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUTBASE=$ROOT/output/janus-pro-7b/ablation_a2
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

QFILE=/tmp/a2_queue.$$; QLOCK=/tmp/a2_queue.lock
build_queue(){
  : > "$QFILE"
  # scales
  echo "SCALE|mllmq|und|$CACHE/mllmq/janus_mae_w4a16_und.pt|$ROOT/data/coco/janus_mixed_calib_undheavy_256.json" >> "$QFILE"
  echo "SCALE|mllmq|gen|$CACHE/mllmq/janus_mae_w4a16_gen.pt|$ROOT/data/coco/janus_mixed_calib_genheavy_256.json" >> "$QFILE"
  echo "SCALE|mllmq|bal|$CACHE/mllmq/janus_mae_w4a16_bal.pt|$ROOT/data/coco/janus_mixed_calib_balanced_256.json" >> "$QFILE"
  echo "SCALE|mbq_scalar|und|$CACHE/mbq/janus_mbq_scalar_w4a16_und.pt|$ROOT/data/coco/janus_mixed_calib_undheavy_256.json" >> "$QFILE"
  echo "SCALE|mbq_scalar|gen|$CACHE/mbq/janus_mbq_scalar_w4a16_gen.pt|$ROOT/data/coco/janus_mixed_calib_genheavy_256.json" >> "$QFILE"
  echo "SCALE|mbq_scalar|bal|$CACHE/mbq/janus_mbq_scalar_w4a16_bal.pt|$ROOT/data/coco/janus_mixed_calib_balanced_256.json" >> "$QFILE"
  # evals
  echo "EVAL|mllmq_und|mllmq_mae|$CACHE/mllmq/janus_mae_w4a16_und.pt" >> "$QFILE"
  echo "EVAL|mllmq_gen|mllmq_mae|$CACHE/mllmq/janus_mae_w4a16_gen.pt" >> "$QFILE"
  echo "EVAL|mllmq_bal|mllmq_mae|$CACHE/mllmq/janus_mae_w4a16_bal.pt" >> "$QFILE"
  echo "EVAL|scalar_und|mbq_scalar|$CACHE/mbq/janus_mbq_scalar_w4a16_und.pt" >> "$QFILE"
  echo "EVAL|scalar_gen|mbq_scalar|$CACHE/mbq/janus_mbq_scalar_w4a16_gen.pt" >> "$QFILE"
  echo "EVAL|scalar_bal|mbq_scalar|$CACHE/mbq/janus_mbq_scalar_w4a16_bal.pt" >> "$QFILE"
}
pop_job(){ local j; exec 9>"$QLOCK"; flock 9; j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp"; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-; echo "$j"; }

run_scale(){ # method mix sp calib gpu
  local m=$1 mix=$2 sp=$3 calib=$4 gpu=$5
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -s "$sp" ] && { good "[GPU$gpu] SKIP scale $mix/$m"; return 0; }
  info "[GPU$gpu] SCALE $m $mix"
  local extra=(); [ "$m" = "mllmq" ] && extra=(--method mllmq --loss_mode mae) || extra=(--method mbq --reweight --reweight_mode scalar --loss_mode mae)
  python main_quant.py --model_path "$MODEL" --device cuda:0 --calib_data coco \
    --data_path "$calib" --image_folder "$IMG" --n_samples 256 --w_bit 4 --a_bit 16 --w_group 128 \
    --run_process "${extra[@]}" --scale_path "$sp" > "$LOG/a2_scale_${m}_${mix}.log" 2>&1
  [ -s "$sp" ] && good "[GPU$gpu] scale OK $mix/$m" || { fail "[GPU$gpu] scale FAIL $mix/$m"; return 1; }
}
run_eval(){ # tag qm sp gpu
  local tag=$1 qm=$2 sp=$3 gpu=$4
  local out="$OUTBASE/$tag" gdir="$OUTBASE/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP eval $tag"; return 0; }
  if [ ! -s "$sp" ]; then local t=0; while [ ! -s "$sp" ] && [ $t -lt 2400 ]; do sleep 20; t=$((t+20)); done; fi
  [ -s "$sp" ] || { fail "[GPU$gpu] $tag 缺 scale"; return 1; }
  mkdir -p "$out"
  info "[GPU$gpu] EVAL $tag（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval --output_dir "$out" \
    --num_samples 4 --limit_per_tag "$PER_TAG" --quant_method "$qm" --w_bit 4 --scale_path "$sp" \
    > "$LOG/a2_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; return 1; }
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" --clip-model-path "$CLIP_BIN" \
    > "$LOG/a2_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}
lane(){ local gpu=$1 job kind a b c d; while :; do job=$(pop_job); [ -z "$job" ] && { log "[GPU$gpu] 队列空"; return; }
  IFS='|' read -r kind a b c d <<< "$job"
  case "$kind" in
    SCALE) run_scale "$a" "$b" "$c" "$d" "$gpu" ;;
    EVAL)  run_eval  "$a" "$b" "$c" "$gpu" ;;
  esac; done; }

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }; trap cleanup SIGINT SIGTERM
log "===== A2：校准配比鲁棒性（und/gen/bal × mllmq/mbq_scalar, n=256 W4）====="
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失"; exit 2; }
mkdir -p "$OUTBASE"; build_queue
for g in ${GPUS:-0 1}; do lane "$g" & done; wait
log "════ 清点 ════"
for t in mllmq_und mllmq_gen mllmq_bal scalar_und scalar_gen scalar_bal; do
  f="$OUTBASE/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
rm -f "$QFILE"; info "ALL DONE"