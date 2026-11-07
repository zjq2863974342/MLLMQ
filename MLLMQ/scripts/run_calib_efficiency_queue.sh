#!/usr/bin/env bash
# ============================================================================
# 校准效率 + W3 扫描 + 机制消融 + 误差棒 · 综合双队列（接在「校准消融」跑完后自动起）
#
# 目标：把 MLLMQ 的卖点从「精度」挪到「校准效率 / 机制」——唯一可能站住的方向。
#
# 队列内容（2 lane 共享 FIFO，GPU 无关，lane 各自导出 CUDA_VISIBLE_DEVICES）：
#   A. 对手曲线：mbq_block_w4 / awq_w4 在 n=64/128（GenEval 分层子集 150prompt×4）
#      → 与已跑的 mllmq_mae_w4 n=64/128/256 直接可比，回答「少数据不降质」是否 mllmq 独有
#   B. 效率测量：s/image + 峰值显存（fp16 / mae_w4 / mae_w3 / mbq_block_w4）
#   C. W3 扫描：mllmq_mae_w3 / mbq_block_w3 在 n=64/128/256（差距最大档）
#   D. 机制消融：n=128 w4 下 channel(mllmq) vs scalar(mbq_scalar) vs off(mbq_block) 三档重加权
#   E. 误差棒：GenEval 子集重复（mae_w4_n64/n128、mae_w3_n128、mbq_block_w3_n64 各 r2）
#
# 产出：output/janus-pro-7b/calib_ablation/<tag>/geneval/results.jsonl
#       output/janus-pro-7b/efficiency/<tag>.json
#       scale → MLLMQ/scale_cache/{mllmq,mbq,awq}/*.pt
#
# Usage: cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_calib_efficiency_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/calibq_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#        PLAN=1 bash ... 只看计划
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUTBASE=$ROOT/output/janus-pro-7b/calib_ablation
EFFDIR=$ROOT/output/janus-pro-7b/efficiency
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
PER_TAG=${PER_TAG:-25}
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

# ── 等当前校准消融结束 ──────────────────────────────────────────────
wait_for_running() {
  if pgrep -f "run_calib_ablation_w4.sh" >/dev/null 2>&1; then
    log "检测到「校准消融」仍在跑 → 等它结束后再起本队列"
    while pgrep -f "run_calib_ablation_w4.sh" >/dev/null 2>&1; do sleep 60; done
    log "「校准消融」已结束，本队列开始"
  else
    log "未检测到「校准消融」进程，直接开跑"
  fi
}

# ── job: 生成 scale ────────────────────────────────────────────────
# $1=method(mllmq|mbq|mbq_scalar|awq) $2=w_bit $3=n $4=scale_path_abs
run_scale() {
  local m=$1 w=$2 n=$3 sp=$4
  [ -s "$sp" ] && { good "SKIP scale $(basename "$sp")（已存在）"; return 0; }
  info "RUN scale [$m w$w n$n] → $(basename "$sp")"
  local extra=()
  case "$m" in
    mllmq)      extra=(--method mllmq --loss_mode mae) ;;
    mbq)        extra=(--method mbq --loss_mode mae) ;;
    mbq_scalar) extra=(--method mbq --reweight --reweight_mode scalar --loss_mode mae) ;;
    awq)        extra=(--method awq) ;;
    *) fail "未知 method $m"; return 1 ;;
  esac
  python main_quant.py --model_path "$MODEL" --device cuda:0 --calib_data coco \
    --data_path "$CALIB" --image_folder "$IMG" \
    --n_samples "$n" --w_bit "$w" --a_bit 16 --w_group 128 \
    --run_process "${extra[@]}" --scale_path "$sp" \
    > "$LOG/calibq_scale_${m}_w${w}_n${n}.log" 2>&1
  local rc=$?
  [ -s "$sp" ] && { good "scale OK $(basename "$sp")"; return 0; }
  fail "scale FAIL [$m w$w n$n] rc=$rc"; tail -4 "$LOG/calibq_scale_${m}_w${w}_n${n}.log"; return 1
}

# ── job: GenEval 分层子集 生成 + 评分 ───────────────────────────────
# $1=tag $2=qmethod $3=w_bit $4=scale_path_abs
run_geneval() {
  local tag=$1 qm=$2 w=$3 sp=$4
  local out="$OUTBASE/$tag" gdir="$OUTBASE/$tag/geneval"
  if [ -f "$gdir/results.jsonl" ]; then good "SKIP $tag（results.jsonl 已存在）"; return 0; fi
  # 等 scale
  if [ ! -s "$sp" ]; then
    log "$tag 等 scale 就绪：$(basename "$sp")"
    local t=0; while [ ! -s "$sp" ] && [ $t -lt 2400 ]; do sleep 20; t=$((t+20)); done
  fi
  [ -s "$sp" ] || { fail "$tag 缺 scale（超时）"; return 1; }
  mkdir -p "$out"
  info "RUN $tag GenEval 生成（${PER_TAG}/tag）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
    > "$LOG/calibq_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "$tag 生成失败"; tail -4 "$LOG/calibq_gen_${tag}.log"; return 1; }
  info "RUN $tag GenEval 评分"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/calibq_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "$tag 评分失败"; tail -4 "$LOG/calibq_score_${tag}.log"; return 1; }
  local ov; ov=$(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')
  good "$tag DONE :: $ov"
}

# ── job: 效率测量 ─────────────────────────────────────────────────
# 参数若干组：tag:qmethod:wbit:scale ('-' 表示 None)
run_eff() {
  local spec
  for spec in "$@"; do
    local tag qm w sp
    IFS=: read -r tag qm w sp <<< "$spec"
    [ "$sp" = "-" ] && sp=""
    [ -f "$EFFDIR/$tag.json" ] && { good "SKIP eff $tag"; continue; }
    info "RUN eff $tag"
    if [ -n "$sp" ]; then
      python "$ROOT/scripts/measure_efficiency.py" --tag "$tag" --quant_method "$qm" --w_bit "$w" \
        --scale_path "$sp" --n_images 50 --out "$EFFDIR/$tag.json" \
        > "$LOG/calibq_eff_${tag}.log" 2>&1
    else
      python "$ROOT/scripts/measure_efficiency.py" --tag "$tag" --quant_method "$qm" \
        --n_images 50 --out "$EFFDIR/$tag.json" \
        > "$LOG/calibq_eff_${tag}.log" 2>&1
    fi
    [ -f "$EFFDIR/$tag.json" ] && good "eff OK $tag :: $(cat "$EFFDIR/$tag.json" | tr -d '\n ')" || fail "eff FAIL $tag"
  done
}

# ── 队列（FIFO；lane 取号时 flock 原子）────────────────────────────
QFILE=$OUTBASE/_queue.txt
QLOCK=$OUTBASE/_queue.lock

build_queue() {
  : > "$QFILE"
  local S=$CACHE/mllmq B=$CACHE/mbq A=$CACHE/awq
  # A. 对手曲线 scale
  echo "SCALE|mbq|4|64|$B/janus_mbq_block_w4a16_n64.pt"       >> "$QFILE"
  echo "SCALE|mbq|4|128|$B/janus_mbq_block_w4a16_n128.pt"     >> "$QFILE"
  echo "SCALE|awq|4|64|$A/janus_awq_w4a16_n64.pt"             >> "$QFILE"
  echo "SCALE|awq|4|128|$A/janus_awq_w4a16_n128.pt"           >> "$QFILE"
  # C. W3 扫描 scale
  echo "SCALE|mllmq|3|64|$S/janus_mae_w3a16_n64.pt"           >> "$QFILE"
  echo "SCALE|mllmq|3|256|$S/janus_mae_w3a16_n256.pt"         >> "$QFILE"
  echo "SCALE|mbq|3|64|$B/janus_mbq_block_w3a16_n64.pt"       >> "$QFILE"
  echo "SCALE|mbq|3|128|$B/janus_mbq_block_w3a16_n128.pt"     >> "$QFILE"
  # D. 机制消融 scale（scalar 重加权 @ n128 w4；off=mqb_n128、channel=mllmq 现有）
  echo "SCALE|mbq_scalar|4|128|$B/janus_mbq_scalar_w4a16_n128.pt" >> "$QFILE"
  # B. 效率
  echo "EFF|fp16:fp16:-:-"                                    >> "$QFILE"
  echo "EFF|mllmq_mae_w4:mllmq_mae:4:$S/janus_mae_w4a16.pt|mllmq_mae_w3:mllmq_mae:3:$S/janus_mae_w3a16.pt" >> "$QFILE"
  echo "EFF|mbq_block_w4:mbq_block:4:$B/janus_mbq_block_w4a16.pt" >> "$QFILE"
  # A. 对手曲线 eval（GenEval 子集）
  echo "GENEVAL|mbq_block_w4_n64|mbq_block|4|$B/janus_mbq_block_w4a16_n64.pt"     >> "$QFILE"
  echo "GENEVAL|mbq_block_w4_n128|mbq_block|4|$B/janus_mbq_block_w4a16_n128.pt"   >> "$QFILE"
  echo "GENEVAL|awq_w4_n64|awq|4|$A/janus_awq_w4a16_n64.pt"                       >> "$QFILE"
  echo "GENEVAL|awq_w4_n128|awq|4|$A/janus_awq_w4a16_n128.pt"                     >> "$QFILE"
  # C. W3 扫描 eval
  echo "GENEVAL|mllmq_mae_w3_n64|mllmq_mae|3|$S/janus_mae_w3a16_n64.pt"           >> "$QFILE"
  echo "GENEVAL|mllmq_mae_w3_n128|mllmq_mae|3|$S/janus_mae_w3a16.pt"              >> "$QFILE"
  echo "GENEVAL|mllmq_mae_w3_n256|mllmq_mae|3|$S/janus_mae_w3a16_n256.pt"         >> "$QFILE"
  echo "GENEVAL|mbq_block_w3_n64|mbq_block|3|$B/janus_mbq_block_w3a16_n64.pt"     >> "$QFILE"
  echo "GENEVAL|mbq_block_w3_n128|mbq_block|3|$B/janus_mbq_block_w3a16_n128.pt"   >> "$QFILE"
  echo "GENEVAL|mbq_block_w3_n256|mbq_block|3|$B/janus_mbq_block_w3a16.pt"        >> "$QFILE"
  # D. 机制消融 eval（n128 w4）
  echo "GENEVAL|mbq_scalar_w4_n128|mbq_scalar|4|$B/janus_mbq_scalar_w4a16_n128.pt" >> "$QFILE"
  # E. 误差棒（重复）
  echo "GENEVAL|mllmq_mae_w4_n64_r2|mllmq_mae|4|$S/janus_mae_w4a16_n64.pt"        >> "$QFILE"
  echo "GENEVAL|mllmq_mae_w4_n128_r2|mllmq_mae|4|$S/janus_mae_w4a16.pt"           >> "$QFILE"
  echo "GENEVAL|mllmq_mae_w3_n128_r2|mllmq_mae|3|$S/janus_mae_w3a16.pt"           >> "$QFILE"
  echo "GENEVAL|mbq_block_w3_n64_r2|mbq_block|3|$B/janus_mbq_block_w3a16_n64.pt"  >> "$QFILE"
}

pop_job() {
  local j
  exec 9>"$QLOCK"; flock 9
  j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp" 2>/dev/null; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-
  echo "$j"
}

lane() {   # $1=gpu
  local gpu=$1 job kind a b c d
  export CUDA_VISIBLE_DEVICES=$gpu
  while :; do
    job=$(pop_job)
    [ -z "$job" ] && { log "[GPU$gpu] 队列已空 → lane 退出"; return 0; }
    IFS='|' read -r kind a b c d <<< "$job"
    case "$kind" in
      SCALE)   run_scale "$a" "$b" "$c" "$d" ;;
      GENEVAL) run_geneval "$a" "$b" "$c" "$d" ;;
      EFF)     IFS='|' read -r _ rest <<< "$job"; run_eff ${rest//|/ } ;;
      *)       fail "[GPU$gpu] 未知 job: $job" ;;
    esac
  done
}

DRIVER_PID=$$
cleanup() { echo "[$(date '+%F %T')] Interrupted — killing children..."; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== 校准效率/机制/误差棒 综合队列 ====="
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失"; exit 2; }
[ -f "$FRCNN" ] || { fail "FRCNN 权重缺失"; exit 2; }
mkdir -p "$OUTBASE" "$EFFDIR"
build_queue
log "队列共 $(wc -l < "$QFILE") 个 job"
if [ "$PLAN" = "1" ]; then cat "$QFILE"; exit 0; fi

wait_for_running
for g in $GPUS; do lane "$g" & done
wait

log "════════ 清点 ════════"
n_ok=0; n_tot=0
for t in mbq_block_w4_n64 mbq_block_w4_n128 awq_w4_n64 awq_w4_n128 \
         mllmq_mae_w3_n64 mllmq_mae_w3_n128 mllmq_mae_w3_n256 \
         mbq_block_w3_n64 mbq_block_w3_n128 mbq_block_w3_n256 \
         mbq_scalar_w4_n128 mllmq_mae_w4_n64_r2 mllmq_mae_w4_n128_r2 \
         mllmq_mae_w3_n128_r2 mbq_block_w3_n64_r2; do
  n_tot=$((n_tot+1))
  f="$OUTBASE/$t/geneval/results.jsonl"
  if [ -f "$f" ]; then n_ok=$((n_ok+1)); good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')"; else fail "$t ✗"; fi
done
info "GenEval 子集 $n_ok/$n_tot 完成；eff: $(ls $EFFDIR/*.json 2>/dev/null | wc -l) 条"
rm -f "$QFILE"
info "ALL DONE"