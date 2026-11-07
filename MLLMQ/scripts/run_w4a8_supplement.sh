#!/usr/bin/env bash
# ============================================================================
# W4A8 supplement driver (TODO 2026-11-04 plan)
#
#   目的：MBQ 论文最大收益出现在 W4A8；补 6 config × 理解/生成两条线。
#
#   Phase:
#     scales  : 5 learned × 2 pool = 10 builds
#               pool und128 = data/coco/janus_understanding_128.json  (n=128)
#               pool mix384 = data/coco/janus_mixed_tri_384.json      (n=384)
#     und     : 6 config × 4 项理解（MMMU/MME/TextVQA/MathVista）
#     geneval : 6 config × 25/tag = 600 图（筛选）
#     dpg     : 6 config × 1065 图（生成后串行评分）
#
#   统一口径：Janus-Pro-7B，W4A8，w_group=128，zero_point=True，仅 LM 30 层。
#   所有阶段 skip-existing；可重复执行。默认等待本机 SEED driver 结束。
#
#   用法：
#     cd /nas3/kangjia/mllmq/MLLMQ
#     # 只补 scale（先确认可跑通）
#     PHASES=scales setsid nohup bash scripts/run_w4a8_supplement.sh \
#       > ../output/logs/w4a8_supplement/driver_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#     # 全量筛选（理解 + GenEval；DPG 需显式加入）
#     PHASES="scales und geneval" ...
#     # 指定卡（主机 2 卡 / 副机 4 卡）
#     GPUS="0 1" PHASES=... ...
#     # 预览
#     PLAN=1 PHASES="scales und geneval" bash scripts/run_w4a8_supplement.sh
#
#   注意：DPG 1065 图/config 约 14h，默认不开；确认 GenEval 有正向后再加。
# ============================================================================
set -o pipefail

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
POOL_UND=$ROOT/data/coco/janus_understanding_128.json
POOL_MIX=$ROOT/data/coco/janus_mixed_tri_384.json
SC=$ROOT/MLLMQ/scale_cache/w4a8
OUT=$ROOT/output/janus-pro-7b/w4a8_supplement
LOG=$ROOT/output/logs/w4a8_supplement
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
DPG_CSV=$ROOT/data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)

PHASES=${PHASES:-"scales und geneval"}   # add "dpg" explicitly
WAIT_SEED=${WAIT_SEED:-1}
PLAN=${PLAN:-0}
GENEVAL_PER_TAG=${GENEVAL_PER_TAG:-25}
GENEVAL_TARGET=$((GENEVAL_PER_TAG * 6 * 4))
DPG_TARGET=${DPG_TARGET:-1065}

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -ge 1 ] || NGPU=1
GPUS=${GPUS:-$(seq -s' ' 0 $((NGPU-1)))}
NGPU=$(echo $GPUS | wc -w)
GPU0=$(echo $GPUS | awk '{print $1}')

TASK_UND=(mmmu_val mme_local textvqa_val_local mathvista_testmini_local)
SEED_TASKS="seedbench_2_local_chunk0 seedbench_2_local_chunk1 seedbench_2_local_chunk2 seedbench_2_local_chunk3 seedbench_2_local_chunk4"
TAGS=(rtn mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img mllmq_both)
LEARNED=(mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img mllmq_both)
# Optional subset / concurrency for eval phases (scales always stay 1/GPU).
JPG=${JOBS_PER_GPU:-1}
TAGS_RUN=("${TAGS[@]}")
if [ -n "${ONLY_TAGS:-}" ]; then
  TAGS_RUN=()
  for _t in $ONLY_TAGS; do TAGS_RUN+=("$_t"); done
fi

mkdir -p "$SC/und128" "$SC/mix384" "$OUT" "$LOG" "$OUT/dpg_scores"

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$EV/.ms_cache}"

cd "$PROJ" || { echo "cannot cd $PROJ"; exit 2; }

log(){ echo "[$(date '+%F %T')] $1"; }
fail(){ echo "[$(date '+%F %T')] ❌ $1"; }

method_of(){ case "$1" in rtn) echo rtn;; mbq_scalar) echo mbq;; *) echo mllmq;; esac; }
qmethod_of(){ case "$1" in rtn) echo rtn;; mbq_scalar) echo mbq_scalar;; *) echo mllmq_mae;; esac; }
env_of(){
  case "$1" in
    mllmq_channel_only) echo "MLLMQ_DISABLE_T2I=1 MLLMQ_DISABLE_GEN_IMG=1";;
    mllmq_t2i_text)     echo "MLLMQ_DISABLE_GEN_IMG=1";;
    mllmq_t2i_img)      echo "MLLMQ_DISABLE_T2I=1";;
    *)                  echo "";;
  esac
}
pool_of(){ case "$1" in und128) echo "$POOL_UND";; mix384) echo "$POOL_MIX";; esac; }
n_of(){ case "$1" in und128) echo 128;; mix384) echo 384;; esac; }
scale_of(){ # kind tag
  local kind=$1 tag=$2
  case "$tag" in rtn) echo ""; return;; esac
  case "$kind" in
    und|seed)
      # Understanding-side / SEED: scalar/channel use the 128-understanding pool;
      # t2i_* variants need the 384 mixed pool (their modality masks are empty
      # in the understanding-only pool).
      case "$tag" in
        mbq_scalar|mllmq_channel_only) echo "$SC/und128/${tag}_w4a8.pt";;
        *)                             echo "$SC/mix384/${tag}_w4a8.pt";;
      esac;;
    *) echo "$SC/mix384/${tag}_w4a8.pt";;
  esac
}

count_png(){ find "$1" -name '*.png' 2>/dev/null | wc -l; }

# ── single-job runner (child process enters here) ───────────────────────────
exec_job(){
  local kind pool tag gpu
  IFS='|' read -r kind pool tag gpu <<< "$1"

  case "$kind" in
    scale)
      local out="$SC/$pool/${tag}_w4a8.pt"
      [ -s "$out" ] && { log "SKIP scale $pool/$tag"; return 0; }
      local method envs extra=()
      method=$(method_of "$tag"); envs=$(env_of "$tag")
      [ "$method" = "mbq" ] && extra=(--reweight --reweight_mode scalar)
      [ "$method" = "mllmq" ] && extra=(--reweight_mode channel)
      log "SCALE $pool/$tag GPU$gpu"
      CUDA_VISIBLE_DEVICES=$gpu env $envs python "$PROJ/main_quant.py" \
        --model_path "$MODEL" --device cuda:0 \
        --calib_data coco --data_path "$(pool_of "$pool")" --image_folder "$IMG" \
        --n_samples "$(n_of "$pool")" \
        --w_bit 4 --a_bit 8 --w_group 128 --run_process \
        --method "$method" --loss_mode mae "${extra[@]}" \
        --scale_path "$out" > "$LOG/scale_${pool}_${tag}.log" 2>&1
      if [ -s "$out" ]; then log "OK scale $pool/$tag"; else fail "scale $pool/$tag (see $LOG/scale_${pool}_${tag}.log)"; return 1; fi
      ;;

    und)
      local method scale
      method=$(method_of "$tag"); scale=$(scale_of und "$tag")
      for t in "${TASK_UND[@]}"; do
        local out="$OUT/und/$tag/$t/${tag}_w4a8"
        if ls "$out"/*_results.json >/dev/null 2>&1; then log "SKIP und $tag/$t"; continue; fi
        log "UND $tag/$t GPU$gpu"
        CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/eval_benchmarks.py" \
          --model_type janus --model_path "$MODEL" \
          --method "$method" --method_tag "$tag" \
          --w_bit 4 --a_bit 8 --w_group 128 \
          --tasks "$t" --output_base "$OUT/und/$tag" \
          --device cuda:0 --log_samples \
          ${scale:+--scale_path "$scale"} > "$LOG/und_${tag}_${t}.log" 2>&1
        if ls "$out"/*_results.json >/dev/null 2>&1; then log "OK und $tag/$t"; else fail "und $tag/$t"; fi
      done
      ;;

    seed)
      local method scale outbase logdir
      method=$(method_of "$tag"); scale=$(scale_of seed "$tag")
      outbase="$OUT/seed/$tag"; logdir="$LOG/seed/$tag"
      mkdir -p "$outbase" "$logdir"
      log "SEED $tag GPU$gpu"
      TAG="$tag" METHOD="$method" SCALE="$scale" GPU="$gpu" W_BIT=4 A_BIT=8 \
        OUTBASE="$OUT/seed" LOG="$logdir" TASKS="$SEED_TASKS" \
        bash "$PROJ/scripts/run_understanding_7.sh" > "$LOG/seed_${tag}.log" 2>&1
      if [ $? -eq 0 ]; then log "OK seed $tag"; else fail "seed $tag (see $LOG/seed_${tag}.log)"; fi
      ;;

    geneval)
      local qm scale outd n
      qm=$(qmethod_of "$tag"); scale=$(scale_of geneval "$tag")
      outd="$OUT/geneval/$tag"
      n=$(count_png "$outd/geneval")
      if [ "$n" -lt "$GENEVAL_TARGET" ]; then
        log "GENEVAL $tag GPU$gpu ($n/$GENEVAL_TARGET)"
        CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/eval_generate_quant.py" \
          --model_path "$MODEL" --benchmark geneval \
          --output_dir "$outd" --num_samples 4 --limit_per_tag "$GENEVAL_PER_TAG" \
          --quant_method "$qm" --w_bit 4 --a_bit 8 \
          ${scale:+--scale_path "$scale"} > "$LOG/geneval_${tag}.log" 2>&1
      fi
      if [ ! -s "$outd/geneval/results.jsonl" ]; then
        if [ -n "$CLIP_BIN" ]; then
          log "SCORE geneval $tag"
          CUDA_VISIBLE_DEVICES=$gpu python "$EV/evaluate_images.py" "$outd/geneval" \
            --outfile "$outd/geneval/results.jsonl" --clip-model-path "$CLIP_BIN" \
            > "$LOG/geneval_score_${tag}.log" 2>&1
        else
          fail "geneval $tag: CLIP weights missing; skip scoring"
        fi
      fi
      if [ -s "$outd/geneval/results.jsonl" ]; then
        log "OK geneval $tag :: $(python "$EV/summary_scores.py" "$outd/geneval/results.jsonl" 2>/dev/null | grep -i '^Overall' | tail -1)"
      else
        fail "geneval $tag ($(count_png "$outd/geneval")/$GENEVAL_TARGET png)"
      fi
      ;;

    dpg)
      local qm scale outd n
      qm=$(qmethod_of "$tag"); scale=$(scale_of dpg "$tag")
      outd="$OUT/dpg/$tag"
      n=$(count_png "$outd/dpg_bench")
      if [ "$n" -lt "$DPG_TARGET" ]; then
        log "DPG $tag GPU$gpu ($n/$DPG_TARGET)"
        CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/eval_generate_quant.py" \
          --model_path "$MODEL" --benchmark dpg_bench \
          --output_dir "$outd" --num_samples 1 --shard_id 0 --num_shards 1 \
          --quant_method "$qm" --w_bit 4 --a_bit 8 \
          ${scale:+--scale_path "$scale"} > "$LOG/dpg_${tag}.log" 2>&1
      fi
      log "OK dpg gen $tag ($(count_png "$outd/dpg_bench")/$DPG_TARGET)"
      ;;

    *) fail "unknown job: $1"; return 2;;
  esac
}

# child dispatch (must come before the main flow)
if [ "${1:-}" = "--run-line" ]; then
  shift
  exec_job "$1"
  exit $?
fi

# ── gates ──────────────────────────────────────────────────────────────────
[ -d "$MODEL" ] || { fail "missing model $MODEL"; exit 2; }
[ -s "$POOL_UND" ] && [ -s "$POOL_MIX" ] || { fail "missing calibration pools"; exit 2; }
grep -q "attention_mask.bool()" "$PROJ/qmllm/models/janus/janus.py" || {
  fail "janus.py is not fixed (attention_mask.bool() missing)"; exit 2; }
log "gate ok: model/pools/fixed wrapper"
log "phases=$PHASES gpus='$GPUS' n_gpu=$NGPU plan=$PLAN"

if [ "$WAIT_SEED" = "1" ] && [ "$PLAN" != "1" ]; then
  while ps -eo args | grep -E 'run_seed_full_remote\.sh|run_understanding_7\.sh|run_seed_dual_isolated\.sh' \
        | grep -v grep >/dev/null; do
    log "waiting for SEED driver ..."
    sleep 120
  done
  log "SEED driver finished"
fi

build_queue(){
  local kind=$1 q=$2
  : > "$q"
  local gpu_idx=0 gpu
  case "$kind" in
    scales)
      for pool in und128 mix384; do
        for tag in "${LEARNED[@]}"; do
          gpu=$(echo "$GPUS" | awk -v i=$((gpu_idx % NGPU)) '{print $(i+1)}')
          echo "scale|$pool|$tag|$gpu" >> "$q"
          gpu_idx=$((gpu_idx+1))
        done
      done
      ;;
    seed)
      # SEED is RAM-bound (~56GB RSS/proc): exactly 1 job per GPU.
      for tag in "${TAGS_RUN[@]}"; do
        gpu=$(echo "$GPUS" | awk -v i=$((gpu_idx % NGPU)) '{print $(i+1)}')
        echo "seed|_|$tag|$gpu" >> "$q"
        gpu_idx=$((gpu_idx+1))
      done
      ;;
    und|geneval|dpg)
      for tag in "${TAGS_RUN[@]}"; do
        # Round-robin across GPUs; xargs -P NGPU*JPG allows up to JPG per GPU.
        gpu=$(echo "$GPUS" | awk -v i=$((gpu_idx % NGPU)) '{print $(i+1)}')
        echo "$kind|_|$tag|$gpu" >> "$q"
        gpu_idx=$((gpu_idx+1))
      done
      ;;
  esac
}

run_queue(){
  local q=$1 par=${2:-$NGPU}
  if [ "$PLAN" = "1" ]; then
    echo "--- $(basename "$q") (parallel=$par)"
    cat "$q"
    return 0
  fi
  xargs -d '\n' -P "$par" -I{} bash "$0" --run-line "{}" < "$q"
}

need_scales(){
  local kind=$1 miss=""
  for tag in "${LEARNED[@]}"; do
    [ -s "$(scale_of "$kind" "$tag")" ] || miss="$miss $tag"
  done
  [ -z "$miss" ] || { fail "missing $kind scales:$miss"; return 1; }
}

for phase in $PHASES; do
  q="/tmp/w4a8_${phase}_$$.q"
  case "$phase" in
    scales)
      build_queue scales "$q"; run_queue "$q"; rm -f "$q"
      ;;
    und)
      [ "$PLAN" = "1" ] || need_scales und || exit 1
      build_queue und "$q"; run_queue "$q" "$((NGPU*JPG))"; rm -f "$q"
      ;;
    seed)
      [ "$PLAN" = "1" ] || need_scales seed || exit 1
      build_queue seed "$q"; run_queue "$q" "$NGPU"; rm -f "$q"
      ;;
    geneval)
      [ "$PLAN" = "1" ] || need_scales geneval || exit 1
      build_queue geneval "$q"; run_queue "$q" "$((NGPU*JPG))"; rm -f "$q"
      ;;
    dpg)
      [ "$PLAN" = "1" ] || need_scales dpg || exit 1
      build_queue dpg "$q"; run_queue "$q" "$((NGPU*JPG))"; rm -f "$q"
      if [ "$PLAN" != "1" ]; then
        log "DPG scoring (serial)"
        for tag in "${TAGS_RUN[@]}"; do
          [ -f "$OUT/dpg_scores/$tag.txt" ] && { log "SKIP dpg score $tag"; continue; }
          [ -d "$OUT/dpg/$tag/dpg_bench" ] || { fail "dpg $tag missing images"; continue; }
          PYTHONPATH="$PROJ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}" \
          CUDA_VISIBLE_DEVICES=$GPU0 python "$PROJ/scripts/score_dpg_coco.py" \
            --image-root-path "$OUT/dpg/$tag/dpg_bench" --resolution 384 --pic-num 1 \
            --csv "$DPG_CSV" --tag "w4a8_${tag}" --out "$OUT/dpg_scores/$tag.txt" \
            > "$LOG/dpg_score_${tag}.log" 2>&1
          [ -s "$OUT/dpg_scores/$tag.txt" ] \
            && log "OK dpg score $tag :: $(grep -i 'DPG-Bench score' "$OUT/dpg_scores/$tag.txt" | tail -1)" \
            || fail "dpg score $tag"
        done
      fi
      ;;
    *) fail "unknown phase $phase"; exit 2;;
  esac
done

if [ "$PLAN" != "1" ]; then
  log "===== status ====="
  for pool in und128 mix384; do
    for tag in "${LEARNED[@]}"; do
      [ -s "$SC/$pool/${tag}_w4a8.pt" ] && log "scale $pool/$tag OK" || log "scale $pool/$tag MISS"
    done
  done
  for tag in "${TAGS[@]}"; do
    u=$(find "$OUT/und/$tag" -name '*_results.json' 2>/dev/null | wc -l)
    g=$(count_png "$OUT/geneval/$tag/geneval")
    d=$(count_png "$OUT/dpg/$tag/dpg_bench")
    log "$tag: und_results=$u/4 geneval_png=$g/$GENEVAL_TARGET dpg_png=$d/$DPG_TARGET"
  done
fi
log "ALL DONE"