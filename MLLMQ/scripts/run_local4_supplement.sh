#!/usr/bin/env bash
# ============================================================================
# LOCAL 4-GPU supplement driver — 2026-11-05 (post container restart)
#
#   LOCAL = complement of the remote machine's 4 configs:
#     W4A8 DPG  local: mllmq_t2i_img, mllmq_both, awq, mllmq_gen_only
#     W4A8 SEED local: rtn, mllmq_both, awq, mllmq_gen_only
#     W3A16     local: awq (DPG resume + score + SEED),
#                      gen_only (und 7 + SEED; DPG is running on the remote)
#     W3A16 t2i_img/both are already complete -> skipped.
#
#   Also fills the previously-missing benchmarks for gen_only:
#     W4A8 gen_only : scale + und4 + GenEval600 + DPG + score + SEED
#     W3A16 gen_only: und7 + SEED (GenEval already done; DPG on remote)
#   W4A8 t2i_img/both DPG (partial) + score + SEED.
#
#   Resource grid:
#     DPG : 8 concurrent slots = 2 procs/GPU (10 shard jobs, queue-packed)
#     und/GenEval: 7 concurrent while one serial DPG score runs on GPU0
#                  -> at most 2 procs/GPU
#     SEED: strictly 1 proc/GPU, 6 tags in 2 waves
#
#   Skip-existing everywhere; safe to rerun after reboot/interrupt.
# ============================================================================
set -o pipefail

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
POOL_UND=$ROOT/data/coco/janus_understanding_128.json
POOL_MIX=$ROOT/data/coco/janus_mixed_tri_384.json
POOL_GEN=$ROOT/data/coco/janus_gen_tf_128.json
SC16=$ROOT/MLLMQ/scale_cache
SC8=$ROOT/MLLMQ/scale_cache/w4a8
OUT16=$ROOT/output/fixed_ablation_20261103_unified
OUT8=$ROOT/output/janus-pro-7b/w4a8_supplement
UND16=$ROOT/output/fixed_ablation_20261103_understanding_5h
EOUT=$ROOT/output/fixed_ablation_20261103_understanding_5h_seed
LOG=$ROOT/output/logs/local4_supplement
LOG16=$ROOT/output/logs/fixed_ablation_20261103_unified
ELOG=$ROOT/output/logs/understanding_5h_seed
L5H=$ROOT/output/logs/understanding_5h
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
DPG_CSV=$ROOT/data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
SEED_TASKS="seedbench_2_local_chunk0 seedbench_2_local_chunk1 seedbench_2_local_chunk2 seedbench_2_local_chunk3 seedbench_2_local_chunk4"
TASK_UND4=(mmmu_val mme_local textvqa_val_local mathvista_testmini_local)
TASK_UND7=(mme_local textvqa_val_local mmmu_val mathvista_testmini_local pope_random_local pope_popular_local pope_adversarial_local)
DPG_TARGET=${DPG_TARGET:-1065}
GENEVAL_TARGET=${GENEVAL_TARGET:-600}
NGPU_REQUIRED=${NGPU_REQUIRED:-4}
DRY_RUN=${DRY_RUN:-0}
SELF=$PROJ/scripts/run_local4_supplement.sh

mkdir -p "$LOG" "$LOG16" "$ELOG" "$L5H" "$SC8/und128" "$SC8/mix384" \
         "$OUT8/und/awq" "$OUT8/und/mllmq_gen_only" "$OUT8/geneval/awq" \
         "$OUT8/geneval/mllmq_gen_only" "$OUT8/dpg/awq" "$OUT8/dpg/mllmq_gen_only" \
         "$OUT8/seed" "$OUT8/dpg_scores" "$OUT16/dpg_scores" "$UND16/mllmq_gen_only"

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$EV/.ms_cache}"

cd "$PROJ" || { echo "cannot cd $PROJ"; exit 2; }

log(){ echo "[$(date '+%F %T')] $1"; }
count_png(){ find "$1" -name '*.png' 2>/dev/null | wc -l; }
seed_chunks(){ find "$1" -name '*_results.json' 2>/dev/null | sed 's/.*chunk\([0-9]\).*/\1/' | sort -u | wc -l; }
und_done(){ ls "$1/$2/$3"/*_results.json >/dev/null 2>&1; }

preflight(){
  local n pool t c=0
  n=$(nvidia-smi -L 2>/dev/null | wc -l)
  log "preflight: GPUs=$n (need $NGPU_REQUIRED)"
  [ "$n" -ge "$NGPU_REQUIRED" ] || { log "❌ need at least $NGPU_REQUIRED GPUs (current $n)"; return 2; }
  for f in "$SC16/unified_384/awq_w3a16.pt" "$SC16/unified_understanding_128/awq_w3a16.pt" \
           "$SC16/unified_generation_only/mllmq_gen_only_w3a16.pt"; do
    [ -s "$f" ] || { log "❌ missing W3A16 scale: $f"; return 2; }
  done
  for pool in und128 mix384; do
    for t in mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img mllmq_both; do
      [ -s "$SC8/$pool/${t}_w4a8.pt" ] || { log "❌ missing W4A8 scale $pool/$t"; return 2; }
      c=$((c+1))
    done
  done
  grep -q "MLLMQ_GEN_ONLY" qmllm/methods/mbq/quantize/pre_quant.py \
    || { log "❌ MLLMQ_GEN_ONLY code path missing"; return 2; }
  log "preflight: W4A8 non-AWQ scales=$c/10; GEN_ONLY path OK"
  return 0
}

# ── stage 0: scales ─────────────────────────────────────────────────────────
build_awq_scale(){ # pool n gpu
  local pool=$1 n=$2 gpu=$3 out="$SC8/$1/awq_w4a8.pt" poolfile
  [ "$pool" = und128 ] && poolfile=$POOL_UND || poolfile=$POOL_MIX
  [ -s "$out" ] && { log "SKIP AWQ scale $pool"; return 0; }
  log "AWQ scale $pool (n=$n) GPU$gpu"
  [ "$DRY_RUN" = "1" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/main_quant.py" \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$poolfile" --image_folder "$IMG" \
    --n_samples "$n" --w_bit 4 --a_bit 8 --w_group 128 --run_process \
    --method awq --loss_mode mae \
    --scale_path "$out" > "$LOG/scale_${pool}_awq.log" 2>&1
  [ -s "$out" ] || { log "❌ AWQ scale $pool failed"; return 1; }
  log "OK AWQ scale $pool"
}
build_genonly_scale(){ # gpu
  local gpu=$1 out="$SC8/mix384/mllmq_gen_only_w4a8.pt"
  [ -s "$out" ] && { log "SKIP gen_only scale"; return 0; }
  log "gen_only scale (gen_tf_128, w4a8) GPU$gpu"
  [ "$DRY_RUN" = "1" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu env MLLMQ_GEN_ONLY=1 python "$PROJ/main_quant.py" \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$POOL_GEN" --image_folder "$IMG" --n_samples 128 \
    --w_bit 4 --a_bit 8 --w_group 128 --run_process \
    --scale_path "$out" --method mllmq --reweight_mode channel --loss_mode mae \
    > "$LOG/scale_gen_only_w4a8.log" 2>&1
  [ -s "$out" ] || { log "❌ gen_only scale failed"; return 1; }
  log "OK gen_only scale"
}
build_scales(){
  build_awq_scale und128 128 0 & local p0=$!
  build_awq_scale mix384 384 1 & local p1=$!
  build_genonly_scale 2 & local p2=$!
  wait $p0 $p1 $p2
}

# ── DPG queue (8 slots -> GPU = SLOT % 4) ───────────────────────────────────
dpg_meta(){ # scope -> outdir method wb ab scale
  case "$1" in
    w4)     echo "$OUT8/dpg/$2|mllmq_mae|4|8|$SC8/mix384/${2}_w4a8.pt";;
    w4awq)  echo "$OUT8/dpg/awq|awq|4|8|$SC8/mix384/awq_w4a8.pt";;
    w4gen)  echo "$OUT8/dpg/mllmq_gen_only|mllmq_mae|4|8|$SC8/mix384/mllmq_gen_only_w4a8.pt";;
    w3awq)  echo "$OUT16/awq|awq|3|16|$SC16/unified_384/awq_w3a16.pt";;
    *)      echo "|||";;
  esac
}
dpg_run_line(){
  local line=$1 scope tag sid ns gpu meta outdir method wb ab scale logf
  IFS='|' read -r scope tag sid ns <<< "$line"
  gpu=$(( ${SLOT:-0} % 4 ))
  meta=$(dpg_meta "$scope" "$tag"); IFS='|' read -r outdir method wb ab scale <<< "$meta"
  if [ -z "$outdir" ] || [ -z "$method" ]; then
    echo "[$(date '+%F %T')] ❌ bad DPG line: $line"; return 2
  fi
  [ "$(count_png "$outdir/dpg_bench")" -ge "$DPG_TARGET" ] && return 0
  logf="$LOG/dpg_${scope}_${tag}_s${sid}.log"
  echo "[$(date '+%F %T')] DPG $scope/$tag shard $sid/$ns GPU$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/eval_generate_quant.py" \
    --model_path "$MODEL" --benchmark dpg_bench \
    --output_dir "$outdir" --num_samples 1 \
    --shard_id "$sid" --num_shards "$ns" \
    --quant_method "$method" --w_bit "$wb" --a_bit "$ab" \
    --scale_path "$scale" > "$logf" 2>&1
}
dpg_all_done(){
  [ "$(count_png "$OUT8/dpg/mllmq_t2i_img/dpg_bench")" -ge "$DPG_TARGET" ] && \
  [ "$(count_png "$OUT8/dpg/mllmq_both/dpg_bench")" -ge "$DPG_TARGET" ] && \
  [ "$(count_png "$OUT8/dpg/awq/dpg_bench")" -ge "$DPG_TARGET" ] && \
  [ "$(count_png "$OUT8/dpg/mllmq_gen_only/dpg_bench")" -ge "$DPG_TARGET" ] && \
  [ "$(count_png "$OUT16/awq/dpg_bench")" -ge "$DPG_TARGET" ]
}
run_dpg_grid(){
  local attempt q="$LOG/dpg_queue.txt"
  if dpg_all_done; then log "DPG grid already complete"; return 0; fi
  {
    echo "w3awq|awq|0|2"; echo "w3awq|awq|1|2"
    echo "w4|mllmq_t2i_img|0|2"; echo "w4|mllmq_t2i_img|1|2"
    echo "w4|mllmq_both|0|2";    echo "w4|mllmq_both|1|2"
    echo "w4awq|awq|0|2";        echo "w4awq|awq|1|2"
    echo "w4gen|mllmq_gen_only|0|2"; echo "w4gen|mllmq_gen_only|1|2"
  } > "$q"
  for attempt in 1 2 3; do
    if dpg_all_done; then break; fi
    log "DPG queue attempt $attempt (10 shard jobs, 8 slots = 2/GPU)"
    [ "$DRY_RUN" = "1" ] && return 0
    xargs -P 8 -d '\n' -n 1 --process-slot-var=SLOT bash "$SELF" --dpg-line < "$q"
    sleep 15
  done
  dpg_all_done || { log "❌ DPG grid incomplete"; return 1; }
  log "DPG grid complete"
}

# ── DPG scores (serial GPU0) ────────────────────────────────────────────────
score_one(){ # image_root out_file tag
  local imgroot=$1 out=$2 tag=$3
  [ -s "$out" ] && { log "SKIP score $tag"; return 0; }
  log "SCORE $tag GPU0"
  [ "$DRY_RUN" = "1" ] && return 0
  PYTHONPATH="$PROJ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES=0 python "$PROJ/scripts/score_dpg_coco.py" \
    --image-root-path "$imgroot" --resolution 384 --pic-num 1 \
    --csv "$DPG_CSV" --tag "$tag" --out "$out" > "$LOG/score_${tag}.log" 2>&1
  [ -s "$out" ] || { log "❌ score $tag failed"; return 1; }
  log "OK score $tag :: $(grep -i 'DPG-Bench score' "$out" | tail -1)"
}
run_scores(){
  score_one "$OUT8/dpg/mllmq_t2i_img/dpg_bench"  "$OUT8/dpg_scores/mllmq_t2i_img.txt" "w4a8_mllmq_t2i_img"
  score_one "$OUT8/dpg/mllmq_both/dpg_bench"     "$OUT8/dpg_scores/mllmq_both.txt"    "w4a8_mllmq_both"
  score_one "$OUT8/dpg/awq/dpg_bench"            "$OUT8/dpg_scores/awq.txt"           "w4a8_awq"
  score_one "$OUT8/dpg/mllmq_gen_only/dpg_bench" "$OUT8/dpg_scores/mllmq_gen_only.txt" "w4a8_mllmq_gen_only"
  score_one "$OUT16/awq/dpg_bench"               "$OUT16/dpg_scores/awq.txt"          "unified_awq"
}

# ── und / GenEval queue (7 slots while score uses GPU0) ─────────────────────
run_und(){ # outbase outroot tag method scale wb ab logdir task gpu
  local outbase=$1 outroot=$2 tag=$3 method=$4 scale=$5 wb=$6 ab=$7 logdir=$8 task=$9 gpu=${10}
  und_done "$outbase" "$task" "${tag}_w${wb}a${ab}" && { log "SKIP und $tag/$task"; return 0; }
  log "UND $tag/$task GPU$gpu"
  [ "$DRY_RUN" = "1" ] && return 0
  mkdir -p "$logdir"
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/eval_benchmarks.py" \
    --model_type janus --model_path "$MODEL" \
    --method "$method" --method_tag "$tag" \
    --w_bit "$wb" --a_bit "$ab" --w_group 128 \
    --tasks "$task" --output_base "$outbase" --device cuda:0 --log_samples \
    --scale_path "$scale" > "$logdir/${tag}_${task}.log" 2>&1
  und_done "$outbase" "$task" "${tag}_w${wb}a${ab}" \
    && { log "OK und $tag/$task"; return 0; } || { log "❌ und $tag/$task"; return 1; }
}
run_geneval(){ # outdir tag method scale logtag gpu
  local outd=$1 tag=$2 method=$3 scale=$4 ltag=$5 gpu=$6 n
  n=$(count_png "$outd/geneval")
  if [ "$n" -lt "$GENEVAL_TARGET" ]; then
    log "GENEVAL $tag GPU$gpu ($n/$GENEVAL_TARGET)"
    [ "$DRY_RUN" = "1" ] && return 0
    CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/eval_generate_quant.py" \
      --model_path "$MODEL" --benchmark geneval --output_dir "$outd" \
      --num_samples 4 --limit_per_tag 25 --quant_method "$method" --w_bit 4 --a_bit 8 \
      --scale_path "$scale" > "$LOG/geneval_${ltag}.log" 2>&1
  fi
  [ -s "$outd/geneval/results.jsonl" ] && { log "SKIP geneval $tag score"; return 0; }
  [ -n "$CLIP_BIN" ] || { log "❌ geneval $tag: CLIP missing"; return 1; }
  log "GENEVAL $tag score GPU$gpu"
  [ "$DRY_RUN" = "1" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu python "$EV/evaluate_images.py" "$outd/geneval" \
    --outfile "$outd/geneval/results.jsonl" --clip-model-path "$CLIP_BIN" \
    > "$LOG/geneval_score_${ltag}.log" 2>&1
  [ -s "$outd/geneval/results.jsonl" ] || { log "❌ geneval $tag score"; return 1; }
  log "OK geneval $tag :: $(python "$EV/summary_scores.py" "$outd/geneval/results.jsonl" 2>/dev/null | grep -i '^Overall' | tail -1)"
}
aux_run_line(){
  local line=$1 kind tag task gpu
  IFS='|' read -r kind tag task <<< "$line"
  # 6 slots -> GPU1-3 (2/GPU); slot 6 -> GPU0 (shares with the serial score)
  if [ "${SLOT:-0}" -lt 6 ]; then gpu=$(( 1 + SLOT % 3 )); else gpu=0; fi
  case "$kind" in
    und_awq)     run_und "$OUT8/und/awq" "$OUT8/und/awq" awq awq "$SC8/und128/awq_w4a8.pt" 4 8 "$LOG/und_awq" "$task" "$gpu";;
    und_w4gen)   run_und "$OUT8/und/mllmq_gen_only" "$OUT8/und/mllmq_gen_only" mllmq_gen_only mllmq "$SC8/mix384/mllmq_gen_only_w4a8.pt" 4 8 "$LOG/und_w4gen" "$task" "$gpu";;
    und_w3gen)   run_und "$UND16/mllmq_gen_only" "$UND16/mllmq_gen_only" mllmq_gen_only mllmq "$SC16/unified_generation_only/mllmq_gen_only_w3a16.pt" 3 16 "$L5H/mllmq_gen_only" "$task" "$gpu";;
    geneval_awq) run_geneval "$OUT8/geneval/awq" awq awq "$SC8/mix384/awq_w4a8.pt" awq "$gpu";;
    geneval_w4gen) run_geneval "$OUT8/geneval/mllmq_gen_only" mllmq_gen_only mllmq_mae "$SC8/mix384/mllmq_gen_only_w4a8.pt" w4gen "$gpu";;
    *) echo "[$(date '+%F %T')] ❌ bad aux line: $line"; return 2;;
  esac
}
run_aux(){
  local q="$LOG/aux_queue.txt" t
  {
    echo "geneval_awq|-|-"; echo "geneval_w4gen|-|-"
    for t in "${TASK_UND4[@]}"; do echo "und_awq|-|$t"; done
    for t in "${TASK_UND4[@]}"; do echo "und_w4gen|-|$t"; done
    for t in "${TASK_UND7[@]}"; do echo "und_w3gen|-|$t"; done
  } > "$q"
  log "aux queue: 2 geneval + 15 und jobs (7 slots)"
  [ "$DRY_RUN" = "1" ] && return 0
  xargs -P 7 -d '\n' -n 1 --process-slot-var=SLOT bash "$SELF" --aux-line < "$q"
}

# ── SEED (strictly 1 proc/GPU, 6 tags in 2 waves) ───────────────────────────
wait_other_seed(){
  if pgrep -f 'run_understanding_7\.sh' >/dev/null 2>&1; then
    log "waiting for existing SEED processes (strict 1/GPU)"
    while pgrep -f 'run_understanding_7\.sh' >/dev/null 2>&1; do sleep 60; done
  fi
}
run_seed(){ # tag method scale wb ab outbase gpu checkdir logdir
  local tag=$1 method=$2 scale=$3 wb=$4 ab=$5 outbase=$6 gpu=$7 checkdir=$8 logdir=$9
  [ "$(seed_chunks "$checkdir")" -ge 5 ] && { log "SKIP SEED $tag ($checkdir)"; return 0; }
  # Never launch a duplicate if the same tag is already running on any GPU
  # (e.g. started manually on otherwise-idle cards).
  if pgrep -f "eval_benchmarks\.py.*--output_base $checkdir" >/dev/null 2>&1; then
    log "SKIP SEED $tag (already running)"; return 0
  fi
  log "SEED $tag GPU$gpu"
  [ "$DRY_RUN" = "1" ] && return 0
  mkdir -p "$logdir"
  TAG="$tag" METHOD="$method" SCALE="$scale" GPU="$gpu" W_BIT="$wb" A_BIT="$ab" \
    OUTBASE="$outbase" LOG="$logdir" TASKS="$SEED_TASKS" \
    bash "$PROJ/scripts/run_understanding_7.sh" > "$logdir/driver.log" 2>&1
  [ "$(seed_chunks "$checkdir")" -ge 5 ] && { log "OK SEED $tag"; return 0; }
  log "❌ SEED $tag incomplete ($(seed_chunks "$checkdir")/5)"; return 0
}
run_seed_all(){
  wait_other_seed
  log "SEED wave 1/2 (1 proc/GPU)"
  run_seed rtn             rtn          ""  4 8 "$OUT8/seed" 0 "$OUT8/seed/rtn"             "$LOG/seed/rtn" &
  run_seed mllmq_both      mllmq        "$SC8/mix384/mllmq_both_w4a8.pt"      4 8 "$OUT8/seed" 1 "$OUT8/seed/mllmq_both"      "$LOG/seed/mllmq_both" &
  run_seed awq             awq          "$SC16/unified_understanding_128/awq_w3a16.pt" 3 16 "$EOUT" 2 "$EOUT/awq" "$ELOG/awq" &
  run_seed mllmq_gen_only  mllmq        "$SC16/unified_generation_only/mllmq_gen_only_w3a16.pt" 3 16 "$EOUT" 3 "$EOUT/mllmq_gen_only" "$ELOG/mllmq_gen_only" &
  wait
  log "SEED wave 2/2 (1 proc/GPU)"
  run_seed awq             awq          "$SC8/und128/awq_w4a8.pt"                4 8 "$OUT8/seed" 0 "$OUT8/seed/awq"            "$LOG/seed/awq" &
  run_seed mllmq_gen_only  mllmq        "$SC8/mix384/mllmq_gen_only_w4a8.pt"     4 8 "$OUT8/seed" 1 "$OUT8/seed/mllmq_gen_only" "$LOG/seed/mllmq_gen_only" &
  wait
}

# ── T2I-CompBench++ P1 (W4A8, 8 tags, 2 procs/GPU) ─────────────────────────
t2i_png(){ find "$1/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }
t2i_root(){ echo "$ROOT/output/janus-pro-7b/t2i_compbench_current/w4a8"; }
t2i_all_done(){
  local t root; root=$(t2i_root)
  for t in rtn mbq_scalar mllmq_channel_only mllmq_t2i_text mllmq_t2i_img mllmq_both awq mllmq_gen_only; do
    [ "$(t2i_png "$root/$t")" -ge 2400 ] || return 1
  done
}
run_t2i_one(){ # tag gpu method wb ab scale envs
  local tag=$1 gpu=$2 method=$3 wb=$4 ab=$5 scale=$6 envs=$7
  local root logd out n n2 rc
  root=$(t2i_root); logd="$LOG/t2i_w4a8"; out="$root/$tag"
  mkdir -p "$logd" "$out"
  n=$(t2i_png "$out")
  [ "$n" -ge 2400 ] && { log "SKIP T2I $tag ($n/2400)"; return 0; }
  if pgrep -f "eval_generate_quant\.py.*--output_dir $out" >/dev/null 2>&1; then
    log "SKIP T2I $tag (already running)"; return 0
  fi
  log "T2I $tag GPU$gpu (${n}/2400)${scale:+ scale=$(basename "$scale")}${envs:+ env=$envs}"
  [ "$DRY_RUN" = "1" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu env $envs python -u "$PROJ/eval_generate_quant.py" \
    --model_path "$MODEL" --benchmark t2i_compbench \
    --output_dir "$out" --num_samples 1 \
    --quant_method "$method" --w_bit "$wb" --a_bit "$ab" ${scale:+--scale_path "$scale"} \
    >> "$logd/$tag.log" 2>&1
  rc=$?
  n2=$(t2i_png "$out")
  if [ "$rc" -ne 0 ] && [ "$n2" -le "$n" ]; then
    log "FAIL T2I $tag rc=$rc no-new ($n2/2400); see $logd/$tag.log"
    touch "$logd/$tag.FAILED"
  else
    log "EXIT T2I $tag rc=$rc ($n2/2400)"
  fi
}
run_t2i_w4a8(){
  local lane="$ROOT/queue/t2i_w4a8_lane_queue.sh"
  t2i_all_done && { log "T2I W4A8 already complete"; return 0; }
  if ! pgrep -f '[t]2i_w4a8_lane_queue\.sh' >/dev/null 2>&1; then
    log "launch T2I W4A8 lane queue (per-GPU, 2 procs/card)"
    mkdir -p "$LOG/t2i_w4a8"
    setsid nohup bash "$lane" >> "$LOG/t2i_w4a8/lane_queue.log" 2>&1 < /dev/null &
    sleep 5
  fi
  # The lane queue may already be running (launched early, before SEED ended);
  # wait for it so this driver does not report completion prematurely.
  while pgrep -f '[t]2i_w4a8_lane_queue\.sh' >/dev/null 2>&1; do sleep 60; done
  t2i_all_done && log "T2I W4A8 ALL DONE" || log "❌ T2I W4A8 incomplete"
}

# ── status / completion ─────────────────────────────────────────────────────
print_status(){
  log "DPG: t2i_img=$(count_png "$OUT8/dpg/mllmq_t2i_img/dpg_bench") both=$(count_png "$OUT8/dpg/mllmq_both/dpg_bench") w4awq=$(count_png "$OUT8/dpg/awq/dpg_bench") w4gen=$(count_png "$OUT8/dpg/mllmq_gen_only/dpg_bench") w3awq=$(count_png "$OUT16/awq/dpg_bench") / $DPG_TARGET"
  log "SCORE: t2i_img=$([ -s "$OUT8/dpg_scores/mllmq_t2i_img.txt" ] && echo y || echo n) both=$([ -s "$OUT8/dpg_scores/mllmq_both.txt" ] && echo y || echo n) w4awq=$([ -s "$OUT8/dpg_scores/awq.txt" ] && echo y || echo n) w4gen=$([ -s "$OUT8/dpg_scores/mllmq_gen_only.txt" ] && echo y || echo n) w3awq=$([ -s "$OUT16/dpg_scores/awq.txt" ] && echo y || echo n)"
  log "UND4 awq=$(for t in "${TASK_UND4[@]}"; do und_done "$OUT8/und/awq" "$t" awq_w4a8 && echo -n "ok "; done) | gen=$(for t in "${TASK_UND4[@]}"; do und_done "$OUT8/und/mllmq_gen_only" "$t" mllmq_gen_only_w4a8 && echo -n "ok "; done)"
  log "UND7 w3gen=$(for t in "${TASK_UND7[@]}"; do und_done "$UND16/mllmq_gen_only" "$t" mllmq_gen_only_w3a16 && echo -n "ok "; done)"
  log "GENEVAL: awq=$(count_png "$OUT8/geneval/awq/geneval") w4gen=$(count_png "$OUT8/geneval/mllmq_gen_only/geneval") / $GENEVAL_TARGET"
  log "SEED: w4rtn=$(seed_chunks "$OUT8/seed/rtn") w4both=$(seed_chunks "$OUT8/seed/mllmq_both") w4awq=$(seed_chunks "$OUT8/seed/awq") w4gen=$(seed_chunks "$OUT8/seed/mllmq_gen_only") w3awq=$(seed_chunks "$EOUT/awq") w3gen=$(seed_chunks "$EOUT/mllmq_gen_only") / 5"
}
all_done(){
  [ -s "$SC8/und128/awq_w4a8.pt" ] && [ -s "$SC8/mix384/awq_w4a8.pt" ] && \
  [ -s "$SC8/mix384/mllmq_gen_only_w4a8.pt" ] && dpg_all_done && \
  [ -s "$OUT8/dpg_scores/mllmq_t2i_img.txt" ] && [ -s "$OUT8/dpg_scores/mllmq_both.txt" ] && \
  [ -s "$OUT8/dpg_scores/awq.txt" ] && [ -s "$OUT8/dpg_scores/mllmq_gen_only.txt" ] && \
  [ -s "$OUT16/dpg_scores/awq.txt" ] && \
  und_done "$OUT8/und/awq" mmmu_val awq_w4a8 && und_done "$OUT8/und/awq" mme_local awq_w4a8 && \
  und_done "$OUT8/und/awq" textvqa_val_local awq_w4a8 && und_done "$OUT8/und/awq" mathvista_testmini_local awq_w4a8 && \
  und_done "$OUT8/und/mllmq_gen_only" mmmu_val mllmq_gen_only_w4a8 && \
  und_done "$OUT8/und/mllmq_gen_only" mme_local mllmq_gen_only_w4a8 && \
  und_done "$OUT8/und/mllmq_gen_only" textvqa_val_local mllmq_gen_only_w4a8 && \
  und_done "$OUT8/und/mllmq_gen_only" mathvista_testmini_local mllmq_gen_only_w4a8 && \
  und_done "$UND16/mllmq_gen_only" mme_local mllmq_gen_only_w3a16 && \
  und_done "$UND16/mllmq_gen_only" textvqa_val_local mllmq_gen_only_w3a16 && \
  und_done "$UND16/mllmq_gen_only" mmmu_val mllmq_gen_only_w3a16 && \
  und_done "$UND16/mllmq_gen_only" mathvista_testmini_local mllmq_gen_only_w3a16 && \
  und_done "$UND16/mllmq_gen_only" pope_random_local mllmq_gen_only_w3a16 && \
  und_done "$UND16/mllmq_gen_only" pope_popular_local mllmq_gen_only_w3a16 && \
  und_done "$UND16/mllmq_gen_only" pope_adversarial_local mllmq_gen_only_w3a16 && \
  [ "$(count_png "$OUT8/geneval/awq/geneval")" -ge "$GENEVAL_TARGET" ] && \
  [ -s "$OUT8/geneval/awq/geneval/results.jsonl" ] && \
  [ "$(count_png "$OUT8/geneval/mllmq_gen_only/geneval")" -ge "$GENEVAL_TARGET" ] && \
  [ -s "$OUT8/geneval/mllmq_gen_only/geneval/results.jsonl" ] && \
  [ "$(seed_chunks "$OUT8/seed/rtn")" -ge 5 ] && [ "$(seed_chunks "$OUT8/seed/mllmq_both")" -ge 5 ] && \
  [ "$(seed_chunks "$OUT8/seed/awq")" -ge 5 ] && [ "$(seed_chunks "$OUT8/seed/mllmq_gen_only")" -ge 5 ] && \
  [ "$(seed_chunks "$EOUT/awq")" -ge 5 ] && [ "$(seed_chunks "$EOUT/mllmq_gen_only")" -ge 5 ] && \
  t2i_all_done
}

main(){
  log "LOCAL4 supplement start (DRY_RUN=$DRY_RUN)"
  preflight || return $?
  print_status
  build_scales || { log "❌ scale stage failed"; return 1; }
  run_dpg_grid || return 1
  run_scores & local sc=$!
  run_aux    & local au=$!
  wait $sc $au
  run_seed_all
  run_t2i_w4a8
  print_status
  if all_done; then log "LOCAL4 supplement ALL DONE"; return 0; fi
  log "LOCAL4 supplement incomplete"
  return 1
}

# child-process dispatch (xargs --process-slot-var=SLOT)
case "${1:-}" in
  --dpg-line) dpg_run_line "$2"; exit $?;;
  --aux-line) aux_run_line "$2"; exit $?;;
esac

main