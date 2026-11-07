#!/usr/bin/env bash
# ============================================================================
# AWQ supplement — LOCAL host (2 GPUs), 2026-11-05
#
#   W3A16 AWQ : finish DPG (resume 140/1065, 2 shards) -> DPG score
#               -> SEED full (5 chunks, unified_understanding_128 scale)
#   W4A8  AWQ : build scales (und128 n=128, mix384 n=384)
#               -> understand 4 tasks -> GenEval 600 -> DPG 1065 -> DPG score
#               -> SEED full (5 chunks, und128 scale)
#
#   Resource rules (same as the existing W4A8 queue):
#     - DPG / und / geneval : at most 1 extra job per GPU (=> 2 procs/GPU cap)
#     - SEED                : strictly 1 proc/GPU; waits for the 6-config SEED driver
#
#   All stages are skip-existing and safe to rerun after reboot/interrupt.
# ============================================================================
set -o pipefail

ROOT=/nas3/kangjia/mllmq
PROJ=$ROOT/MLLMQ
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
POOL_UND=$ROOT/data/coco/janus_understanding_128.json
POOL_MIX=$ROOT/data/coco/janus_mixed_tri_384.json
SC16=$ROOT/MLLMQ/scale_cache
SC8=$ROOT/MLLMQ/scale_cache/w4a8
OUT16=$ROOT/output/fixed_ablation_20261103_unified
OUT8=$ROOT/output/janus-pro-7b/w4a8_supplement
LOG=$ROOT/output/logs/awq_supplement
LOG16=$ROOT/output/logs/fixed_ablation_20261103_unified
EOUT=$ROOT/output/fixed_ablation_20261103_understanding_5h_seed
ELOG=$ROOT/output/logs/understanding_5h_seed
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
DPG_CSV=$ROOT/data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
SEED_TASKS="seedbench_2_local_chunk0 seedbench_2_local_chunk1 seedbench_2_local_chunk2 seedbench_2_local_chunk3 seedbench_2_local_chunk4"
TASK_UND=(mmmu_val mme_local textvqa_val_local mathvista_testmini_local)
DPG_TARGET=${DPG_TARGET:-1065}
GENEVAL_TARGET=${GENEVAL_TARGET:-600}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$LOG" "$LOG16" "$ELOG/awq" "$SC8/und128" "$SC8/mix384" \
         "$OUT8/und/awq" "$OUT8/geneval/awq" "$OUT8/dpg/awq" \
         "$OUT8/seed/awq" "$OUT8/dpg_scores" "$OUT16/dpg_scores"

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$EV/.ms_cache}"

cd "$PROJ" || { echo "cannot cd $PROJ"; exit 2; }

log(){ echo "[$(date '+%F %T')] $1"; }
count_png(){ find "$1" -name '*.png' 2>/dev/null | wc -l; }
seed_chunks(){ find "$1" -name '*_results.json' 2>/dev/null | sed 's/.*chunk\([0-9]\).*/\1/' | sort -u | wc -l; }
und_done(){ local t=$1; ls "$OUT8/und/awq/$t/awq_w4a8"/*_results.json >/dev/null 2>&1; }

other_seed_running(){ pgrep -f 'run_understanding_7\.sh' >/dev/null 2>&1; }
wait_other_seed(){
  if other_seed_running; then
    log "waiting for existing SEED driver (strict 1 proc/GPU)"
    while other_seed_running; do sleep 60; done
    log "existing SEED driver finished"
  fi
}

# ── W3A16 AWQ DPG (resume 2 shards) ─────────────────────────────────────────
run_w3_dpg(){
  local n attempt i
  for attempt in 1 2 3; do
    n=$(count_png "$OUT16/awq/dpg_bench")
    [ "$n" -ge "$DPG_TARGET" ] && return 0
    log "W3A16 AWQ DPG attempt $attempt ($n/$DPG_TARGET)"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi
    for i in 0 1; do
      CUDA_VISIBLE_DEVICES=$i python "$PROJ/eval_generate_quant.py" \
        --model_path "$MODEL" --benchmark dpg_bench \
        --output_dir "$OUT16/awq" --num_samples 1 \
        --shard_id "$i" --num_shards 2 \
        --quant_method awq --w_bit 3 --a_bit 16 \
        --scale_path "$SC16/unified_384/awq_w3a16.pt" \
        > "$LOG/w3_dpg_s${i}_a${attempt}.log" 2>&1 &
    done
    wait
    sleep 10
  done
  n=$(count_png "$OUT16/awq/dpg_bench")
  [ "$n" -ge "$DPG_TARGET" ] || { log "❌ W3A16 AWQ DPG incomplete ($n/$DPG_TARGET)"; return 1; }
  return 0
}

score_w3_dpg(){
  [ -s "$OUT16/dpg_scores/awq.txt" ] && { log "SKIP W3A16 AWQ DPG score"; return 0; }
  log "W3A16 AWQ DPG score"
  [ "$DRY_RUN" = "1" ] && return 0
  PYTHONPATH="$PROJ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES=0 python "$PROJ/scripts/score_dpg_coco.py" \
    --image-root-path "$OUT16/awq/dpg_bench" --resolution 384 --pic-num 1 \
    --csv "$DPG_CSV" --tag "unified_awq" --out "$OUT16/dpg_scores/awq.txt" \
    > "$LOG16/dpg_score_awq.log" 2>&1
  [ -s "$OUT16/dpg_scores/awq.txt" ] || { log "❌ W3A16 AWQ DPG score failed"; return 1; }
  log "OK W3A16 AWQ DPG score :: $(grep -i 'DPG-Bench score' "$OUT16/dpg_scores/awq.txt" | tail -1)"
}

# ── W4A8 AWQ scales ─────────────────────────────────────────────────────────
build_scale(){ # pool n gpu
  local pool=$1 n=$2 gpu=$3
  local out="$SC8/$pool/awq_w4a8.pt"
  [ -s "$out" ] && { log "SKIP W4A8 AWQ scale $pool"; return 0; }
  log "W4A8 AWQ scale $pool (n=$n) GPU$gpu"
  [ "$DRY_RUN" = "1" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/main_quant.py" \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$( [ "$pool" = und128 ] && echo "$POOL_UND" || echo "$POOL_MIX" )" \
    --image_folder "$IMG" --n_samples "$n" \
    --w_bit 4 --a_bit 8 --w_group 128 --run_process \
    --method awq --loss_mode mae \
    --scale_path "$out" > "$LOG/scale_${pool}_awq.log" 2>&1
  [ -s "$out" ] || { log "❌ W4A8 AWQ scale $pool failed"; return 1; }
  log "OK W4A8 AWQ scale $pool"
}

# ── W4A8 AWQ understanding ──────────────────────────────────────────────────
run_und(){ # task gpu
  local t=$1 gpu=$2
  und_done "$t" && { log "SKIP und awq/$t"; return 0; }
  log "UND awq/$t GPU$gpu"
  [ "$DRY_RUN" = "1" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu python "$PROJ/eval_benchmarks.py" \
    --model_type janus --model_path "$MODEL" \
    --method awq --method_tag awq \
    --w_bit 4 --a_bit 8 --w_group 128 \
    --tasks "$t" --output_base "$OUT8/und/awq" \
    --device cuda:0 --log_samples \
    --scale_path "$SC8/und128/awq_w4a8.pt" > "$LOG/und_awq_${t}.log" 2>&1
  und_done "$t" || { log "❌ und awq/$t"; return 1; }
  log "OK und awq/$t"
}
run_und_pair(){ run_und "$1" 0 & local p0=$!; run_und "$2" 1 & local p1=$!; wait $p0 $p1; }

# ── W4A8 AWQ GenEval ────────────────────────────────────────────────────────
run_geneval(){
  local outd="$OUT8/geneval/awq"
  local n; n=$(count_png "$outd/geneval")
  if [ "$n" -lt "$GENEVAL_TARGET" ]; then
    log "GENEVAL awq GPU0 ($n/$GENEVAL_TARGET)"
    [ "$DRY_RUN" = "1" ] && return 0
    CUDA_VISIBLE_DEVICES=0 python "$PROJ/eval_generate_quant.py" \
      --model_path "$MODEL" --benchmark geneval \
      --output_dir "$outd" --num_samples 4 --limit_per_tag 25 \
      --quant_method awq --w_bit 4 --a_bit 8 \
      --scale_path "$SC8/mix384/awq_w4a8.pt" > "$LOG/geneval_awq.log" 2>&1
  fi
  [ -s "$outd/geneval/results.jsonl" ] && { log "SKIP geneval awq score"; return 0; }
  [ -n "$CLIP_BIN" ] || { log "❌ geneval awq: CLIP weights missing"; return 1; }
  log "GENEVAL awq score"
  [ "$DRY_RUN" = "1" ] && return 0
  CUDA_VISIBLE_DEVICES=0 python "$EV/evaluate_images.py" "$outd/geneval" \
    --outfile "$outd/geneval/results.jsonl" --clip-model-path "$CLIP_BIN" \
    > "$LOG/geneval_score_awq.log" 2>&1
  [ -s "$outd/geneval/results.jsonl" ] || { log "❌ geneval awq score"; return 1; }
  log "OK geneval awq :: $(python "$EV/summary_scores.py" "$outd/geneval/results.jsonl" 2>/dev/null | grep -i '^Overall' | tail -1)"
}

# ── W4A8 AWQ DPG (2 shards) + score ─────────────────────────────────────────
run_w4_dpg(){
  local n attempt i
  for attempt in 1 2 3; do
    n=$(count_png "$OUT8/dpg/awq/dpg_bench")
    [ "$n" -ge "$DPG_TARGET" ] && break
    wait_other_seed
    log "W4A8 AWQ DPG attempt $attempt ($n/$DPG_TARGET)"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi
    for i in 0 1; do
      CUDA_VISIBLE_DEVICES=$i python "$PROJ/eval_generate_quant.py" \
        --model_path "$MODEL" --benchmark dpg_bench \
        --output_dir "$OUT8/dpg/awq" --num_samples 1 \
        --shard_id "$i" --num_shards 2 \
        --quant_method awq --w_bit 4 --a_bit 8 \
        --scale_path "$SC8/mix384/awq_w4a8.pt" \
        > "$LOG/w4_dpg_s${i}_a${attempt}.log" 2>&1 &
    done
    wait
    sleep 10
  done
  n=$(count_png "$OUT8/dpg/awq/dpg_bench")
  [ "$n" -ge "$DPG_TARGET" ] || { log "❌ W4A8 AWQ DPG incomplete ($n/$DPG_TARGET)"; return 1; }
}
score_w4_dpg(){
  [ -s "$OUT8/dpg_scores/awq.txt" ] && { log "SKIP W4A8 AWQ DPG score"; return 0; }
  wait_other_seed
  log "W4A8 AWQ DPG score"
  [ "$DRY_RUN" = "1" ] && return 0
  PYTHONPATH="$PROJ/scripts/pyshim${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES=0 python "$PROJ/scripts/score_dpg_coco.py" \
    --image-root-path "$OUT8/dpg/awq/dpg_bench" --resolution 384 --pic-num 1 \
    --csv "$DPG_CSV" --tag "w4a8_awq" --out "$OUT8/dpg_scores/awq.txt" \
    > "$LOG/dpg_score_awq.log" 2>&1
  [ -s "$OUT8/dpg_scores/awq.txt" ] || { log "❌ W4A8 AWQ DPG score failed"; return 1; }
  log "OK W4A8 AWQ DPG score :: $(grep -i 'DPG-Bench score' "$OUT8/dpg_scores/awq.txt" | tail -1)"
}

# ── AWQ SEED full (final stage, 1 proc/GPU) ─────────────────────────────────
run_seed_w3(){
  [ "$(seed_chunks "$EOUT/awq")" -ge 5 ] && { log "SKIP W3A16 AWQ SEED"; return 0; }
  log "SEED W3A16 AWQ GPU0"
  [ "$DRY_RUN" = "1" ] && return 0
  TAG=awq METHOD=awq SCALE="$SC16/unified_understanding_128/awq_w3a16.pt" GPU=0 \
    W_BIT=3 A_BIT=16 OUTBASE="$EOUT" LOG="$ELOG/awq" TASKS="$SEED_TASKS" \
    bash "$PROJ/scripts/run_understanding_7.sh" > "$LOG/seed_w3_awq.log" 2>&1
}
run_seed_w4(){
  [ "$(seed_chunks "$OUT8/seed/awq")" -ge 5 ] && { log "SKIP W4A8 AWQ SEED"; return 0; }
  log "SEED W4A8 AWQ GPU1"
  [ "$DRY_RUN" = "1" ] && return 0
  TAG=awq METHOD=awq SCALE="$SC8/und128/awq_w4a8.pt" GPU=1 \
    W_BIT=4 A_BIT=8 OUTBASE="$OUT8/seed" LOG="$LOG/seed/awq" TASKS="$SEED_TASKS" \
    bash "$PROJ/scripts/run_understanding_7.sh" > "$LOG/seed_w4_awq.log" 2>&1
}

# ── status helpers (shared with the scheduler) ──────────────────────────────
w3_done(){ [ "$(count_png "$OUT16/awq/dpg_bench")" -ge "$DPG_TARGET" ] && [ -s "$OUT16/dpg_scores/awq.txt" ] && [ "$(seed_chunks "$EOUT/awq")" -ge 5 ]; }
w4_done(){
  [ -s "$SC8/und128/awq_w4a8.pt" ] && [ -s "$SC8/mix384/awq_w4a8.pt" ] || return 1
  for t in "${TASK_UND[@]}"; do und_done "$t" || return 1; done
  [ "$(count_png "$OUT8/geneval/awq/geneval")" -ge "$GENEVAL_TARGET" ] && [ -s "$OUT8/geneval/awq/geneval/results.jsonl" ] || return 1
  [ "$(count_png "$OUT8/dpg/awq/dpg_bench")" -ge "$DPG_TARGET" ] && [ -s "$OUT8/dpg_scores/awq.txt" ] || return 1
  [ "$(seed_chunks "$OUT8/seed/awq")" -ge 5 ] || return 1
}

print_status(){
  log "status: W3 dpg=$(count_png "$OUT16/awq/dpg_bench")/$DPG_TARGET score=$([ -s "$OUT16/dpg_scores/awq.txt" ] && echo yes || echo no) seed_chunks=$(seed_chunks "$EOUT/awq")/5"
  log "status: W4 scales und128=$([ -s "$SC8/und128/awq_w4a8.pt" ] && echo yes || echo no) mix384=$([ -s "$SC8/mix384/awq_w4a8.pt" ] && echo yes || echo no) und=$(for t in "${TASK_UND[@]}"; do und_done "$t" && echo -n "$t,OK "; done) geneval=$(count_png "$OUT8/geneval/awq/geneval")/$GENEVAL_TARGET dpg=$(count_png "$OUT8/dpg/awq/dpg_bench")/$DPG_TARGET score=$([ -s "$OUT8/dpg_scores/awq.txt" ] && echo yes || echo no) seed_chunks=$(seed_chunks "$OUT8/seed/awq")/5"
}

main(){
  log "AWQ supplement start (DRY_RUN=$DRY_RUN)"
  print_status

  run_w3_dpg          || return 1
  score_w3_dpg        || return 1
  build_scale und128 128 0 & local p0=$!
  build_scale mix384 384 1 & local p1=$!
  wait $p0 $p1

  run_und_pair mmmu_val mme_local
  run_und_pair textvqa_val_local mathvista_testmini_local
  run_geneval         || return 1

  run_w4_dpg          || return 1
  score_w4_dpg        || return 1

  wait_other_seed
  run_seed_w3 & local s0=$!
  run_seed_w4 & local s1=$!
  wait $s0 $s1

  print_status
  if w3_done && w4_done; then log "AWQ supplement ALL DONE"; return 0; fi
  log "AWQ supplement incomplete"
  return 1
}

main