#!/usr/bin/env bash
# ============================================================================
# Score ONE current T2I-CompBench tag on ONE GPU (all 8 categories).
#
# Usage:
#   WIDTH=w3a16 TAG=rtn GPU=0 bash queue/t2i_current_score_one.sh
#   WIDTH=w4a8  TAG=rtn GPU=0 WAIT_PNG=2400 bash queue/t2i_current_score_one.sh
#
# WIDTH=w3a16 -> output/janus-pro-7b/t2i_compbench_current/<tag>
# WIDTH=w4a8  -> output/janus-pro-7b/t2i_compbench_current/w4a8/<tag>
# Scores     -> output/janus-pro-7b/t2i_compbench_current_scores/<width>/<tag>/<cat>.json
#
# Categories: color/shape/texture (BLIP), non_spatial (CLIP),
#             spatial/3d_spatial/numeracy (UniDet), complex (3-in-1)
# skip-existing: a category with score != null is skipped.
# ============================================================================
set -o pipefail

ROOT=/nas3/kangjia/mllmq
WIDTH=${WIDTH:?set WIDTH=w3a16|w4a8}
TAG=${TAG:?set TAG}
GPU=${GPU:?set GPU}
WAIT_PNG=${WAIT_PNG:-2400}
TARGET_PER_CAT=${TARGET_PER_CAT:-300}

case "$WIDTH" in
  w3a16) IN="$ROOT/output/janus-pro-7b/t2i_compbench_current/$TAG" ;;
  w4a8)  IN="$ROOT/output/janus-pro-7b/t2i_compbench_current/w4a8/$TAG" ;;
  *) echo "bad WIDTH=$WIDTH (w3a16|w4a8)"; exit 2 ;;
esac

RES="$ROOT/output/janus-pro-7b/t2i_compbench_current_scores/$WIDTH/$TAG"
WORK="$RES/work"
LOGD="$ROOT/output/logs/t2i_compbench_current"
mkdir -p "$RES" "$WORK" "$LOGD"

export HF_HOME="$ROOT/checkpoints/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME="$ROOT/checkpoints/torch_home"

PYC="$ROOT/venvs/t2icomp/bin/python"
PYD="$ROOT/venvs/t2idet/bin/python"
SMOKE="$ROOT/scripts/smoke_t2i_compbench_score.py"
CX="$ROOT/scripts/score_t2i_compbench_complex.py"

log(){ echo "[$(date '+%F %T')] [score $WIDTH/$TAG GPU$GPU] $*"; }
count_all(){ find "$IN/t2i_compbench" -name '*.png' 2>/dev/null | wc -l; }
count_cat(){ find "$IN/t2i_compbench/$1/samples" -name '*.png' 2>/dev/null | wc -l; }

wait_ready(){
  local n ok c
  while :; do
    n=$(count_all)
    if [ "$WAIT_PNG" -le 0 ] || [ "$n" -ge "$WAIT_PNG" ]; then
      ok=1
      for c in color shape texture spatial non_spatial complex 3d_spatial numeracy; do
        [ "$(count_cat "$c")" -ge "$TARGET_PER_CAT" ] || ok=0
      done
      if [ "$ok" = 1 ]; then log "samples ready ($n total)"; return 0; fi
    fi
    log "wait samples ($n/$WAIT_PNG)"
    sleep 120
  done
}

has_score(){ # <category>
  local f="$RES/$1.json"
  [ -s "$f" ] || return 1
  "$PYD" - "$f" <<'PYEOF' >/dev/null 2>&1
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("score") is not None:
    sys.exit(0)
rows = d.get("rows", [])
sys.exit(0 if rows and rows[0].get("score") is not None else 1)
PYEOF
}

run_smoke(){ # <python> <category>
  local py=$1
  local cat=$2
  local w="$WORK/$cat"
  local lf="$LOGD/score_${WIDTH}_${TAG}_${cat}.log"
  local rc
  if has_score "$cat"; then log "SKIP $cat (existing)"; return 0; fi
  log "RUN $cat"
  CUDA_VISIBLE_DEVICES="$GPU" "$py" "$SMOKE" \
      --tag-dir "$IN" --work "$w" --categories "$cat" \
      --device cuda:0 --python "$py" --force > "$lf" 2>&1
  rc=$?
  [ -s "$w/_smoke_summary.json" ] && cp "$w/_smoke_summary.json" "$RES/$cat.json"
  if has_score "$cat"; then
    log "DONE $cat $(grep -o '"score": [^,]*' "$RES/$cat.json" 2>/dev/null | head -1)"
    return 0
  fi
  log "FAIL $cat rc=$rc (see $lf)"
  return 1
}

run_complex(){
  local w="$WORK/complex" lf="$LOGD/score_${WIDTH}_${TAG}_complex.log" rc
  if has_score complex; then log "SKIP complex (existing)"; return 0; fi
  log "RUN complex"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYD" "$CX" \
      --tag-dir "$IN" --work "$w" --num 1 --force > "$lf" 2>&1
  rc=$?
  [ -s "$w/_summary.json" ] && cp "$w/_summary.json" "$RES/complex.json"
  if has_score complex; then
    log "DONE complex $(grep -o '"score": [^,]*' "$RES/complex.json" 2>/dev/null | head -1)"
    return 0
  fi
  log "FAIL complex rc=$rc (see $lf)"
  return 1
}

main(){
  if [ "${PLAN:-0}" = "1" ]; then
    log "PLAN in=$IN"
    log "PLAN out=$RES"
    for c in color shape texture spatial non_spatial complex 3d_spatial numeracy; do
      has_score "$c" && log "  $c: scored" || log "  $c: TODO ($(count_cat "$c") samples)"
    done
    return 0
  fi
  exec 9>"$RES/.score.lock"
  flock -n 9 || { log "another scorer for this tag is running"; return 0; }
  wait_ready
  local rc=0
  run_smoke "$PYC" color       || rc=1
  run_smoke "$PYC" shape       || rc=1
  run_smoke "$PYC" texture     || rc=1
  run_smoke "$PYC" non_spatial || rc=1
  run_smoke "$PYD" spatial     || rc=1
  run_smoke "$PYD" 3d_spatial  || rc=1
  run_smoke "$PYD" numeracy    || rc=1
  run_complex                  || rc=1

  local ok=0 c
  for c in color shape texture spatial non_spatial complex 3d_spatial numeracy; do
    has_score "$c" && ok=$((ok+1))
  done
  log "score complete $ok/8 -> $RES"
  [ "$ok" -eq 8 ] && return 0
  return 1
}

main
exit $?