#!/bin/bash
# ============================================================================
# T2I-CompBench 评分 · UniDet 双队列（spatial / 3d_spatial / numeracy）
#
# 环境：独立 conda env **venvs/t2idet**（torch2.4.0+cu121 + detectron2 0.6 自己编译）
#       故意与已跑通的 BLIP/CLIP 环境 venvs/t2icomp 隔离（见 dialog_1027_detectron2.md）
# 范围：13 config × {spatial(2D), 3d_spatial, numeracy} = 39 jobs
# 产出：RES=<...>/t2i_compbench_scores/<tag>/<category>.json（含 score）
#
# 🔴 关键：UniDet 官方脚本**不认 T2I_DEVICE**（detectron2 DefaultPredictor 用 cfg.MODEL.DEVICE，
#    默认 cuda:0）→ 必须用 CUDA_VISIBLE_DEVICES 把子进程钉到指定卡，否则两 lane 抢 GPU0。
#
# Usage: cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_t2i_compbench_score_unidet_dual_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/t2iuscore_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#        只看计划： PLAN=1 bash ...
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
PY=$ROOT/venvs/t2idet/bin/python
SMOKE=$ROOT/scripts/smoke_t2i_compbench_score.py
OUT=$ROOT/output/janus-pro-7b/t2i_compbench
RES=$ROOT/output/janus-pro-7b/t2i_compbench_scores
WORK=$RES/work_u
LOG=$ROOT/output/logs

export HF_HOME=/nas3/kangjia/mllmq/checkpoints/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

COOLDOWN=15
GPU_FREE_MIB=6000
POLL=60
PLAN=${PLAN:-0}
GPUS="0 1"

CATS="spatial 3d_spatial numeracy"
TAGS="fp16 rtn_w3 awq_w3 mbq_block_w3 mbq_scalar_w3 mllmq_mae_w3 mllmq_mse_w3 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4"

QFILE=/tmp/t2iuscore_queue.$$
QLOCK=/tmp/t2iuscore_queue.lock
DRIVER_PID=$$

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date '+%F %T')] $1"; }
good() { echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date '+%F %T')] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

cleanup() { warn "Interrupted — killing children..."; pkill -P $DRIVER_PID 2>/dev/null; log "Driver exiting."; exit 130; }
trap cleanup SIGINT SIGTERM

count_samples() { find "${OUT}/$1/t2i_compbench/$2/samples" -name '*.png' 2>/dev/null | wc -l; }
gpu_used() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

already_scored() {
  local f="$RES/$1/$2.json"
  [ -f "$f" ] || return 1
  "$PY" - "$f" <<'PYEOF' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]))
rows=d.get("rows",[])
sys.exit(0 if rows and rows[0].get("score") is not None else 1)
PYEOF
}

build_queue() {
  : > "$QFILE"
  local t c
  for t in $TAGS; do
    for c in $CATS; do
      already_scored "$t" "$c" && continue
      echo "$t $c" >> "$QFILE"
    done
  done
}

pop_job() {
  local j
  exec 9>"$QLOCK"; flock 9
  j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp" 2>/dev/null; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-
  echo "$j"
}

run_job() {   # $1=tag  $2=cat  $3=gpu
  local tag=$1 cat=$2 gpu=$3 n
  n=$(count_samples "$tag" "$cat")
  if [ "$n" -eq 0 ]; then warn "[GPU$gpu] SKIP $tag/$cat — 无样本"; return 0; fi
  if already_scored "$tag" "$cat"; then good "[GPU$gpu] SKIP $tag/$cat — 已有分"; return 0; fi

  sleep $COOLDOWN
  mkdir -p "$RES/$tag"
  local w="$WORK/$tag/$cat"
  local logf="$LOG/t2iuscore_${tag}_${cat}.log"
  info "[GPU$gpu] RUN $tag/$cat (n=$n)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$SMOKE" --tag-dir "$OUT/$tag" --work "$w" \
        --categories "$cat" --device cuda:0 --python "$PY" --force > "$logf" 2>&1
  local rc=$?
  [ -f "$w/_smoke_summary.json" ] && cp "$w/_smoke_summary.json" "$RES/$tag/$cat.json"
  if already_scored "$tag" "$cat"; then
    good "[GPU$gpu] DONE $tag/$cat — $(grep -o '\"score\": [^,]*' "$RES/$tag/$cat.json" | head -1)"
    return 0
  fi
  fail "[GPU$gpu] FAIL $tag/$cat rc=$rc"; tail -5 "$logf" 2>/dev/null
  return 1
}

lane() {
  local gpu=$1 job tag cat u
  while :; do
    while :; do
      u=$(gpu_used "$gpu"); [ "${u:-0}" -lt "$GPU_FREE_MIB" ] && break
      log "[GPU$gpu] 忙（used=${u}MiB）→ ${POLL}s 后再探"; sleep $POLL
    done
    job=$(pop_job)
    [ -z "$job" ] && { log "[GPU$gpu] 队列已空 → lane 退出"; return 0; }
    tag=${job%% *}; cat=${job##* }
    run_job "$tag" "$cat" "$gpu"
  done
}

# ── main ──
log "===== T2I-CompBench UniDet 评分 · 双队列（spatial/3d_spatial/numeracy）====="
log "环境：$PY"
mkdir -p "$RES" "$WORK"

for f in "$SMOKE" "$PY"; do [ -e "$f" ] || { fail "缺 $f"; exit 2; }; done
"$PY" -c "import detectron2" 2>/dev/null || { fail "venvs/t2idet 无 detectron2"; exit 2; }
for g in 0 1; do
  if CUDA_VISIBLE_DEVICES=$g timeout -s 9 60 "$PY" -c \
       "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize()" >/dev/null 2>&1; then
    good "GPU${g} CUDA 上下文自检通过"
  else
    fail "GPU${g} 无法创建 CUDA 上下文 → 拒绝启动"; exit 2
  fi
done

for pass in 1 2 3; do
  build_queue
  total=$(wc -l < "$QFILE")
  if [ "$total" -eq 0 ]; then good "无待评 job（全部已出分）"; break; fi
  log "── pass $pass：待评 $total jobs ──"
  if [ "$PLAN" = "1" ]; then info "PLAN=1 → 仅打印计划，未执行。"; head -5 "$QFILE"; exit 0; fi
  for g in $GPUS; do lane "$g" & done
  wait
done

echo ""; log "════════ 最终清点 ════════"
done_n=0; tot=0
for t in $TAGS; do
  line="$t:"
  for c in $CATS; do
    tot=$((tot+1))
    if already_scored "$t" "$c"; then done_n=$((done_n+1)); line="$line $c=✅"; else line="$line $c=❌"; fi
  done
  log "$line"
done
if [ "$done_n" -eq "$tot" ]; then info "🎉 UniDet 3 类 × 13 config 全部出分（$done_n/$tot）"; else fail "出分 $done_n/$tot —— 重跑本脚本即可续"; fi
rm -f "$QFILE"