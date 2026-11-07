#!/bin/bash
# ============================================================================
# T2I-CompBench complex（3-in-1）评分 · 双队列
#
# 范围：13 config × {complex} = 13 jobs
# 每 job：BLIP(attribute,t2icomp) + CLIP(action,--complex,t2icomp) + UniDet-2D(--complex,t2idet)
#          → 按官方 3_in_1 逻辑聚合（num 改 1 适配；官方硬编码 10）
# 底层：scripts/score_t2i_compbench_complex.py
#
# ⚠️ 口径：本实验每 prompt 1 张；官方 10 张 → 结果不可与官方直接对标，仅内部横向可比
#
# 产出：RES=<...>/t2i_compbench_scores/<tag>/complex.json（含 score）
#       WORK=<...>/t2i_compbench_scores/work_cx/<tag>/ 官方产物 + _run.log
#
# Usage: cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_t2i_compbench_score_complex_dual_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/t2icxscore_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#        只看计划： PLAN=1 bash ...
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
PY=$ROOT/venvs/t2idet/bin/python
JOB=$ROOT/scripts/score_t2i_compbench_complex.py
OUT=$ROOT/output/janus-pro-7b/t2i_compbench
RES=$ROOT/output/janus-pro-7b/t2i_compbench_scores
WORK=$RES/work_cx
LOG=$ROOT/output/logs

export HF_HOME=/nas3/kangjia/mllmq/checkpoints/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# CLIP 权重预热（clip.load 只认 ~/.cache/clip，无 env 覆盖；容器重启会清空 → 自愈）
stage_clip() {
  local dst="$HOME/.cache/clip/ViT-B-32.pt" src="$ROOT/checkpoints/t2i_compbench/clip/ViT-B-32.pt"
  if [ ! -s "$dst" ]; then
    mkdir -p "$(dirname "$dst")"
    [ -s "$src" ] && cp "$src" "$dst" && log "已预热 CLIP 权重 → $dst" || warn "缺 $src"
  fi
}

COOLDOWN=15
GPU_FREE_MIB=6000
POLL=60
PLAN=${PLAN:-0}
GPUS="0 1"

TAGS="fp16 rtn_w3 awq_w3 mbq_block_w3 mbq_scalar_w3 mllmq_mae_w3 mllmq_mse_w3 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4"
CAT=complex

QFILE=/tmp/t2icx_queue.$$
QLOCK=/tmp/t2icx_queue.lock
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

already_scored() {   # $1=tag
  local f="$RES/$1/$CAT.json"
  [ -f "$f" ] || return 1
  "$PY" - "$f" <<'PYEOF' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get("score") is not None else 1)
PYEOF
}

build_queue() {
  : > "$QFILE"
  local t
  for t in $TAGS; do already_scored "$t" && continue; echo "$t" >> "$QFILE"; done
}

pop_job() {
  local j
  exec 9>"$QLOCK"; flock 9
  j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp" 2>/dev/null; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-
  echo "$j"
}

run_job() {   # $1=tag  $2=gpu
  local tag=$1 gpu=$2 n
  n=$(count_samples "$tag" "$CAT")
  if [ "$n" -eq 0 ]; then warn "[GPU$gpu] SKIP $tag/$CAT — 无样本"; return 0; fi
  if already_scored "$tag"; then good "[GPU$gpu] SKIP $tag/$CAT — 已有分"; return 0; fi

  sleep $COOLDOWN
  mkdir -p "$RES/$tag"
  local w="$WORK/$tag" logf="$LOG/t2icxscore_${tag}_${CAT}.log"
  info "[GPU$gpu] RUN $tag/$CAT (n=$n)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$JOB" --tag-dir "$OUT/$tag" --work "$w" --force > "$logf" 2>&1
  local rc=$?
  [ -f "$w/_summary.json" ] && cp "$w/_summary.json" "$RES/$tag/$CAT.json"
  if already_scored "$tag"; then
    good "[GPU$gpu] DONE $tag/$CAT — $(grep -o '"score": [^,]*' "$RES/$tag/$CAT.json" | head -1)"
    return 0
  fi
  fail "[GPU$gpu] FAIL $tag/$CAT rc=$rc"; tail -5 "$logf" 2>/dev/null
  return 1
}

lane() {
  local gpu=$1 job u
  while :; do
    while :; do
      u=$(gpu_used "$gpu"); [ "${u:-0}" -lt "$GPU_FREE_MIB" ] && break
      log "[GPU$gpu] 忙（used=${u}MiB）→ ${POLL}s 后再探"; sleep $POLL
    done
    job=$(pop_job)
    [ -z "$job" ] && { log "[GPU$gpu] 队列已空 → lane 退出"; return 0; }
    run_job "$job" "$gpu"
  done
}

# ── main ──
log "===== T2I-CompBench complex 评分 · 双队列（3-in-1）====="
log "环境：$PY（unidet）+ $ROOT/venvs/t2icomp（blip/clip）"
mkdir -p "$RES" "$WORK"
stage_clip

for f in "$JOB" "$PY" "$ROOT/venvs/t2icomp/bin/python"; do [ -e "$f" ] || { fail "缺 $f"; exit 2; }; done
"$ROOT/venvs/t2idet/bin/python" -c "import detectron2" 2>/dev/null || { fail "venvs/t2idet 无 detectron2"; exit 2; }
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
  tot=$((tot+1))
  if already_scored "$t"; then done_n=$((done_n+1)); good "$t: complex=✅"; else fail "$t: complex=❌"; fi
done
if [ "$done_n" -eq "$tot" ]; then info "🎉 complex × 13 config 全部出分（$done_n/$tot）"; else fail "出分 $done_n/$tot —— 重跑本脚本即可续"; fi
rm -f "$QFILE"