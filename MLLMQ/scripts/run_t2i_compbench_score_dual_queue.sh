#!/bin/bash
# ============================================================================
# T2I-CompBench 评分 · 双队列（BLIP×3 + CLIP×1，仅 4/8 类可跑）
#
# 范围：13 config × {color, shape, texture} (BLIP VQA) + {non_spatial} (CLIP Score)
#       = 13 × 4 = 52 jobs
#   🔴 spatial/3d_spatial/numeracy (UniDet) 仍需 detectron2 + 3 个 expert 权重 → 未纳入
#   🔴 complex 需 3-in-1（官方硬编码 num=10）→ 未纳入
#
# 底层：复用 scripts/smoke_t2i_compbench_score.py（--tag-dir / --categories / --device）
#   · 官方脚本按文件名反解 prompt（file_name.split('_')[0]）→ 每类必须单独跑
#   · BLIP 8 个 noun-phrase 通道、每通道重载一次模型 → 单 job ~20-30min
#   · 需要 bert-base-uncased tokenizer（离线）→ 已置于 NFS，经 HF_HOME 暴露
#
# 产出：RES=<...>/t2i_compbench_scores/<tag>/<category>.json（含 score）
#       RES/work/<tag>/<category>/ 为官方产物 + _run.log
#       skip-existing：<tag>/<category>.json 已存在且有 score 则跳过
#
# Usage: cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_t2i_compbench_score_dual_queue.sh \
#          > /nas3/kangjia/mllmq/output/logs/t2iscore_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#        只看计划： PLAN=1 bash scripts/run_t2i_compbench_score_dual_queue.sh
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
PY=/nas3/kangjia/mllmq/venvs/t2icomp/bin/python
SMOKE=$ROOT/scripts/smoke_t2i_compbench_score.py
OUT=$ROOT/output/janus-pro-7b/t2i_compbench
RES=$ROOT/output/janus-pro-7b/t2i_compbench_scores
WORK=$RES/work
LOG=$ROOT/output/logs

export HF_HOME=/nas3/kangjia/mllmq/checkpoints/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

COOLDOWN=20
GPU_FREE_MIB=6000        # 平台保活 ~1663MiB；真实任务 ≥6GB
POLL=60
PLAN=${PLAN:-0}
GPUS="0 1"

CATS="color shape texture non_spatial"
TAGS="fp16 rtn_w3 awq_w3 mbq_block_w3 mbq_scalar_w3 mllmq_mae_w3 mllmq_mse_w3 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4"

# CLIP 权重预热：clip.load("ViT-B/32") 只认 ~/.cache/clip（无 env 覆盖），容器重启会清空 → 自愈
stage_clip() {
  local dst="$HOME/.cache/clip/ViT-B-32.pt"
  local src="$ROOT/checkpoints/t2i_compbench/clip/ViT-B-32.pt"
  if [ ! -s "$dst" ]; then
    mkdir -p "$(dirname "$dst")"
    [ -s "$src" ] && cp "$src" "$dst" && log "已预热 CLIP 权重 → $dst" || warn "缺 $src，CLIP 类将失败"
  fi
}

QFILE=/tmp/t2iscore_queue.$$
QLOCK=/tmp/t2iscore_queue.lock
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

# 已有分：<tag>/<category>.json 且 score 非 null
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

build_queue() {   # 只排「尚无分」的 job（skip-existing）
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
  local logf="$LOG/t2iscore_${tag}_${cat}.log"
  info "[GPU$gpu] RUN $tag/$cat (n=$n)"
  "$PY" "$SMOKE" --tag-dir "$OUT/$tag" --work "$w" \
        --categories "$cat" --device "cuda:$gpu" --force > "$logf" 2>&1
  local rc=$?
  if [ -f "$w/_smoke_summary.json" ]; then
    cp "$w/_smoke_summary.json" "$RES/$tag/$cat.json"
  fi
  if already_scored "$tag" "$cat"; then
    good "[GPU$gpu] DONE $tag/$cat — $(grep -o '"score": [^,]*' "$RES/$tag/$cat.json" | head -1)"
    return 0
  fi
  fail "[GPU$gpu] FAIL $tag/$cat rc=$rc"; tail -5 "$logf" 2>/dev/null
  return 1
}

lane() {      # $1=gpu
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
log "===== T2I-CompBench 评分 · 双队列（BLIP×3 + CLIP×1）====="
log "范围：13 config × {$CATS} = 52 jobs；UniDet 3 类 + complex 未纳入"
mkdir -p "$RES" "$WORK"

echo ""; log "── 已出分现状 ──"
for t in $TAGS; do printf "   %-18s %s\n" "$t" "$(for c in $CATS; do s=·; already_scored "$t" "$c" && s=$(grep -o '"score": [^,]*' "$RES/$t/$c.json" 2>/dev/null|head -1|cut -d' ' -f2); printf "%s=%s " "$c" "$s"; done)"; done
echo ""

# 依赖自检
for f in "$SMOKE" "$PY"; do [ -e "$f" ] || { fail "缺 $f"; exit 2; }; done
[ -d "$ROOT/data/benchmarks_gen/T2I-CompBench-main" ] || { fail "缺 T2I-CompBench 仓库"; exit 2; }
stage_clip
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
  if [ "$PLAN" = "1" ]; then info "PLAN=1 → 仅打印计划，未执行。"; exit 0; fi
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
if [ "$done_n" -eq "$tot" ]; then info "🎉 4 类 × 13 config 全部出分（$done_n/$tot）"; else fail "出分 $done_n/$tot —— 重跑本脚本即可续"; fi
rm -f "$QFILE"