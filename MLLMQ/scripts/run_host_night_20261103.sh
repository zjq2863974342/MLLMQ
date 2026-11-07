#!/usr/bin/env bash
# ============================================================================
# 主机夜间队列（2026-11-03）：等 s2 收尾后依次跑
#   Wave 1: mllmq_channel_only (GPU0) + mbq_scalar (GPU1)   理解侧 7 项
#   Wave 2: mllmq_both         (GPU0) + mbq_block  (GPU1)   理解侧 7 项
#   Wave 3: k12 full seed-3（GPU0+1，2 分片）
# 全部 skip-existing；只写 output/fixed_ablation_20261103/ 与
# output/janus-pro-7b/ablation_mixed_full/w34_spread_k12_full_s3/。
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_host_night_20261103.sh \
#     > ../output/logs/host_night_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
LOG=$ROOT/output/logs
SC=$ROOT/MLLMQ/scale_cache/fixed_ablation
S2DIR=$ROOT/output/janus-pro-7b/ablation_mixed_full/w34_spread_k12_full_s2/geneval

log(){ echo "[$(date '+%F %T')] $1"; }

# ── 0. 等 s2（driver 退出）────────────────────────────────────────────────
log "等待 s2 driver 退出 ..."
n=0
while pgrep -f "bash scripts/run_exp3_full_s2.sh" >/dev/null; do
  sleep 60; n=$((n+1))
  [ $((n % 10)) -eq 0 ] && log "  仍在跑 s2（${n}min）"
  [ "$n" -ge 120 ] && { log "⚠️ s2 超过 2h 未结束，继续夜间队列"; break; }
done
if [ -f "$S2DIR/results.jsonl" ]; then
  log "✅ s2 已评分：$(python "$ROOT/data/benchmarks_gen/geneval/evaluation/summary_scores.py" "$S2DIR/results.jsonl" 2>/dev/null | grep -i '^Overall')"
else
  log "⚠️ s2 无 results.jsonl（可能失败），仍继续夜间队列"
fi

# ── helper ────────────────────────────────────────────────────────────────
run_cfg(){ # tag method scale gpu
  TAG="$1" METHOD="$2" SCALE="$3" GPU="$4" \
    bash scripts/run_fixed_understanding.sh > "$LOG/night_und_$1.log" 2>&1
}

wave(){ # tag0 method0 scale0 tag1 method1 scale1
  log "===== Wave: $1 + $4 ====="
  run_cfg "$1" "$2" "$3" 0 &
  p0=$!
  run_cfg "$4" "$5" "$6" 1 &
  p1=$!
  wait $p0; r0=$?
  wait $p1; r1=$?
  log "Wave 完成：$1(rc=$r0) + $4(rc=$r1)"
}

# ── 1/2. 两波理解侧 ───────────────────────────────────────────────────────
wave mllmq_channel_only mllmq "$SC/mllmq_channel_only.pt" \
     mbq_scalar         mbq   "$SC/mbq_scalar.pt"

wave mllmq_both         mllmq "$SC/mllmq_both.pt" \
     mbq_block          mbq   "$SC/mbq_block.pt"

# ── 3. k12 full seed-3 ────────────────────────────────────────────────────
log "===== k12 full seed-3（2 分片）====="
bash scripts/run_exp3_full_s3.sh > "$LOG/host_night_s3_driver.log" 2>&1
log "s3 完成：$(ls "$ROOT/output/janus-pro-7b/ablation_mixed_full/w34_spread_k12_full_s3/geneval/results.jsonl" 2>/dev/null || echo '无 results.jsonl')"

log "🏁 夜间队列结束"