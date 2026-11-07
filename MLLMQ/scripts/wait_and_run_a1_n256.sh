#!/usr/bin/env bash
# ============================================================================
# 守夜接跑器：等 ★正确对照 与 A13 都结束后，再启动 A1（mbq_block 全量 n256）。
#   A1 driver 已内置 skip（n64/n128 有 results.jsonl → SKIP），n256 从 696 张续跑。
#   目的：不在核心对照期间抢 GPU。
#
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/wait_and_run_a1_n256.sh \
#     > /nas3/kangjia/mllmq/output/logs/a1_waiter_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

log() { echo "[$(date '+%F %T')] $1"; }

if pgrep -f "run_mllmq_correct_comparison.sh" >/dev/null 2>&1; then
  log "等 ★正确对照 结束..."
  while pgrep -f "run_mllmq_correct_comparison.sh" >/dev/null 2>&1; do sleep 60; done
  log "正确对照已结束"
fi

if pgrep -f "run_ablation_a13_t2i_rw.sh" >/dev/null 2>&1; then
  log "等 A13 结束..."
  while pgrep -f "run_ablation_a13_t2i_rw.sh" >/dev/null 2>&1; do sleep 60; done
  log "A13 已结束"
fi

log "→ 启动 A1（mbq_block 全量，n256 续跑）"
bash scripts/run_ablation_a1_full.sh