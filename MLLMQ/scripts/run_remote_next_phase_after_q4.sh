#!/usr/bin/env bash
# ============================================================================
# 副机第二阶段启动器：等「4-GPU 第一阶段队列」结束后，再执行官方第二阶段脚本。
#
# 背景：scripts/run_remote_next_phase.sh 内部用
#   pgrep -f "run_remote_priority_queue.sh"
# 作为第一阶段探针，但本机第一阶段已改用 run_remote_priority_queue_4gpu.sh，
# 该探针识别不到。此包装器只是补上等待，随后原样 exec 官方脚本（不修改它）。
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
log(){ echo -e "[$(date '+%F %T')] $1"; }

if pgrep -f '[r]un_remote_priority_queue_4gpu.sh' >/dev/null 2>&1; then
  log "第一阶段 4-GPU 队列仍在跑，等它结束..."
  while pgrep -f '[r]un_remote_priority_queue_4gpu.sh' >/dev/null 2>&1; do sleep 120; done
  log "第一阶段结束 → 启动第二阶段"
else
  log "未检测到第一阶段 4-GPU 队列，直接启动第二阶段"
fi

exec bash scripts/run_remote_next_phase.sh