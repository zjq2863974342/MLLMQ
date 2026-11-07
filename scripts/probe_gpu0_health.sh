#!/bin/bash
# GPU0 健康探测（2026-10-09 事故后）
# 背景: 平台重启实例后 nvidia-smi idx0 (PCI 48:00.0) 无法创建 CUDA 上下文
#       ("CUDA-capable device(s) is/are busy or unavailable")，而 nvidia-smi 常规查询全绿，
#       所以只能用真实 CUDA 分配来探测恢复。idx1 (4d:00.0) 健康。
#
# 🔴 安全护栏（dialog.md 重点注意事项 §0）: 探测会创建 CUDA 上下文，
#    而"其他 CUDA 进程满载运行时并发建上下文"正是 2026-10-09 11:48 驱动全局锁挂死的触发形态。
#    因此本脚本只要发现任何 GPU 任务在跑就立即退出，绝不探测。
#
# Usage: bash /nas3/kangjia/mllmq/scripts/probe_gpu0_health.sh
# Exit:  0 = GPU0 已恢复   2 = GPU0 仍故障   3 = 有 GPU 任务在跑，跳过探测（护栏生效）

set -o pipefail

# ── 护栏 1: 任何 CUDA 任务在跑 → 直接跳过 ──
BUSY=$(ps aux | grep -E '[e]val_generate_quant|[e]valuate_images|[m]ain_quant|[e]val_benchmarks|[p]ytorch|[t]akeup' | grep -v grep)
if [ -n "$BUSY" ]; then
  echo "[SKIP] $(date '+%m-%d %H:%M:%S') 检测到 GPU 任务在跑，按纪律 §0 不创建新 CUDA 上下文:"
  echo "$BUSY" | awk '{printf "       pid=%s cpu=%s%% %s %s\n", $2, $3, $11, $12}'
  exit 3
fi

# ── 护栏 2: nvidia-smi 必须带 timeout -s 9（D 状态下 TERM 杀不掉） ──
SMI=$(timeout -s 9 8 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1)
SMI_RC=$?
if [ $SMI_RC -ne 0 ]; then
  echo "[ALERT] $(date '+%m-%d %H:%M:%S') nvidia-smi 挂住 (rc=$SMI_RC) → 疑似驱动锁挂死复发，按 TODO Phase 3 请平台重启"
  exit 2
fi

# ── 真实 CUDA 上下文探测 GPU0 ──
RES=$(CUDA_VISIBLE_DEVICES=0 timeout 60 python -c "
import torch
torch.zeros(8, device='cuda'); torch.cuda.synchronize()
print('HEALTHY')
" 2>&1 | tail -1)

if [ "$RES" = "HEALTHY" ]; then
  echo "[OK] $(date '+%m-%d %H:%M:%S') ✅ GPU0 CUDA 上下文创建成功 → 已恢复！可以拉起双队列:"
  echo "       cd /nas3/kangjia/mllmq/MLLMQ && nohup bash scripts/run_geneval_dual_queue.sh > /nas3/kangjia/mllmq/output/logs/geneval_dual_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
  echo "       nvidia-smi: $(echo "$SMI" | tr '\n' ' ')"
  exit 0
else
  echo "[BAD] $(date '+%m-%d %H:%M:%S') ❌ GPU0 仍不可用: ${RES:0:90}"
  echo "       nvidia-smi: $(echo "$SMI" | tr '\n' ' ')"
  exit 2
fi
