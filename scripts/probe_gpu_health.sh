#!/bin/bash
# 双卡健康探测（keepalive 感知版）— 2026-10-10 新增
#
# 为什么不用 scripts/probe_gpu0_health.sh:
#   旧脚本的护栏把平台保活进程 `_takeup.py` 也算作"GPU 任务在跑" → 只要保活在跑就永远 exit 3，
#   探测形同虚设。而保活进程**不能杀**（dialog.md 重点注意事项 §0f：杀了平台可能回收 GPU）。
#   所以本脚本区分两类进程:
#     ① 本项目重负载 CUDA 任务（eval_generate_quant / evaluate_images / main_quant / eval_benchmarks）
#        → 在跑就跳过探测（并发建上下文 = 2026-10-09 11:48 驱动锁挂死的触发形态）
#     ② 平台保活 `_takeup.py`（randn 心跳，734MiB/卡，GPU util>80% 时自动让路）
#        → 长期共存、历史上从未诱发事故，**允许其在场**，仅记录不退出
#
# 判 GPU 可用 = 真实 CUDA 分配探测，不能只看 nvidia-smi（dialog 问题 10：NVML 假绿）。
#
# Usage: bash /nas3/kangjia/mllmq/scripts/probe_gpu_health.sh
# Exit:  0 = 双卡均可用   2 = 至少一张卡建不出 CUDA 上下文   3 = 有本项目 GPU 任务在跑，跳过探测

set -o pipefail

# ── 护栏 1: 本项目 GPU 任务在跑 → 直接跳过（保活进程不在此列） ──
BUSY=$(ps aux | grep -E '[e]val_generate_quant|[e]valuate_images|[m]ain_quant|[e]val_benchmarks')
if [ -n "$BUSY" ]; then
  echo "[SKIP] $(date '+%m-%d %H:%M:%S') 本项目 GPU 任务在跑，按纪律 §0 不创建新 CUDA 上下文:"
  echo "$BUSY" | awk '{printf "       pid=%s cpu=%s%% %s %s %s\n", $2, $3, $11, $12, $13}'
  exit 3
fi

# 平台保活在场是正常状态，只记录
KEEP=$(pgrep -f '_takeup.py' | tr '\n' ' ')
[ -n "$KEEP" ] && echo "[KEEPALIVE] 平台保活 _takeup.py 在跑 (pid: ${KEEP% }) — 不杀、不算冲突（§0f）"

# ── 护栏 2: nvidia-smi 必须 timeout -s 9（D 状态下 TERM 杀不掉） ──
SMI=$(timeout -s 9 8 nvidia-smi --query-gpu=index,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader 2>&1)
if [ $? -ne 0 ]; then
  echo "[ALERT] $(date '+%m-%d %H:%M:%S') nvidia-smi 挂住 → 疑似驱动锁挂死复发（dialog 问题 9），请平台重启实例"
  exit 2
fi
echo "[nvidia-smi] $(echo "$SMI" | tr '\n' ' ')"

# ── 真实 CUDA 上下文探测（逐卡串行，不并发） ──
FAIL=0
for g in 0 1; do
  RES=$(CUDA_VISIBLE_DEVICES=$g timeout 60 python -c "
import torch
torch.zeros(8, device='cuda'); torch.cuda.synchronize()
a = torch.randn(256, 256, device='cuda') @ torch.randn(256, 256, device='cuda')
torch.cuda.synchronize()
print('HEALTHY ' + torch.cuda.get_device_name(0))
" 2>&1 | tail -1)
  case "$RES" in
    HEALTHY*) echo "[OK]  GPU${g} ✅ ${RES#HEALTHY }" ;;
    *)        echo "[BAD] GPU${g} ❌ ${RES:0:100}"; FAIL=1 ;;
  esac
done

if [ $FAIL -eq 0 ]; then
  echo "[RESULT] $(date '+%m-%d %H:%M:%S') ✅ 双卡均可创建 CUDA 上下文 → 可拉双队列:"
  echo "         cd /nas3/kangjia/mllmq/MLLMQ && nohup bash scripts/run_geneval_dual_queue.sh > /nas3/kangjia/mllmq/output/logs/geneval_dual_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
  exit 0
else
  echo "[RESULT] $(date '+%m-%d %H:%M:%S') ❌ 有卡不可用 → 不要拉双队列（会连环秒失败后谎报 Queue complete）"
  exit 2
fi
