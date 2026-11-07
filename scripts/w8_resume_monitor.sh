#!/bin/bash
# W8 续跑 + 评分 状态监控（10-20 01:10 建）
# 用法: setsid nohup bash scripts/w8_resume_monitor.sh > output/logs/w8_resume_monitor.log 2>&1 < /dev/null &
# 或每隔一段手动跑一次（幂等）

OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/coco_val
LOG=/nas3/kangjia/mllmq/output/logs
DRIVER='run_coco_val_w8_full_dual_queue'

ts() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $1"; }

echo "=== W8 resume monitor @ $(date '+%Y-%m-%d %H:%M:%S') ==="
for t in awq_w8 mbq_block_w8 mbq_scalar_w8 mllmq_mse_w8 mllmq_mae_w8; do
  n=$(find "$OUT/$t" -name '*.png' 2>/dev/null | wc -l)
  u=$(find "$OUT/$t" -name '*_0000.png' 2>/dev/null | sed 's|.*/||;s|_.*||' | sort -u | wc -l)
  echo "  $t: total=$n uniq0000=$u/5000"
done

gen=$(pgrep -fc 'eval_generate_quant' 2>/dev/null || echo 0)
drv=$(pgrep -fc "$DRIVER" 2>/dev/null || echo 0)
scal=$(find "$OUT/mbq_scalar_w8" -name '*.png' | wc -l)
mse=$(find "$OUT/mllmq_mse_w8" -name '*.png' | wc -l)
echo "  gen procs=$gen driver=$drv"
if [ $gen -eq 0 ] && [ $scal -ge 5000 ] && [ $mse -ge 5000 ]; then
  echo "  ✅ 生成全停且 2 个 config 都到 5000 → 可以评分"
elif [ $gen -eq 0 ]; then
  echo "  ⚠️ 无生成进程但还没到 5000: scalar=$scal mse=$mse"
else
  echo "  ⏳ 仍在生成: scalar=$scal mse=$mse (gen=$gen)"
fi
