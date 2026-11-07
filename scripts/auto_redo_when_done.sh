#!/usr/bin/env bash
# ============================================================================
# Watchdog: 等主队列（run_janus_full_matrix.sh 的两个实例）全部结束后，
# 自动启动 w8 补跑脚本（Pope + MathVista 格式修复后）。
#
# 启动方式：
#   nohup bash scripts/auto_redo_when_done.sh > output/logs/auto_redo_watchdog.log 2>&1 &
# ============================================================================
set -u

cd /nas3/kangjia/mllmq || exit 1

echo "[watchdog] $(date) started. Waiting for main queue (run_janus_full_matrix.sh) to finish..."

# 每 60s 检查一次主队列进程是否还存在
while pgrep -f "run_janus_full_matrix\.sh" >/dev/null 2>&1; do
    sleep 60
done

echo "[watchdog] $(date) main queue finished. Waiting 30s, then launching redo..."
sleep 30

nohup bash scripts/run_janus_redo_w8.sh 0 mae > output/logs/redo_gpu0_mae.log 2>&1 &
MAE_PID=$!
sleep 10
nohup bash scripts/run_janus_redo_w8.sh 1 mse > output/logs/redo_gpu1_mse.log 2>&1 &
MSE_PID=$!

echo "[watchdog] $(date) redo launched: mae PID=$MAE_PID mse PID=$MSE_PID"
echo "[watchdog] done."