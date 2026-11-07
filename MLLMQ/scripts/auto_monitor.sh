#!/bin/bash
# Auto-monitor: appends a brief status line to dialog.md
# Usage in cron: */30 * * * * bash /nas3/kangjia/mllmq/MLLMQ/scripts/auto_monitor.sh

D=/nas3/kangjia/mllmq
COUNT_FILE="$D/output/janus-pro-7b/geneval"
TOTAL=$(for t in fp16 rtn_w8 rtn_w4 rtn_w3 awq_w8 awq_w4 awq_w3 mbq_block_w8 mbq_block_w4 mbq_block_w3 mbq_scalar_w8 mbq_scalar_w4 mbq_scalar_w3 mllmq_mae_w8 mllmq_mae_w4 mllmq_mae_w3 mllmq_mse_w8 mllmq_mse_w4 mllmq_mse_w3; do
  [ "$(find "$COUNT_FILE/$t" -name '*.png' 2>/dev/null | wc -l)" -ge 2212 ] && echo 1
done | wc -l)

GPU=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>/dev/null | awk -F', ' '{printf "GPU%s=%s%% ", $1, $2}')

PROCS=$(ps aux 2>/dev/null | grep -c 'eval_generate_quant')

cat >> "$D/dialog.md" << EOF
### [AUTO] — $(date '+%m-%d %H:%M')
📊 $GPU | $PROCS process(es) | $TOTAL/19 complete
EOF
