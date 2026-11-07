#!/bin/bash
# 守护：让本机 GPU0 跑完 mbq_block_w4 后，阻止它接着起 mllmq_mse_w4（该 tag 已由另一台机子跑）。
# 一旦 mbq_block_w4 满 2400 或本机出现 mllmq_mse_w4 python，就杀掉 GPU0 队列子shell 及其 python。
TAGDIR=/nas3/kangjia/mllmq/output/janus-pro-7b/t2i_compbench
MBQ=$TAGDIR/mbq_block_w4/t2i_compbench
SUB=28296                      # GPU0 队列子shell（mbq_block_w4 python 的父）
SUB_MBQ_PY=28946               # mbq_block_w4 python
LOG=/nas3/kangjia/mllmq/output/logs/guard_no_mse_w4_$(date +%Y%m%d_%H%M%S).log
log(){ echo "[$(date '+%F %T')] $1" >> "$LOG"; }

log "guard start: 待 mbq_block_w4 满或 mse_w4 冒头 → 掐 GPU0 子shell $SUB"

while true; do
  n=$(find "$MBQ" -name '*.png' 2>/dev/null | wc -l)
  mse_pid=$(pgrep -f 'eval_generate_quant.*t2i_compbench/mllmq_mse_w4')
  if [ -n "$mse_pid" ]; then
    log "🔴 本机出现 mllmq_mse_w4 python $mse_pid → 立即 kill（防双写）"
    kill -9 $mse_pid 2>/dev/null
    sleep 2
    kill -TERM "$SUB" 2>/dev/null; sleep 2; kill -9 "$SUB" 2>/dev/null
    log "已杀 GPU0 子shell $SUB"
    break
  fi
  # mbq_block_w4 已完成 → 趁 30s cooldown 掐子shell，mse_w4 永不起
  if [ "$n" -ge 2400 ]; then
    log "✅ mbq_block_w4 已满 2400/$n → 趁 cooldown 掐 GPU0 子shell $SUB，阻止 mllmq_mse_w4 起跑"
    kill -TERM "$SUB" 2>/dev/null; sleep 3; kill -9 "$SUB" 2>/dev/null
    log "已杀 GPU0 子shell $SUB（mbq_block_w4 python $SUB_MBQ_PY 可能残留，稍后清）"
    break
  fi
  sleep 3
done
# 收尾：清掉可能残留的 mbq_block_w4 python 与任何 mse_w4 python
sleep 3
for p in $(pgrep -f 'eval_generate_quant.*t2i_compbench/mbq_block_w4'); do log "清理残留 mbq python $p"; kill -9 $p 2>/dev/null; done
for p in $(pgrep -f 'eval_generate_quant.*t2i_compbench/mllmq_mse_w4'); do log "清理残留 mse python $p"; kill -9 $p 2>/dev/null; done
log "guard done（GPU1 mllmq_mae_w4 未受影响，继续跑）"