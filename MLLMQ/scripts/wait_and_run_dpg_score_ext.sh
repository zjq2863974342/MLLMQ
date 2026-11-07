#!/bin/bash
# DPG 段3 评分「守夜接跑器」
#   用途：本机=评分容器。段2 grid4 评分把两卡占住到 ~07:00，对端生成容器此刻仍在产图。
#   本脚本每 5min 探一次：一旦本机没有任何 CUDA 任务，且有任一 段3 tag 已生成满 1065 张，
#   就 detached 启动 run_dpg_score_ext_dual_queue.sh（该队列自带「等 tag 生成齐再评」）。
#   —— 不依赖任何交互式会话，setsid 后可跨会话存活。
#
# 用法：cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/wait_and_run_dpg_score_ext.sh \
#          > /nas3/kangjia/mllmq/output/logs/dpg_score_ext_waiter_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

cd /nas3/kangjia/mllmq

IMG_ROOT=/nas3/kangjia/mllmq/output/janus-pro-7b/dpg_bench
OUT=/nas3/kangjia/mllmq/output/dpg_scores
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=1065
POLL=300          # 5min 探一次
MAX_HOURS=40      # 最长守 40h
DRIVER="/nas3/kangjia/mllmq/MLLMQ/scripts/run_dpg_score_ext_dual_queue.sh"
TAGS="mbq_scalar_w3 mllmq_mse_w3 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4"

# 本机任何其它 CUDA 任务（含 grid4 评分）→ 不能起。
# 注意也纳入两级 driver 名：config 切换有 30s COOLDOWN + 权重加载窗口，
# 那段时间 worker 进程已退、driver 还在——只匹配 worker 会误判「两卡已空」而抢卡。
BUSY_PAT='[e]val_generate_quant|[s]core_pref|[s]core_fid|[s]core_clip|[m]ain_quant|[e]val_benchmarks|[s]core_dpg_coco|[c]ompute_dpg_bench|[r]un_dpg_score_dual_queue|[r]un_dpg_score_grid4_dual_queue'

log() { echo "[$(date '+%F %T')] $1"; }

count_pngs() { find "${IMG_ROOT}/$1/dpg_bench" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l; }
done_scores() { local n=0 f; for t in $TAGS; do f="${OUT}/${t}_single_results.txt"; [ -s "$f" ] && grep -q "DPG-Bench score:" "$f" && n=$((n+1)); done; echo $n; }

T0=$(date +%s)
log "=== 段3 评分守夜接跑器启动 (poll=${POLL}s, max=${MAX_HOURS}h) ==="

while :; do
  elapsed=$(( ($(date +%s) - T0) / 60 ))
  [ "$elapsed" -ge $(( MAX_HOURS * 60 )) ] && { log "守夜超时 ${MAX_HOURS}h，退出"; exit 1; }

  scored=$(done_scores)
  [ "$scored" -ge 8 ] && { log "8/8 段3 config 均已评分完成，接跑器退出"; exit 0; }

  if pgrep -f "$BUSY_PAT" >/dev/null 2>&1; then
    log "本机仍有 CUDA 任务（段2 grid4 评分未完）→ 继续等 (已守 ${elapsed}min, 段3 已评 ${scored}/8)"
    sleep $POLL; continue
  fi

  # 本机已空：看是否有 tag 生成齐
  ready=""; ready_n=0
  for t in $TAGS; do
    n=$(count_pngs "$t")
    [ "$n" -ge "$TARGET" ] && { ready="$ready $t(${n})"; ready_n=$((ready_n+1)); }
  done

  if [ "$ready_n" -gt 0 ]; then
    log "本机两卡已空 ✅ 且有 ${ready_n} 个 tag 生成齐：${ready}"
    log "→ detached 启动段3 评分队列"
    setsid nohup bash "$DRIVER" > "${LOG}/dpg_score_ext_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
    sleep 30
    if pgrep -f 'run_dpg_score_ext_dual_queue' >/dev/null 2>&1; then
      log "✅ 段3 评分队列已接上（守夜接跑器退出，后续由评分队列 driver 自主跑完）"
      exit 0
    else
      log "⚠️ 启动后未见 driver 进程，60s 后重试"
      sleep 60
    fi
  else
    log "本机已空，但暂无 tag 生成齐（对端仍在产图）→ 继续等 (已守 ${elapsed}min)"
    sleep $POLL
  fi
done