#!/usr/bin/env bash
# ============================================================================
# 主机 5h 理解侧：2 波 × 2 卡，每 config 7 个 benchmark（串行）
#   Wave1: RTN (128 und) + AWQ (128 und)
#   Wave2: MLLMQ-t2i_text (384) + MLLMQ-t2i_img (384)
# 输出：output/fixed_ablation_20261103_understanding_5h/<tag>/
# 日志：output/logs/understanding_5h/
# 用法（副机 GenEval 评分完成后启动）：
#   cd /nas3/kangjia/mllmq/MLLMQ
#   setsid nohup bash scripts/run_understanding_host_5h.sh \
#     > ../output/logs/understanding_5h/host_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
SC=$ROOT/MLLMQ/scale_cache
export OUTBASE=$ROOT/output/fixed_ablation_20261103_understanding_5h
export LOG=$ROOT/output/logs/understanding_5h
mkdir -p "$OUTBASE" "$LOG"

run(){ # tag method scale gpu
  TAG=$1 METHOD=$2 SCALE=$3 GPU=$4
  export TAG METHOD SCALE GPU TASKS
  bash scripts/run_understanding_7.sh
}

echo "===== Wave1: RTN + AWQ $(date '+%F %T') ====="
run rtn rtn "" 0 > "$LOG/driver_rtn.log" 2>&1 &
check_rtn=$!
run awq awq "$SC/unified_understanding_128/awq_w3a16.pt" 1 > "$LOG/driver_awq.log" 2>&1 &
check_awq=$!
wait $check_rtn; wait $check_awq

echo "===== Wave2: MLLMQ-t2i_text + t2i_img $(date '+%F %T') ====="
run mllmq_t2i_text mllmq "$SC/unified_384/mllmq_t2i_text_w3a16.pt" 0 > "$LOG/driver_mllmq_t2i_text.log" 2>&1 &
check_t=$!
run mllmq_t2i_img mllmq "$SC/unified_384/mllmq_t2i_img_w3a16.pt" 1 > "$LOG/driver_mllmq_t2i_img.log" 2>&1 &
check_i=$!
wait $check_t; wait $check_i

echo "[HOST UNDERSTANDING DONE] $(date '+%F %T')"