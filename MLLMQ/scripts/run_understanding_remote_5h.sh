#!/usr/bin/env bash
# ============================================================================
# 副机 5h 理解侧：Wave1 4 卡 × 4 config；Wave2 用 SEED-Bench-2 chunk0/1 填充
#   Wave1 (4 卡, 7 benchmarks/config):
#     GPU0 MBQ-block-masked (128)
#     GPU1 MBQ-scalar (128)
#     GPU2 MLLMQ-channel (128)
#     GPU3 MLLMQ-both (384)
#   Wave2 (可选, SEED=1): 同样 4 个 config 跑 seedbench_2_local_chunk0/1
#
# 输出：output/fixed_ablation_20261103_understanding_5h/<tag>/
# 日志：output/logs/understanding_5h/
# 用法（等 GenEval 评分完成后）：
#   cd /nas3/kangjia/mllmq/MLLMQ
#   SEED=1 setsid nohup bash scripts/run_understanding_remote_5h.sh \
#     > ../output/logs/understanding_5h/remote_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
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

echo "===== Wave1: 4 config × 7 benchmarks $(date '+%F %T') ====="
run mbq_block_masked mbq "$SC/unified_understanding_128/mbq_block_masked_w3a16.pt" 0 > "$LOG/driver_mbq_block_masked.log" 2>&1 &
p0=$!
run mbq_scalar mbq "$SC/unified_understanding_128/mbq_scalar_w3a16.pt" 1 > "$LOG/driver_mbq_scalar.log" 2>&1 &
p1=$!
run mllmq_channel_only mllmq "$SC/unified_understanding_128/mllmq_channel_only_w3a16.pt" 2 > "$LOG/driver_mllmq_channel_only.log" 2>&1 &
p2=$!
run mllmq_both mllmq "$SC/unified_384/mllmq_both_w3a16.pt" 3 > "$LOG/driver_mllmq_both.log" 2>&1 &
p3=$!
wait $p0; wait $p1; wait $p2; wait $p3

echo "===== Wave1 done $(date '+%F %T') ====="

if [ "${SEED:-0}" = "1" ]; then
  SEED_TASKS=${SEED_TASKS:-"seedbench_2_local_chunk0 seedbench_2_local_chunk1"}
  echo "===== Wave2: SEED-Bench-2 ($SEED_TASKS) $(date '+%F %T') ====="
  TASKS="$SEED_TASKS" run mbq_block_masked mbq "$SC/unified_understanding_128/mbq_block_masked_w3a16.pt" 0 > "$LOG/driver_seed_mbq_block_masked.log" 2>&1 &
  s0=$!
  TASKS="$SEED_TASKS" run mbq_scalar mbq "$SC/unified_understanding_128/mbq_scalar_w3a16.pt" 1 > "$LOG/driver_seed_mbq_scalar.log" 2>&1 &
  s1=$!
  TASKS="$SEED_TASKS" run mllmq_channel_only mllmq "$SC/unified_understanding_128/mllmq_channel_only_w3a16.pt" 2 > "$LOG/driver_seed_mllmq_channel_only.log" 2>&1 &
  s2=$!
  TASKS="$SEED_TASKS" run mllmq_both mllmq "$SC/unified_384/mllmq_both_w3a16.pt" 3 > "$LOG/driver_seed_mllmq_both.log" 2>&1 &
  s3=$!
  wait $s0; wait $s1; wait $s2; wait $s3
fi

echo "[REMOTE UNDERSTANDING DONE] $(date '+%F %T')"