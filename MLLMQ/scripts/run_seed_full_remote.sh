#!/usr/bin/env bash
# ============================================================================
# 副机 SEED-Bench-2 全量补齐：4 config 只跑 chunk2–4
#   （chunk0/1 已由 run_seed_breadth_remote.sh 完成，skip-existing 安全）
#   GPU0 MBQ-scalar(128) / GPU1 MLLMQ-t2i_text(384)
#   GPU2 MLLMQ-t2i_img(384) / GPU3 MLLMQ-both(384)
#   预计 ~2.9h（3 chunk × ~58 min/config，4 并行）
# 输出：output/fixed_ablation_20261103_understanding_5h_seed/<tag>/
# 用法：
#   cd /nas3/kangjia/mllmq/MLLMQ
#   setsid nohup bash scripts/run_seed_full_remote.sh \
#     > ../output/logs/understanding_5h_seed/remote_full_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
SC=$ROOT/MLLMQ/scale_cache
export OUTBASE=$ROOT/output/fixed_ablation_20261103_understanding_5h_seed
export LOG=$ROOT/output/logs/understanding_5h_seed
export TASKS=${SEED_TASKS:-"seedbench_2_local_chunk2 seedbench_2_local_chunk3 seedbench_2_local_chunk4"}
mkdir -p "$OUTBASE" "$LOG"

for f in \
  "$SC/unified_understanding_128/mbq_scalar_w3a16.pt" \
  "$SC/unified_384/mllmq_t2i_text_w3a16.pt" \
  "$SC/unified_384/mllmq_t2i_img_w3a16.pt" \
  "$SC/unified_384/mllmq_both_w3a16.pt" ; do
  [ -s "$f" ] || { echo "GATE FAIL: missing $f"; exit 2; }
done

run(){ # tag method scale gpu
  TAG=$1 METHOD=$2 SCALE=$3 GPU=$4
  export TAG METHOD SCALE GPU TASKS
  bash scripts/run_understanding_7.sh
}

echo "===== SEED full chunks: $TASKS $(date '+%F %T') ====="
run mbq_scalar     mbq   "$SC/unified_understanding_128/mbq_scalar_w3a16.pt" 0 > "$LOG/full_mbq_scalar.log" 2>&1 &
p0=$!
run mllmq_t2i_text mllmq "$SC/unified_384/mllmq_t2i_text_w3a16.pt"           1 > "$LOG/full_mllmq_t2i_text.log" 2>&1 &
p1=$!
run mllmq_t2i_img  mllmq "$SC/unified_384/mllmq_t2i_img_w3a16.pt"            2 > "$LOG/full_mllmq_t2i_img.log" 2>&1 &
p2=$!
run mllmq_both     mllmq "$SC/unified_384/mllmq_both_w3a16.pt"               3 > "$LOG/full_mllmq_both.log" 2>&1 &
p3=$!
wait $p0; wait $p1; wait $p2; wait $p3
echo "[SEED FULL REMOTE DONE] $(date '+%F %T')"