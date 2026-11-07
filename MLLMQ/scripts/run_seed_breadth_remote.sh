#!/usr/bin/env bash
# ============================================================================
# 副机 SEED-Bench-2 广度方案（少 chunk、多 config）
#   4 卡 × 1 config：MBQ-scalar / MLLMQ-t2i_text / MLLMQ-t2i_img / MLLMQ-both
#   chunk: 默认 chunk0 + chunk1（各 ~70 min/config；4 个并行 ~2.3h）
#   输出：output/fixed_ablation_20261103_understanding_5h_seed/<tag>/
#   日志：output/logs/understanding_5h_seed/
#
# 用法：
#   cd /nas3/kangjia/mllmq/MLLMQ
#   setsid nohup bash scripts/run_seed_breadth_remote.sh \
#     > ../output/logs/understanding_5h_seed/remote_driver_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#
# 可选：SEED_TASKS="seedbench_2_local_chunk0"（只跑 1 个 chunk，约 1.2h）
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
SC=$ROOT/MLLMQ/scale_cache
export OUTBASE=$ROOT/output/fixed_ablation_20261103_understanding_5h_seed
export LOG=$ROOT/output/logs/understanding_5h_seed
export TASKS=${SEED_TASKS:-"seedbench_2_local_chunk0 seedbench_2_local_chunk1"}
mkdir -p "$OUTBASE" "$LOG"

for f in \
  "$SC/unified_understanding_128/mbq_scalar_w3a16.pt" \
  "$SC/unified_384/mllmq_t2i_text_w3a16.pt" \
  "$SC/unified_384/mllmq_t2i_img_w3a16.pt" \
  "$SC/unified_384/mllmq_both_w3a16.pt" ; do
  [ -s "$f" ] || { echo "GATE FAIL: missing $f"; exit 2; }
done

# 可选：等 GenEval 8 个 results.jsonl 齐了再跑，避免和 GenEval 抢卡
if [ "${WAIT_GENEVAL:-0}" = "1" ]; then
  for i in $(seq 1 720); do
    miss=""
    for t in rtn mbq_block_masked mbq_scalar mllmq_channel_only \
             mllmq_t2i_text mllmq_t2i_img mllmq_both mllmq_gen_only; do
      [ -f "$ROOT/output/fixed_ablation_20261103_unified/$t/geneval/results.jsonl" ] || miss="$miss $t"
    done
    [ -z "$miss" ] && { echo "[gate] GenEval done $(date '+%F %T')"; break; }
    echo "[gate $(date '+%H:%M')] waiting GenEval:$miss"
    sleep 60
  done
fi

run(){ # tag method scale gpu
  TAG=$1 METHOD=$2 SCALE=$3 GPU=$4
  export TAG METHOD SCALE GPU
  bash scripts/run_understanding_7.sh
}

echo "===== SEED breadth (chunks: $TASKS) $(date '+%F %T') ====="
run mbq_scalar         mbq   "$SC/unified_understanding_128/mbq_scalar_w3a16.pt" 0 > "$LOG/driver_mbq_scalar.log" 2>&1 &
p0=$!
run mllmq_t2i_text     mllmq "$SC/unified_384/mllmq_t2i_text_w3a16.pt"           1 > "$LOG/driver_mllmq_t2i_text.log" 2>&1 &
p1=$!
run mllmq_t2i_img      mllmq "$SC/unified_384/mllmq_t2i_img_w3a16.pt"            2 > "$LOG/driver_mllmq_t2i_img.log" 2>&1 &
p2=$!
run mllmq_both         mllmq "$SC/unified_384/mllmq_both_w3a16.pt"               3 > "$LOG/driver_mllmq_both.log" 2>&1 &
p3=$!
wait $p0; wait $p1; wait $p2; wait $p3
echo "[SEED BREADTH DONE] $(date '+%F %T')"