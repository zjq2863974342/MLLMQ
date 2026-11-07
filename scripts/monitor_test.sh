#!/usr/bin/env bash
# Quick monitoring test: run 1 MMMU cell (--limit 100) while logging CPU+GPU
set -uo pipefail

PROJ_ROOT=/nas3/kangjia/mllmq
LOG_DIR="$PROJ_ROOT/output/logs"
TS=$(date +%Y%m%d_%H%M%S)
MON_LOG="$LOG_DIR/monitor_${TS}.log"
EVAL_LOG="$LOG_DIR/eval_test_rtn_w4_mmmu_${TS}.log"
STATS_LOG="$LOG_DIR/monitor_stats_${TS}.csv"

mkdir -p "$LOG_DIR"

# ── System info header ──
{
echo "# Monitor Test — $(date)"
echo "# CPU: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "# Cores: $(nproc) threads"
echo "# GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "# VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | tr '\n' ' ')MiB"
echo "# Env: OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset}"
echo ""
} > "$MON_LOG"

# ── CSV header ──
echo "timestamp,elapsed_s,gpu0_util%,gpu0_mem_mib,gpu0_mem_%,gpu1_util%,gpu1_mem_mib,mpstat_usr%,mpstat_sys%,mpstat_idle%" > "$STATS_LOG"

START_TS=$(date +%s)

# ── Background monitoring loop ──
monitor_loop() {
    while true; do
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TS))
        TS_STR=$(date +%H:%M:%S)

        # GPU stats
        GPU_STATS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,utilization.memory --format=csv,noheader,nounits 2>/dev/null)
        G0_UTIL=$(echo "$GPU_STATS" | grep "^0," | cut -d',' -f2 | tr -d ' ')
        G0_MEM=$(echo "$GPU_STATS" | grep "^0," | cut -d',' -f3 | tr -d ' ')
        G0_MEMP=$(echo "$GPU_STATS" | grep "^0," | cut -d',' -f4 | tr -d ' ')
        G1_UTIL=$(echo "$GPU_STATS" | grep "^1," | cut -d',' -f2 | tr -d ' ')
        G1_MEM=$(echo "$GPU_STATS" | grep "^1," | cut -d',' -f3 | tr -d ' ')

        # CPU stats (1-second snapshot)
        CPU_STATS=$(mpstat 1 1 2>/dev/null | tail -1)
        CPU_USR=$(echo "$CPU_STATS" | awk '{print $3}')
        CPU_SYS=$(echo "$CPU_STATS" | awk '{print $5}')
        CPU_IDLE=$(echo "$CPU_STATS" | awk '{print $12}')

        echo "${TS_STR},${ELAPSED},${G0_UTIL:-0},${G0_MEM:-0},${G0_MEMP:-0},${G1_UTIL:-0},${G1_MEM:-0},${CPU_USR:-0},${CPU_SYS:-0},${CPU_IDLE:-0}" >> "$STATS_LOG"

        # Also log human-readable
        printf "[%s t+%04ds] GPU0: %3s%% util %5s MiB | GPU1: %3s%% util %5s MiB | CPU: usr=%5s sys=%5s idle=%5s\n" \
            "$TS_STR" "$ELAPSED" "$G0_UTIL" "$G0_MEM" "$G1_UTIL" "$G1_MEM" "$CPU_USR" "$CPU_SYS" "$CPU_IDLE" >> "$MON_LOG"

        sleep 5
    done
}

monitor_loop &
MON_PID=$!
echo "Monitor PID: $MON_PID" | tee -a "$MON_LOG"

# ── Run test eval ──
cd "$PROJ_ROOT/MLLMQ"

export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=== Starting test eval (MMMU RTN W4, limit=100) ===" | tee -a "$MON_LOG"
echo "Log: $EVAL_LOG" | tee -a "$MON_LOG"

python -u eval_benchmarks.py \
    --method rtn --w_bit 4 --tasks mmmu_val --device cuda:0 --batch_size 1 \
    --limit 100 \
    --output_base /nas3/kangjia/mllmq/output/internvl2-8b/eval_test \
    > "$EVAL_LOG" 2>&1

EVAL_RC=$?
echo "Eval exit code: $EVAL_RC" | tee -a "$MON_LOG"

# ── Stop monitor ──
kill $MON_PID 2>/dev/null
wait $MON_PID 2>/dev/null

# ── Summary ──
echo "" | tee -a "$MON_LOG"
echo "=== Test Complete — $(date) ===" | tee -a "$MON_LOG"
echo "Monitor log: $MON_LOG" | tee -a "$MON_LOG"
echo "Eval log:   $EVAL_LOG" | tee -a "$MON_LOG"
echo "Stats CSV:  $STATS_LOG" | tee -a "$MON_LOG"

# Quick stats from CSV
echo "" | tee -a "$MON_LOG"
echo "=== GPU Utilization Summary ===" | tee -a "$MON_LOG"
tail -n +2 "$STATS_LOG" | awk -F',' '{
    gpu0_sum+=$3; gpu0_mem_sum+=$4; count++
    if($3+0>0) {gpu0_active++; gpu0_active_sum+=$3}
}
END {
    printf "GPU0 avg util: %.1f%%  avg mem: %.0f MiB  active samples: %d/%d (active avg util: %.1f%%)\n",
        gpu0_sum/count, gpu0_mem_sum/count, gpu0_active, count, (gpu0_active>0?gpu0_active_sum/gpu0_active:0)
}' | tee -a "$MON_LOG"

echo "" | tee -a "$MON_LOG"
echo "=== CPU Utilization Summary ===" | tee -a "$MON_LOG"
tail -n +2 "$STATS_LOG" | awk -F',' '{
    usr_sum+=$8; sys_sum+=$9; idle_sum+=$10; count++
}
END {
    printf "CPU avg: usr=%.1f%% sys=%.1f%% idle=%.1f%% (all 128 threads)\n",
        usr_sum/count, sys_sum/count, idle_sum/count
}' | tee -a "$MON_LOG"
