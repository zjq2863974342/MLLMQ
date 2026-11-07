#!/usr/bin/env bash
# ============================================================================
# Serial pending-eval runner for InternVL2-8B W4A16.
#
# Queues (in order):
#   1. MME     — MBQ / AWQ     (RTN done; MLLMQ may be running or done)
#   2. TextVQA — fp16/RTN/AWQ/MBQ/MLLMQ  (local cache ready)
#
# Properties:
#   * GPU gate:  at startup, waits until no OTHER eval_benchmarks.py process
#                is holding the GPU (so we never collide with a running MME).
#   * Idempotent: a cell is SKIPPED if a *_results.json already exists AND
#                no live process is writing the same <task>/<tag> right now.
#   * Serial:    one cell on one GPU at a time.
#   * Resume:    if interrupted, re-run skips completed cells.
#
# Usage:  bash scripts/run_pending_eval.sh
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJ_ROOT=$(pwd)
cd "$PROJ_ROOT/MLLMQ" || exit 1

MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
LOG_DIR=/nas3/kangjia/mllmq/output/logs
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/pending_eval_${TIMESTAMP}.log"

export HF_DATASETS_OFFLINE=1          # 本地 task 配置，禁联网验证

# ---------------------------------------------------------------------------
# 进程检测：任何 eval_benchmarks eval 进程（含我们自己的子进程）
# ---------------------------------------------------------------------------
any_eval_proc() {
    pgrep -f "eval_benchmarks.py" >/dev/null 2>&1
}

# 指定 task 是否正有 eval 进程在写（避免把"进行中"当成"已完成"）
# 只匹配 --tasks <task> 参数，tag 不参与判断，避免 set -u $2 报 unbound variable
cell_running() {
    local task=$1
    pgrep -f "eval_benchmarks.py.*--tasks ${task}" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# 已完成判定：dir 存在且有 *_results.json，且没有同名进程正在写
# ---------------------------------------------------------------------------
is_done() {
    local task=$1 tag=$2
    local d="$OUTPUT_BASE/$task/$tag"
    if cell_running "$task"; then
        return 1          # 有进程正在写该 task -> 未完成
    fi
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}

run_cell() {
    local task=$1 method=$2 w_bit=$3 scale=$4
    local tag
    if [ "$method" = "fp16" ]; then
        tag="fp16"
    else
        tag="${method}_w${w_bit}a16"
    fi

    if is_done "$task" "$tag"; then
        echo ">>> SKIP $task/$tag — already done"
        return 0
    fi

    local log="$LOG_DIR/${tag}_${task}_$(date +%Y%m%d_%H%M%S).log"
    echo ">>> [$task/$tag] start $(date +%H:%M) — log $log"
    python -u eval_benchmarks.py \
        --model_path "$MODEL_PATH" \
        --method "$method" --w_bit "$w_bit" --a_bit 16 \
        --tasks "$task" --device cuda:0 --batch_size 1 \
        --output_base "$OUTPUT_BASE" \
        ${scale:+--scale_path "$scale"} \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo ">>> [$task/$tag] OK $(date +%H:%M)"
    else
        echo ">>> [$task/$tag] FAILED rc=$rc — see $log"
    fi
    return $rc
}

SCALE_DIR="$PROJ_ROOT/MLLMQ/scale_cache"

echo "=============================================="
echo "Pending Eval — $(date)"
echo "Log: $MAIN_LOG"
echo "=============================================="

# ── GPU 门控：等所有外部 eval 进程结束（最多等 6h，5min 间隔）──
WAITED=0
while any_eval_proc; do
    if [ "$WAITED" -ge 360 ]; then
        echo ">>> [gate] waited 6h for eval processes, aborting"
        exit 1
    fi
    sleep 300
    WAITED=$((WAITED + 5))
    echo ">>> [gate] $(date +%H:%M) waiting for other eval process (${WAITED}min)..."
done
echo ">>> [gate] GPU clear — starting queue"

# ── Queue 1: MME (MBQ / AWQ; RTN done; MLLMQ may be running->will skip) ──
run_cell mme_local mbq   4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"
run_cell mme_local awq   4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"

# ── Queue 2: TextVQA (5 methods) ──
run_cell textvqa_val_local fp16  16 ""
run_cell textvqa_val_local rtn   4 ""
run_cell textvqa_val_local awq   4 "$SCALE_DIR/awq/internvl2_InternVL2-8B_w4a16.pt"
run_cell textvqa_val_local mbq   4 "$SCALE_DIR/mbq/internvl2_w4a16.pt"
run_cell textvqa_val_local mllmq 4 "$SCALE_DIR/mllmq/internvl2_w4a16.pt"

echo "=============================================="
echo "Pending Eval DONE — $(date)"
echo "Log: $MAIN_LOG"
echo "=============================================="