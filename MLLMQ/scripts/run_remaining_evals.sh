#!/usr/bin/env bash
# Batch MMMU evals for the remaining W3/W8 cells, resilient to single-cell failure.
# Mirrors scripts/run_all_evals.sh but skips already-completed outputs and tolerates errors.
#
# Usage: bash scripts/run_remaining_evals.sh
#
set -u
cd /nas3/kangjia/mllmq/MLLMQ

declare -A SCALES
SCALES[mbq_w8]="scale_cache/mbq/internvl2_w8a16.pt"
SCALES[mbq_w3]="scale_cache/mbq/internvl2_w3a16.pt"
SCALES[mllmq_w8]="scale_cache/mllmq/internvl2_w8a16.pt"
SCALES[mllmq_w3]="scale_cache/mllmq/internvl2_w3a16.pt"
SCALES[awq_w8]="scale_cache/awq/internvl2_InternVL2-8B_w8a16.pt"
SCALES[awq_w3]="scale_cache/awq/internvl2_InternVL2-8B_w3a16.pt"

OUT=/nas3/kangjia/mllmq/output

# Cells to run: only the ones not already covered by the W4 batch + earlier RTN runs
# Format: "method:w_bit"
TODO=(
    "rtn:3"      # safety: keep RTN W3 in the queue
    "awq:8"
    "awq:3"
    "mbq:8"
    "mbq:3"
    "mllmq:8"
    "mllmq:3"
)

is_done() {
    # $1 = eval dir; consider done if a *_results.json exists in it
    local d="$1"
    if ls "$d"/*results.json >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

echo "========== Continuation MMMU Evals =========="
echo "Start: $(date)"
echo ""

for cell in "${TODO[@]}"; do
    method="${cell%:*}"
    w_bit="${cell#*:}"
    eval_dir="${OUT}/eval_internvl2_${method}_w${w_bit}a16"
    scale=""
    if [[ "$method" != "rtn" ]]; then
        scale="${SCALES[${method}_w${w_bit}]:-}"
    fi

    if is_done "$eval_dir"; then
        echo "------------------------------------------------------------"
        echo "[${method} W${w_bit}A16] SKIP — already evaluated at ${eval_dir}"
        echo "------------------------------------------------------------"
        continue
    fi

    echo "============================================================"
    echo "[${method} W${w_bit}A16] eval -> ${eval_dir}"
    echo "  scale: ${scale:-<none>}"
    echo "  start: $(date)"
    echo "============================================================"

    mkdir -p "${eval_dir}"
    set +e
    python -W ignore eval_mmmu.py \
        --method "${method}" --w_bit "${w_bit}" \
        ${scale:+--scale_path "${scale}"} \
        --output_dir "${eval_dir}" 2>&1 | tee "${eval_dir}/run.log"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ ${rc} -ne 0 ]]; then
        echo "[${method} W${w_bit}] FAILED (rc=${rc}) at $(date) — continuing to next cell"
    else
        echo "[${method} W${w_bit}] DONE at $(date)"
    fi
    echo ""
done

echo "========== All attempted evals finished: $(date) =========="