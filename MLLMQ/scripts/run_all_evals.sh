#!/usr/bin/env bash
# Batch MMMU evals for all 8 completed quant models + MBQ W4A16 redo
set -euo pipefail
cd /nas3/kangjia/mllmq/MLLMQ

declare -A SCALES
SCALES[mbq_w8]="scale_cache/mbq/internvl2_w8a16.pt"
SCALES[mbq_w3]="scale_cache/mbq/internvl2_w3a16.pt"
SCALES[mllmq_w8]="scale_cache/mllmq/internvl2_w8a16.pt"
SCALES[mllmq_w3]="scale_cache/mllmq/internvl2_w3a16.pt"
SCALES[awq_w8]="scale_cache/awq/internvl2_InternVL2-8B_w8a16.pt"
SCALES[awq_w3]="scale_cache/awq/internvl2_InternVL2-8B_w3a16.pt"

OUT=/nas3/kangjia/mllmq/output

echo "========== Batch MMMU Evals =========="
echo "Start: $(date)"
echo ""

for method in rtn awq mbq mllmq; do
  for w_bit in 8 3; do
    eval_dir="${OUT}/eval_internvl2_${method}_w${w_bit}a16"
    scale=""
    if [[ "$method" != "rtn" ]]; then
      scale="${SCALES[${method}_w${w_bit}]}"
    fi

    echo "------------------------------------------------------------"
    echo "[${method} W${w_bit}A16] eval -> ${eval_dir}"
    echo "  scale: ${scale:-<none>}"
    echo "  start: $(date)"
    echo "------------------------------------------------------------"

    mkdir -p "${eval_dir}"
    python -W ignore eval_mmmu.py \
      --method "${method}" --w_bit "${w_bit}" \
      ${scale:+--scale_path "${scale}"} \
      --output_dir "${eval_dir}" 2>&1 | tee "${eval_dir}/run.log"

    echo "[${method} W${w_bit}] DONE at $(date)"
    echo ""
  done
done

echo "========== All evals complete: $(date) =========="