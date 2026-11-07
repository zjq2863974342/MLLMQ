#!/bin/bash
# ============================================================================
# MLLMQ VL Understanding Model - All Baselines Runner
#
# Runs Phase 1 experiments for VL understanding: all methods at W4A16,
# plus the key reweight ablation. Each uses COCO multimodal calibration.
# Matches the Janus run_all_baselines.sh conventions.
#
# Usage:
#   ./scripts/run_all_baselines_vl.sh <model_type> <model_path> \\
#       <data_path> <image_folder> [device]
#
# Example:
#   ./scripts/run_all_baselines_vl.sh qwen2_vl /path/to/Qwen2-VL-7B \\
#       /data/coco_train.json /data/coco/train2014 cuda:0
# ============================================================================

set -euo pipefail

MODEL_TYPE="${1:-qwen2_vl}"
MODEL_PATH="${2:?Error: model_path required}"
DATA_PATH="${3:?Error: data_path required}"
IMAGE_FOLDER="${4:?Error: image_folder required}"
DEVICE="${5:-cuda:0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_experiment_vl.sh"

echo "============================================================"
echo "MLLMQ VL Phase 1 - ${MODEL_TYPE} All Baselines Runner"
echo "============================================================"
echo "  Model type:  ${MODEL_TYPE}"
echo "  Model path:  ${MODEL_PATH}"
echo "  Data:        ${DATA_PATH}"
echo "  Images:      ${IMAGE_FOLDER}"
echo "  Device:      ${DEVICE}"
echo "  Start:       $(date)"
echo "============================================================"

# ---- Experiment Matrix ----
# Format: method w_bit a_bit extra_args description
# MBQ = per-layer (no gradient or uniform gradient mean)
# MLLMQ = per-channel gradient reweighting (our method)
EXPERIMENTS=(
    # ---- W8A16 upper bound ----
    "mbq 8 16 ''                        'MBQ W8A16 near-lossless'"

    # ---- W4A16 Baselines ----
    "rtn 4 16 ''                        'RTN W4A16 lower bound'"
    "awq 4 16 ''                        'AWQ W4A16'"

    # ---- Core comparison: MBQ vs MLLMQ ----
    "mbq   4 16 ''                      'MBQ W4A16 (per-layer, no reweight)'"
    "mllmq 4 16 ''                      'MLLMQ W4A16 (per-channel reweight)'"
)

echo ""
echo "Experiments to run: ${#EXPERIMENTS[@]}"
echo ""

for exp_line in "${EXPERIMENTS[@]}"; do
    # Parse the line
    read -r method w_bit a_bit extra_args label <<< "${exp_line}"

    echo "------------------------------------------------------------"
    echo "Running: ${method} W${w_bit}A${a_bit} : ${label}"
    echo "------------------------------------------------------------"

    # Run with or without extra_args
    if [ "${extra_args}" = "''" ] || [ -z "${extra_args}" ]; then
        bash "${RUNNER}" "${MODEL_TYPE}" "${method}" "${w_bit}" "${a_bit}" \
            "${MODEL_PATH}" "${DATA_PATH}" "${IMAGE_FOLDER}" 128 "${DEVICE}" ""
    else
        bash "${RUNNER}" "${MODEL_TYPE}" "${method}" "${w_bit}" "${a_bit}" \
            "${MODEL_PATH}" "${DATA_PATH}" "${IMAGE_FOLDER}" 128 "${DEVICE}" "${extra_args}"
    fi

    echo ""
    echo "Completed: ${method} W${w_bit}A${a_bit} at $(date)"
    echo ""
done

echo "============================================================"
echo "All ${MODEL_TYPE} baselines complete at $(date)"
echo "============================================================"

# ---- Collect results ----
echo ""
echo "Collecting results..."
python "${SCRIPT_DIR}/collect_results.py" --vl --output_dir "${EVAL_OUTPUT:-/data/results/vl_eval}"