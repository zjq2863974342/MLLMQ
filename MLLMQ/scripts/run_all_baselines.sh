#!/bin/bash
# ============================================================================
# MLLMQ All Baselines Runner
#
# Runs Phase 1 experiments for Janus: all methods at W4A16.
# MBQ and MLLMQ are run as separate methods for proper comparison.
#
# Usage:
#   ./scripts/run_all_baselines.sh [device] [coco_data_path] [coco_image_folder]
#
# Examples:
#   # Text-only baselines only (RTN, AWQ, MBQ)
#   ./scripts/run_all_baselines.sh cuda:0
#
#   # Full comparison including MLLMQ with COCO multimodal calib
#   ./scripts/run_all_baselines.sh cuda:0 \\
#       /data/coco/annotations/captions_train2014.json \\
#       /data/coco/train2014
# ============================================================================

set -euo pipefail

DEVICE="${1:-cuda:0}"
COCO_DATA="${2:-}"
COCO_IMAGES="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_experiment.sh"

echo "============================================================"
echo "MLLMQ Phase 1 - All Baselines + MLLMQ Runner"
echo "============================================================"
echo "  Device:      ${DEVICE}"
echo "  COCO data:   ${COCO_DATA:-N/A (MLLMQ skipped)}"
echo "  COCO images: ${COCO_IMAGES:-N/A}"
echo "  Start:       $(date)"
echo "============================================================"

# ---- Text-only baselines (pileval calib) ----
# These use pileval text-only calibration — no vision_mask, no gradient reweighting.
EXPERIMENTS=(
    # Method  W-bit  A-bit  Calib    N-samples  Extra   Label
    "rtn       4      16     pileval  128        ''      'RTN W4A16 lower bound'"
    "awq       4      16     pileval  128        ''      'AWQ W4A16'"
    "mbq       4      16     pileval  128        ''      'MBQ W4A16 (no reweight, per-layer)'"
)

# ---- MLLMQ with COCO multimodal calibration ----
# MLLMQ uses per-channel gradient reweighting — requires COCO for vision_mask.
if [ -n "${COCO_DATA}" ] && [ -n "${COCO_IMAGES}" ]; then
    EXPERIMENTS+=(
        "mllmq     4      16     coco      128        '--data_path ${COCO_DATA} --image_folder ${COCO_IMAGES}'  'MLLMQ W4A16 (per-channel reweight)'"
    )
fi

echo ""
echo "Experiments to run: ${#EXPERIMENTS[@]}"
echo ""

for exp_line in "${EXPERIMENTS[@]}"; do
    read -r method w_bit a_bit calib n_samples extra label <<< "${exp_line}"

    echo "------------------------------------------------------------"
    echo "Running: ${method} W${w_bit}A${a_bit} on ${calib} (${n_samples} samples)"
    echo "  ${label}"
    echo "------------------------------------------------------------"

    if [ "${extra}" = "''" ] || [ -z "${extra}" ]; then
        bash "${RUNNER}" "${method}" "${w_bit}" "${a_bit}" "${calib}" "${n_samples}" "${DEVICE}" ""
    else
        bash "${RUNNER}" "${method}" "${w_bit}" "${a_bit}" "${calib}" "${n_samples}" "${DEVICE}" "${extra}"
    fi

    echo ""
    echo "Completed: ${method} W${w_bit}A${a_bit} at $(date)"
    echo ""
done

echo "============================================================"
echo "All baselines complete at $(date)"
echo "============================================================"