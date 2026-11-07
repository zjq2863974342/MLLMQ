#!/bin/bash
# ============================================================================
# run_mbq_vs_mllmq.sh — Controlled MBQ vs MLLMQ Comparison Runner
#
# Runs quantization experiments for a single model with controlled variables.
# Core comparison: MBQ W4A16 vs MLLMQ W4A16 — same model, same calib data,
# same w_bit, ONLY --method differs (mbq vs mllmq).
#
# Usage:
#   ./scripts/run_mbq_vs_mllmq.sh <model_type> <model_path> \\
#       <data_path> <image_folder> [device] [mode]
#
# mode: full (default, runs all 5 experiments) | core (only MBQ vs MLLMQ W4)
#
# Examples:
#   # InternVL2-8B full matrix
#   ./scripts/run_mbq_vs_mllmq.sh internvl2 /path/to/InternVL2-8B \\
#       /data/coco_train.json /data/coco/train2014 cuda:0
#
#   # Qwen2-VL-7B core comparison only
#   ./scripts/run_mbq_vs_mllmq.sh qwen2_vl /path/to/Qwen2-VL-7B \\
#       /data/coco_train.json /data/coco/train2014 cuda:0 core
#
#   # Janus-Pro-7B (uses main_quant.py, needs COCO for fair comparison)
#   ./scripts/run_mbq_vs_mllmq.sh janus /path/to/Janus-Pro-7B \\
#       /data/coco_train.json /data/coco/train2014 cuda:0
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

# ---- Arguments ----
MODEL_TYPE="${1:?Error: model_type required (qwen2_vl|internvl2|janus)}"
MODEL_PATH="${2:?Error: model_path required}"
DATA_PATH="${3:?Error: data_path required (COCO JSON/JSONL)}"
IMAGE_FOLDER="${4:?Error: image_folder required}"
DEVICE="${5:-cuda:0}"
MODE="${6:-full}"  # full | core

MODEL_NAME=$(basename "${MODEL_PATH}")

# ---- Output root ----
OUTPUT_ROOT="${OUTPUT_ROOT:-/nas3/kangjia/mllmq/output}"
mkdir -p "${OUTPUT_ROOT}"

# Determine script to use
if [ "${MODEL_TYPE}" = "janus" ]; then
    QUANT_SCRIPT="main_quant.py"
else
    QUANT_SCRIPT="main_quant_vl.py"
fi

echo "============================================================"
echo "MBQ vs MLLMQ — Controlled Comparison"
echo "============================================================"
echo "  Model type:  ${MODEL_TYPE}"
echo "  Model path:  ${MODEL_PATH}"
echo "  Model name:  ${MODEL_NAME}"
echo "  Data:        ${DATA_PATH}"
echo "  Images:      ${IMAGE_FOLDER}"
echo "  Device:      ${DEVICE}"
echo "  Mode:        ${MODE}"
echo "  Quant script: ${QUANT_SCRIPT}"
echo "  Output root:  ${OUTPUT_ROOT}"
echo "  Start:        $(date)"
echo "============================================================"

# ---- Quantize function ----
run_quantize() {
    local method="$1"
    local w_bit="$2"
    local a_bit="$3"
    local n_samples="${4:-128}"
    local extra_args="${5:-}"

    # Use coco by default for VL models, pileval for Janus baselines
    # For Janus MLLMQ, COCO is required (for vision_mask)
    local calib="coco"
    if [ "${MODEL_TYPE}" = "janus" ] && [ "${method}" != "mllmq" ] && [ "${method}" != "mbq" ] && \
       ! echo "${extra_args}" | grep -q "data_path"; then
        # Janus baselines (RTN, AWQ) default to pileval unless coco paths provided
        # But if data_path is provided, use coco
        if [ -n "${DATA_PATH}" ] && [ -n "${IMAGE_FOLDER}" ]; then
            calib="coco"
        else
            calib="pileval"
        fi
    fi

    local suffix="w${w_bit}a${a_bit}"
    local experiment_dir="${OUTPUT_ROOT}/${MODEL_TYPE}_${method}_${suffix}"
    local scale_dir="${PROJ_DIR}/scale_cache/${method}"
    local scale_path="${scale_dir}/${MODEL_TYPE}_${MODEL_NAME}_${suffix}.pt"

    mkdir -p "${scale_dir}" "${experiment_dir}"

    echo ""
    echo "------------------------------------------------------------"
    echo "[${method}] W${w_bit}A${a_bit} — ${experiment_dir}"
    echo "------------------------------------------------------------"

    local args=(
        --model_type "${MODEL_TYPE}"
        --model_path "${MODEL_PATH}"
        --device "${DEVICE}"
        --calib_data "${calib}"
        --n_samples "${n_samples}"
        --method "${method}"
        --w_bit "${w_bit}"
        --a_bit "${a_bit}"
        --w_group 128
        --run_process
        --pack_quant
        --scale_path "${scale_path}"
        --output_path "${experiment_dir}"
    )

    # Add COCO data paths for multimodal calibration
    if [ "${calib}" = "coco" ]; then
        args+=(--data_path "${DATA_PATH}")
        args+=(--image_folder "${IMAGE_FOLDER}")
    fi

    # Append extra args
    if [ -n "${extra_args}" ]; then
        for arg in ${extra_args}; do
            args+=("${arg}")
        done
    fi

    python "${QUANT_SCRIPT}" "${args[@]}"

    echo ""
    echo "Completed: [${method}] W${w_bit}A${a_bit} → ${experiment_dir}"
    echo "  Scale cache: ${scale_path}"

    # Return experiment dir for later use
    echo "${experiment_dir}" > /tmp/mllmq_last_experiment.txt
}

# ---- Experiment Matrix ----
echo ""
echo "============================================================"
echo "Experiment Matrix"
echo "============================================================"

if [ "${MODE}" = "full" ]; then
    echo ""
    echo "Running FULL matrix (5 experiments):"
    echo "  #1  RTN    W4A16  — lower bound"
    echo "  #2  AWQ    W4A16  — activation-aware baseline"
    echo "  #3  MBQ    W8A16  — near-lossless upper bound"
    echo "  #4  MBQ    W4A16  — [CONTROL] per-layer, no reweight"
    echo "  #5  MLLMQ  W4A16  — [TREATMENT] per-channel gradient reweight"
    echo ""
    echo "  ** Core comparison: #4 vs #5 (only --method differs) **"
    echo ""

    # 1. RTN W4A16 (lower bound)
    run_quantize "rtn" 4 16

    # 2. AWQ W4A16 (baseline)
    run_quantize "awq" 4 16

    # 3. MBQ W8A16 (upper bound)
    run_quantize "mbq" 8 16

elif [ "${MODE}" = "core" ]; then
    echo ""
    echo "Running CORE comparison only (2 experiments):"
    echo "  #1  MBQ    W4A16  — [CONTROL]"
    echo "  #2  MLLMQ  W4A16  — [TREATMENT]"
    echo ""
fi

# ---- Core comparison: MBQ W4A16 vs MLLMQ W4A16 ----
# These two share identical settings except --method

echo ""
echo "============================================================"
echo "CORE COMPARISON"
echo "============================================================"
echo "  Both use: model=${MODEL_TYPE}, calib=coco, w_bit=4, n_samples=128"
echo "  ONLY difference: --method mbq vs --method mllmq"
echo ""

# 4 / C1. MBQ W4A16 (control — per-layer, no gradient reweighting)
MBQ_DIR="${OUTPUT_ROOT}/${MODEL_TYPE}_mbq_w4a16"
echo "[MBQ W4A16] CONTROL — per-layer, no per-channel gradient"
run_quantize "mbq" 4 16

# 5 / C2. MLLMQ W4A16 (treatment — per-channel gradient reweighting)
MLLMQ_DIR="${OUTPUT_ROOT}/${MODEL_TYPE}_mllmq_w4a16"
echo "[MLLMQ W4A16] TREATMENT — per-channel gradient reweighting"
run_quantize "mllmq" 4 16

# ---- Summary ----
echo ""
echo "============================================================"
echo "Quantization Complete — $(date)"
echo "============================================================"
echo ""
echo "Output directories:"
echo "  MBQ W4A16:    ${MBQ_DIR}"
echo "  MLLMQ W4A16:  ${MLLMQ_DIR}"
echo ""
echo "Scale caches:"
echo "  scale_cache/mbq/${MODEL_TYPE}_${MODEL_NAME}_w4a16.pt"
echo "  scale_cache/mllmq/${MODEL_TYPE}_${MODEL_NAME}_w4a16.pt"
echo ""
echo "============================================================"
echo "NEXT: Run evaluation (e.g., MMMU)"
echo "============================================================"