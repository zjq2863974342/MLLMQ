#!/bin/bash
# ============================================================================
# MLLMQ Single Experiment Pipeline
#
# Runs the full pipeline: scale search -> pack -> evaluation
# for a single (method, w_bit, a_bit) configuration.
#
# Usage:
#   ./scripts/run_experiment.sh <method> <w_bit> <a_bit> [calib_data] [n_samples] [device] [extra_args]
#
# Examples:
#   # Janus MBQ W4A16 (pileval, no reweight)
#   ./scripts/run_experiment.sh mbq 4 16 pileval 128 cuda:1
#
#   # Janus MLLMQ W4A16 (COCO multimodal calib + per-channel reweight)
#   ./scripts/run_experiment.sh mllmq 4 16 coco 128 cuda:0 \\
#       "--data_path /data/coco_train.json --image_folder /data/coco/train2014"
#
#   # Janus AWQ W4A16 baseline
#   ./scripts/run_experiment.sh awq 4 16 pileval 128 cuda:1
#
# Output directory naming: {method}-w{w_bit}a{a_bit}-{calib_data}
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

# ---- Arguments ----
METHOD="${1:-mbq}"
W_BIT="${2:-4}"
A_BIT="${3:-16}"
CALIB="${4:-pileval}"
N_SAMPLES="${5:-128}"
DEVICE="${6:-cuda:0}"
EXTRA_ARGS="${7:-}"

# ---- Paths ----
MODEL_PATH="${MODEL_PATH:-/DATA/DATA4/zjq_data/Janus/checkpoints/Janus-Pro-7B}"
OUTPUT_BASE="${OUTPUT_BASE:-/DATA/DATA4/zjq_data/Janus/checkpoints}"
SCALE_CACHE="${PROJ_DIR}/scale_cache/${METHOD}"

EXPERIMENT_NAME="${METHOD}-w${W_BIT}a${A_BIT}-${CALIB}"
SCALE_PATH="${SCALE_CACHE}/janus_${EXPERIMENT_NAME}.pt"
OUTPUT_PATH="${OUTPUT_BASE}/Janus-Pro-7B-${EXPERIMENT_NAME}"
DEMO_OUTPUT="${DEMO_OUTPUT:-/DATA/DATA4/zjq_data/demo_output}/${EXPERIMENT_NAME}"

echo "============================================================"
echo "MLLMQ Experiment: ${EXPERIMENT_NAME}"
echo "============================================================"
echo "  Method:      ${METHOD}"
echo "  W-bit:       ${W_BIT}"
echo "  A-bit:       ${A_BIT}"
echo "  Calib data:  ${CALIB}"
echo "  N samples:   ${N_SAMPLES}"
echo "  Device:      ${DEVICE}"
echo "  Output:      ${OUTPUT_PATH}"
echo "  Scale cache: ${SCALE_PATH}"
echo "  Demo output: ${DEMO_OUTPUT}"
echo "============================================================"

# ---- Step 1: Scale Search + Packing ----
echo ""
echo "[Step 1/3] Quantization scale search + packing ..."

QUANT_ARGS=(
    --model_path "${MODEL_PATH}"
    --device "${DEVICE}"
    --calib_data "${CALIB}"
    --n_samples "${N_SAMPLES}"
    --method "${METHOD}"
    --w_bit "${W_BIT}"
    --a_bit "${A_BIT}"
    --w_group 128
    --run_process
    --pack_quant
    --scale_path "${SCALE_PATH}"
    --output_path "${OUTPUT_PATH}"
)

# Append extra args (e.g., --data_path, --image_folder, --reweight)
if [ -n "${EXTRA_ARGS}" ]; then
    for arg in ${EXTRA_ARGS}; do
        QUANT_ARGS+=("${arg}")
    done
fi

python main_quant.py "${QUANT_ARGS[@]}"
echo "[Step 1/3] Quantization complete."

# ---- Step 2: T2I Evaluation ----
echo ""
echo "[Step 2/3] T2I evaluation (FID) ..."
mkdir -p "${DEMO_OUTPUT}"

DEMO_ARGS_T2I=(
    --orig_model_path "${MODEL_PATH}"
    --mbq_model_path "${OUTPUT_PATH}"
    --model_type both
    --task t2i
    --n_samples 50
    --device "${DEVICE}"
    --output_dir "${DEMO_OUTPUT}"
    --w_bit "${W_BIT}"
    --a_bit "${A_BIT}"
)

python demo.py "${DEMO_ARGS_T2I[@]}"
echo "[Step 2/3] T2I evaluation complete."

# ---- Step 3: IQA Evaluation ----
echo ""
echo "[Step 3/3] IQA evaluation (BLEU-1) ..."

DEMO_ARGS_IQA=(
    --orig_model_path "${MODEL_PATH}"
    --mbq_model_path "${OUTPUT_PATH}"
    --model_type both
    --task iqa
    --n_samples 50
    --device "${DEVICE}"
    --output_dir "${DEMO_OUTPUT}"
    --w_bit "${W_BIT}"
    --a_bit "${A_BIT}"
)

python demo.py "${DEMO_ARGS_IQA[@]}"
echo "[Step 3/3] IQA evaluation complete."

# ---- Summary ----
echo ""
echo "============================================================"
echo "Experiment '${EXPERIMENT_NAME}' complete."
echo "  Quantized model: ${OUTPUT_PATH}"
echo "  Demo output:     ${DEMO_OUTPUT}"
echo "============================================================"