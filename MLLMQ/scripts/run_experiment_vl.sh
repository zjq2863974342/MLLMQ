#!/bin/bash
# ============================================================================
# MLLMQ VL Understanding Model Single Experiment Pipeline
#
# Runs: scale search -> pack -> evaluation (perplexity) for one config.
# Matches the Janus run_experiment.sh conventions.
#
# Usage:
#   ./scripts/run_experiment_vl.sh <model_type> <method> <w_bit> <a_bit> \
#       [model_path] [data_path] [image_folder] [n_samples] [device] [extra_args]
#
# Examples:
#   ./scripts/run_experiment_vl.sh qwen2_vl mbq 4 16 \
#       /path/to/Qwen2-VL-7B /data/coco_train.json /data/coco/train2014 128 cuda:0
#
#   # With reweight
#   ./scripts/run_experiment_vl.sh qwen2_vl mbq 4 16 \
#       /path/to/Qwen2-VL-7B /data/coco_train.json /data/coco/train2014 128 cuda:0 "--reweight"
#
#   # Baseline methods
#   ./scripts/run_experiment_vl.sh qwen2_vl awq 4 16 \
#       /path/to/Qwen2-VL-7B /data/coco_train.json /data/coco/train2014 128 cuda:0
#   ./scripts/run_experiment_vl.sh qwen2_vl rtn 4 16 \
#       /path/to/Qwen2-VL-7B /data/coco_train.json /data/coco/train2014 128 cuda:0
#
# Output directory naming: {method}-w{w_bit}a{a_bit}-coco{suffix}
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

# ---- Arguments ----
MODEL_TYPE="${1:-qwen2_vl}"
METHOD="${2:-mbq}"
W_BIT="${3:-4}"
A_BIT="${4:-16}"
MODEL_PATH="${5:-}"
DATA_PATH="${6:-}"
IMAGE_FOLDER="${7:-}"
N_SAMPLES="${8:-128}"
DEVICE="${9:-cuda:0}"
EXTRA_ARGS="${10:-}"

# ---- Validate required args ----
if [ -z "${MODEL_PATH}" ]; then
    echo "Error: model_path is required (arg 5)"
    exit 1
fi
if [ -z "${DATA_PATH}" ]; then
    echo "Error: data_path is required (arg 6)"
    exit 1
fi
if [ -z "${IMAGE_FOLDER}" ]; then
    echo "Error: image_folder is required (arg 7)"
    exit 1
fi

# ---- Paths ----
MODEL_NAME=$(basename "${MODEL_PATH}")

# Determine suffix: include flags present in extra_args
SUFFIX="w${W_BIT}a${A_BIT}"
if echo "${EXTRA_ARGS}" | grep -q "reweight"; then
    SUFFIX="${SUFFIX}-rw"
fi
if echo "${EXTRA_ARGS}" | grep -q "distort"; then
    SUFFIX="${SUFFIX}-distort"
fi

EXPERIMENT_NAME="${MODEL_TYPE}-${METHOD}-${SUFFIX}"

SCALE_CACHE_DIR="${PROJ_DIR}/scale_cache/${METHOD}"
mkdir -p "${SCALE_CACHE_DIR}"
SCALE_PATH="${SCALE_CACHE_DIR}/${EXPERIMENT_NAME}.pt"

OUTPUT_BASE="${OUTPUT_BASE:-/data/models/quantized}"
OUTPUT_PATH="${OUTPUT_BASE}/${EXPERIMENT_NAME}"

EVAL_OUTPUT="${EVAL_OUTPUT:-/data/results/vl_eval}/${EXPERIMENT_NAME}"

# ---- Eval data (default to calib data if not overridden) ----
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DATA_PATH}}"
EVAL_IMAGE_FOLDER="${EVAL_IMAGE_FOLDER:-${IMAGE_FOLDER}}"

echo "============================================================"
echo "MLLMQ VL Experiment: ${EXPERIMENT_NAME}"
echo "============================================================"
echo "  Model type:  ${MODEL_TYPE}"
echo "  Method:      ${METHOD}"
echo "  W-bit:       ${W_BIT}"
echo "  A-bit:       ${A_BIT}"
echo "  Data:        ${DATA_PATH}"
echo "  Images:      ${IMAGE_FOLDER}"
echo "  N samples:   ${N_SAMPLES}"
echo "  Device:      ${DEVICE}"
echo "  Extra args:  ${EXTRA_ARGS}"
echo "  Scale cache: ${SCALE_PATH}"
echo "  Output:      ${OUTPUT_PATH}"
echo "  Eval output: ${EVAL_OUTPUT}"
echo "============================================================"

# ---- Step 1: Scale Search + Packing ----
echo ""
echo "[Step 1/3] Quantization scale search + packing ..."

QUANT_ARGS=(
    --model_type "${MODEL_TYPE}"
    --model_path "${MODEL_PATH}"
    --device "${DEVICE}"
    --calib_data coco
    --data_path "${DATA_PATH}"
    --image_folder "${IMAGE_FOLDER}"
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

# Append extra args (e.g., --reweight, --distort)
if [ -n "${EXTRA_ARGS}" ]; then
    # Use xargs-style splitting for whitespace-separated extra args
    for arg in ${EXTRA_ARGS}; do
        QUANT_ARGS+=("${arg}")
    done
fi

python main_quant_vl.py "${QUANT_ARGS[@]}"
echo "[Step 1/3] Quantization complete."

# ---- Step 2: Pseudo-quant evaluation (PPL) ----
echo ""
echo "[Step 2/3] Pseudo-quant PPL evaluation ..."

EVAL_PSEUDO_ARGS=(
    --model_type "${MODEL_TYPE}"
    --model_path "${MODEL_PATH}"
    --scale_path "${SCALE_PATH}"
    --data_path "${EVAL_DATA_PATH}"
    --image_folder "${EVAL_IMAGE_FOLDER}"
    --n_samples 50
    --device "${DEVICE}"
    --w_bit "${W_BIT}"
    --a_bit "${A_BIT}"
    --eval_pseudo
    --output_dir "${EVAL_OUTPUT}"
)

python eval_vl_understanding.py "${EVAL_PSEUDO_ARGS[@]}" 2>&1 | tee "${EVAL_OUTPUT}/eval_pseudo.log"
echo "[Step 2/3] Pseudo-quant evaluation complete."

# ---- Step 3: Packed model evaluation (PPL) ----
echo ""
echo "[Step 3/3] Packed model PPL evaluation ..."

EVAL_PACKED_ARGS=(
    --model_type "${MODEL_TYPE}"
    --model_path "${MODEL_PATH}"
    --quant_model_path "${OUTPUT_PATH}"
    --data_path "${EVAL_DATA_PATH}"
    --image_folder "${EVAL_IMAGE_FOLDER}"
    --n_samples 50
    --device "${DEVICE}"
    --eval_packed
    --output_dir "${EVAL_OUTPUT}"
)

python eval_vl_understanding.py "${EVAL_PACKED_ARGS[@]}" 2>&1 | tee "${EVAL_OUTPUT}/eval_packed.log"
echo "[Step 3/3] Packed model evaluation complete."

# ---- Summary ----
echo ""
echo "============================================================"
echo "Experiment '${EXPERIMENT_NAME}' complete."
echo "  Quantized model: ${OUTPUT_PATH}"
echo "  Scale cache:     ${SCALE_PATH}"
echo "  Eval results:    ${EVAL_OUTPUT}"
echo "============================================================"