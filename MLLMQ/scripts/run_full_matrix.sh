#!/usr/bin/env bash
# OBSOLETE (2026-11-07): MBQ original repo removed; lmms-eval moved to third_party/lmms-eval.
# This InternVL2 full-matrix script is kept for reference only; the Janus CVPR pipeline is in MLLMQ/.
# ============================================================================
# MLLMQ — fill W3A16 / W8A16 matrix for InternVL2-8B
#
# Runs 8 quant + 8 eval jobs. All jobs share IDENTICAL calibration data,
# IDENTICAL quant config (group=128, act=16), IDENTICAL evaluation
# command (lmms-eval --pseudo_quant), so the only differences across
# the matrix are:
#     --method {rtn, awq, mbq, mllmq}   ×   --w_bit {3, 8}
#
# Methods are dispatched inside main_quant_vl.py via run_awq/run_mbq/
# apply_mbq/apply_awq + RTN's no-calibration quantizer. Scales for
# AWQ/MBQ/MLLMQ are cached under scale_cache/<method>/*.pt, so re-runs
# skip the expensive scale search.
#
# Usage:
#     bash scripts/run_full_matrix.sh
#
# Hardware: 4 GPUs recommended; otherwise runs sequentially (~12–14h).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

# ---- Fixed variables --------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/nas3/kangjia/mllmq/checkpoints/InternVL2}"
MODEL_TYPE="internvl2"
DATA_PATH="${DATA_PATH:-/nas3/kangjia/mllmq/data/coco/sharegpt4v_coco_calib.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/nas3/kangjia/mllmq/data/coco/train2017}"
N_SAMPLES=128
W_BIT_LIST="${W_BIT_LIST:-8 3}"           # space-separated
METHOD_LIST="${METHOD_LIST:-rtn awq mbq mllmq}"
W_GROUP=128
A_BIT=16
DEVICE="${DEVICE:-cuda:0}"

# MLLMQ eval task config (must match existing W4A16 eval)
TASKS="mmmu_val"
BATCH_SIZE=1
DEVICE_MAP_LMM_EVAL="cuda:1"

# ---- MBQ main.py (unified pseudo-quant eval) -------------------------------
THIRD_PARTY="/nas3/kangjia/mllmq/third_party"
MBQ_DIR="$THIRD_PARTY"  # OBSOLETE: MBQ repo removed
LMM_EVAL="${THIRD_PARTY}/lmms-eval/lmms_eval/__main__.py"

# ---- Iteration -------------------------------------------------------------
RUN_QUANT=true
RUN_EVAL=true

mkdir -p output

echo "============================================================"
echo "MLLMQ full matrix: ${METHOD_LIST} x W${W_BIT_LIST// /, W} (A${A_BIT})"
echo "Model:    ${MODEL_PATH}"
echo "Calib:    ${N_SAMPLES} COCO samples (seed=42)"
echo "Output:   ${PROJ_DIR}/output/internvl2_<method>_w<bit>a${A_BIT}"
echo "Quant cmd: python main_quant_vl.py"
echo "Eval  cmd: OBSOLETE (MBQ main.py removed; use lmms_eval CLI)"
echo "Start:    $(date)"
echo "============================================================"

for METHOD in ${METHOD_LIST}; do
  for W_BIT in ${W_BIT_LIST}; do
    OUT_DIR="${PROJ_DIR}/output/internvl2_${METHOD}_w${W_BIT}a${A_BIT}"
    SCALE_PATH="${PROJ_DIR}/scale_cache/${METHOD}/${MODEL_TYPE}_${MODEL_PATH##*/}_w${W_BIT}a${A_BIT}${'_rw' if [ "${METHOD}" = "mllmq" ] else ''}.pt"
    LOG_FILE="${OUT_DIR}/run.log"

    echo ""
    echo "============================================================"
    echo "[${METHOD} W${W_BIT}] -> ${OUT_DIR}"
    echo "============================================================"

    mkdir -p "${OUT_DIR}"

    # --- Quantization ------------------------------------------------------
    if [ "${RUN_QUANT}" = true ]; then
      echo "[QWANT] method=${METHOD} bit=${W_BIT}"
      case "${METHOD}" in
        rtn)
          # RTN: --run_process=False (no scale search)
          EXTRA_QUANT=("--method" "rtn" "--run_process")
          # No --scale_path needed for rtn; main_quant_vl handles internally
          ;;
        awq|mbq|mllmq)
          EXTRA_QUANT=("--method" "${METHOD}" "--run_process" "--scale_path" "${SCALE_PATH}")
          ;;
      esac

      python -W ignore main_quant_vl.py \
        --model_type "${MODEL_TYPE}" \
        --model_path "${MODEL_PATH}" \
        --calib_data coco \
        --data_path "${DATA_PATH}" \
        --image_folder "${IMAGE_FOLDER}" \
        --n_samples "${N_SAMPLES}" \
        --w_bit "${W_BIT}" \
        --a_bit "${A_BIT}" \
        --w_group "${W_GROUP}" \
        --device "${DEVICE}" \
        --output_path "${OUT_DIR}" \
        "${EXTRA_QUANT[@]}" \
        2>&1 | tee "${LOG_FILE}.quant"
    fi

    # --- Evaluation --------------------------------------------------------
    if [ "${RUN_EVAL}" = true ]; then
      # OBSOLETE: MBQ main.py removed. The Janus paper uses MLLMQ/eval_benchmarks.py.
      echo "[EVAL] SKIP method=${METHOD} bit=${W_BIT} (MBQ main.py removed; script obsolete)"
    fi

    echo "[DONE] ${METHOD} W${W_BIT} -> ${OUT_DIR}"
  done
done

echo ""
echo "============================================================"
echo "All matrix entries complete at $(date)"
echo "============================================================"