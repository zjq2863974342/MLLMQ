#!/usr/bin/env bash
# ============================================================================
# Run MME + TextVQA on InternVL2-8B for the W3 and W8 cells (all 4 methods).
# Uses existing scale caches (mllmq/mbq/awq have w3+w8; rtn needs none).
# Usage:  bash scripts/run_more_bits.sh
# Output: output/internvl2-8b/eval/{mme,textvqa_val}/<method>_w{n}a16/*_results.json
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PROJ_ROOT=$(pwd)
MODEL_PATH=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval
SCALE_DIR="$PROJ_ROOT/scale_cache"
LOG_DIR=/nas3/kangjia/mllmq/output/logs
mkdir -p "$LOG_DIR"

TASKS="mme,textvqa_val"

# method:bit:scale_path  (empty scale => none)
CELLS=(
  "rtn:8:"
  "rtn:3:"
  "awq:8:$SCALE_DIR/awq/internvl2_InternVL2-8B_w8a16.pt"
  "awq:3:$SCALE_DIR/awq/internvl2_InternVL2-8B_w3a16.pt"
  "mbq:8:$SCALE_DIR/mbq/internvl2_w8a16.pt"
  "mbq:3:$SCALE_DIR/mbq/internvl2_w3a16.pt"
  "mllmq:8:$SCALE_DIR/mllmq/internvl2_w8a16.pt"
  "mllmq:3:$SCALE_DIR/mllmq/internvl2_w3a16.pt"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/more_bits_${TIMESTAMP}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=============================================="
echo "W3+W8 Benchmarks — $(date)   tasks=$TASKS"
echo "=============================================="

for cell in "${CELLS[@]}"; do
    IFS=: read -r method wbit scale <<< "$cell"    tag="${method}_w${wbit}"
    echo ">>> [$(date +%H:%M)] $tag"
    python eval_benchmarks.py \
        --model_path "$MODEL_PATH" \
        --method "$method" --w_bit "$wbit" --a_bit 16 \
        ${scale:+--scale_path "$scale"} \
        --tasks "$TASKS" \
        --device cuda:0 --batch_size 1 \
        --output_base "$OUTPUT_BASE" 2>&1 | sed "s/^/  [$tag] /"
done

echo "=============================================="
echo "W3+W8 Benchmarks DONE — $(date)"
echo "Log: $MAIN_LOG"
echo "=============================================="