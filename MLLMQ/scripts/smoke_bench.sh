#!/usr/bin/env bash
# ============================================================================
# Smoke test: verify the eval_benchmarks.py MME/TextVQA path end-to-end
# with a tiny limit, before launching the full cells.
# Usage:  bash scripts/smoke_bench.sh [limit]
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
LIMIT=${1:-8}

echo ">>> [smoke] fp16  tasks=mme  limit=$LIMIT"
python eval_benchmarks.py \
    --method fp16 --w_bit 16 \
    --tasks mme \
    --limit "$LIMIT" \
    --device cuda:0 --batch_size 1 \
    --output_base /tmp/smoke_bench 2>&1 | tail -20

echo
echo ">>> [smoke] mllmq w4  tasks=mme  limit=$LIMIT"
python eval_benchmarks.py \
    --method mllmq --w_bit 4 --a_bit 16 \
    --scale_path scale_cache/mllmq/internvl2_w4a16.pt \
    --tasks mme \
    --limit "$LIMIT" \
    --device cuda:0 --batch_size 1 \
    --output_base /tmp/smoke_bench 2>&1 | tail -20

echo
echo "SMOKE DONE"