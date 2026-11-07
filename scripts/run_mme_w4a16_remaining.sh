#!/usr/bin/env bash
# Run remaining W4A16 methods for MME local
# Run each sequentially to avoid GPU contention
set -uo pipefail
cd /nas3/kangjia/mllmq/MLLMQ

export HF_DATASETS_OFFLINE=1

# MLLMQ W4A16
python -u eval_benchmarks.py \
    --method mllmq --w_bit 4 \
    --scale_path scale_cache/mllmq/internvl2_w4a16.pt \
    --tasks mme_local \
    --batch_size 1 --device cuda:0 \
    --output_base /nas3/kangjia/mllmq/output/internvl2-8b/eval \
    > /nas3/kangjia/mllmq/output/logs/eval_mllmq_w4a16_mme_local.log 2>&1

echo "MLLMQ MME done"

# MBQ W4A16
python -u eval_benchmarks.py \
    --method mbq --w_bit 4 \
    --scale_path scale_cache/mbq/internvl2_w4a16.pt \
    --tasks mme_local \
    --batch_size 1 --device cuda:0 \
    --output_base /nas3/kangjia/mllmq/output/internvl2-8b/eval \
    > /nas3/kangjia/mllmq/output/logs/eval_block_w4a16_mme_local.log 2>&1

echo "MBQ MME done"

# AWQ W4A16
python -u eval_benchmarks.py \
    --method awq --w_bit 4 \
    --scale_path scale_cache/awq/internvl2_InternVL2-8B_w4a16.pt \
    --tasks mme_local \
    --batch_size 1 --device cuda:0 \
    --output_base /nas3/kangjia/mllmq/output/internvl2-8b/eval \
    > /nas3/kangjia/mllmq/output/logs/eval_awq_w4a16_mme_local.log 2>&1

echo "AWQ MME done"
