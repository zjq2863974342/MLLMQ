#!/bin/bash
# 临时启动器：CPU 冒烟跑 T2I-CompBench 评分链路（不占 GPU）
cd /nas3/kangjia/mllmq
exec venvs/t2icomp/bin/python scripts/smoke_t2i_compbench_score.py \
  --tag-dir output/janus-pro-7b/t2i_compbench/_smoke_fp16 \
  --work /tmp/t2i_smoke --device cpu \
  --python /nas3/kangjia/mllmq/venvs/t2icomp/bin/python --force