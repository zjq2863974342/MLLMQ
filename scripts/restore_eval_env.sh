#!/bin/bash
# 容器重启后恢复 GenEval 评分环境（pip 装在 ephemeral /opt/conda，重启即丢）
# 权重不在这里下：已在 /nas3 持久缓存（evaluation/.torch_home、evaluation/.ms_cache）
set -e
pip install -q open_clip_torch clip_benchmark modelscope "numpy<2"
python -c "import torch,open_clip,clip_benchmark,modelscope;print('OK torch',torch.__version__)"
