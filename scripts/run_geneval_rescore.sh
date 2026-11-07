#!/bin/bash
# GenEval 重评（2026-10-09）：evaluate_images.py:227 off-by-one 修复后，重评 6 个已完成 config
# 旧 results.jsonl 备份为 results.jsonl.offbyone.bak；权重走 /nas3 持久缓存（容器重启不丢）
# Usage: nohup bash /nas3/kangjia/mllmq/scripts/run_geneval_rescore.sh > /nas3/kangjia/mllmq/output/logs/rescore_all.log 2>&1 &
set -o pipefail
cd /nas3/kangjia/mllmq/data/benchmarks_gen/geneval
EV=/nas3/kangjia/mllmq/data/benchmarks_gen/geneval/evaluation
export TORCH_HOME="$EV/.torch_home"
export MODELSCOPE_CACHE="$EV/.ms_cache"
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/geneval
LOG=/nas3/kangjia/mllmq/output/logs

# OMP 限制：否则 torch 吃满 128 核触发线程风暴（dialog §10.3 / 顺带发现，8min/config → 预期显著提速）
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# GPU0 未恢复则钉 GPU1（evaluate_images.py 的 DEVICE="cuda" 默认落 GPU0 → 会 rc=1）
bash /nas3/kangjia/mllmq/scripts/probe_gpu0_health.sh >/dev/null 2>&1 || export CUDA_VISIBLE_DEVICES=1

CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
if [ -z "$CLIP_BIN" ]; then echo "❌ CLIP 权重未就绪: $EV/.ms_cache"; exit 1; fi
if [ ! -f "$EV/.torch_home/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" ]; then
  echo "❌ Faster R-CNN 权重未就绪"; exit 1; fi
echo "CLIP_BIN=$CLIP_BIN"

for tag in fp16 rtn_w8 awq_w8 mbq_block_w8 mbq_scalar_w8 mllmq_mae_w8; do
  if [ -f "$OUT/$tag/results.jsonl" ] && [ ! -f "$OUT/$tag/results.jsonl.offbyone.bak" ]; then
    cp "$OUT/$tag/results.jsonl" "$OUT/$tag/results.jsonl.offbyone.bak"
    echo "[$(date +%H:%M:%S)] backed up $tag old results.jsonl"
  fi
  echo "=== [$(date +%H:%M:%S)] rescoring $tag ==="
  python evaluation/evaluate_images.py "$OUT/$tag/geneval" \
    --outfile "$OUT/$tag/results.jsonl" --clip-model-path "$CLIP_BIN" \
    > "$LOG/rescore_${tag}.log" 2>&1
  rc=$?
  echo "rc=$rc"
  python evaluation/summary_scores.py "$OUT/$tag/results.jsonl" 2>/dev/null | tail -11
  echo ""
done
echo "=== [$(date +%H:%M:%S)] RESCORE ALL DONE ==="
