#!/usr/bin/env bash
# ============================================================================
# A14：干净消融 —— channel 重加权「有/无 T2I」
#   目的：现有对比 channel(vis+cap+T2I) vs scalar(vis+cap) 同时改变了两件事：
#         (a) 逐通道 vs 标量  (b) 有 T2I vs 无 T2I。
#         本实验用 MLLMQ_DISABLE_T2I=1 关掉 T2I，得到 channel(vis+cap)，
#         从而把 colors/counting 上的差异归因到「T2I」还是「逐通道」。
#
#   口径：W3（信号出现在 W3）、n=128、只评 GenEval 的 colors+counting
#         （174 prompt ×4 = 696 张，~2.5h；与已有全量 W3 逐 tag 结果可比）
#   产出：output/janus-pro-7b/ablation_a14/not2i_w3/geneval/results.jsonl
#         scale: scale_cache/mllmq/janus_mae_w3a16_n128_not2i.pt
#
#   会等本机现有 driver 全部结束（W4 正确对照 + W3 n128）再开始。
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_ablation_a14_channel_not2i.sh \
#     > /nas3/kangjia/mllmq/output/logs/a14_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/janus-pro-7b/ablation_a14
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
TAGS=${TAGS:-colors,counting}

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
[ -n "$CLIP_BIN" ] || { echo "CLIP 权重缺失"; exit 2; }

SP=$CACHE/mllmq/janus_mae_w3a16_n128_not2i.pt
TAG=not2i_w3
OUTD=$OUT/$TAG; GDIR=$OUT/$TAG/geneval

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

log "===== A14：channel(vis+cap) vs channel(vis+cap+T2I)，W3 n=128，tags=$TAGS ====="
for pat in run_mllmq_correct_comparison.sh run_local_w3n128.sh run_remote_w3_n256.sh; do
  if pgrep -f "$pat" >/dev/null 2>&1; then
    log "等待 $pat 结束..."
    while pgrep -f "$pat" >/dev/null 2>&1; do sleep 120; done
  fi
done

if [ ! -s "$SP" ]; then
  info "生成 scale（MLLMQ_DISABLE_T2I=1，w3 n=128）"
  MLLMQ_DISABLE_T2I=1 CUDA_VISIBLE_DEVICES=0 python main_quant.py \
    --model_path "$MODEL" --device cuda:0 --calib_data coco \
    --data_path "$CALIB" --image_folder "$IMG" --n_samples 128 \
    --w_bit 3 --a_bit 16 --w_group 128 --run_process \
    --method mllmq --loss_mode mae --scale_path "$SP" \
    > "$LOG/a14_scale_not2i_w3.log" 2>&1
  [ -s "$SP" ] && good "scale OK" || { fail "scale 失败"; exit 1; }
else good "scale 已存在"; fi

if [ ! -f "$GDIR/results.jsonl" ]; then
  mkdir -p "$OUTD"
  info "生成（${TAGS}）"
  CUDA_VISIBLE_DEVICES=0 python eval_generate_quant.py --model_path "$MODEL" \
    --benchmark geneval --output_dir "$OUTD" --num_samples 4 --tags "$TAGS" \
    --quant_method mllmq_mae --w_bit 3 --scale_path "$SP" \
    > "$LOG/a14_gen_not2i_w3.log" 2>&1
  [ $? -ne 0 ] && { fail "生成失败"; tail -3 "$LOG/a14_gen_not2i_w3.log"; exit 1; }
  info "评分"
  CUDA_VISIBLE_DEVICES=0 python "$EV/evaluate_images.py" "$GDIR" \
    --outfile "$GDIR/results.jsonl" --clip-model-path "$CLIP_BIN" \
    > "$LOG/a14_score_not2i_w3.log" 2>&1
  [ -f "$GDIR/results.jsonl" ] || { fail "评分失败"; exit 1; }
else good "已评测"; fi

log "════ A14 结果（逐 tag accuracy）════"
python - "$GDIR/results.jsonl" <<'PY'
import json,sys
from collections import defaultdict
t=defaultdict(lambda:[0,0])
for line in open(sys.argv[1]):
    r=json.loads(line); t[r['tag']][0]+=int(bool(r['correct'])); t[r['tag']][1]+=1
for k in sorted(t): print(f"  {k}: {t[k][0]/t[k][1]:.4f} (n={t[k][1]})")
PY
log "对照（已有全量 W3 逐 tag）：mllmq_mae colors=0.7527 counting=0.3500 | mbq_scalar colors=0.6995 counting=0.3000 | mbq_block colors=0.7261 counting=0.3469"
info "ALL DONE"