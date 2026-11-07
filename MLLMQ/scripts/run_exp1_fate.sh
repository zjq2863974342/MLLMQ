#!/usr/bin/env bash
# ============================================================================
# Exp 1：T2I prompt 表示的“命运实验”
#   1a) probe_prompt_sensitivity.py：测 W4/W3 量化对 prompt token hidden state
#       的相对扰动幅度（逐层），作为噪声 eps 的锚点。
#   1b) 在 fp16 生成时给 prompt embedding 注入相对 RMS 噪声 eps，测 GenEval 掉多少。
#       若在 eps* 处几乎不掉 → prompt 表示不是瓶颈，T2I 方向不必再投入。
#
#   口径：GenEval 10 prompt/tag ×6 tag ×4 图 = 240 图/eps（~50min/eps）
#        eps ∈ {0.05, 0.10, 0.20, 0.40}，双卡并行（两波）
#   产出：output/exp1_prompt_sensitivity/{summary.json,noise_eps*/geneval/results.jsonl}
#
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_exp1_fate.sh \
#     > ../output/logs/exp1_fate_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache
OUT=$ROOT/output/exp1_prompt_sensitivity
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
PER_TAG=${PER_TAG:-10}
EPS_LIST=${EPS_LIST:-"0.05 0.10 0.20 0.40"}

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
mkdir -p "$OUT"

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

# ---------- 1a probe ----------
log "===== Exp1a：prompt hidden-state 量化扰动 ==="
if [ ! -f "$OUT/summary.json" ]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_prompt_sensitivity.py \
    --device cuda:0 --n_prompts 8 \
    --scales "w4=$CACHE/mllmq/janus_mae_w4a16.pt,w3=$CACHE/mllmq/janus_mae_w3a16.pt" \
    > "$LOG/exp1_probe.log" 2>&1
  [ -f "$OUT/summary.json" ] && good "probe OK" || { fail "probe 失败"; tail -5 "$LOG/exp1_probe.log"; exit 1; }
else good "probe 已存在"; fi
python - "$OUT/summary.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for k,v in r["variants"].items():
    print(f"  {k}: mean relative hidden-state error = {v['mean']:.4f} (max {v['max']:.4f} @layer{v['layer_of_max']})")
PY

# ---------- 1b noise sweep ----------
run_eps(){ # eps gpu
  local eps=$1 gpu=$2 tag=eps${1/./}; local out="$OUT/noise_$tag" gdir="$OUT/noise_$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP eps=$eps"; return 0; }
  mkdir -p "$out"
  info "[GPU$gpu] eps=$eps 生成 ${PER_TAG}/tag"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --limit_per_tag "$PER_TAG" \
    --prompt_noise_eps "$eps" \
    > "$LOG/exp1_gen_eps${eps}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] eps=$eps 生成失败"; return 1; }
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/exp1_score_eps${eps}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] eps=$eps 评分失败"; return 1; }
  good "[GPU$gpu] eps=$eps DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}

log "===== Exp1b：prompt embedding 噪声扫描 ====="
idx=0
for eps in $EPS_LIST; do
  run_eps "$eps" $((idx % 2)) &
  idx=$((idx+1))
  if [ $((idx % 2)) -eq 0 ]; then wait; fi
done
wait

log "════ Exp1 结果 ════"
python - "$OUT/summary.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for k,v in r["variants"].items():
    print(f"probe {k}: mean hidden rel err = {v['mean']:.4f}")
PY
for eps in $EPS_LIST; do
  f="$OUT/noise_eps${eps/./}/geneval/results.jsonl"
  [ -f "$f" ] && good "eps=$eps :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "eps=$eps ✗"
done
info "ALL DONE"