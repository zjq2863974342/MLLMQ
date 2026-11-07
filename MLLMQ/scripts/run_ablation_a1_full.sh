#!/usr/bin/env bash
# ============================================================================
# A1（另一台机专用）：补全"无重加权"基线的【全量 GenEval】三点
#   目的：与正在本机跑的 ★正确对照（mbq_scalar / mllmq_mae 全量）拼成完整的三模态消融
#         none(mbq_block) vs scalar(mbq_scalar) vs channel(mllmq)
#   规模：mbq_block_w4 × n{64,128,256}，每点 553 prompt ×4 = 2212 张
#   产出：output/janus-pro-7b/ablation_a1_full/<tag>/geneval/results.jsonl
#   ⚠️ 与其它任务无冲突：只读现有 scale，输出走独立目录
#
# Usage（另一台机）：
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_ablation_a1_full.sh \
#     > /nas3/kangjia/mllmq/output/logs/a1_full_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CACHE=$ROOT/MLLMQ/scale_cache
OUTBASE=$ROOT/output/janus-pro-7b/ablation_a1_full
LOG=$ROOT/output/logs
EV=$ROOT/data/benchmarks_gen/geneval/evaluation

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)

QFILE=/tmp/a1_full_queue.$$; QLOCK=/tmp/a1_full_queue.lock
build_queue(){
  : > "$QFILE"
  echo "mbq_block_w4_n64_full|mbq_block|4|$CACHE/mbq/janus_mbq_block_w4a16_n64.pt" >> "$QFILE"
  echo "mbq_block_w4_n128_full|mbq_block|4|$CACHE/mbq/janus_mbq_block_w4a16_n128.pt" >> "$QFILE"
  echo "mbq_block_w4_n256_full|mbq_block|4|$CACHE/mbq/janus_mbq_block_w4a16.pt" >> "$QFILE"
}
pop_job(){ local j; exec 9>"$QLOCK"; flock 9; j=$(head -1 "$QFILE" 2>/dev/null)
  [ -n "$j" ] && { tail -n +2 "$QFILE" > "$QFILE.$$.tmp"; mv "$QFILE.$$.tmp" "$QFILE"; }
  flock -u 9; exec 9>&-; echo "$j"; }

run_job(){ # tag qm w scale gpu
  local tag=$1 qm=$2 w=$3 sp=$4 gpu=$5
  local out="$OUTBASE/$tag" gdir="$OUTBASE/$tag/geneval"
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f "$gdir/results.jsonl" ] && { good "[GPU$gpu] SKIP $tag"; return 0; }
  [ -s "$sp" ] || { fail "[GPU$gpu] 缺 scale $sp"; return 1; }
  mkdir -p "$out"
  info "[GPU$gpu] RUN $tag 全量 GenEval（2212 张）"
  python eval_generate_quant.py --model_path "$MODEL" --benchmark geneval \
    --output_dir "$out" --num_samples 4 --quant_method "$qm" --w_bit "$w" --scale_path "$sp" \
    > "$LOG/a1_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "[GPU$gpu] $tag 生成失败"; return 1; }
  info "[GPU$gpu] RUN $tag 评分"
  python "$EV/evaluate_images.py" "$gdir" --outfile "$gdir/results.jsonl" \
    --clip-model-path "$CLIP_BIN" > "$LOG/a1_score_${tag}.log" 2>&1
  [ -f "$gdir/results.jsonl" ] || { fail "[GPU$gpu] $tag 评分失败"; return 1; }
  good "[GPU$gpu] $tag DONE :: $(python "$EV/summary_scores.py" "$gdir/results.jsonl" 2>/dev/null | grep -i '^Overall')"
}
lane(){ local gpu=$1 job tag qm w sp; while :; do job=$(pop_job); [ -z "$job" ] && { log "[GPU$gpu] 队列空"; return; }
  IFS='|' read -r tag qm w sp <<< "$job"; run_job "$tag" "$qm" "$w" "$sp" "$gpu"; done; }

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }; trap cleanup SIGINT SIGTERM
log "===== A1：mbq_block（none reweight）全量 GenEval × n{64,128,256} ====="
[ -n "$CLIP_BIN" ] || { fail "CLIP 权重缺失"; exit 2; }
mkdir -p "$OUTBASE"; build_queue
for g in 0 1; do lane "$g" & done; wait
log "════ 清点 ════"
for t in mbq_block_w4_n64_full mbq_block_w4_n128_full mbq_block_w4_n256_full; do
  f="$OUTBASE/$t/geneval/results.jsonl"
  [ -f "$f" ] && good "$t :: $(python "$EV/summary_scores.py" "$f" 2>/dev/null | grep -i '^Overall')" || fail "$t ✗"
done
rm -f "$QFILE"; info "ALL DONE"