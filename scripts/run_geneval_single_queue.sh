#!/usr/bin/env bash
# GenEval 单卡队列 (GPU1 only)
# GPU0 wedged (exit 2)，用 GPU1 顺序跑所有剩余 task
# 断点续跑：count_pngs >=2212 跳过，<2212 续跑
set -u

cd /nas3/kangjia/mllmq/MLLMQ
export PYTHONPATH="/nas3/kangjia/mllmq/MLLMQ:/nas3/kangjia/mllmq/checkpoints/Janus:$PYTHONPATH"
export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

GPU=1
MODEL=/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B
OUT=/nas3/kangjia/mllmq/output/janus-pro-7b/geneval
SCALE=/nas3/kangjia/mllmq/MLLMQ/scale_cache
LOG=/nas3/kangjia/mllmq/output/logs
TARGET=2212
LOGFILE="${LOG}/geneval_single_q.log"

mkdir -p "$LOG"

count_pngs() { find "${OUT}/$1" -name '*.png' 2>/dev/null | wc -l; }

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "[$(date +%H:%M:%S)] $1"; }
ok()   { echo -e "[$(date +%H:%M:%S)] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date +%H:%M:%S)] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date +%H:%M:%S)] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date +%H:%M:%S)] ${CYAN}📌 $1${NC}"; }

run_one() {
  local tag="$1" method="$2" w="$3" scale_path="${4:-}"
  local outd="${OUT}/${tag}"
  local logf="${LOG}/geneval_${tag}.log"

  local n
  n=$(count_pngs "$tag")

  if [ "$n" -ge "$TARGET" ]; then
    ok "SKIP $tag — $n/$TARGET done"
    return 0
  fi

  local remaining=$(( TARGET - n ))
  local est_min=$(( remaining / 4 / 60 ))
  info "LAUNCH $tag ($method w${w}, $n/$TARGET existing, ~${est_min}h est)"

  local cmd=(python eval_generate_quant.py
    --model_path "$MODEL" --benchmark geneval
    --output_dir "$outd" --num_samples 4
    --quant_method "$method" --w_bit "$w"
  )
  if [ -n "$scale_path" ] && [ -f "$scale_path" ]; then
    cmd+=(--scale_path "$scale_path")
  fi

  CUDA_VISIBLE_DEVICES=$GPU "${cmd[@]}" >> "$logf" 2>&1
  local rc=$?
  local final
  final=$(count_pngs "$tag")
  if [ "$final" -ge "$TARGET" ]; then
    ok "DONE $tag — $final/$TARGET (rc=$rc)"
  else
    fail "PARTIAL $tag — $final/$TARGET (rc=$rc), will resume next run"
  fi
  return 0  # queue continues on failure
}

echo "============================================================"
echo " GenEval Single-GPU Queue (GPU1)"
echo " GPU0 wedged (exit 2) => single GPU1 sequential queue"
echo " Mode: skip >=2212 png, resume partials, sequential"
echo " Start: $(date)"
echo "============================================================" | tee -a "$LOGFILE"

# ── Pre-flight: verify GPU1 CUDA ──
CUDA_VISIBLE_DEVICES=$GPU timeout -s 9 30 python -c "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize(); print('GPU1 OK:', torch.cuda.get_device_name(0))" 2>&1 | tee -a "$LOGFILE" && ok "GPU1 CUDA 上下文自检通过" || warn "GPU1 CUDA init failed"

# ── No conflicting CUDA process check ──
if pgrep -f 'eval_generate_quant|evaluate_images|main_quant|eval_benchmarks' >/dev/null 2>&1; then
  warn "Other CUDA processes detected, waiting..."
  while pgrep -f 'eval_generate_quant|evaluate_images|main_quant|eval_benchmarks' >/dev/null 2>&1; do
    log "  Waiting... (1/30s)"
    sleep 30
  done
  ok "No conflicting CUDA processes"
fi

# ── Tasks (organised by priority for partials first, then fresh) ──
# Format: tag|method|w_bit|scale_path  (scale_path=__NONE__ for rtn)
CAMPAIGN_1_PARTIALS=(
  "mllmq_mse_w8|mbq|8|${SCALE}/mllmq/janus_mse_w8a16.pt"
  "awq_w4|awq|4|${SCALE}/awq/janus_awq_w4a16.pt"
)
CAMPAIGN_2_QUICKS=(
  "rtn_w3|rtn|3|__NONE__"
  "rtn_w4|rtn|4|__NONE__"
)
CAMPAIGN_3_W4_FRESH=(
  "mbq_block_w4|mbq|4|${SCALE}/mbq/janus_mbq_block_w4a16.pt"
  "mbq_scalar_w4|mbq|4|${SCALE}/mbq/janus_mbq_scalar_w4a16.pt"
  "mllmq_mae_w4|mbq|4|${SCALE}/mllmq/janus_mae_w4a16.pt"
  "mllmq_mse_w4|mbq|4|${SCALE}/mllmq/janus_mse_w4a16.pt"
)
CAMPAIGN_4_W3_FRESH=(
  "awq_w3|awq|3|${SCALE}/awq/janus_awq_w3a16.pt"
  "mbq_block_w3|mbq|3|${SCALE}/mbq/janus_mbq_block_w3a16.pt"
  "mbq_scalar_w3|mbq|3|${SCALE}/mbq/janus_mbq_scalar_w3a16.pt"
  "mllmq_mae_w3|mbq|3|${SCALE}/mllmq/janus_mae_w3a16.pt"
  "mllmq_mse_w3|mbq|3|${SCALE}/mllmq/janus_mse_w3a16.pt"
)

ALL_TASKS=("${CAMPAIGN_1_PARTIALS[@]}" "${CAMPAIGN_2_QUICKS[@]}" "${CAMPAIGN_3_W4_FRESH[@]}" "${CAMPAIGN_4_W3_FRESH[@]}")

TOTAL=${#ALL_TASKS[@]}
idx=0
START=$(date +%s)

for item in "${ALL_TASKS[@]}"; do
  IFS='|' read -r tag method w scale <<< "$item"

  n=$(count_pngs "$tag")
  if [ "$n" -ge "$TARGET" ]; then
    ok "[${idx=$((idx+1))}/${TOTAL}] SKIP $tag — $n/$TARGET already done"
    continue
  fi

  info "[${idx=$((idx+1))}/${TOTAL}] Launch $tag ($method w${w}, $n/$TARGET existing)"
  T0=$(date +%s)
  run_one "$tag" "$method" "$w" "$scale"

  # Quick progress snapshot
  done_count=0
  for ct in fp16 rtn_w8 rtn_w4 rtn_w3 awq_w8 awq_w4 awq_w3 \
            mbq_block_w8 mbq_block_w4 mbq_block_w3 \
            mbq_scalar_w8 mbq_scalar_w4 mbq_scalar_w3 \
            mllmq_mae_w8 mllmq_mae_w4 mllmq_mae_w3 \
            mllmq_mse_w8 mllmq_mse_w4 mllmq_mse_w3; do
    [ "$(count_pngs "$ct")" -ge "$TARGET" ] && done_count=$((done_count+1))
  done
  info "Progress: $done_count/19 complete"
done

ELAPSED=$(( $(date +%s) - START ))
echo "" | tee -a "$LOGFILE"
echo "===== Final Matrix =====" | tee -a "$LOGFILE"

for tag in fp16 rtn_w8 rtn_w4 rtn_w3 awq_w8 awq_w4 awq_w3 \
           mbq_block_w8 mbq_block_w4 mbq_block_w3 \
           mbq_scalar_w8 mbq_scalar_w4 mbq_scalar_w3 \
           mllmq_mae_w8 mllmq_mae_w4 mllmq_mae_w3 \
           mllmq_mse_w8 mllmq_mse_w4 mllmq_mse_w3; do
  n=$(count_pngs "$tag")
  if [ "$n" -ge "$TARGET" ]; then ok "  $tag: $n/$TARGET"; else warn "  $tag: $n/$TARGET"; fi
done

COMPLETED=0
for tag in fp16 rtn_w8 rtn_w4 rtn_w3 awq_w8 awq_w4 awq_w3 \
           mbq_block_w8 mbq_block_w4 mbq_block_w3 \
           mbq_scalar_w8 mbq_scalar_w4 mbq_scalar_w3 \
           mllmq_mae_w8 mllmq_mae_w4 mllmq_mae_w3 \
           mllmq_mse_w8 mllmq_mse_w4 mllmq_mse_w3; do
  [ "$(count_pngs "$tag")" -ge "$TARGET" ] && COMPLETED=$((COMPLETED + 1))
done
info "TOTAL: $COMPLETED/19 configs complete | Elapsed: $((ELAPSED/60))m $((ELAPSED%60))s"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"
echo "============================================================" | tee -a "$LOGFILE"
