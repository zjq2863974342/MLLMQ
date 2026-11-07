#!/usr/bin/env bash
# ============================================================================
# Exp2 诊断（不评测，只判断“值不值得评测”）
#   背景：发现 GradCacheHook 旧逻辑会跳过 vis/cap 都为空的样本 → 纯文本生成样本
#         的 t2i 梯度从未被使用（这就是 A13/L0/A14 全部无效的根因）。现已修正。
#   本脚本做三件事：
#     1) 用旧 calib（128 理解 + 128 纯文本生成）重跑一次 mllmq scale，
#        打开 MLLMQ_DEBUG_GEN_RW=1，确认 t2i 梯度是否终于出现；
#     2) 用 128 理解 + 128 教师强制生成（gen_tf）合并 calib 跑一次 scale，
#        打开 gen 梯度统计；
#     3) 对比两份 scale 与基线 janus_mae_w4a16.pt 的逐层差异（inspect_scale_diff）。
#   **只有 scale 移动了才值得进入评测**。
#
# 产出：/tmp/exp2_scale_oldcalib.pt, /tmp/exp2_scale_gentf.pt, 日志
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_exp2_diagnostic.sh \
#     > ../output/logs/exp2_diag_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
LOG=$ROOT/output/logs
BASE_SCALE=$ROOT/MLLMQ/scale_cache/mllmq/janus_mae_w4a16.pt
OLD_CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
NEW_CALIB=$ROOT/data/coco/janus_mixed_gen_tf_256.json
GPU=${GPU:-0}

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

DRIVER_PID=$$; cleanup(){ echo interrupted; pkill -P $DRIVER_PID 2>/dev/null; exit 130; }
trap cleanup SIGINT SIGTERM

run_scale(){ # calib out logname debug
  local calib=$1 out=$2 name=$3 dbg=$4
  info "scale: $name"
  MLLMQ_DEBUG_GEN_RW=$dbg CUDA_VISIBLE_DEVICES=$GPU python main_quant.py \
    --model_path "$MODEL" --device cuda:0 --calib_data coco \
    --data_path "$calib" --image_folder "$IMG" --n_samples 128 \
    --w_bit 4 --a_bit 16 --w_group 128 --run_process \
    --method mllmq --loss_mode mae --scale_path "$out" \
    > "$LOG/exp2_${name}.log" 2>&1
  [ -s "$out" ] && good "$name scale OK" || { fail "$name scale 失败"; tail -5 "$LOG/exp2_${name}.log"; return 1; }
  grep -E "\[gen_rw\]" "$LOG/exp2_${name}.log" || echo "  (no [gen_rw] lines)"
}

log "===== Exp2 诊断 ====="
[ -s "$BASE_SCALE" ] || { fail "缺基线 scale"; exit 1; }

# 0) 合并 calib
python scripts/build_exp2_mixed_calib.py > "$LOG/exp2_build_calib.log" 2>&1
[ -s "$NEW_CALIB" ] && good "calib 合并 OK → $NEW_CALIB" || { fail "calib 合并失败"; tail -5 "$LOG/exp2_build_calib.log"; exit 1; }

# 1) 旧 calib（纯文本生成）——验证 t2i 梯度是否终于出现
run_scale "$OLD_CALIB" /tmp/exp2_scale_oldcalib.pt oldcalib 1
# 2) 新 calib（教师强制生成）
run_scale "$NEW_CALIB" /tmp/exp2_scale_gentf.pt gentf 1

# 3) scale diff
log "════ scale diff vs 基线 kn128 W4 ════"
for n in oldcalib gentf; do
  echo "--- $n ---"
  python scripts/inspect_scale_diff.py "$BASE_SCALE" "/tmp/exp2_scale_${n}.pt" 2>/dev/null | head -8
done

log "判定：若两份 scale 都 0 层移动 → 停止 Exp2 评测；否则进入 25/tag 评测"
info "ALL DONE"