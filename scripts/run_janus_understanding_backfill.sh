#!/usr/bin/env bash
# ============================================================================
# Janus-Pro-7B 理解侧主表「补跑」双队列（在副机执行）
#
# 背景：output/janus-pro-7b/eval/ 下 MLLMQ 6 config 仅 8/42 格有结果，
#       另有 42 格停在 eval/bak_old_calib/（10-02 重校准之前的旧数据，不可用）；
#       fp16 理解侧基线 0/7 格。本脚本按当前 scale cache 补齐。
#
# 只跑 7 个 non-SEED benchmark —— SEED-Bench-2 已因 RAM OOM 废弃
# （原 run_janus_full_matrix.sh / run_janus_fp16_baseline.sh 含 5 个 seed chunk，勿直接用）。
#
# 队列（逐 task is_done，可随时中断续跑）：
#   GPU0: mllmq_mae_w3 → mllmq_mae_w4 → mllmq_mae_w8 → fp16
#   GPU1: mllmq_mse_w3 → mllmq_mse_w4 → mllmq_mse_w8
#
# 可选 BACKFILL_SCOPE=all 追加 awq / mbq_block / mbq_scalar × w3/w4/w8（6 config）——
#   ⚠️ 仅当确认「10-02 重校准改变了校准文件内容」时才需要（否则 awq/mbq 现有结果是有效的，重跑纯属浪费）。
#
# 🔴 纪律：本机不得有其它 CUDA 任务；且**不要与评分容器同时跑同一 config**
#          （eval 结果落共享 NFS 的 output/janus-pro-7b/eval/，同 config 并发会互相覆盖）。
#
# 用法（副机）：
#   cd /nas3/kangjia/mllmq && setsid nohup bash scripts/run_janus_understanding_backfill.sh \
#        > output/logs/understanding_backfill_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   只看计划不执行：  PLAN=1 bash scripts/run_janus_understanding_backfill.sh
# ============================================================================
set -o pipefail

PROJ_ROOT=/nas3/kangjia/mllmq
cd "$PROJ_ROOT/MLLMQ" || { echo "❌ 无法进入 $PROJ_ROOT/MLLMQ"; exit 1; }

MODEL=$PROJ_ROOT/checkpoints/Janus/models/Janus-Pro-7B
OUT=$PROJ_ROOT/output/janus-pro-7b/eval
CACHE=$PROJ_ROOT/MLLMQ/scale_cache
LOGS=$PROJ_ROOT/output/logs
TASKS=(mme_local textvqa_val_local mmmu_val mathvista_testmini_local \
       pope_random_local pope_popular_local pope_adversarial_local)
TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")
MAX_RETRY=2
COOLDOWN=30
PLAN=${PLAN:-0}

# 🔴 环境（照抄 run_janus_full_matrix.sh，缺一不可）：
#   HF_DATASETS_OFFLINE=1 — 离线机不设会去连 HF hub 挂住
#   OMP/MKL/OPENBLAS=4   — 本机 nproc=128，torch 默认起 128 线程 → 小张量逐条前向 thrashing
#                          （COCO 侧实测同一批 300 张：OMP=128 → 93s，OMP=4 → 8s）
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "[$(date '+%F %T')] $1"; }
good() { echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
warn() { echo -e "[$(date '+%F %T')] ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info() { echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }

# 门禁：本机（=运行本脚本的这台）不得有其它 CUDA 任务。
# 含两级 driver 名，避免 config 切换的 COOLDOWN/权重加载窗口误判「卡已空」。
BUSY_PAT='[e]val_generate_quant|[e]val_benchmarks|[s]core_dpg_coco|[c]ompute_dpg_bench|[s]core_pref|[s]core_fid|[s]core_clip|[m]ain_quant|[r]un_dpg_score_dual_queue|[r]un_dpg_score_grid4_dual_queue|[r]un_janus_full_matrix|[r]un_janus_fp16_baseline|[r]un_janus_baselines|[r]un_janus_mllmq_new|[r]un_janus_redo_w8'

is_done() { ls "$OUT/$1/$2"/*_results.json >/dev/null 2>&1; }
done_count() {
  local tag="$1" n=0 t
  for t in "${TASKS[@]}"; do is_done "$t" "$tag" && n=$((n+1)); done
  echo $n
}

run_task() {
    local gpu="$1" tag="$2" task="$3" method="$4" wbit="$5" scale="$6"

    if is_done "$task" "$tag"; then
        info "SKIP  $task/$tag（已有结果）"; return 0
    fi

    local attempt=1
    while [ "$attempt" -le "$MAX_RETRY" ]; do
        local lg="$LOGS/eval_${tag}_${task}_$(date +%Y%m%d_%H%M%S).log"
        info "[GPU$gpu] LAUNCH $task/$tag (attempt ${attempt}/${MAX_RETRY}) → $lg"
        if [ "$method" = "fp16" ]; then
            CUDA_VISIBLE_DEVICES=$gpu python -u eval_benchmarks.py \
                --model_type janus --model_path "$MODEL" \
                --method fp16 --tasks "$task" \
                --task_include_path tasks_local --output_base "$OUT" \
                --device cuda:0 --log_samples --batch_size 1 >> "$lg" 2>&1
        else
            CUDA_VISIBLE_DEVICES=$gpu python -u eval_benchmarks.py \
                --model_type janus --model_path "$MODEL" \
                --method "$method" --method_tag "$(method_tag_of "$tag")" --w_bit "$wbit" --w_group 128 \
                --scale_path "$scale" --tasks "$task" \
                --task_include_path tasks_local --output_base "$OUT" \
                --device cuda:0 --log_samples --batch_size 1 >> "$lg" 2>&1
        fi
        local rc=$?

        if is_done "$task" "$tag"; then
            good "[GPU$gpu] DONE  $task/$tag"
            return 0
        fi
        fail "[GPU$gpu] FAIL  $task/$tag rc=$rc"
        attempt=$(( attempt + 1 ))
        [ "$attempt" -le "$MAX_RETRY" ] && sleep $COOLDOWN
    done
    fail "[GPU$gpu] GAVE-UP $task/$tag"
    return 1
}

# mllmq_mae_w3a16 -> mllmq_mae
method_tag_of() { echo "${1%_w[0-9]a16}"; }

run_variant() {
    local gpu="$1" tag="$2" method="$3" wbit="$4" scale="$5"
    local t
    for t in "${TASKS[@]}"; do
        run_task "$gpu" "$tag" "$t" "$method" "$wbit" "$scale"
    done
    good "[GPU$gpu] $tag 完成 —— $(done_count "$tag")/7 格"
}

queue_gpu0() {
    run_variant 0 mllmq_mae_w3a16 mllmq 3 "$CACHE/mllmq/janus_mae_w3a16.pt"
    run_variant 0 mllmq_mae_w4a16 mllmq 4 "$CACHE/mllmq/janus_mae_w4a16.pt"
    run_variant 0 mllmq_mae_w8a16 mllmq 8 "$CACHE/mllmq/janus_mae_w8a16.pt"
    run_variant 0 fp16          fp16  "" ""            # fp16 无 scale cache
    if [ "${BACKFILL_SCOPE:-mllmq}" = "all" ]; then
        warn "BACKFILL_SCOPE=all → 追加 awq / mbq 6 config"
        local w
        for w in 3 4 8; do run_variant 0 "awq_w${w}a16"       awq  "$w" "$CACHE/awq/janus_awq_w${w}a16.pt"; done
        for w in 3 4 8; do run_variant 0 "mbq_block_w${w}a16" mbq  "$w" "$CACHE/mbq/janus_mbq_block_w${w}a16.pt"; done
        for w in 3 4 8; do run_variant 0 "mbq_scalar_w${w}a16" mbq "$w" "$CACHE/mbq/janus_mbq_scalar_w${w}a16.pt"; done
    fi
    good "GPU0 队列全部完成"
}
queue_gpu1() {
    run_variant 1 mllmq_mse_w3a16 mllmq 3 "$CACHE/mllmq/janus_mse_w3a16.pt"
    run_variant 1 mllmq_mse_w4a16 mllmq 4 "$CACHE/mllmq/janus_mse_w4a16.pt"
    run_variant 1 mllmq_mse_w8a16 mllmq 8 "$CACHE/mllmq/janus_mse_w8a16.pt"
    good "GPU1 队列全部完成"
}

# ────────────────────────── main ──────────────────────────
log "===== Janus 理解侧主表补跑 · 双队列（$(hostname)）====="
log "GPU0: mae_w3 → mae_w4 → mae_w8 → fp16"
log "GPU1: mse_w3 → mse_w4 → mse_w8"
log "scope=${BACKFILL_SCOPE:-mllmq}（all 则追加 awq/mbq 6 config）· benchmark=${#TASKS[@]} 个（不含已废弃的 SEED-Bench-2）"

echo ""; log "── 补跑前现状 ──"
for tag in mllmq_mae_w3a16 mllmq_mae_w4a16 mllmq_mae_w8a16 \
           mllmq_mse_w3a16 mllmq_mse_w4a16 mllmq_mse_w8a16 fp16; do
    printf "   %-22s %s/7 格\n" "$tag" "$(done_count "$tag")"
done
echo ""

# 依赖自检
[ -d "$MODEL" ]  || { fail "模型目录不存在: $MODEL"; exit 2; }
[ -d "$OUT" ]    || { fail "结果目录不存在: $OUT"; exit 2; }
miss=0
for w in 3 4 8; do
    for m in mae mse; do
        f="$CACHE/mllmq/janus_${m}_w${w}a16.pt"
        [ -f "$f" ] || { fail "缺 scale cache: $f"; miss=1; }
    done
done
[ "$miss" -eq 0 ] && good "mllmq scale cache 6/6 齐备（10-02 重校准后重建）"
for w in 3 4 8; do
    ls "$OUT/mme_local"/*"w${w}a16" >/dev/null 2>&1 || warn "w${w} 的 mme_local 结果目录尚未建立（首次跑即会建）"
done
good "huggingface_hub: $(python -c 'import huggingface_hub;print(huggingface_hub.__version__)' 2>/dev/null)"

if [ "$PLAN" = "1" ]; then
    echo ""; info "PLAN=1 → 仅打印计划，未执行。"
    exit 0
fi

# 门禁
if pgrep -f "$BUSY_PAT" >/dev/null 2>&1; then
    fail "检测到其它 GPU 任务在跑 → 拒绝启动"
    pgrep -af "$BUSY_PAT" | head -5
    exit 3
fi
for g in 0 1; do
    if CUDA_VISIBLE_DEVICES=$g timeout -s 9 60 python -c \
         "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize()" >/dev/null 2>&1; then
        good "GPU${g} CUDA 上下文自检通过"
    else
        fail "GPU${g} 无法创建 CUDA 上下文 → 拒绝启动"; exit 2
    fi
done

queue_gpu0 & PID0=$!
queue_gpu1 & PID1=$!
wait $PID0; RC0=$?
wait $PID1; RC1=$?

echo ""; log "════════ 最终清点 ════════"
for tag in mllmq_mae_w3a16 mllmq_mae_w4a16 mllmq_mae_w8a16 \
           mllmq_mse_w3a16 mllmq_mse_w4a16 mllmq_mse_w8a16 fp16; do
    n=$(done_count "$tag")
    if [ "$n" -eq 7 ]; then good "  $tag  $n/7"; else fail "  $tag  $n/7"; fi
done
if [ "$RC0" -eq 0 ] && [ "$RC1" -eq 0 ]; then
    info "🎉 理解侧补跑 ALL DONE"
else
    fail "队列出错 rc0=$RC0 rc1=$RC1（逐 task is_done，重跑本脚本即可续）"
fi
echo ""; info "提示：结果落共享 NFS → 在评分容器执行 python3 scripts/build_score_json.py 刷新 output/score.json"