#!/usr/bin/env bash
# ============================================================================
# 10-min progress monitor — ALL 17 cells (13 TextVQA + 4 MMMU W4 retry)
# Reports: ✅ done (acc), 🔄 running (%), 💀 crashed, ⏳ queued
#updates dialog.md progress table.
# ============================================================================
set -uo pipefail

PROJ_ROOT=/nas3/kangjia/mllmq
OUTPUT_BASE="$PROJ_ROOT/output/internvl2-8b/eval"
LOG_DIR="$PROJ_ROOT/output/logs"
DIALOG="$PROJ_ROOT/dialog.md"
TS=$(date '+%Y-%m-%d %H:%M:%S')

# ── Check if any eval is actually running ──
RUNNING=$(ps aux | grep "eval_benchmarks.py" | grep -v grep | wc -l)
GPU0=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | grep "^0," || echo "0, 0 MiB, 0 MiB, 0 %")
GPU1=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | grep "^1," || echo "1, 0 MiB, 0 MiB, 0 %")
GPU0_MEM=$(echo "$GPU0" | cut -d',' -f2 | tr -d ' ')
GPU1_MEM=$(echo "$GPU1" | cut -d',' -f2 | tr -d ' ')
GPU0_UTIL=$(echo "$GPU0" | cut -d',' -f4 | tr -d ' ')
GPU1_UTIL=$(echo "$GPU1" | cut -d',' -f4 | tr -d ' ')

# ── Helper: get result accuracy ──
get_result() {
    local tag=$1 task=$2
    local d="$OUTPUT_BASE/$task/$tag"
    local rf=$(ls "$d"/*_results.json 2>/dev/null | head -1)
    if [ -z "$rf" ]; then echo ""; return; fi
    python3 -c "
import json
d=json.load(open('$rf'))
r=d.get('results',{}).get('$task',{})
for k in ['exact_match,none','exact_match','accuracy,none','accuracy','acc,none','acc','mmmu_acc,none','mmmu_acc']:
    if k in r: print(f'{r[k]:.4f}'); break
else: print('?')
" 2>/dev/null
}

# ── Helper: check if done ──
is_done() {
    local task=$1 tag=$2
    local d="$OUTPUT_BASE/$task/$tag"
    [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1
}

# ── Helper: get cell status ──
cell_status() {
    local tag=$1 task=$2
    if is_done "$task" "$tag"; then
        local acc=$(get_result "$tag" "$task")
        echo "✅ ${acc}"
        return
    fi
    # Search for log files (newest first)
    local logf=$(ls "$LOG_DIR"/final_${tag}_${task}*.log \
                   "$LOG_DIR"/quad2_${tag}_${task}*.log \
                   "$LOG_DIR"/quad_tvqa_${tag}_*.log \
                   "$LOG_DIR"/quad_${tag}_${task}*.log \
                   "$LOG_DIR"/mmmu_w4_${tag}*.log \
                   2>/dev/null | tail -1)
    if [ -n "$logf" ] && [ -f "$logf" ]; then
        # Check for CUDA crash
        if grep -q 'CUDA driver error' "$logf" 2>/dev/null || \
           grep -q 'RuntimeError' "$logf" 2>/dev/null; then
            local mod_time=$(stat -c "%y" "$logf" 2>/dev/null | cut -d'.' -f1 | cut -d' ' -f2)
            local lines=$(wc -l < "$logf" 2>/dev/null || echo 0)
            echo "💀 CRASHED ${mod_time} (${lines}L)"
            return
        fi
        if grep -q 'AttributeError' "$logf" 2>/dev/null; then
            echo "💀 attr_bug (submission key)"
            return
        fi
        local lines=$(wc -l < "$logf" 2>/dev/null || echo 0)
        if [ "$lines" -lt 30 ] 2>/dev/null; then
            echo "💀 crash (${lines}L)"
            return
        fi
        # Try tqdm progress
        local responding=$(tail -20 "$logf" | grep -oP '\d+/\d+' | tail -1)
        if [ -n "$responding" ]; then
            echo "🔄 ${responding}"
            return
        fi
        local pct=$(grep -oP '\d+%' "$logf" | tail -1)
        if [ -n "$pct" ]; then
            echo "🔄 ${pct}"
            return
        fi
        echo "🔄 running"
    else
        echo "⏳ queued"
    fi
}

# ── Build TextVQA table (13 cells) ──
METHODS=("fp16" "rtn" "awq" "mbq" "mllmq")
BITS=(4 8 3)
TVQA_TASK="textvqa_val_local"

TVQA_ROWS=""
for method in "${METHODS[@]}"; do
    row="| **${method}** |"
    for w_bit in "${BITS[@]}"; do
        if [ "$method" = "fp16" ] && [ "$w_bit" != "4" ]; then
            row+=" — |"
            continue
        fi
        if [ "$method" = "fp16" ]; then
            tag="fp16"
        else
            tag="${method}_w${w_bit}a16"
        fi
        status=$(cell_status "$tag" "$TVQA_TASK")
        row+=" ${status} |"
    done
    TVQA_ROWS+="$row
"
done

# ── Build MMMU table (4 cells) ──
MMMU_TASK="mmmu_val"
MMMU_METHODS=("rtn" "awq" "mbq" "mllmq")
MMMU_ROWS=""
for method in "${MMMU_METHODS[@]}"; do
    tag="${method}_w4a16"
    status=$(cell_status "$tag" "$MMMU_TASK")
    MMMU_ROWS+="| **${method}** | ${status} |
"
done

# Check which workers are active
W0_LOG=$(ls "$LOG_DIR"/final_tvqa_w0.log "$LOG_DIR"/quad2_w0_gpu0.log 2>/dev/null | head -1)
W1_LOG=$(ls "$LOG_DIR"/final_tvqa_w1.log "$LOG_DIR"/quad2_w1_gpu0.log 2>/dev/null | head -1)
W2_LOG=$(ls "$LOG_DIR"/final_tvqa_w2.log "$LOG_DIR"/quad2_w2_gpu1.log 2>/dev/null | head -1)
W3_LOG=$(ls "$LOG_DIR"/final_tvqa_w3.log "$LOG_DIR"/quad2_w3_gpu1.log 2>/dev/null | head -1)

get_worker_current() {
    local wlog=$1
    if [ -z "$wlog" ] || [ ! -f "$wlog" ]; then echo "-"; return; fi
    tail -3 "$wlog" 2>/dev/null | grep -E "START|OK|FAIL|SKIP" | tail -1 | sed 's/^[[:space:]]*//'
}

W0_CUR=$(get_worker_current "$W0_LOG")
W1_CUR=$(get_worker_current "$W1_LOG")
W2_CUR=$(get_worker_current "$W2_LOG")
W3_CUR=$(get_worker_current "$W3_LOG")

# ═══════════════════════════════════════════════════
# Generate the update block
# ═══════════════════════════════════════════════════

MONITOR_STDOUT="═══════════════════════════════════════════════════
TextVQA Progress — $TS
═══════════════════════════════════════════════════

**GPU**: GPU0=${GPU0_MEM}/81251 MiB (${GPU0_UTIL}%) | GPU1=${GPU1_MEM}/81251 MiB (${GPU1_UTIL}%) | $RUNNING eval process(es) running

**Workers**:
- W0 (GPU0): $W0_CUR
- W1 (GPU0): $W1_CUR
- W2 (GPU1): $W2_CUR
- W3 (GPU1): $W3_CUR

"

echo "$MONITOR_STDOUT"

# Build the MARKDOWN block for dialog.md update
DIALOG_BLOCK="### 📊 实验进展全矩阵（自动更新，$TS）

**GPU**: GPU0=${GPU0_MEM}/81251 MiB (${GPU0_UTIL}%) | GPU1=${GPU1_MEM}/81251 MiB (${GPU1_UTIL}%) | $RUNNING eval process(es) running

---

### TextVQA 多 Benchmark 进度（13 cells: fp16 + 4 methods x 3 bit widths）

| Method | W4A16 | W8A16 | W3A16 |
|--------|-------|-------|-------|
$TVQA_ROWS

---

### MMMU W4A16 进度（4 cells — $((4 - $(echo "$MMMU_ROWS" | grep -c '✅')) retries needed)

| Method | W4A16 |
|--------|-------|
$MMMU_ROWS

---

### 🔴 2026-09-17 故障诊断与修复记录

| # | 故障 | 时间 | 影响 | 根因 | 修复 |
|---|------|------|------|------|------|
| 1 | **TextVQA W4x4+fp16 全崩** | 14:55-15:12 | 78min 推理白费 | \`process_results()\` 返回 \`submission\` key，聚合时 \`args.output_path=None\` 崩溃 | yaml 已删 submission metric；\`utils.py\` + \`textvqa_utils.py\` 防御性移除；\`run_final_queue.sh\` 刷新版 |
| 2 | **TextVQA W8x4 log 停写** | 16:12-16:15 | 每 cell 仅 1 个 sample | 终端会话断开 + GPU context corruption（MMMU 同时启动叠加 VRAM） | 队列改用 nohup + MMMU 1proc/GPU 先跑 |
| 3 | **MMMU RTN/MBQ CUDA driver error** | 16:25-16:36 | 90% 崩，0 results | 2 进程/GPU MMMU → GPU ~74-75 GB → softmax 显存超限 | Retry: 1 process/GPU, 60s stagger → \`run_final_queue.sh\` |
| 4 | **MMMU AWQ/MLLMQ ok** | 16:25-16:36 | AWQ=0.4633, MLLMQ=0.4900 | 正常完成 | - |

**综合教训**：

1. **\`process_results\` 必须防御性**：即使 yaml \`metric_list\` 未定义 submission，utils 返回 dict 里多一个无法聚合的 key 也会导致 pipeline 崩。所有 task utils 函数只返回 yaml 声明的 metric key。
2. **MMMU 和 TextVQA 不能同 GPU 混合运行**：MMMU（大 batch 多图）和 TextVQA 同时跑同一个 GPU 会 VRAM 叠加导致崩溃。队列必须串行化不同 benchmark。
3. **MMMU W4  Trials 必须 1 process/GPU**（A800 80 GB 在 74 GB 就开始 CUDA driver error，不是稳态 OOM）。
4. **长远会话管理**：所有 long-running queue 必须用 \`nohup\` 启动，避免终端断开导致全部子进程被杀。
"

# Update dialog.md using Python
python3 << 'PYEOF'
import re, os

path = os.environ["DIALOG"]
block = os.environ["DIAG_BLOCK"]

with open(path, 'r') as f:
    content = f.read()

if '📊 实验进展全矩阵' in content:
    # Replace: from header to end of failure section, up to next "---\n\n##" or EOF
    pattern = r'### 📊 实验进展全矩阵.*?(?=\n---\n\n## |\Z)'
    new_content = re.sub(pattern, block.strip(), content, flags=re.DOTALL)
else:
    # Insert before the-notes section
    anchor = '## ⚠️ 重点注意事项（每次操作前必读）'
    idx = content.find(anchor)
    if idx > 0:
        new_content = content[:idx] + block.strip() + '\n\n---\n\n' + content[idx:]
    else:
        new_content = content + '\n\n' + block
        print('[monitor] Appended (no anchor)')

with open(path, 'w') as f:
    f.write(new_content)
print('[monitor] dialog.md updated')
PYEOF