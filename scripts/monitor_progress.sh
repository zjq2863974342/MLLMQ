#!/usr/bin/env bash
# Monitor MLLMQ repair queue progress — output table format
# Usage: bash scripts/monitor_progress.sh
set -u
OUTPUT_BASE=/nas3/kangjia/mllmq/output/internvl2-8b/eval

# echo done/todo based on whether a results.json exists for (task, tag)
cell_done() {
  local task=$1 tag=$2
  local d="$OUTPUT_BASE/$task/$tag"
  if [ -d "$d" ] && ls "$d"/*_results.json >/dev/null 2>&1; then
    echo "done"
  else
    echo "todo"
  fi
}

# print metric value: scan all *_results.json newest-first, return value of first that has the key (skips deprecated/empty files)
cell_score() {
  local task=$1 tag=$2 key=$3
  local d="$OUTPUT_BASE/$task/$tag"
  python -c "
import json, os, glob
p='$d'
files=sorted(glob.glob(p+'/*_results.json'), key=os.path.getmtime, reverse=True)
for f in files:
    try:
        r=json.load(open(f))
        v=r.get('results',{}).get('$task',{}).get('$key',None)
        if v is not None:
            print(round(v,1) if isinstance(v,(int,float)) else v); break
    except Exception:
        pass
else:
    print('—')
" 2>/dev/null
}

# one status cell: "<score>" if done else ⬜
cell_show() {
  local task=$1 tag=$2 key=$3
  if [ "$(cell_done "$task" "$tag")" = "done" ]; then
    printf '%12s' "$(cell_score "$task" "$tag" "$key")"
  else
    printf '        ⬜'
  fi
}

echo "══════════════════════════════════════════════════════════"
echo "📊 MLLMQ 修复队列进度 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════"

echo ""
echo "── 运行中进程 ──"
eval_n=$(pgrep -f "eval_benchmarks.py" 2>/dev/null | wc -l)
echo "eval_benchmarks.py 进程数: $eval_n"
pgrep -fa "eval_benchmarks.py" 2>/dev/null | sed 's/^/  /'

echo ""
echo "── GPU 状态 ──"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'

echo ""
echo "── MME 进度表 (目标 12 cell, perc score) ──"
printf '%-8s%12s%12s%12s\n' "Method" "W4" "W8" "W3"
for m in rtn awq block mllmq; do
  printf '%-8s' "$m"
  for w in 4 8 3; do
    tag="${m}_w${w}a16"
    cell_show mme_local "$tag" "mme_percetion_score,none"
  done
  echo ""
done

echo ""
echo "── TextVQA 进度表 (目标 13 cell, acc) ──"
printf '%-8s%12s%12s%12s%12s\n' "Method" "fp16" "W4" "W8" "W3"
for m in fp16 rtn awq block mllmq; do
  printf '%-8s' "$m"
  if [ "$m" = "fp16" ]; then
    cell_show textvqa_val_local "fp16" "exact_match,none"
  else
    for w in 4 8 3; do
      tag="${m}_w${w}a16"
      cell_show textvqa_val_local "$tag" "exact_match,none"
    done
  fi
  echo ""
done

echo ""
echo "═══ 完成 ═══"