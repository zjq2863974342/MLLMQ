#!/usr/bin/env python3
"""定期实验状态快照 → 写入 memory，防上下文 100% 被清空后丢失关键信息。

由 cron 每 N 分钟调用：
    /opt/conda/bin/python3 /nas3/kangjia/mllmq/scripts/snapshot.py

快照写入 memory/experiment-status.md（每次会话自动加载索引里的一行），
内容是：完成/进行中/待跑 batch、当前 GPU/CPU 状态、已知风险、下一步。
这样即使对话 context 被压缩或清空，(memory) 也能恢复全局。
"""
import json, os, re, subprocess, sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supervisor import (
    ALL_CELLS, pending_cells, cell_acc, tag_for, count_procs,
    gpu_info, alive_tags, latest_log, run,
)

MEMDIR = "/root/.codefuse/engine/cc/projects/-nas3-kj/mllmq/memory"
SNAP = os.path.join(MEMDIR, "experiment-status.md")
INDEX = os.path.join(MEMDIR, "MEMORY.md")


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpus = gpu_info()
    nproc = count_procs()
    alive = alive_tags()

    done = sum(1 for t, m, w in ALL_CELLS if cell_acc(t, tag_for(m, w)) is not None)
    pend = pending_cells()
    running = sorted(set(
        "%s/%s" % (t, tag) for t, m, w in pend if False
    ))

    # 运行中：从 alive_tags 拿到 (tag, task) 对
    run_list = []
    for tag, task in alive:
        run_list.append("%s=%s" % (task.replace("_local", ""), tag))
    if not run_list and nproc > 0:
        run_list.append("(procs=%d but no tag parsed)" % nproc)

    # 待跑：按任务分组
    from collections import defaultdict
    by_task = defaultdict(list)
    for t, m, w in pend:
        by_task[t.replace("_local", "")].append("%s%s" % (
            "fp16" if m == "fp16" else m + "_w%d" % w, ""))
    pend_summary = " | ".join("%s(%d)" % (k, len(v)) for k, v in by_task.items()) or "无"

    # GPU 摘要
    gpu_str = " / ".join("GPU%d %sMB util=%s%%" % (i, g[0], g[2])
                         for i, g in sorted(gpus.items()))

    # 最近的运行日志进度
    prog = []
    today_cut = datetime.now().replace(hour=0, minute=0, second=0,
                                       microsecond=0).timestamp()
    for tag, task in list(alive)[:4]:
        l = latest_log(tag, task, alive, today_cut)
        if l:
            f = os.path.join("/nas3/kangjia/mllmq/output/logs", l[0])
            try:
                lines = open(f, errors="ignore").readlines()
            except Exception:
                continue
            m = None
            for line in lines[-40:][::-1]:
                mm = re.search(r"(\d+)/(\d+)\s*\[", line)
                if mm:
                    m = mm; break
            if m:
                prog.append("%s/%s %s%%" % (task.replace("_local", ""), tag,
                                            int(m.group(1)) * 100 // int(m.group(2))))
            else:
                prog.append("%s/%s booting" % (task.replace("_local", ""), tag))

    snap = f"""---
name: experiment-status
description: 量化评估调度的实时快照 — 完成/运行/待跑 batch、GPU 状态、下一步
metadata:
  type: project
---

# 实验调度快照 @ {ts}

- **完成**: {done}/{len(ALL_CELLS)} cells
- **GPU**: {gpu_str}
- **进程**: {nproc} eval proc(s)
- **运行中**: {", ".join(run_list) if run_list else "无"}
  - 进度: {"; ".join(prog) if prog else "—"}
- **待跑**: {pend_summary}
- **已知风险**: SEED 12h/cell 固有耗时 | MathVista 需 OPENAI_API_KEY |
  调度锁已防并发 | 死 cell ≥3 次自动跳过（failures.txt）
- **下一步**: 当前批跑完后 cron supervisor 自动拉下一批，无需人工干预
"""
    os.makedirs(MEMDIR, exist_ok=True)
    with open(SNAP, "w") as f:
        f.write(snap.strip() + "\n")

    # 更新 memory 索引
    line = "- [experiment-status](experiment-status.md) — 量化评估调度实时快照（完成/运行/待跑/风险/下一步）"
    try:
        idx = open(INDEX).read()
    except Exception:
        idx = "# Memory Index\n\n"
    if "experiment-status" in idx:
        idx = re.sub(r"- \[xperiment-status\].*", line, idx)
        idx = re.sub(r"- \[experiment-status\].*", line, idx)
    else:
        idx = idx.rstrip() + "\n" + line + "\n"
    with open(INDEX, "w") as f:
        f.write(idx)

    print("[snapshot] done=%d/%d gpu=%s run=%s" % (done, len(ALL_CELLS),
                                                   gpu_str, ",".join(run_list)))


if __name__ == "__main__":
    main()