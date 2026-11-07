#!/usr/bin/env python3
"""10-min progress monitor for ALL 17 eval cells. Updates dialog.md.
Fixes: rotates stale logs by last-modified time; cross-checks with live processes.
"""

import json, os, re, subprocess
from pathlib import Path
from datetime import datetime

P = "/nas3/kangjia/mllmq"
OUT_DIR = os.path.join(P, "output/internvl2-8b/eval")
LOG_DIR = os.path.join(P, "output/logs")
DIALOG_FILE = os.path.join(P, "dialog.md")
TASK_TVQA = "textvqa_val_local"
TASK_MMMU = "mmmu_val"

TVQA_METHODS = ["fp16", "rtn", "awq", "mbq", "mllmq"]
TVQA_BITS = {"fp16": [4], "rtn": [4, 8, 3], "awq": [4, 8, 3],
             "mbq": [4, 8, 3], "mllmq": [4, 8, 3]}
MMMU_METHODS = ["rtn", "awq", "mbq", "mllmq"]

WORKER_LOGS = {
    "W0(GPU0)": ["final_tvqa_w0.log", "final_mmmu_gpu0.log",
                  "quad2_w0_gpu0.log", "quad_tvqa_w0_gpu0.log"],
    "W1(GPU0)": ["final_tvqa_w1.log",
                  "quad2_w1_gpu0.log", "quad_tvqa_w1_gpu0.log"],
    "W2(GPU1)": ["final_tvqa_w2.log", "final_mmmu_gpu1.log",
                  "quad2_w2_gpu1.log", "quad_tvqa_w2_gpu1.log"],
    "W3(GPU1)": ["final_tvqa_w3.log",
                  "quad2_w3_gpu1.log", "quad_tvqa_w3_gpu1.log"],
}

# How long (seconds) before a log is considered stale (no new writes)
STALE_THRESHOLD = 180  # 3 minutes - if log hasn't been updated in 3 min, process likely dead


def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, shell=True)
        return r.stdout
    except Exception:
        return ""


def get_accuracy(tag, task):
    d = Path(OUT_DIR) / task / tag
    js = list(d.glob("*_results.json"))
    if not js:
        return None
    data = json.load(open(js[0]))
    r = data.get("results", {}).get(task, {})
    for key in ["exact_match,none", "exact_match", "mmmu_acc,none", "mmmu_acc",
                "accuracy,none", "accuracy", "acc,none"]:
        if key in r:
            return "%.4f" % r[key]
    return "?"


def get_alive_tags():
    """Get set of tags that have active eval_benchmarks.py processes."""
    alive = set()
    out = run_cmd("ps aux | grep eval_benchmarks.py | grep -v grep")
    for line in out.splitlines():
        if "eval_benchmarks.py" not in line:
            continue
        # Extract method from --method
        m = re.search(r"--method\s+(\w+)", line)
        w = re.search(r"--w_bit\s+(\d+)", line)
        t = re.search(r"--tasks\s+(\w+)", line)
        if not m or not w:
            continue
        method = m.group(1)
        w_bit = w.group(1)
        if method == "fp16":
            tag = "fp16"
        else:
            tag = "%s_w%sa16" % (method, w_bit)
        alive.add(tag)
    return alive


def find_log_file(tag, task, alive_set):
    """Find the most recent TODAY log file for this (tag, task) pair.
    Prioritizes logs associated with currently-alive processes.
    Returns (path, is_stale) where is_stale means the process seems dead."""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today_start.timestamp()
    candidates = []

    for f in sorted(Path(LOG_DIR).glob("*.log")):
        fn = f.name
        if tag not in fn:
            continue
        if task == TASK_TVQA and "tvqa" not in fn and "textvqa" not in fn:
            continue
        if task == TASK_MMMU and "mmmu" not in fn:
            continue
        mt = f.stat().st_mtime
        if mt < cutoff:
            continue  # Skip logs from before today
        mt_dt = datetime.fromtimestamp(mt)
        age_seconds = (datetime.now() - mt_dt).total_seconds()
        is_stale = age_seconds > STALE_THRESHOLD
        candidates.append((fn, mt, is_stale, age_seconds))

    if not candidates:
        return None, False

    # Sort: prefer alive+non-stale first, then newest
    def sort_key(c):
        fn, mt, stale, age = c
        alive = 0 if tag in alive_set else 1
        stale_priority = 0 if not stale else 1
        return (alive, stale_priority, -mt)

    candidates.sort(key=sort_key)
    best = candidates[0]
    return best[0], best[2]  # (filename, is_stale)


def check_cell_status(tag, task, alive_set):
    """Return (display_emoji, display_detail) for a cell.
    Handles: completed, running, crashed, stale-dead, queued."""
    acc = get_accuracy(tag, task)
    if acc is not None:
        return ("OK", acc)

    lf_name, is_stale = find_log_file(tag, task, alive_set)

    if not lf_name:
        # Check if a brand-new log exists (just started)
        logf = find_log_file(tag, task, set())  # ignore alive check
        if not logf[0]:
            return ("?_", "queued")

    lf = os.path.join(LOG_DIR, lf_name)
    if not os.path.isfile(lf):
        return ("?_", "queued")

    try:
        lines = open(lf).readlines()
    except Exception:
        return ("ERR", "unreadable log")

    mod = datetime.fromtimestamp(os.path.getmtime(lf)).strftime("%H:%M")
    nl = len(lines)
    mt = os.path.getmtime(lf)
    age_min = (datetime.now() - datetime.fromtimestamp(mt)).total_seconds() / 60

    # If stale (log last modified >3 min ago) and no result.json -> process is dead
    if is_stale:
        # Could be a proper crash or terminal died
        # Check if there are crash markers
        has_crash = False
        for line in lines[-100:]:
            if "CUDA driver error" in line or "AttributeError" in line or \
               "RuntimeError" in line or "killed" in line.lower() or \
               "OOM" in line.upper():
                has_crash = True
                break
        if has_crash:
            crash_line = ""
            for line in lines[-100:]:
                if "CUDA driver error" in line:
                    crash_line = "CRASH:%s CUDA(%dL)" % (mod, nl)
                    break
                if "AttributeError" in line:
                    crash_line = "CRASH:%s submission(%dL)" % (mod, nl)
                    break
                if "RuntimeError" in line:
                    crash_line = "CRASH:%s RuntimeError(%dL)" % (mod, nl)
                    break
            return ("ERR", crash_line)
        else:
            # No crash marker, log stopped being written >3 min ago -> stale/dead
            # Check max progress from the log
            max_done = 0
            max_total = 0
            for line in lines[-50:]:
                m = re.search(r"(\d+)/(\d+)\s*\[", line)
                if m:
                    d2 = int(m.group(1))
                    t2 = int(m.group(2))
                    if d2 > max_done:
                        max_done = d2
                        max_total = t2
            pct_str = ""
            if max_total > 0:
                pct = max_done * 100 // max_total
                pct_str = " %d%%(%d/%d)" % (pct, max_done, max_total)
            return ("ERR", "DIED:%s stale(dead),%dL%s" % (mod, nl, pct_str))

    if nl < 25:
        return ("ERR", "Crash early(%dL)" % nl)

    # Active process with recent log — extract tqdm progress
    for line in lines[-30:][::-1]:
        m = re.search(r"(\d+)/(\d+)\s*\[", line)
        if m:
            done = int(m.group(1))
            total = int(m.group(2))
            if total > 0:
                pct = done * 100 // total
                return ("XX", "%d%%(%d/%d)" % (pct, done, total))

    return ("XX", "running")


def worker_current_activity(wlog_paths):
    best_line = "-"
    best_mtime = 0
    for pname in wlog_paths:
        lp = os.path.join(LOG_DIR, pname)
        if not os.path.isfile(lp):
            continue
        mt = os.path.getmtime(lp)
        if mt <= best_mtime:
            continue
        try:
            lines = open(lp).readlines()[-10:]
        except Exception:
            continue
        for ln in lines[::-1]:
            s = ln.strip()
            if not s:
                continue
            if (s.startswith("[W") or "START [" in s or "OK   [" in s
                    or "FAIL [" in s or "SKIP " in s):
                best_line = s[:140]
                best_mtime = mt
                break
    return best_line


def get_gpu_info():
    try:
        out = run_cmd("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader")
        gpus = {}
        for line in out.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                mem = parts[1].replace("MiB", "").strip()
                tot = parts[2].replace("MiB", "").strip()
                ut = parts[3].replace("%", "").strip()
                gpus[int(parts[0])] = (mem, tot, ut)
        return gpus
    except Exception:
        return {}


def count_running():
    out = run_cmd("ps aux | grep eval_benchmarks.py | grep -v grep | wc -l")
    try:
        return int(out.strip())
    except Exception:
        return 0


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpus = get_gpu_info()
    nr = count_running()
    alive_set = get_alive_tags()

    g0 = gpus.get(0, ("?", "81251", "?"))
    g1 = gpus.get(1, ("?", "81251", "?"))

    print("=" * 64)
    print("ALL Benchmarks Progress  %-21s" % ts)
    print("=" * 64)
    print("GPU0=%sMiB/util=%s  GPU1=%sMiB/util=%s  |  %d eval processes" % (
        g0[0], g0[2], g1[0], g1[2], nr))
    print("Live process tags: %s" % ", ".join(sorted(alive_set)) if alive_set else "(none)")
    print()

    for wk_name, wpaths in WORKER_LOGS.items():
        act = worker_current_activity(wpaths)
        print("  %-12s %s" % (wk_name, act))
    print()

    # ---- Build dialog.md content --------------------------------
    D = []
    D.append("### EXP Progress (%s)" % ts)
    D.append("")
    D.append("**GPU**: GPU0=%s MiB (util %s%%) | GPU1=%s MiB (util %s%%) | %d eval processes" % (
        g0[0], g0[2], g1[0], g1[2], nr))
    D.append("")
    D.append("*0 processes + low GPU util = queue done or all crashed*")
    D.append("")

    # --- TextVQA table ---
    D.append("### TextVQA (13 cells)")
    D.append("")
    D.append("| Method | W4A16 | W8A16 | W3A16 |")
    D.append("|--------|-------|-------|-------|")

    for method in TVQA_METHODS:
        bits = TVQA_BITS.get(method, [4])
        cols = []
        for b in bits:
            if method == "fp16" and b != 4:
                cols.append("-")
                continue
            tag = "fp16" if method == "fp16" else "%s_w%da16" % (method, b)
            cat, detail = check_cell_status(tag, TASK_TVQA, alive_set)
            if cat == "OK":
                cols.append("%.4f" % float(detail))
            elif cat == "ERR":
                # Clean up the detail for display
                if "DIED:" in detail:
                    # stale/dead process
                    m2 = re.search(r"(\d+)%", detail)
                    pct = m2.group(1) + "%" if m2 else "--"
                    cols.append("X stale %s" % pct)
                elif "CRASH:" in detail:
                    cols.append("X CRASH")
                else:
                    cols.append("X error")
            elif cat == "XX":
                cols.append("[%s]" % detail[:22])
            else:
                cols.append("-")
        colstr = "  |  ".join(cols)
        D.append("| **%s**  | %s |" % (method, colstr))

    D.append("")

    # --- MMMU table ---
    mmmu_done = sum(1 for m in MMMU_METHODS if get_accuracy("%s_w4a16" % m, TASK_MMMU) is not None)
    D.append("### MMMU W4A16 (%d retries pending)" % (4 - mmmu_done))
    D.append("")
    D.append("| Method  | W4A16      |")
    D.append("|---------|------------|")

    for method in MMMU_METHODS:
        tag = "%s_w4a16" % method
        cat, detail = check_cell_status(tag, TASK_MMMU, alive_set)
        if cat == "OK":
            D.append("| **%s**  | %.4f     |" % (method, float(detail)))
        elif cat == "ERR":
            clean = detail.replace("CRASH:", "X ").replace("DIED:", "X ").replace(":", " ")
            D.append("| **%s**  | %s   |" % (method, clean[:18]))
        elif cat == "XX":
            D.append("| **%s**  | %s   |" % (method, detail[:18]))
        else:
            D.append("| **%s**  | -        |" % method)

    D.append("")
    D.append("### ERRORS AND FIXES (2026-09-17)")
    D.append("")
    D.append("| # | Bug | Time | Impact | Root Cause | Fix |")
    D.append("|---|-----|------|--------|------------|-----|")
    D.append("| 1 | **TextVQA W4x4+fp16 crash** | 14:55 | 78min gone | `process_results()` returned `submission` key -> `textvqa_aggregate_submissions` crashed on `args.output_path=None` | Removed `submission` from yaml; fixed `utils.py` + `textvqa_utils.py` to only return `exact_match` key |")
    D.append("| 2 | **TextVQA W8x4 logs stopped** | 16:12 | 4 cells lost | Terminal session died + GPU VRAM contention with concurrent MMMU launch | Queue: MMMU 1proc/GPU first, then TVQA 2proc/GPU; use nohup |")
    D.append("| 3 | **MMMU RTN/MBQ CUDA driver error** | 16:25-16:36 | 90% progress lost | 2 MMMU processes on GPU (~74 GB) -> CUDA driver error at softmax | Retry: 1 process/GPU, 60s stagger |")
    D.append("| 4 | **MMMU AWQ/MLLMQ OK** | 16:25-16:36 | AWQ=0.4633, MLLMQ=0.4900 | Normal completion | -- |")
    D.append("")
    D.append("Key Lessons:")
    D.append("1. `process_results` must only return yaml-declared metric keys (defense-in-depth).")
    D.append("2. Never mix TextVQA and MMMU on the same GPU -- VRAM is additive; causes CUDA driver error.")
    D.append("3. MMMU W4 retry: use 1 process/GPU (A800 80GB hits CUDA driver error above ~72 GB).")
    D.append("4. All long-running queues must be `nohup`-ed or daemonized to survive terminal disconnect.")
    D.append("")
    D.append("Status Legend:")
    D.append("`✅ X.XXXX` = completed (accuracy) | `console:[X%/d/n]` = running | `console[X ...]` = crashed or stale/dead | `-` = queued")
    D.append("")

    block = "\n".join(D)

    # ---- Update dialog.md ----
    try:
        content = open(DIALOG_FILE, "r").read()
        if "EXP Progress" in content:
            pat = r"#{1,3}\s+EXP Progress.*?(?=\n## |\Z)"
            new_content = re.sub(pat, block.strip(), content, flags=re.DOTALL)
        else:
            anchor = "### Any Important"
            idx = content.find(anchor)
            if idx > 0:
                new_content = (content[:idx] + "\n\n" + block.strip() + "\n\n---\n\n" + content[idx:])
            else:
                new_content = content + "\n\n" + block
        open(DIALOG_FILE, "w").write(new_content)
        print("[monitor] dialog.md updated OK")
    except Exception as e:
        print("[monitor] dialog.md update failed: %s" % e)


if __name__ == "__main__":
    main()