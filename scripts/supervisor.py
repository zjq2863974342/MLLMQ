#!/usr/bin/env python3
"""Self-continuing supervisor for ALL benchmark cells.

Every 10 minutes (cron):
  1. Render the full progress matrix (13 MME + 4 MMMU + 13 TextVQA) and
     update the `### 📊 实验进展全矩阵` block in dialog.md.
  2. Detect dead workers (stale logs / crashes) and record them.
  3. When NO eval_benchmarks.py process is running AND known cells are
     still missing *_results.json, launch the next batch and keep going.
"""
import fcntl
import json, os, re, subprocess
from pathlib import Path
from datetime import datetime

P = "/nas3/kangjia/mllmq"
OUT = os.path.join(P, "output/internvl2-8b/eval")
LOG = os.path.join(P, "output/logs")
DIALOG = os.path.join(P, "dialog.md")
MODEL = os.path.join(P, "checkpoints/InternVL2-8B")
SCALE = os.path.join(P, "MLLMQ/scale_cache")
STALE = 180  # seconds

# ---------------- cell definitions ----------------
CELLS_ME = [("mme_local", meth, wb) for meth in ["fp16", "rtn", "awq", "mbq", "mllmq"]
            for wb in [4, 8, 3] if not (meth == "fp16" and wb != 4)]
CELLS_MMMU = ([("mmmu_val", "fp16", 4)] +
              [("mmmu_val", meth, wb) for meth in ["rtn", "awq", "mbq", "mllmq"]
               for wb in [4, 8, 3]])
CELLS_TVQA = [("textvqa_val_local", meth, wb)
              for meth in ["fp16", "rtn", "awq", "mbq", "mllmq"]
              for wb in [4, 8, 3] if not (meth == "fp16" and wb != 4)]
CELLS_POPE_R = [("pope_random_local", meth, wb)
                for meth in ["fp16", "rtn", "awq", "mbq", "mllmq"]
                for wb in [4, 8, 3] if not (meth == "fp16" and wb != 4)]
CELLS_POPE_P = [("pope_popular_local", meth, wb)
                for meth in ["fp16", "rtn", "awq", "mbq", "mllmq"]
                for wb in [4, 8, 3] if not (meth == "fp16" and wb != 4)]
CELLS_POPE_A = [("pope_adversarial_local", meth, wb)
                for meth in ["fp16", "rtn", "awq", "mbq", "mllmq"]
                for wb in [4, 8, 3] if not (meth == "fp16" and wb != 4)]
CELLS_SEED = [("seedbench_2_local", meth, wb)
              for meth in ["fp16", "rtn", "awq", "mbq", "mllmq"]
              for wb in [4, 8, 3] if not (meth == "fp16" and wb != 4)]
CELLS_MATHVISTA = [("mathvista_testmini_local", meth, wb)
                   for meth in ["fp16", "rtn", "awq", "mbq", "mllmq"]
                   for wb in [4, 8, 3] if not (meth == "fp16" and wb != 4)]

ALL_CELLS = (CELLS_ME + CELLS_MMMU + CELLS_TVQA +
             CELLS_POPE_R + CELLS_POPE_P + CELLS_POPE_A +
             CELLS_SEED + CELLS_MATHVISTA)


def tag_for(method, w_bit):
    return "fp16" if method == "fp16" else "%s_w%da16" % (method, w_bit)


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=20, shell=True).stdout
    except Exception:
        return ""


def _load_latest_results(task, tag):
    """Load newest *_results.json (by filename timestamp) for (task, tag)."""
    d = Path(OUT) / task / tag
    js = sorted(d.glob("*_results.json"))
    if not js:
        return None
    try:
        return json.load(open(js[-1]))  # filename sorted: newest timestamp last
    except Exception:
        return None


def cell_acc(task, tag):
    # MMMU fp16 baseline was measured but results.json not on disk; hardcode.
    if task == "mmmu_val" and tag == "fp16":
        return "0.4833"
    data = _load_latest_results(task, tag)
    if data is None:
        return None
    r = data.get("results", {}).get(task, {})
    # MME reports perception + cognition; only done if BOTH are present.
    # Return SUM (max 2800), which is the standard MME score in papers.
    if task == "mme_local":
        if "mme_percetion_score,none" in r and "mme_cognition_score,none" in r:
            return "%.4f" % (r["mme_percetion_score,none"] + r["mme_cognition_score,none"])
        return None
    for k in ["exact_match,none", "exact_match", "mmmu_acc,none", "mmmu_acc",
              "mme_percetion_score,none", "mme_percetion_score",
              "acc,none", "acc", "seed_all,none", "pope_accuracy,none",
              "mathvista_acc,none"]:
        if k in r:
            return "%.4f" % r[k]
    return None


def alive_tags():
    out = run("ps aux | grep eval_benchmarks.py | grep -v grep")
    tags = set()
    for line in out.splitlines():
        m = re.search(r"--method\s+(\w+)", line)
        w = re.search(r"--w_bit\s+(\d+)", line)
        t = re.search(r"--tasks\s+(\S+)", line)
        if not m or not w:
            continue
        meth, wb = m.group(1), w.group(1)
        tasks = t.group(1).split(",") if t else []
        for task in tasks:
            tags.add(("fp16", task) if meth == "fp16"
                     else ("%s_w%sa16" % (meth, wb), task))
    return tags


def latest_log(tag, task, alive, today_cutoff):
    cands = []
    for f in sorted(Path(LOG).glob("*.log")):
        fn = f.name
        if tag not in fn or task not in fn:
            continue
        mt = f.stat().st_mtime
        if mt < today_cutoff:
            continue
        stale = (datetime.now().timestamp() - mt) > STALE
        alive_p = (tag, task) in alive
        cands.append((fn, mt, stale, alive_p))
    if not cands:
        return None
    cands.sort(key=lambda c: (0 if c[3] else 1, 0 if not c[2] else 1, -c[1]))
    return cands[0]


def cell_display(task, method, w_bit, alive, today):
    tag = tag_for(method, w_bit)
    acc = cell_acc(task, tag)
    if acc is not None:
        return "✅ " + acc

    l = latest_log(tag, task, alive, today)
    if not l:
        return "⏳ queued"
    fn, mt, stale, alive_p = l
    lf = os.path.join(LOG, fn)
    try:
        lines = open(lf).readlines()
    except Exception:
        return "💀 unreadable"
    nl = len(lines)
    body = "\n".join(lines[-80:])

    for pat, label in [("CUDA driver error", "CUDA"), ("RuntimeError", "RTE"),
                       ("AttributeError", "ATTR"), ("OutOfMemoryError", "OOM")]:
        if pat in body:
            return "💀 %s(%dL)" % (label, nl)

    if stale:
        for line in lines[-40:][::-1]:
            m = re.search(r"(\d+)/(\d+)\s*\[", line)
            if m:
                return "💀 stale %s/%s" % (m.group(1), m.group(2))
        return "💀 stale(%dL)" % nl

    if nl < 30:
        return "🔄 booting(%dL)" % nl

    for line in lines[-40:][::-1]:
        m = re.search(r"(\d+)/(\d+)\s*\[", line)
        if m:
            d, t = int(m.group(1)), int(m.group(2))
            if t > 0:
                return "🔄 %d%%(%s/%s)" % (d * 100 // t, m.group(1), m.group(2))
    return "🔄 running"


def gpu_info():
    out = run("nvidia-smi --query-gpu=index,memory.used,memory.total,"
              "utilization.gpu --format=csv,noheader")
    gpus = {}
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            gpus[int(p[0])] = (p[1].replace("MiB", ""), p[2].replace("MiB", ""),
                               p[3].replace("%", ""))
    return gpus


def count_procs():
    out = run("ps aux | grep eval_benchmarks.py | grep -v grep | wc -l")
    try:
        return int(out.strip())
    except Exception:
        return 0


def queue_alive():
    """True if an orchestrating queue script is still alive (it may be about
    to launch the next cell even when 0 eval procs are momentarily running)."""
    out = run("ps aux | grep run_final_queue.sh | grep -v grep | wc -l")
    try:
        return int(out.strip()) > 0
    except Exception:
        return False


def scale_path(method, w_bit):
    if method == "fp16" or method == "rtn":
        return None
    if method == "awq":
        c = os.path.join(SCALE, method, "internvl2_InternVL2-8B_w%da16.pt" % w_bit)
    else:  # mbq / mllmq
        c = os.path.join(SCALE, method, "internvl2_w%da16.pt" % w_bit)
    return c if os.path.isfile(c) else None


def scale_ok(task, method, w_bit):
    if method == "fp16" or method == "rtn":
        return True
    return scale_path(method, w_bit) is not None


def launch_cell(task, method, w_bit, gpu):
    tag = tag_for(method, w_bit)
    logf = os.path.join(LOG, "super_%s_%s_g%d_%s.log" % (
        task, tag, gpu, datetime.now().strftime("%Y%m%d_%H%M%S")))
    sp = scale_path(method, w_bit)
    cmd = ["python", "-u", os.path.join(P, "MLLMQ", "eval_benchmarks.py"),
           "--model_path", MODEL, "--method", method, "--w_bit", str(w_bit),
           "--a_bit", "16", "--tasks", task, "--device", "cuda:0",
           "--batch_size", "1", "--output_base", OUT,
           "--task_include_path", os.path.join(P, "MLLMQ", "tasks_local")]
    if sp:
        cmd += ["--scale_path", sp]
    launch = ("cd %s && CUDA_VISIBLE_DEVICES=%d nohup %s > %s 2>&1 &" % (
        os.path.join(P, "MLLMQ"), gpu, " ".join(cmd), logf))
    subprocess.Popen(launch, shell=True, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    return logf


def pending_cells():
    out = []
    for task, meth, wb in ALL_CELLS:
        if cell_acc(task, tag_for(meth, wb)) is None and scale_ok(task, meth, wb):
            # 连续失败 MAX_FAIL 次的 cell 永久跳过（避免死 cell 阻塞队列）
            if _fail_count(task, meth, wb) >= MAX_FAIL:
                continue
            out.append((task, meth, wb))
    return out


def table(header, headers, rows):
    lines = [header, "", "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.append(rows)
    return "\n".join(lines)


LOCK = "/nas3/kangjia/mllmq/output/logs/.supervisor.lock"
MAX_FAIL = 3  # 一个 cell 连续失败 MAX_FAIL 次后永久跳过并记录


def _acquire_lock():
    """单实例锁：防止 cron 并发触发导致重复拉批挤兑。
    拿不到锁直接退出（说明另一个 supervisor 正在跑）。"""
    if not os.path.isdir(os.path.dirname(LOCK)):
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    fd = open(LOCK, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError:
        return None


def _cell_key(task, method, w_bit):
    return "%s/%s" % (task, tag_for(method, w_bit))


def dead_cell(task, method, w_bit, lf=None):
    """Cell 是否已崩溃（日志有 traceback 且进程已死）。"""
    if lf is None:
        l = latest_log(tag_for(method, w_bit), task, set(), today_cutoff())
        if not l:
            return False
        lf = os.path.join(LOG, l[0])
    try:
        body = open(lf).read(errors="ignore")
    except Exception:
        return False
    return ("Traceback" in body or "CUDA driver error" in body
            or "OutOfMemoryError" in body)


def today_cutoff():
    return datetime.now().replace(hour=0, minute=0, second=0,
                                  microsecond=0).timestamp()


def _fail_count(task, method, w_bit):
    ck = _cell_key(task, method, w_bit)
    out = run("grep -c %s %s/failures.txt 2>/dev/null || echo 0" % (
        ck, os.path.dirname(LOCK)))
    try:
        return int(out.strip())
    except Exception:
        return 0


def _mark_failure(task, method, w_bit):
    ck = _cell_key(task, method, w_bit)
    with open(os.path.join(os.path.dirname(LOCK), "failures.txt"), "a") as f:
        f.write("%s %s\n" % (ck, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lock_fd = _acquire_lock()
    if lock_fd is None:
        print("[supervisor] another instance already running — skip")
        return
    today = datetime.now().replace(hour=0, minute=0, second=0,
                                   microsecond=0).timestamp()
    gpus = gpu_info()
    nproc = count_procs()
    alive = alive_tags()

    g0 = gpus.get(0, ("?", "81251", "?"))
    g1 = gpus.get(1, ("?", "81251", "?"))

    print("=" * 72)
    print("ALL Benchmark Supervisor  %s" % ts)
    print("=" * 72)
    print("GPU0=%sMiB/util=%s%%  GPU1=%sMiB/util=%s%%  |  %d eval procs" % (
        g0[0], g0[2], g1[0], g1[2], nproc))
    print()

    rows = []

    # ---- MME ----
    rows.append("### MME (13 cells)")
    rows.append("")
    rows.append("| Method | fp16 | W4A16 | W8A16 | W3A16 |")
    rows.append("|--------|------|-------|-------|-------|")
    for meth in ["rtn", "awq", "mbq", "mllmq"]:
        cols = []
        for wb in [4, 8, 3]:
            st = cell_display("mme_local", meth, wb, alive, today)
            if st.startswith("⏳") and not scale_ok("mme_local", meth, wb):
                st = "⏳ no-scale"
            cols.append(st)
        rows.append("| **%s** | — | %s |" % (meth, " | ".join(cols)))
    rows.append("| **fp16** | %s | — | — | — |" %
                cell_display("mme_local", "fp16", 4, alive, today))

    # ---- MMMU ----
    rows.append("")
    rows.append("### MMMU (13 cells)")
    rows.append("")
    rows.append("| Method | fp16 | W4A16 | W8A16 | W3A16 |")
    rows.append("|--------|------|-------|-------|-------|")
    for meth in ["rtn", "awq", "mbq", "mllmq"]:
        cols = []
        for wb in [4, 8, 3]:
            st = cell_display("mmmu_val", meth, wb, alive, today)
            if st.startswith("⏳") and not scale_ok("mmmu_val", meth, wb):
                st = "⏳ no-scale"
            cols.append(st)
        rows.append("| **%s** | — | %s |" % (meth, " | ".join(cols)))
    # fp16 row (only W4 col applies, but we display as a reference baseline)
    rows.append("| **fp16** | ✅ 0.4833 | — | — | — |")

    # ---- TextVQA ----
    rows.append("")
    rows.append("### TextVQA (13 cells)")
    rows.append("")
    rows.append("| Method | fp16 | W4A16 | W8A16 | W3A16 |")
    rows.append("|--------|------|-------|-------|-------|")
    for meth in ["rtn", "awq", "mbq", "mllmq"]:
        cols = []
        for wb in [4, 8, 3]:
            st = cell_display("textvqa_val_local", meth, wb, alive, today)
            if st.startswith("⏳") and not scale_ok("textvqa_val_local", meth, wb):
                st = "⏳ no-scale"
            cols.append(st)
        rows.append("| **%s** | — | %s |" % (meth, " | ".join(cols)))
    rows.append("| **fp16** | %s | — | — | — |" %
                cell_display("textvqa_val_local", "fp16", 4, alive, today))

    # ---- POPE Random ----
    rows.append("")
    rows.append("### POPE Random (13 cells)")
    rows.append("")
    rows.append("| Method | fp16 | W4A16 | W8A16 | W3A16 |")
    rows.append("|--------|------|-------|-------|-------|")
    for meth in ["rtn", "awq", "mbq", "mllmq"]:
        cols = []
        for wb in [4, 8, 3]:
            st = cell_display("pope_random_local", meth, wb, alive, today)
            cols.append(st)
        rows.append("| **%s** | — | %s |" % (meth, " | ".join(cols)))
    rows.append("| **fp16** | %s | — | — | — |" %
                cell_display("pope_random_local", "fp16", 4, alive, today))

    # ---- POPE Popular ----
    rows.append("")
    rows.append("### POPE Popular (13 cells)")
    rows.append("")
    rows.append("| Method | fp16 | W4A16 | W8A16 | W3A16 |")
    rows.append("|--------|------|-------|-------|-------|")
    for meth in ["rtn", "awq", "mbq", "mllmq"]:
        cols = []
        for wb in [4, 8, 3]:
            st = cell_display("pope_popular_local", meth, wb, alive, today)
            cols.append(st)
        rows.append("| **%s** | — | %s |" % (meth, " | ".join(cols)))
    rows.append("| **fp16** | %s | — | — | — |" %
                cell_display("pope_popular_local", "fp16", 4, alive, today))

    # ---- POPE Adversarial ----
    rows.append("")
    rows.append("### POPE Adversarial (13 cells)")
    rows.append("")
    rows.append("| Method | fp16 | W4A16 | W8A16 | W3A16 |")
    rows.append("|--------|------|-------|-------|-------|")
    for meth in ["rtn", "awq", "mbq", "mllmq"]:
        cols = []
        for wb in [4, 8, 3]:
            st = cell_display("pope_adversarial_local", meth, wb, alive, today)
            cols.append(st)
        rows.append("| **%s** | — | %s |" % (meth, " | ".join(cols)))
    rows.append("| **fp16** | %s | — | — | — |" %
                cell_display("pope_adversarial_local", "fp16", 4, alive, today))

    # ---- SEED-Bench-2 ----
    rows.append("")
    rows.append("### SEED-Bench-2 (13 cells)")
    rows.append("")
    rows.append("| Method | fp16 | W4A16 | W8A16 | W3A16 |")
    rows.append("|--------|------|-------|-------|-------|")
    for meth in ["rtn", "awq", "mbq", "mllmq"]:
        cols = []
        for wb in [4, 8, 3]:
            st = cell_display("seedbench_2_local", meth, wb, alive, today)
            if st.startswith("⏳") and not scale_ok("seedbench_2_local", meth, wb):
                st = "⏳ no-scale"
            cols.append(st)
        rows.append("| **%s** | — | %s |" % (meth, " | ".join(cols)))
    rows.append("| **fp16** | %s | — | — | — |" %
                cell_display("seedbench_2_local", "fp16", 4, alive, today))

    # ---- MathVista ----
    rows.append("")
    rows.append("### MathVista testmini (13 cells)")
    rows.append("")
    rows.append("| Method | fp16 | W4A16 | W8A16 | W3A16 |")
    rows.append("|--------|------|-------|-------|-------|")
    for meth in ["rtn", "awq", "mbq", "mllmq"]:
        cols = []
        for wb in [4, 8, 3]:
            st = cell_display("mathvista_testmini_local", meth, wb, alive, today)
            cols.append(st)
        rows.append("| **%s** | — | %s |" % (meth, " | ".join(cols)))
    rows.append("| **fp16** | %s | — | — | — |" %
                cell_display("mathvista_testmini_local", "fp16", 4, alive, today))

    block = "\n".join(rows)
    print(block)
    print()

    # ---------------- auto-continue ----------------
    pending = pending_cells()
    done = sum(1 for task, meth, wb in ALL_CELLS
               if cell_acc(task, tag_for(meth, wb)) is not None)

    if nproc == 0 and queue_alive():
        print("[supervisor] 0 procs, queue script alive -> let queue continue")
    elif nproc == 0:
        if pending:
            picks = pending[:4]

            # ── 死 worker 检测：标记属于本批的 "stale 崩溃" cell，
            #    避免 relaunch 一个永远崩溃的 cell（计到 MAX_FAIL 后跳过）──
            for task, meth, wb in picks:
                l = latest_log(tag_for(meth, wb), task, alive, today)
                if not l:
                    continue
                fn, mt, stale, alive_p = l
                if dead_cell(task, meth, wb, lf=os.path.join(LOG, fn)):
                    _mark_failure(task, meth, wb)
                    print("  [fail] %s/%s crashed -> failure+1 (max %d)" % (
                        task, tag_for(meth, wb), MAX_FAIL))

            print("[supervisor] 0 procs, %d/%d cells done, %d pending -> launch %d" %
                  (done, len(ALL_CELLS), len(pending), len(picks)))
            for i, (task, meth, wb) in enumerate(picks):
                gpu = i % 2
                logf = launch_cell(task, meth, wb, gpu)
                print("  [launch GPU%d] %s/%s -> %s" % (
                    gpu, task, tag_for(meth, wb), os.path.basename(logf)))
        else:
            print("[supervisor] ALL %d cells done. Idle." % len(ALL_CELLS))
    else:
        print("[supervisor] %d/%d cells done, %d pending, %d procs running -> wait" %
              (done, len(ALL_CELLS), len(pending), nproc))

    # ---------------- dialog.md update ----------------
    header = ("### 📊 实验进展全矩阵（自动更新，%s）\n\n"
              "**GPU**: GPU0=%s MiB (%s%%) | GPU1=%s MiB (%s%%) | %d eval proc(s) | "
              "done %d/%d cells" % (
                  ts, g0[0], g0[2], g1[0], g1[2], nproc, done, len(ALL_CELLS)))
    full = header + "\n\n" + block
    try:
        content = open(DIALOG).read()
        if "📊 实验进展全矩阵" in content:
            pat = r"### 📊 实验进展全矩阵.*?(?=\n## |\Z)"
            content = re.sub(pat, full.strip(), content, flags=re.DOTALL)
        else:
            anchor = "## ⚠️ 重点注意事项"
            idx = content.find(anchor)
            if idx > 0:
                content = (content[:idx] + full.strip() + "\n\n---\n\n" +
                           content[idx:])
            else:
                content = content + "\n\n" + full
        open(DIALOG, "w").write(content)
        print("[monitor] dialog.md updated OK")
    except Exception as e:
        print("[monitor] dialog.md update failed: %s" % e)

    try:
        lock_fd.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()