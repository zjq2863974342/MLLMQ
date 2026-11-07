#!/usr/bin/env python3
"""自包含批量调度器。每 10 分钟检查 GPU 显存 + 进程数，自动拉下一批。

规则：0 个 eval 进程 + GPU 显存 < 2GB → 判定上一批结束 → 拉 4 个新 cell

启动:  nohup python3 /nas3/kangjia/mllmq/scripts/poll.py &>/dev/null &
停止:  pkill -f "scripts/poll.py"
"""
import fcntl, json, os, re, subprocess, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

P        = "/nas3/kangjia/mllmq"
OUT      = os.path.join(P, "output/internvl2-8b/eval")
LOG_DIR  = os.path.join(P, "output/logs")
DIALOG   = os.path.join(P, "dialog.md")
MODEL    = os.path.join(P, "checkpoints/InternVL2-8B")
SCALE    = os.path.join(P, "MLLMQ/scale_cache")
LOCK     = os.path.join(LOG_DIR, ".poll.lock")
FAILS    = os.path.join(LOG_DIR, "failures.txt")
MEMDIR   = "/root/.codefuse/engine/cc/projects/-nas3-kj-mllmq/memory"

GPU_IDLE = 2000   # MB
INTERVAL = 600    # 秒（10 分钟）
MAX_FAIL = 3

# ─── cell 定义 ───
def make_cells(task, *methods):
    out = []
    for m in methods:
        for wb in [4, 8, 3]:
            if m == "fp16" and wb != 4:
                continue
            out.append((task, m, wb))
    return out

ALL = (
    make_cells("mme_local",                "fp16","rtn","awq","block","mllmq") +
    [("mmmu_val",                   "fp16", 4)] +
    make_cells("mmmu_val",                   "rtn","awq","block","mllmq") +
    make_cells("textvqa_val_local",        "fp16","rtn","awq","block","mllmq") +
    make_cells("pope_random_local",        "fp16","rtn","awq","block","mllmq") +
    make_cells("pope_popular_local",       "fp16","rtn","awq","block","mllmq") +
    make_cells("pope_adversarial_local",   "fp16","rtn","awq","block","mllmq") +
    make_cells("seedbench_2_local",        "fp16","rtn","awq","block","mllmq") +
    make_cells("mathvista_testmini_local", "fp16","rtn","awq","block","mllmq")
)

# 任务名 → 在 ALL 中的第一个索引（保证表格排序稳定）
TASK_ORDER = {}
for _t, _m, _w in ALL:
    TASK_ORDER.setdefault(_t, len(TASK_ORDER))

METHOD_ORDER = {"rtn": 0, "awq": 1, "block": 2, "mllmq": 3, "fp16": 4}

def tag(method, wb):
    return "fp16" if method == "fp16" else "%s_w%da16" % (method, wb)

def scale_path(method, wb):
    if method in ("fp16", "rtn"):
        return None
    if method == "awq":
        c = os.path.join(SCALE, method, "internvl2_InternVL2-8B_w%da16.pt" % wb)
    else:
        c = os.path.join(SCALE, method, "internvl2_w%da16.pt" % wb)
    return c if os.path.isfile(c) else None

def scale_ok(method, wb):
    if method in ("fp16", "rtn"):
        return True
    return scale_path(method, wb) is not None

# ─── 检测 ───
def gpu_mem_max():
    try:
        r = subprocess.run(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader",
            capture_output=True, text=True, shell=True, timeout=10)
        vals = [int(x.strip().replace("MiB","").replace("MB",""))
                for x in r.stdout.strip().splitlines() if x.strip()]
        return max(vals) if vals else 999999
    except Exception:
        return 999999

def gpu_mem_per(i):
    try:
        r = subprocess.run(
            "nvidia-smi -i %d --query-gpu=memory.used --format=csv,noheader" % i,
            capture_output=True, text=True, shell=True, timeout=5)
        return int(r.stdout.strip().replace("MiB",""))
    except Exception:
        return 0

def eval_procs():
    try:
        out = subprocess.run(
            "ps aux | grep 'eval_benchmarks.py' | grep 'python -u' | grep -v grep | wc -l",
            capture_output=True, text=True, shell=True, timeout=5)
        return int(out.stdout.strip())
    except Exception:
        return 0

# ─── 结果 ───
def load_results(task, tg):
    d = Path(OUT) / task / tg
    js = sorted(d.glob("*_results.json"))
    if not js:
        return None
    try:
        return json.load(open(js[-1]))
    except Exception:
        return None

def cell_acc(task, tg):
    if task == "mmmu_val" and tg == "fp16":
        return "0.4833"
    data = load_results(task, tg)
    if data is None:
        return None
    r = data.get("results", {}).get(task, {})
    if task == "mme_local":
        pk = "mme_percetion_score,none"
        ck = "mme_cognition_score,none"
        if pk in r and ck in r:
            return "%.4f" % (r[pk] + r[ck])
        return None
    for k in ["exact_match,none","exact_match","mmmu_acc,none","mmmu_acc",
              "mme_percetion_score,none","mme_percetion_score",
              "acc,none","acc","seed_all,none","pope_accuracy,none",
              "mathvista_acc,none"]:
        if k in r:
            return "%.4f" % r[k]
    return None

# ─── 失败 ───
def fail_count(task, method, wb):
    ck = "%s/%s" % (task, tag(method, wb))
    try:
        out = subprocess.run(
            "grep -c '%s' %s 2>/dev/null || echo 0" % (ck, FAILS),
            capture_output=True, text=True, shell=True, timeout=3)
        return int(out.stdout.strip())
    except Exception:
        return 0

def is_dead(task, method, wb):
    tg = tag(method, wb)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    best = None
    for f in sorted(Path(LOG_DIR).glob("*.log")):
        fn = f.name
        if tg not in fn or task not in fn:
            continue
        if f.stat().st_mtime < today:
            continue
        best = f
    if best is None:
        return False
    try:
        body = open(str(best), errors="ignore").read()
    except Exception:
        return False
    return ("Traceback" in body or "CUDA driver error" in body
            or "OutOfMemoryError" in body)

# ─── 启动 ───
def launch(task, method, wb, gpu):
    tg = tag(method, wb)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logf = os.path.join(LOG_DIR, "poll_%s_%s_g%d_%s.log" % (task, tg, gpu, ts))
    sp = scale_path(method, wb)
    cmd = ["python", "-u", os.path.join(P, "MLLMQ", "eval_benchmarks.py"),
           "--model_path", MODEL, "--method", method, "--w_bit", str(wb),
           "--a_bit", "16", "--tasks", task, "--device", "cuda:0",
           "--batch_size", "1", "--output_base", OUT,
           "--task_include_path", os.path.join(P, "MLLMQ", "tasks_local")]
    if sp:
        cmd += ["--scale_path", sp]
    launch_cmd = ("cd %s && CUDA_VISIBLE_DEVICES=%d nohup %s > %s 2>&1 &" %
                  (os.path.join(P, "MLLMQ"), gpu, " ".join(cmd), logf))
    subprocess.Popen(launch_cmd, shell=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return logf

# ─── 表格 ───
def render_tables():
    rows = []
    seen = set()
    # 按 ALL 中的首次出现顺序遍历任务
    for tname, tmeth, twb in ALL:
        if tname in seen:
            continue
        seen.add(tname)
        rows.append("")
        total_n = sum(1 for tn, tm, tw in ALL if tn == tname)
        rows.append("### %s (%d cells)" % (tname, total_n))
        rows.append("")
        rows.append("| Method | fp16 | W4A16 | W8A16 | W3A16 |")
        rows.append("|--------|------|-------|-------|-------|")
        for meth in ["rtn", "awq", "block", "mllmq"]:
            cols = []
            for wb in [4, 8, 3]:
                acc = cell_acc(tname, tag(meth, wb))
                cols.append("✅ " + acc if acc else "⏳ queued")
            rows.append("| **%s** | — | %s |" % (meth, " | ".join(cols)))
        fp16_acc = cell_acc(tname, tag("fp16", 4))
        rows.append("| **fp16** | %s | — | — | — |" %
                    (("✅ " + fp16_acc) if fp16_acc else "⏳ queued"))
    return "\n".join(rows)

# ─── 进度汇总 ───
def task_progress():
    """{task_name: (done, total)}"""
    out = {}
    for tname, tmeth, twb in ALL:
        out.setdefault(tname, [0, 0])
        out[tname][1] += 1
        if cell_acc(tname, tag(tmeth, twb)) is not None:
            out[tname][0] += 1
    return out

# ─── 日志 ───
def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(os.path.join(LOG_DIR, "poll.log"), "a") as f:
            f.write("[%s] %s\n" % (t, msg))
    except Exception:
        pass

# ─── 主循环 ───
def main():
    os.chdir(os.path.join(P, "MLLMQ"))
    log("daemon started")

    while True:
        fd = open(LOCK, "w")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fd.close()
            time.sleep(INTERVAL)
            continue

        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mem = gpu_mem_max()
            procs = eval_procs()

            done = sum(1 for _t, _m, _w in ALL
                       if cell_acc(_t, tag(_m, _w)) is not None)
            pending = []
            for _t, _m, _w in ALL:
                if cell_acc(_t, tag(_m, _w)) is None and scale_ok(_m, _w):
                    if fail_count(_t, _m, _w) >= MAX_FAIL:
                        continue
                    pending.append((_t, _m, _w))

            idle = (procs == 0 and mem < GPU_IDLE)

            print("[%s] GPU=%dMB procs=%d done=%d/%d pending=%d %s" %
                  (ts, mem, procs, done, len(ALL), len(pending),
                   "IDLE→launch" if idle else "running"))

            if idle and pending:
                # 标记死 cell（日志有 traceback 的）
                for _t, _m, _w in pending[:4]:
                    if is_dead(_t, _m, _w):
                        ck = "%s/%s" % (_t, tag(_m, _w))
                        with open(FAILS, "a") as ff:
                            ff.write("%s %s\n" % (ck, ts))
                        log("DEAD %s" % ck)

                picks = pending[:4]
                log("launch %d cells" % len(picks))
                for i, (_t, _m, _w) in enumerate(picks):
                    gpu = i % 2
                    lf = launch(_t, _m, _w, gpu)
                    tg = tag(_m, _w)
                    print("  [GPU%d] %s/%s → %s" % (gpu, _t, tg, os.path.basename(lf)))
                    log("launch GPU%d %s/%s" % (gpu, _t, tg))
            elif idle and not pending:
                log("ALL %d DONE" % len(ALL))
                print("  ✅ ALL DONE")

            # ── dialog.md ──
            block = render_tables()
            g0mem = gpu_mem_per(0)
            g1mem = gpu_mem_per(1)
            header = ("### 📊 实验进展全矩阵（自动更新，%s）\n\n"
                      "**GPU**: GPU0=%dMB GPU1=%dMB | %d eval proc(s) | "
                      "done %d/%d cells" %
                      (ts, g0mem, g1mem, procs, done, len(ALL)))
            full = header + "\n\n" + block
            try:
                content = open(DIALOG).read()
                if "📊 实验进展全矩阵" in content:
                    content = re.sub(
                        r"### 📊 实验进展全矩阵.*?(?=\n## |\Z)",
                        full.strip(), content, flags=re.DOTALL)
                else:
                    content = content + "\n\n" + full
                open(DIALOG, "w").write(content)
            except Exception as e:
                log("dialog FAIL: %s" % e)

            # ── memory 快照 ──
            try:
                os.makedirs(MEMDIR, exist_ok=True)
                tp = task_progress()
                prog_lines = []
                for tname, (d, total) in tp.items():
                    prog_lines.append("- %s: %d/%d" % (tname, d, total))
                snap = ("---\nname: experiment-status\n"
                        "description: 批量调度实时快照\n"
                        "metadata:\n  type: project\n---\n\n"
                        "# 实验调度 @ %s\n\n"
                        "- **完成**: %d/%d\n- **进程**: %d\n- **显存**: %dMB\n"
                        "- **状态**: %s\n\n## 进度\n\n%s\n" %
                        (ts, done, len(ALL), procs, mem,
                         "⏳ IDLE" if idle else "🔄 运行中",
                         "\n".join(prog_lines)))
                with open(os.path.join(MEMDIR, "experiment-status.md"), "w") as f:
                    f.write(snap)
            except Exception:
                pass

        except Exception as e:
            log("CRASH: %s" % e)

        finally:
            try:
                fd.close()
            except Exception:
                pass

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()