#!/usr/bin/env python3
"""COCO-val dual-queue hourly monitor. Appends a snapshot to dialog.md.

Usage (via cron every hour):
  python3 /nas3/kangjia/mllmq/scripts/monitor_coco_val.py --tag COCOVAL_MON

Appends one '[MONITOR]' entry per invocation to /nas3/kangjia/mllmq/dialog.md.
Idempotent per existing tag; caller passes a unique tag to dedupe retries.
"""
import argparse
import datetime
import os
import re
import subprocess

ROOT = "/nas3/kangjia/mllmq"
OUT = os.path.join(ROOT, "output/janus-pro-7b/coco_val")
DIALOG = os.path.join(ROOT, "dialog.md")

CONFIG_SETS = {
    "w4": ["fp16", "rtn_w4", "awq_w4", "mbq_block_w4",
           "mbq_scalar_w4", "mllmq_mae_w4", "mllmq_mse_w4"],
    # W3 队列跑在另一容器(A100 双卡)；fp16 与位宽无关，W3 复用 W4 队列的 fp16，故不计入本组
    "w3": ["mllmq_mae_w3", "mbq_block_w3", "rtn_w3",
           "awq_w3", "mbq_scalar_w3", "mllmq_mse_w3"],
    # W8 档 2026-10-16 20:22 起补齐（本容器 A800；mae_w8 已于 10-16 11:31 完工）
    # 注：**不含 rtn_w8** —— 10-16 20:34 用户指示不跑（GenEval W8 上 rtn 与 awq 并列第一、
    #     COCO 上信息量最低）。COCO 的 W8 行因此只有 5 config，GenEval 侧是全的 → 论文需注明。
    "w8": ["mllmq_mae_w8", "mbq_block_w8", "awq_w8",
           "mbq_scalar_w8", "mllmq_mse_w8"],
}
CONFIG_SETS["all"] = CONFIG_SETS["w4"] + CONFIG_SETS["w3"]

# 默认组 = w4，保证历史调用方的输出格式与内容逐字节不变（scripts/ 在共享 NAS 上，
# 另一容器的 cron 也读这个文件）
CONFIGS = CONFIG_SETS["w4"]
TARGET = 5000

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

def count_pngs(tag):
    d = os.path.join(OUT, tag)
    if not os.path.isdir(d):
        return 0
    n = 0
    for root, _dirs, files in os.walk(d):
        n += sum(1 for f in files if f.endswith(".png"))
    return n

def pids(cmd_pat):
    try:
        out = subprocess.run(["pgrep", "-f", cmd_pat], capture_output=True, text=True)
        return [p for p in out.stdout.split() if p]
    except Exception:
        return []

def running_python():
    return pids(r"eval_generate_quant\.py")

def driver_pids():
    # 同时匹配 W4 (run_coco_val_dual_queue.sh) 与 W3 (run_coco_val_w3_dual_queue.sh)
    return pids(r"run_coco_val_.*dual_queue\.sh")

def active_task():
    """Return list of running config tags (from process cmdline)."""
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
        tags = []
        for line in out.stdout.splitlines():
            if "eval_generate_quant.py" in line and "--benchmark coco_val" in line:
                m = re.search(r"coco_val/([a-z0-9_]+)", line)
                if m:
                    tags.append(m.group(1))
        return sorted(set(tags))
    except Exception:
        return []

def compute_md5(block):
    import hashlib
    return hashlib.md5(block.encode()).hexdigest()[:10]

def load_prev_state(configs_name):
    """上次快照的 png 计数，用于跨容器(--remote)时以吞吐而非进程判定存活。"""
    import json
    path = os.path.join(ROOT, "output/logs", f".monitor_state_{configs_name}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_state(configs_name, counts):
    import json
    path = os.path.join(ROOT, "output/logs", f".monitor_state_{configs_name}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                       "counts": counts}, f)
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="COCOVAL_MON")
    ap.add_argument("--configs", default="w4", choices=sorted(CONFIG_SETS),
                    help="监控哪一组 config（默认 w4，保持历史行为不变）")
    ap.add_argument("--remote", action="store_true",
                    help="该组跑在另一容器：本容器 ps/pgrep 看不到其进程，"
                         "改用 png 增量判定存活，避免把健康的任务误报成 STALLED")
    args = ap.parse_args()

    configs = CONFIG_SETS[args.configs]
    prev = load_prev_state(args.configs)

    counts = {t: count_pngs(t) for t in configs}
    done = [t for t in configs if counts[t] >= TARGET]
    running = active_task()
    drivers = driver_pids()
    pythons = running_python()
    ts = now_str()

    pct_global = sum(min(counts[t], TARGET) for t in configs) / (TARGET * len(configs))

    # GPU usage
    gpu_info = ""
    try:
        g = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        gpu_info = "GPU: " + " | ".join(g.stdout.splitlines())
    except Exception:
        gpu_info = "GPU: n/a"

    # png 增量（跨容器唯一可信的存活信号；nvidia-smi 的利用率字段本项目已实测不可信）
    delta = None
    minutes = None
    if prev and prev.get("counts"):
        delta = {t: counts[t] - prev["counts"].get(t, 0) for t in configs}
        try:
            t_prev = datetime.datetime.fromisoformat(prev["ts"])
            minutes = max((datetime.datetime.now() - t_prev).total_seconds() / 60.0, 1e-9)
        except Exception:
            minutes = None

    state = "RUNNING"
    if args.remote:
        # 本机看不到对端进程属正常，不能据此判 STALLED — 只看 png 是否在长
        if len(done) == len(configs):
            state = "DONE"
        elif delta is not None and sum(v for v in delta.values() if v > 0) == 0:
            state = "STALLED(png-frozen)"
        elif delta is None:
            state = "REMOTE(首次快照)"
        else:
            state = "REMOTE-RUNNING(png-advancing)"
    elif not drivers and not pythons:
        state = "DONE" if len(done) == len(configs) else "STALLED"
    elif not pythons and drivers:
        state = "IDLE-BETWEEN"

    save_state(args.configs, counts)

    lines = []
    if args.configs != "w4":
        lines.append(f"### [MONITOR] COCO-val [{args.configs.upper()}] — {ts} · {state}")
    else:
        lines.append(f"### [MONITOR] COCO-val — {ts} · {state}")
    # --remote 时本容器 ps 里的 tag 属于**另一组**任务（如 W3），印在 W4 条目下会误导，
    # 改用 png 增量推出"谁在产出"
    if args.remote:
        if delta is None:
            who = "首次快照无基线（对端进程跨容器不可见）"
        else:
            adv = [t for t in configs if delta.get(t, 0) > 0]
            who = ", ".join(adv) if adv else "无（png 全冻结 → 疑 STALLED）"
    else:
        who = ", ".join(running) if running else "无"

    lines.append(f"- **队列进度** {len(done)}/{len(configs)} 完成 · 全局 {(pct_global * 100):.0f}% · "
                 f"正在跑: {who}")
    for t in configs:
        d = delta.get(t) if delta else None
        rate = ""
        if d is not None and d > 0 and minutes:
            rate = f"  (+{d}, {d / minutes:.1f}/min)"
        elif d is not None and counts[t] > 0 and counts[t] < TARGET:
            # 已开始但 png 不动 = 真冻结；0 张的只是还没轮到，不该显示 (+0) 以免误读成 STALLED
            rate = "  (+0 ⚠️冻结)"
        live = (d or 0) > 0 if args.remote else (t in running or (d or 0) > 0)
        mark = "✅" if counts[t] >= TARGET else ("🔄" if live else ("⏳" if counts[t] > 0 else "⬜"))
        lines.append(f"  {mark} `{t:<14s}` {counts[t]:>5,}/{TARGET}{rate}")
    if args.remote:
        lines.append(f"- **进程**: 跨容器不可见（本容器 python {len(pythons)} · driver {len(drivers)}）→ 判据为 png 增量")
    else:
        lines.append(f"- **进程**: python {len(pythons)} · driver {len(drivers)}")
    if args.remote:
        lines.append(f"- {gpu_info}  ← ⚠️ 本容器(A100)的卡，**不是对端 W4 的 A800**；对端显存/利用率从这里量不到")
    else:
        lines.append(f"- {gpu_info}")

    # ETA estimate based on running task's recent throughput (num pngs / elapsed)
    etas = []
    for t in TARGET,:
        pass
    if args.tag:
        lines.append(f"- tag={args.tag}")

    block = "\n".join(lines) + "\n"
    mid = compute_md5(block)

    # Dedup: skip if identical block already present in dialog.md
    existed = False
    if os.path.exists(DIALOG):
        cur = open(DIALOG, encoding="utf-8").read()
        if f"md5={mid}" in cur:
            existed = True
    if existed:
        print("duplicate snapshot, not writing")
        return

    block += f"> md5={mid}\n"
    with open(DIALOG, "a", encoding="utf-8") as f:
        f.write("\n" + block)
    print("appended to dialog.md:", mid)

if __name__ == "__main__":
    main()
