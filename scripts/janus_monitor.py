#!/usr/bin/env python3
"""
Janus GPU monitor — checks every 20 min via cron.
Writes anomalies to dialog.md under ## [MONITOR] section.
"""
import subprocess, time, os, sys, json

DIALOG = "/nas3/kangjia/mllmq/dialog.md"
LOG_DIR = "/nas3/kangjia/mllmq/output/logs"

def run(cmd):
    return subprocess.getoutput(cmd)

def check_gpu():
    """Return GPU info dict."""
    out = run("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader")
    gpus = []
    for line in out.strip().split("\n"):
        if not line.strip(): continue
        idx, name, total, used, free, util, temp = [x.strip() for x in line.split(",")]
        gpus.append({
            "idx": idx, "name": name, "total_mb": int(total.split()[0]),
            "used_mb": int(used.split()[0]), "free_mb": int(free.split()[0]),
            "util_pct": int(util.split()[0].replace("%","")),
            "temp_c": int(temp.replace("C",""))
        })
    return gpus

def check_ram():
    out = run("free -h | grep Mem")
    parts = out.split()
    return {"total": parts[1], "used": parts[2], "free": parts[3], "avail": parts[6]}

def check_processes():
    out = run("ps aux | grep -E 'main_quant|eval_benchmarks|run_janus' | grep -v grep")
    procs = []
    for line in out.strip().split("\n"):
        if not line.strip(): continue
        parts = line.split()
        if len(parts) >= 11:
            procs.append({"pid": parts[1], "cpu": parts[2], "mem": parts[3], "cmd": " ".join(parts[10:])[:120]})
    return procs

def check_logs():
    """Check recent log files for errors."""
    errors = []
    for f in sorted(os.listdir(LOG_DIR)):
        if not f.endswith('.log'): continue
        fpath = os.path.join(LOG_DIR, f)
        try:
            stat = os.stat(fpath)
            if time.time() - stat.st_mtime > 600: continue  # skip old logs
            with open(fpath) as fh:
                lines = fh.readlines()
                last_lines = lines[-20:]
                for l in last_lines:
                    if 'Traceback' in l or 'CUDA out of memory' in l or 'SIGKILL' in l or 'rc=137' in l:
                        errors.append(f"{f}: {l.strip()[:100]}")
        except: pass
    return errors

def append_to_dialog(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n### [MONITOR] — {timestamp}\n\n{msg}\n"
    with open(DIALOG, "a") as f:
        f.write(entry)
    print(f"[MONITOR] {entry[:200]}")

def main():
    gpus = check_gpu()
    ram = check_ram()
    procs = check_processes()
    errors = check_logs()

    issues = []

    # GPU checks
    for g in gpus:
        if g["util_pct"] < 10 and g["used_mb"] > 5000:
            issues.append(f"⚠️  GPU{g['idx']} LOW UTIL: {g['util_pct']}% but VRAM={g['used_mb']}MB (stuck?)")
        if g["used_mb"] > g["total_mb"] * 0.95:
            issues.append(f"🔴 GPU{g['idx']} VRAM HIGH: {g['used_mb']}/{g['total_mb']}MB ({g['used_mb']*100//g['total_mb']}%)")
        if g["temp_c"] > 80:
            issues.append(f"🔴 GPU{g['idx']} TEMP HIGH: {g['temp_c']}C")

    # RAM check
    avail_str = ram["avail"].replace("Gi","")
    try:
        avail_gb = float(avail_str)
        if avail_gb < 10:
            issues.append(f"🔴 LOW RAM: {ram['avail']} available (total {ram['total']})")
    except: pass

    # Process checks
    if not procs:
        issues.append("📌 No eval_benchmarks or main_quant processes running")
    else:
        for p in procs:
            mem_pct = float(p["mem"])
            if mem_pct > 50:
                issues.append(f"🔴 Process PID={p['pid']} MEM HIGH: {mem_pct}%")

    # Error checks
    for e in errors[:5]:
        issues.append(f"🔴 {e}")

    # Build report
    lines = []
    lines.append(f"📊 GPU: " + " | ".join(
        f"GPU{g['idx']}={g['used_mb']}MB/{g['util_pct']}%/{g['temp_c']}C" for g in gpus
    ))
    lines.append(f"💾 RAM: {ram['avail']} avail / {ram['total']} total")
    lines.append(f"📌 {len(procs)} eval process(es): " + ", ".join(
        f"PID={p['pid']}" for p in procs[:5]
    ))
    for issue in issues:
        lines.append(issue)

    msg = "\n".join(lines)

    if issues:
        append_to_dialog(msg)
    print(msg)
    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(main())