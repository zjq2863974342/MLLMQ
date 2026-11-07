#!/usr/bin/env python3
"""
GPU Monitoring Script — runs every 10 minutes via cron.
- Checks GPU status and running processes
- Analyzes experiment logs for errors
- Logs findings to dialog.md
- If no errors, outputs experiment matrix summary
"""
import subprocess, json, re, time, os, glob
from datetime import datetime

DIALOG_PATH = "/nas3/kangjia/mllmq/dialog.md"
OUTPUT_BASE = "/nas3/kangjia/mllmq/output/internvl2-8b/eval/seedbench_2_local"
QUEUE_LOG_DIR = "/nas3/kangjia/mllmq/output/logs"

def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def append_dialog(section, lines):
    entry = f"\n\n### [MONITOR-{section}] — {now()}\n\n"
    for line in lines:
        entry += f"- {line}\n"
    try:
        with open(DIALOG_PATH, "a") as f:
            f.write(entry)
    except Exception as e:
        print(f"ERROR writing dialog: {e}")

def check_gpu():
    out, err, rc = run(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu --format=csv,noheader,nounits"
    )
    if rc != 0:
        return ["❌ nvidia-smi FAILED: " + err], True

    findings = []
    has_issue = False
    for line in out.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            idx, name, util, mem_used, mem_total, temp = parts[:6]
            util_n = int(util)
            mem_pct = int(mem_used) / int(mem_total) * 100
            temp_n = int(temp)

            if temp_n > 85:
                findings.append(f"⚠️  GPU{idx} HOT: {temp_n}°C | util={util}% | mem={mem_used}/{mem_total} MiB ({mem_pct:.0f}%)")
                has_issue = True
            elif mem_pct > 95:
                findings.append(f"⚠️  GPU{idx} VRAM HIGH: {mem_pct:.0f}% ({mem_used}/{mem_total} MiB) | util={util}%")
                has_issue = True
            elif util_n == 0 and int(mem_used) > 5000:
                findings.append(f"⚠️  GPU{idx} IDLE with VRAM: util=0% | mem={mem_used} MiB (stuck?)")
                has_issue = True
            else:
                findings.append(f"✅ GPU{idx}: {name} | util={util}% | mem={mem_used}/{mem_total} MiB ({mem_pct:.0f}%) | temp={temp_n}°C")

    return findings, has_issue

def check_processes():
    out, _, _ = run("ps aux | grep eval_benchmarks | grep -v grep")
    lines = [l for l in out.split("\n") if l.strip()]

    if not lines:
        return ["📌 No eval_benchmarks processes running"], True

    findings = [f"📌 {len(lines)} eval process(es) running:"]
    for l in lines:
        parts = l.split()
        if len(parts) >= 11:
            pid, cpu, mem = parts[1], parts[2], parts[3]
            cmd = " ".join(parts[10:])
            method_m = re.search(r'--method\s+(\S+)', cmd)
            wbit_m = re.search(r'--w_bit\s+(\S+)', cmd)
            tasks_m = re.search(r'--tasks\s+(\S+)', cmd)
            method = method_m.group(1) if method_m else "?"
            wbit = wbit_m.group(1) if wbit_m else "?"
            tasks = tasks_m.group(1) if tasks_m else "?"
            findings.append(f"  PID={pid} CPU={cpu}% MEM={mem}% | {method}_w{wbit} on {tasks}")
    return findings, False

def check_logs():
    # Find recent seed logs
    log_files = sorted(
        glob.glob(f"{QUEUE_LOG_DIR}/seed_full_a100_*.log") + glob.glob(f"{QUEUE_LOG_DIR}/seed_*_*.log"),
        key=os.path.getmtime, reverse=True
    )
    if not log_files:
        return ["📌 No SEED log files found"], False

    latest_log = log_files[0]
    mtime = os.path.getmtime(latest_log)
    age_min = (time.time() - mtime) / 60

    try:
        with open(latest_log, "r") as f:
            # Read last 10KB
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 10240))
            content = f.read()
    except Exception as e:
        return [f"❌ Cannot read log: {e}"], True

    findings = []
    has_issue = False

    # Check for critical errors
    if "CUDA out of memory" in content or "OutOfMemoryError" in content:
        findings.append("🔴 CUDA OOM detected!")
        has_issue = True
    if re.search(r'rc=13[7-9]|SIGKILL', content):
        findings.append("🔴 Process killed (rc=137 SIGKILL) — likely OOM Killer")
        has_issue = True
    if "ModuleNotFoundError" in content:
        missing = re.findall(r"No module named '(\S+)'", content)
        findings.append(f"🔴 Missing module: {', '.join(missing) if missing else 'unknown'}")
        has_issue = True
    if "RuntimeError" in content or "Traceback" in content:
        last_tb = re.findall(r'(\w+Error:.*)', content)
        if last_tb:
            findings.append(f"⚠️  Runtime error: {last_tb[-1][:120]}")
            has_issue = True

    # Progress summary
    done_c = content.count("[DONE]")
    fail_c = content.count("[FAIL]")
    skip_c = content.count("[SKIP]")
    findings.append(f"📊 Log progress: {done_c} done | {fail_c} failed | {skip_c} skipped | age={age_min:.0f}min | file={os.path.basename(latest_log)}")

    return findings, has_issue

def check_results():
    result_files = glob.glob(f"{OUTPUT_BASE}/**/*_results.json", recursive=True)

    expected = [
        "fp16", "rtn_w4a16", "rtn_w3a16", "rtn_w8a16",
        "awq_w3a16", "awq_w4a16", "awq_w8a16",
        "block_w3a16", "block_w4a16", "block_w8a16",
        "mllmq_w3a16", "mllmq_w4a16", "mllmq_w8a16",
    ]
    completed = []
    for cell in expected:
        tag_dir = os.path.join(OUTPUT_BASE, cell)
        if os.path.isdir(tag_dir) and glob.glob(f"{tag_dir}/*_results.json"):
            completed.append(cell)

    findings = [f"📊 SEED-Bench-2: {len(completed)}/13 cells done ({len(result_files)} result files)"]
    pending = [c for c in expected if c not in completed]
    if pending:
        findings.append(f"  Pending ({len(pending)}): {', '.join(pending)}")

    # Extract accuracies from completed results
    for r in sorted(result_files):
        try:
            with open(r, "r") as f:
                data = json.load(f)
            tag = os.path.basename(os.path.dirname(r))
            results_d = data.get("results", {})
            for k, v in results_d.items():
                if isinstance(v, (int, float)):
                    findings.append(f"  ✅ {tag}: {k} = {v:.4f}")
        except:
            pass

    return findings, len(pending) > 0

def matrix_summary():
    """Generate full experiment matrix summary"""
    bench_map = {
        "mme_local": "/nas3/kangjia/mllmq/output/internvl2-8b/eval/mme_local",
        "mmmu_val": "/nas3/kangjia/mllmq/output/internvl2-8b/eval/mmmu_val",
        "textvqa_val_local": "/nas3/kangjia/mllmq/output/internvl2-8b/eval/textvqa_val_local",
        "pope_random": "/nas3/kangjia/mllmq/output/internvl2-8b/eval/pope_random_local",
        "pope_popular": "/nas3/kangjia/mllmq/output/internvl2-8b/eval/pope_popular_local",
        "pope_adversarial": "/nas3/kangjia/mllmq/output/internvl2-8b/eval/pope_adversarial_local",
        "mathvista": "/nas3/kangjia/mllmq/output/internvl2-8b/eval/mathvista_testmini_local",
        "seedbench_2": OUTPUT_BASE,
    }

    lines = [f"### 📊 Full Matrix — {now()}", ""]
    lines.append("| Benchmark | Done | Pending |")
    lines.append("|-----------|:----:|:-------:|")

    total_done = 0
    for name, path in bench_map.items():
        results = glob.glob(f"{path}/**/*_results.json", recursive=True)
        # Count unique cells
        cells = set()
        for r in results:
            cells.add(os.path.basename(os.path.dirname(r)))
        done = len(cells)
        total_done += done
        pending = 13 - done
        lines.append(f"| {name} | **{done}/13** | {pending} |")

    total = 8 * 13
    lines.append(f"| **TOTAL** | **{total_done}/{total}** | {total-total_done} |")
    lines.append("")
    return lines

def main():
    print(f"=== GPU Monitor {now()} ===")

    all_findings = []
    any_error = False

    gpu_findings, gpu_err = check_gpu()
    all_findings.extend(gpu_findings)
    any_error = any_error or gpu_err

    proc_findings, proc_err = check_processes()
    all_findings.extend(proc_findings)
    any_error = any_error or proc_err

    log_findings, log_err = check_logs()
    all_findings.extend(log_findings)
    any_error = any_error or log_err

    res_findings, res_err = check_results()
    all_findings.extend(res_findings)
    any_error = any_error or res_err

    # Print
    for f in all_findings:
        print(f"  {f}")

    if any_error:
        # Log errors/problems to dialog.md
        error_lines = [l for l in all_findings if any(c in l for c in ["⚠️", "🔴", "❌", "📌"])]
        if not error_lines:
            error_lines = all_findings
        append_dialog("ISSUE", error_lines)
        print(f"\n  → Issues logged to dialog.md")
    else:
        # All healthy — log matrix
        matrix = matrix_summary()
        append_dialog("HEALTHY", matrix)
        print(f"\n  → Matrix logged to dialog.md")

if __name__ == "__main__":
    main()