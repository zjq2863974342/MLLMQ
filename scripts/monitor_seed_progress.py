#!/usr/bin/env python3
"""
SEED-Bench-2 双队列进度监控脚本

Usage:
    python3 scripts/monitor_seed_progress.py
    python3 scripts/monitor_seed_progress.py --json   # JSON 输出
"""
import os, sys, json, re, glob, time
from datetime import datetime, timedelta
from collections import defaultdict
from subprocess import check_output

PROJ_ROOT = "/nas3/kangjia/mllmq"
LOG_DIR = f"{PROJ_ROOT}/output/logs"
OUTPUT_BASE = f"{PROJ_ROOT}/output/internvl2-8b/eval/seedbench_2_local"
QUEUE_LOG_PATTERN = f"{LOG_DIR}/seed_dual_isolated_*.log"

# ── Config ─────────────────────────────────────────────────
NUM_CHUNKS = 5
TOTAL_CELLS = 12  # 6 Q1 + 6 Q2

Q1_CELLS = [
    ("fp16", "16"), ("rtn", "3"), ("rtn", "8"),
    ("awq", "3"), ("awq", "4"), ("awq", "8"),
]
Q2_CELLS = [
    ("block", "3"), ("block", "4"), ("block", "8"),
    ("mllmq", "3"), ("mllmq", "4"), ("mllmq", "8"),
]
ALL_CELLS = [("GPU0", m, w) for m, w in Q1_CELLS] + [("GPU1", m, w) for m, w in Q2_CELLS]


def tag(method, w_bit):
    return "fp16" if method == "fp16" else f"{method}_w{w_bit}a16"


def get_gpu_status():
    """Return GPU info dict."""
    try:
        out = check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,"
             "utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=30
        ).decode().strip().split("\n")
    except Exception:
        return [{"error": "nvidia-smi failed"}]

    gpus = []
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append({
                "index": parts[0], "name": parts[1],
                "vram_used_mib": int(parts[2]), "vram_total_mib": int(parts[3]),
                "util_pct": int(parts[4]), "temp_c": int(parts[5]),
                "power_w": float(parts[6]) if parts[6] and parts[6].replace('.','',1).replace('-','',1).isdigit() else 0,
            })
    return gpus


def get_system_ram():
    """Return (used_gb_str, available_gb_str)."""
    try:
        out = check_output(["free", "-h"], timeout=5).decode()
        for line in out.split("\n"):
            if line.startswith("Mem:"):
                parts = line.split()
                # Mem:   total   used    free   shared  buff/cache  available
                # parts:  0       1       2      3       4           5
                if len(parts) >= 7:
                    return (parts[2], parts[6])  # used, available
    except Exception:
        pass
    return ("?", "?")


def check_done(task_name, t):
    """Check if a chunk has valid results."""
    d = f"{OUTPUT_BASE}/{task_name}/{t}"
    if not os.path.isdir(d):
        return False
    for f in glob.glob(f"{d}/*_results.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            limit = data.get("config", {}).get("limit")
            if limit is not None and limit < 100:
                continue
            return True
        except Exception:
            continue
    return False


def get_chunk_progress():
    """Scan all chunk results and return done/pending counts per cell."""
    cell_status = {}
    for gpu, method, w_bit in ALL_CELLS:
        t = tag(method, w_bit)
        done_chunks = []
        pending_chunks = []
        for chunk in range(NUM_CHUNKS):
            task_name = f"seedbench_2_local_chunk{chunk}"
            if check_done(task_name, t):
                done_chunks.append(chunk)
            else:
                pending_chunks.append(chunk)
        cell_status[f"{gpu}:{t}"] = {
            "gpu": gpu, "tag": t, "method": method, "w_bit": w_bit,
            "done": done_chunks, "pending": pending_chunks,
            "done_count": len(done_chunks), "pending_count": len(pending_chunks),
        }
    return cell_status


def parse_queue_log():
    """Parse the queue supervisor log to find which cell each queue is on."""
    logs = sorted(glob.glob(QUEUE_LOG_PATTERN), key=os.path.getmtime, reverse=True)
    if not logs:
        return {}

    active = {}
    with open(logs[0]) as f:
        content = f.read()

    # Find current CELL lines for each GPU
    gpu0_cells = list(re.finditer(r"=== \[(GPU\d)\]\s*CELL:\s*(\S+)\s*\(method=(\S+)", content))
    gpu1_cells = [m for m in gpu0_cells if m.group(1) == "GPU1"]
    gpu0_cells = [m for m in gpu0_cells if m.group(1) == "GPU0"]

    # Find START lines
    starts = list(re.finditer(
        r"\[(GPU\d)\]\[START\]\s+(\S+)/(\S+)\s+(\d+:\d+:\d+)", content))
    dones = list(re.finditer(
        r"\[(GPU\d)\]\[DONE\]\s+(\S+)/(\S+)\s+(\d+:\d+:\d+)", content))
    fails = list(re.finditer(
        r"\[(GPU\d)\]\[FAIL\]\s+(\S+)/(\S+)\s+rc=(\d+)\s+(\d+:\d+:\d+)", content))

    for gpu_id in ["GPU0", "GPU1"]:
        gpu_starts = [s for s in starts if s.group(1) == gpu_id]
        gpu_dones = [s for s in dones if s.group(1) == gpu_id]
        gpu_fails = [s for s in fails if s.group(1) == gpu_id]

        # Latest START without matching DONE/FAIL = current running task
        running = None
        for s in reversed(gpu_starts):
            task_tag = f"{s.group(2)}/{s.group(3)}"
            has_result = any(
                d.group(2) == s.group(2) and d.group(3) == s.group(3)
                for d in gpu_dones + gpu_fails
            )
            if not has_result:
                running = {
                    "task": s.group(2), "tag": s.group(3),
                    "start_time": s.group(4),
                }
                break

        active[gpu_id] = {
            "running": running,
            "total_starts": len(gpu_starts),
            "total_dones": len(gpu_dones),
            "total_fails": len(gpu_fails),
        }

    return active


def get_active_chunk_progress(queue_info=None):
    """Parse active chunk log files for real-time progress."""
    if queue_info is None:
        queue_info = {}

    progress = {}
    for gpu_id in ["GPU0", "GPU1"]:
        qi = queue_info.get(gpu_id, {})
        running = qi.get("running", {})
        tag = running.get("tag", "")

        if not tag or not running:
            progress[gpu_id] = {"status": "idle"}
            continue

        # Find the latest chunk log for this tag
        task_part = running.get("task", "seedbench_2_local_chunk")
        chunk_logs = sorted(
            glob.glob(f"{LOG_DIR}/{task_part}_{tag}_*.log"),
            key=os.path.getmtime, reverse=True
        )
        if not chunk_logs:
            # Try broader pattern
            chunk_logs = sorted(
                glob.glob(f"{LOG_DIR}/seedbench_2_local_chunk*_{tag}_*.log"),
                key=os.path.getmtime, reverse=True
            )
        if not chunk_logs:
            progress[gpu_id] = {"status": "no log found", "tag": tag}
            continue

        log_file = chunk_logs[0]
        mtime = os.path.getmtime(log_file)
        is_active = (time.time() - mtime) < 120

        if not is_active:
            progress[gpu_id] = {"status": "idle (log inactive)", "tag": tag,
                                "log": os.path.basename(log_file)}
            continue

        try:
            with open(log_file, errors='replace') as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 8192))
                tail = f.read()

            progress_lines = re.findall(
                r"Model Responding:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[(?:[\d:]+)<([\d:]+),\s*([\d.]+)it/s\]",
                tail
            )
            if progress_lines:
                last = progress_lines[-1]
                pct = int(last[0])
                current = int(last[1])
                total = int(last[2])
                eta_str = last[3]
                it_per_s = float(last[4])

                # Parse ETA
                eta_parts = eta_str.split(":")
                eta_seconds = 0
                if len(eta_parts) == 2:
                    eta_seconds = int(eta_parts[0]) * 60 + int(eta_parts[1])
                elif len(eta_parts) == 3:
                    eta_seconds = int(eta_parts[0]) * 3600 + int(eta_parts[1]) * 60 + int(eta_parts[2])

                progress[gpu_id] = {
                    "status": "running",
                    "tag": tag,
                    "current": current, "total": total, "pct": pct,
                    "it_per_s": it_per_s, "eta_seconds": eta_seconds,
                    "eta_str": eta_str,
                    "log": os.path.basename(log_file),
                }
            elif "Loading checkpoint shards" in tail:
                progress[gpu_id] = {"status": "loading model", "tag": tag}
            elif "Model Responding" in tail:
                progress[gpu_id] = {"status": "starting inference", "tag": tag}
            else:
                progress[gpu_id] = {"status": "initializing", "tag": tag,
                                    "log": os.path.basename(log_file)}
        except Exception as e:
            progress[gpu_id] = {"status": f"error: {e}", "tag": tag}

    return progress


def estimate_completion(chunk_status, active_progress):
    """Estimate total remaining time."""
    # Count total pending chunks
    total_pending = sum(
        cs["pending_count"] for cs in chunk_status.values()
    )

    # Get average it/s from active processes
    speeds = []
    for gpu_id, info in active_progress.items():
        if info.get("status") == "running" and info.get("it_per_s"):
            speeds.append(info["it_per_s"])

    avg_speed = sum(speeds) / len(speeds) if speeds else 2.0  # default ~2 it/s

    # Each chunk is ~4875 samples, time per chunk = 4875 / speed
    # But we have 2 GPUs, so effective throughput = avg_speed * 2
    samples_per_chunk = 4875
    seconds_per_chunk = samples_per_chunk / avg_speed
    effective_seconds_per_chunk = seconds_per_chunk / 2  # 2 GPUs

    # Also account for model loading overhead per cell (~120s per cell switch)
    # Each cell has up to 5 chunks, and we switch cells roughly every 5 chunks
    # Cells with all chunks done = 0 switching
    # Remaining cells = cells with any pending chunk

    remaining_cells = set()
    for cs in chunk_status.values():
        if cs["pending_count"] > 0:
            remaining_cells.add(f"{cs['gpu']}:{cs['tag']}")

    # Each remaining cell adds ~120s overhead for model loading + scale apply
    loading_overhead = len(remaining_cells) * 120 / 2  # divided by 2 GPUs

    total_seconds = total_pending * seconds_per_chunk + loading_overhead
    total_hours = total_seconds / 3600

    eta = datetime.now() + timedelta(seconds=total_seconds)

    return {
        "total_pending_chunks": total_pending,
        "remaining_cells": len(remaining_cells),
        "avg_it_per_s": round(avg_speed, 2),
        "seconds_per_chunk": round(seconds_per_chunk),
        "total_est_hours": round(total_hours, 1),
        "eta": eta.strftime("%m-%d %H:%M"),
        "loading_overhead_hours": round(loading_overhead / 3600, 1),
    }


def render_text(gpus, ram_used, ram_avail, chunk_status, queue_info, active_progress, estimate):
    """Render human-readable report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 72)
    lines.append(f"  SEED-Bench-2 双队列进度监控  |  {now}")
    lines.append("=" * 72)

    # ── GPU ──
    lines.append("")
    lines.append("  🖥️  GPU 状态")
    lines.append("  " + "-" * 44)
    for g in gpus:
        if "error" in g:
            lines.append(f"  ⚠️  GPU 状态不可用: {g['error']}")
            continue
        vram_pct = g["vram_used_mib"] / g["vram_total_mib"] * 100
        bar_len = int(vram_pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  GPU{g['index']} | {g['name'][:28]:28s} | VRAM [{bar}] {vram_pct:4.0f}% | 利用率 {g['util_pct']:3d}% | {g['temp_c']}°C")
    lines.append(f"  系统内存: 已用 {ram_used} / 可用 {ram_avail}")

    # ── Active Progress ──
    lines.append("")
    lines.append("  🔄 实时推理进度")
    lines.append("  " + "-" * 44)
    for gpu_id in ["GPU0", "GPU1"]:
        info = active_progress.get(gpu_id, {})
        qi = queue_info.get(gpu_id, {})
        running = qi.get("running", {})

        if info.get("status") == "running":
            pct = info["pct"]
            bar_len = pct // 5
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {gpu_id} | {info['tag'] if 'tag' in info else running.get('tag','?')} | [{bar}] {pct:3d}% | {info['current']}/{info['total']} | {info['it_per_s']:.1f} it/s | ETA {info['eta_str']}")
        elif info.get("status") == "loading model":
            lines.append(f"  {gpu_id} | {running.get('tag','?')} | 🔄 模型加载中...")
        elif info.get("status") == "starting inference":
            lines.append(f"  {gpu_id} | {running.get('tag','?')} | 🚀 推理启动中...")
        elif info.get("status") == "initializing":
            lines.append(f"  {gpu_id} | {running.get('tag','?')} | ⏳ 初始化中...")
        else:
            lines.append(f"  {gpu_id} | 空闲 ({info.get('status','等待中')})")

    # ── Cell Matrix ──
    lines.append("")
    lines.append("  📊 Cell 完成矩阵")
    lines.append("  " + "-" * 44)
    # Header
    header = "  " + " " * 16
    for c in range(NUM_CHUNKS):
        header += f" C{c}"
    header += "  完成"
    lines.append(header)
    lines.append("  " + "-" * 44)

    # Group by method
    methods_order = ["fp16", "rtn_w3a16", "rtn_w8a16", "awq_w3a16", "awq_w4a16", "awq_w8a16",
                     "block_w3a16", "block_w4a16", "block_w8a16", "mllmq_w3a16", "mllmq_w4a16", "mllmq_w8a16"]
    for mtag in methods_order:
        # Find matching status
        matched = None
        for key, cs in chunk_status.items():
            if cs["tag"] == mtag:
                matched = cs
                break
        if not matched:
            continue
        row = f"  {mtag:14s} |"
        for c in range(NUM_CHUNKS):
            if c in matched["done"]:
                row += " ✅"
            else:
                row += " ❌"
        row += f"  {matched['done_count']}/{NUM_CHUNKS}"
        lines.append(row)

    # ── Queue Summary ──
    lines.append("")
    lines.append("  📈 队列汇总")
    lines.append("  " + "-" * 44)
    for gpu_id in ["GPU0", "GPU1"]:
        qi = queue_info.get(gpu_id, {})
        ap = active_progress.get(gpu_id, {})
        if qi:
            running_tag = qi.get("running", {}).get("tag", ap.get("tag", "?"))
            status_icon = "✅" if ap.get("status") == "running" else ("🔄" if qi.get("running") else "⏸️")
            lines.append(f"  {gpu_id}: 完成 {qi['total_dones']}  |  失败 {qi['total_fails']}  |  当前 {running_tag} {status_icon}")

    # ── ETA ──
    lines.append("")
    lines.append("  ⏱️  预估完成时间")
    lines.append("  " + "-" * 44)
    lines.append(f"  剩余 chunks: {estimate['total_pending_chunks']}  ({estimate['remaining_cells']} 个 cell)")
    lines.append(f"  平均速度: {estimate['avg_it_per_s']} it/s/GPU")
    lines.append(f"  预估单 chunk 耗时: ~{estimate['seconds_per_chunk']//60} 分钟")
    lines.append(f"  预估总剩余时间: ~{estimate['total_est_hours']} 小时")
    lines.append(f"  预计完成: {estimate['eta']}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    json_out = "--json" in sys.argv

    gpus = get_gpu_status()
    ram_used, ram_avail = get_system_ram()
    chunk_status = get_chunk_progress()
    queue_info = parse_queue_log()
    active_progress = get_active_chunk_progress(queue_info)
    estimate = estimate_completion(chunk_status, active_progress)

    if json_out:
        print(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "gpus": gpus,
            "ram": {"used": ram_used, "available": ram_avail},
            "chunk_status": {k: {"done": v["done"], "pending": v["pending"]} for k, v in chunk_status.items()},
            "queue_info": queue_info,
            "active_progress": active_progress,
            "estimate": estimate,
        }, indent=2))
    else:
        print(render_text(gpus, ram_used, ram_avail, chunk_status, queue_info, active_progress, estimate))


if __name__ == "__main__":
    main()