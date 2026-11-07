#!/usr/bin/env python3
"""GPU 心跳保活守护进程

保护 GPU1 不被平台回收/抢占。当 GPU1 利用率低于阈值时，创建一个轻量 CUDA
上下文（5000x5000 tensor + matmul）告诉驱动"有人在用这张卡"。

安全规则（11:48 死锁教训）：
  - 绝不在有 gen/eval 进程运行时创建 CUDA 上下文
  - 每次心跳前都用 pgrep 双重校验
  - nvidia-smi 查询一律 timeout -s 9 3，绝不让 smi 挂住阻塞进程

用法:
  nohup python3 -u gpu_heartbeat.py &          # 后台启动（-u = 无缓冲）
  pkill -f gpu_heartbeat.py                    # 停止
"""

import subprocess
import signal
import sys
import os
import time

try:
    import torch
except ImportError:
    print("❌ torch 未安装，心跳脚本无法运行", flush=True)
    sys.exit(1)

# ── 配置 ──────────────────────────────────────────────────────────────

GPU_IDX      = 1
UTIL_THRESH  = 70
CHECK_INTERVAL = 10
HEARTBEAT_DUR  = 3
TIMOUT_SMI     = 9
GEN_PROCS_RE   = r'eval_generate_quant|evaluate_images|main_quant|eval_benchmarks'

# ── 工具函数（不创建 CUDA 上下文）─────────────────────────────────────

def has_gen_process():
    try:
        r = subprocess.run(
            ['pgrep', '-x', '-f', GEN_PROCS_RE],
            capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0
    except Exception:
        return False

def get_gpu_util(idx):
    try:
        r = subprocess.run(
            ['timeout', '-s', '9', str(TIMOUT_SMI), 'nvidia-smi',
             '--query-gpu=index,utilization.gpu,memory.used',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode != 0:
            return None, None
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3 and parts[0] == str(idx):
                try:
                    return int(parts[1]), int(parts[2].replace(' MiB', ''))
                except (ValueError, IndexError):
                    return None, None
    except Exception:
        pass
    return None, None

def get_gpu_mem_total_mb(idx):
    try:
        r = subprocess.run(
            ['timeout', '-s', '9', '3', 'nvidia-smi',
             '--query-gpu=index,memory.total',
             '--format=csv,noheader', '-i', str(idx)],
            capture_output=True, text=True, timeout=15
        )
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if parts and parts[0] == str(idx):
                return int(parts[1].replace(' MiB', ''))
    except Exception:
        pass
    return None

def log(msg):
    print(msg, flush=True)

# ── 心跳类 ────────────────────────────────────────────────────────────

class GpuHeartbeat:
    def __init__(self, idx):
        self.idx = idx
        self.dev = torch.device(f"cuda:{idx}")
        self._alive = False
        self._peak_mb = 0

        self.x = torch.randn(5000, 5000, device=self.dev, dtype=torch.float32)
        self.y = torch.randn(5000, 5000, device=self.dev, dtype=torch.float32)

        total = torch.cuda.get_device_properties(idx).total_memory / 1024**2
        used = torch.cuda.memory_allocated(idx) / 1024**2
        self._peak_mb = used
        self._alive = True
        log(f"   [HEARTBEAT] GPU {idx} 心跳就绪: {total:.0f} MB 总显存, "
            f"张量占用 {used:.0f} MB")

    def pulse(self):
        if not self._alive or not torch.cuda.is_available():
            return
        z = torch.matmul(self.x, self.y)
        del z
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    def peek_peak_mb(self):
        if self._alive:
            self._peak_mb = max(self._peak_mb, torch.cuda.memory_allocated(self.idx) / 1024**2)
        return self._peak_mb

    def release(self):
        if not self._alive:
            return
        try:
            del self.x, self.y
            torch.cuda.empty_cache()
        except Exception:
            pass
        used = torch.cuda.memory_allocated(self.idx) / 1024**2
        self._alive = False
        log(f"   → 心跳释放 | 峰值显存 {self._peak_mb:.0f} MB, 当前显存 {used:.0f} MB")

# ── 主循环 ────────────────────────────────────────────────────────────

def main():
    def handler(signum, frame):
        hb.release()
        log(f"\n[HEARTBEAT] 收到信号 {signum}，安全退出。\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    log(f"\n🫀 GPU {GPU_IDX} 心跳保活守护进程启动")
    log(f"   利用率阈值: {UTIL_THRESH}%          检查间隔: {CHECK_INTERVAL}s")
    log(f"   心跳保持: {HEARTBEAT_DUR}s           安全规则: pgrep 双重校验")
    nv_total = get_gpu_mem_total_mb(GPU_IDX)
    if nv_total:
        log(f"   GPU {GPU_IDX} 总显存: {nv_total} MB\n")

    hb = GpuHeartbeat(GPU_IDX)
    since_last_log = time.time()

    try:
        while True:
            ubefore = time.time()

            # Step 1: gen 有进程？跳过
            if has_gen_process():
                procs = subprocess.run(
                    ['pgrep', '-a', '-f', GEN_PROCS_RE],
                    capture_output=True, text=True, timeout=3
                )
                info = procs.stdout.strip().split('\n')[0][:90] if procs.stdout.strip() else 'gen在跑'
                util, mem = get_gpu_util(GPU_IDX)
                now = time.strftime('%H:%M:%S')
                if time.time() - since_last_log > 45:
                    log(f"[{now}] Gen进程在跑，跳过 | {info} | GPU{GPU_IDX} util={util}% mem={mem}MiB")
                    since_last_log = time.time()
                elapsed = time.time() - ubefore
                time.sleep(max(0, CHECK_INTERVAL - elapsed))
                continue

            # Step 2: 读利用率
            util, mem = get_gpu_util(GPU_IDX)
            if util is None:
                now = time.strftime('%H:%M:%S')
                if time.time() - since_last_log > 60:
                    log(f"[{now}] ⚠️ nvidia-smi查询失败，等待...")
                    since_last_log = time.time()
                elapsed = time.time() - ubefore
                time.sleep(max(0, CHECK_INTERVAL - elapsed))
                continue

            # Step 3: util ≥ 阈值？不保活
            if util >= UTIL_THRESH:
                now = time.strftime('%H:%M:%S')
                if time.time() - since_last_log > 45:
                    log(f"[{now}] GPU{GPU_IDX} util={util}% ≥ {UTIL_THRESH}%，无需心跳 | mem={mem}MiB")
                    since_last_log = time.time()
                elapsed = time.time() - ubefore
                time.sleep(max(0, CHECK_INTERVAL - elapsed))
                continue

            # Step 4: 双重 pgrep 校验
            if has_gen_process():
                now = time.strftime('%H:%M:%S')
                procs = subprocess.run(
                    ['pgrep', '-a', '-f', GEN_PROCS_RE],
                    capture_output=True, text=True, timeout=3
                )
                info = procs.stdout.strip().split('\n')[0][:90] if procs.stdout.strip() else ''
                log(f"[{now}] Gen进程中途出现，跳过 | {info} | GPU{GPU_IDX} util={util}%")
                since_last_log = time.time()
                elapsed = time.time() - ubefore
                time.sleep(max(0, CHECK_INTERVAL - elapsed))
                continue

            # Step 5: 安全！心跳保活
            now = time.strftime('%H:%M:%S')
            log(f"[{now}] 🫀 GPU{GPU_IDX} util={util}% < {UTIL_THRESH}%, mem={mem}MiB → 心跳保活中...")
            peak = hb.peek_peak_mb()
            hb.pulse()
            time.sleep(HEARTBEAT_DUR)
            peak = hb.peek_peak_mb()
            hb.release()
            since_last_log = time.time()
            elapsed = time.time() - ubefore
            time.sleep(max(0, CHECK_INTERVAL - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        hb.release()
        log("[HEARTBEAT] 退出\n")

if __name__ == '__main__':
    main()
