#!/usr/bin/env python3
"""
Supplement SEED-Bench-2 parquet shards (ModelScope, ignore-TLS) that were
interrupted at 284/418. Downloads the missing test-<idx>-of-00418.parquet
files that are NOT already cached, into data/seed_patch/ keyed by shard index.

Why ignore-TLS: the sandbox's CA chain is incomplete for modelscope.cn, but
the endpoint is reachable. This is I/O-only (no GPU).

Usage:
    python scripts/continue_seed_download.py [--mirror modelscope|hf]
"""
import argparse, hashlib, json, os, re, ssl, sys, glob, urllib.request, urllib.parse

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DL = os.path.join(PROJ, "data", "benchmarks", "downloads")
PATCH = os.path.join(PROJ, "data", "seed_patch")
os.makedirs(PATCH, exist_ok=True)

TOTAL = 418

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mirror", default="modelscope",
                   choices=["modelscope", "hf"])
    p.add_argument("--max-shards", type=int, default=None,
                   help="download at most N missing shards (dry test)")
    p.add_argument("--only-missing", action="store_true", default=True)
    return p.parse_args()

def existing_seed_indices():
    """Return the set of shard indices already cached (via *.json descriptors)."""
    idxs = set()
    for j in glob.glob(os.path.join(DL, "*.json")):
        try:
            u = json.load(open(j)).get("url", "")
        except Exception:
            continue
        if "SEED-Bench-2" not in u:
            continue
        m = re.search(r"test-(\d+)-of-\d+\.parquet", urllib.parse.unquote(u))
        if m:
            base = j[:-5]
            if os.path.isfile(base) and os.path.getsize(base) > 1000:
                idxs.add(int(m.group(1)))
    return idxs

def download_one(url, dest, ctx, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "mllmq-seed-continue/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return os.path.getsize(dest)

def main():
    args = parse()
    have = existing_seed_indices()
    missing = sorted(set(range(TOTAL)) - have)
    print("cached:", len(have), " missing:", len(missing))

    if args.mirror == "modelscope":
        url_tpl = ("https://modelscope.cn/api/v1/datasets/lmms-lab/SEED-Bench-2/repo"
                   "?Source=SDK&Revision=master&FilePath=data%2Ftest-{idx:05d}-of-00418.parquet")
    else:
        url_tpl = "https://huggingface.co/datasets/lmms-lab/SEED-Bench-2/resolve/main/data/test-{idx:05d}-of-00418.parquet"

    if args.only_missing:
        todo = missing
    else:
        todo = list(range(TOTAL))
    if args.max_shards is not None:
        todo = todo[:args.max_shards]

    ctx = ssl._create_unverified_context()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch_one(idx):
        url = url_tpl.format(idx=idx)
        dest = os.path.join(PATCH, f"test-{idx:05d}-of-00418.parquet")
        if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
            return ("skip", idx, 0)
        try:
            sz = download_one(url, dest, ctx)
            return ("ok", idx, sz)
        except Exception as e:
            return ("fail", idx, f"{type(e).__name__}: {e}")

    ok, fail, conn_err = 0, [], 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_one, i): i for i in todo}
        for fu in as_completed(futs):
            status, idx, info = fu.result()
            if status == "skip":
                print(f"  [skip] already in patch: {idx}")
                ok += 1
            elif status == "ok":
                print(f"  [OK] {idx:03d} -> {info//1024} KB")
                ok += 1
            else:
                print(f"  [FAIL] {idx:03d}: {info}")
                fail.append(idx)
                conn_err += 1
                if conn_err >= 10:   # 网络级失败太多则停（避免浪费）
                    print("too many failures (%d), aborting" % conn_err)
                    for f in futs: f.cancel()
                    break
    print(f"done: ok={ok} fail={len(fail)}")
    if fail:
        print("failed idx:", fail[:30])
    return 0 if not fail else 1

if __name__ == "__main__":
    sys.exit(main())