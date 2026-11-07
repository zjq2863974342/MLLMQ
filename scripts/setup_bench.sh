#!/usr/bin/env bash
# ============================================================================
# SEED-Bench-2 offline cache rebuild + local task config
# Usage:  bash scripts/setup_bench.sh seedbench2
#
# No HF token needed. Sources:
#   * data/benchmarks/downloads/         — ModelScope shards already cached (0-283)
#   * data/seed_patch/                   — supplemented via continue_seed_download.py
#   * any other test-*.parquet shard both dirs know about
# ============================================================================
set -uo pipefail
ROOT=/nas3/kangjia/mllmq
MLLMQ=$ROOT/MLLMQ
DL=$ROOT/data/benchmarks/downloads
PATCH=$ROOT/data/seed_patch
OUT=$ROOT/data/benchmarks_local/lmms-lab/SEED-Bench-2
export HF_DATASETS_OFFLINE=1

seedbench2_rebuild() {
    echo ">>> [SEED-Bench-2] rebuild local DatasetDict ..."
    python - "$DL" "$PATCH" "$OUT" <<'PY'
import sys, glob, json, os, re
from datasets import load_dataset

dl, patch, out = sys.argv[1], sys.argv[2], sys.argv[3]
seed_files = {}

import urllib.parse
def shard_idx_from_url(u):
    # url 是 URL 编码路径（FilePath=data%2Ftest-...），先 unquote
    uu = urllib.parse.unquote(u)
    m = re.search(r"test-(\d+)-of-(\d+)\.parquet", uu)
    return int(m.group(1)) if m else None

# 1) downloads/: index 由 json url 推导
for j in glob.glob(os.path.join(dl, "*.json")):
    try:
        u = json.load(open(j)).get("url", "")
    except Exception:
        continue
    if "SEED-Bench-2" in u:  # url 已含 SEED-Bench-2，匹配即读
        base = j[:-5]
        if os.path.isfile(base) and os.path.getsize(base) > 1000:
            idx = shard_idx_from_url(u)
            if idx is not None and idx not in seed_files:
                seed_files[idx] = base

# 2) seed_patch/: 文件名自带 index
if os.path.isdir(patch):
    for p in sorted(glob.glob(os.path.join(patch, "test-*.parquet"))):
        m = re.search(r"test-(\d+)-of-\d+\.parquet", os.path.basename(p))
        if m and os.path.getsize(p) > 1000:
            idx = int(m.group(1))
            if idx not in seed_files:
                seed_files[idx] = p

# 完整性检查
total = 418
have = sorted(seed_files.keys())
missing = [i for i in range(total) if i not in seed_files]
print("  %d/%d shards present" % (len(have), total))
if missing:
    print("  WARN missing %d shards: first10=%s (will still build partial)" %
          (len(missing), missing[:10]))

ordered = [seed_files[i] for i in sorted(seed_files)]
os.makedirs(os.path.dirname(out), exist_ok=True)
ds = load_dataset("parquet", data_files={"test": ordered},
                  split="test")
ds.save_to_disk(out)
print("  saved:", out, "rows:", len(ds))
if missing:
    print("  NOTE: partial dataset (%d/%d shards)" % (len(have), total))
PY
}

case "${1:-}" in
    seedbench2) seedbench2_rebuild ;;
    *) echo "Usage: $0 seedbench2"; exit 1 ;;
esac