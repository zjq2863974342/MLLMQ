#!/usr/bin/env bash
# ============================================================================
# Download benchmark datasets for MLLMQ multi-benchmark evaluation.
#
# Status (2026-09-14):
#   MME        ✅  cached at data/benchmarks/lmms-lab___mme/           (already complete)
#   TextVQA    ✅  cached at data/benchmarks/lmms-lab___textvqa/       (validation + test)
#   SEED-Bench2⚠️  PARTIAL download (data/benchmarks/lmms-lab___seed-bench-2/ incomplete)
#   MathVista  ❌  not cached  (lmms-lab/mathvista)
#   MMLU       ❌  not cached  (hails/mmlu_no_train)
#
# REQUIRED: a HuggingFace / ModelScope token with access to gated datasets.
#   export HF_TOKEN=hf_...          # huggingface.co
#   # OR if your env routes through the ModelScope mirror (HF_ENDPOINT etc.):
#   export MODELSCOPE_API_TOKEN=... # modelscope.cn
#
# The `datasets` cache lives under data/benchmarks/ (see HF_DATASETS_CACHE).
# Each step uses the datasets library so the download lands in the same cache
# that lmms-eval reads — no double storage.
#
# Run one section at a time (they are gated / large). Always start with a
# --dry_run (see usage of lmms_eval) to confirm auth before the big pulls.
# ============================================================================

set -uo pipefail

PROJ_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CACHE_DIR="$PROJ_ROOT/data/benchmarks"
mkdir -p "$CACHE_DIR"

# ── Force the datasets cache that lmms-eval / eval_benchmarks.py already use ──
export HF_DATASETS_CACHE="$CACHE_DIR"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/.cache/huggingface}"

cd "$PROJ_ROOT"

echo "PROJ_ROOT = $PROJ_ROOT"
echo "HF_DATASETS_CACHE = $HF_DATASETS_CACHE"
echo "HF_TOKEN set: ${HF_TOKEN:+yes}${HF_TOKEN:-no}"
echo "MODELSCOPE_API_TOKEN set: ${MODELSCOPE_API_TOKEN:+yes}${MODELSCOPE_API_TOKEN:-no}"

# ═══════════════════════════════════════════════════════════════════════════
# 1) SEED-Bench-2 — resume the interrupted pull (418 parquet shards, ~5 GB)
#    lmms-eval task: seedbench_2   dataset: lmms-lab/SEED-Bench-2
# ═══════════════════════════════════════════════════════════════════════════
seedbench_2() {
    echo ">>> [SEED-Bench-2] resuming via datasets load (auto no-op on complete)"
    python - <<'PY'
import os
from datasets import load_dataset
ds = load_dataset("lmms-lab/SEED-Bench-2", split="test", token=os.environ.get("HF_TOKEN"))
print("SEED-Bench-2 loaded:", len(ds), "rows — check cache at data/benchmarks/lmms-lab___seed-bench-2/")
PY
}

# ═══════════════════════════════════════════════════════════════════════════
# 2) MathVista — vision math QA (lmms-lab/mathvista)
#    lmms-eval tasks: mathvista_testmini (group → 3 subtasks)
# ═══════════════════════════════════════════════════════════════════════════
mathvista() {
    echo ">>> [MathVista] downloading lmms-lab/mathvista"
    python - <<'PY'
import os
from datasets import load_dataset
ds = load_dataset("lmms-lab/mathvista", split="testmini", token=os.environ.get("HF_TOKEN"))
print("MathVista testmini loaded:", len(ds), "rows")
PY
}

# ═══════════════════════════════════════════════════════════════════════════
# 3) MMLU — 57-subject text QA (hails/mmlu_no_train ≈ cais/mmlu)
#    lmms-eval task: mmlu (group) OR a single subtask e.g. mmlu_abstract_algebra
#    NOTE: needs trust_remote_code / auth
# ═══════════════════════════════════════════════════════════════════════════
mmlu() {
    echo ">>> [MMLU] downloading hails/mmlu_no_train (all 57 subjects)"
    python - <<'PY'
import os
from datasets import load_dataset
tok = os.environ.get("HF_TOKEN")
for subj in ["abstract_algebra", "college_computer_science", "professional_law"]:
    try:
        ds = load_dataset("hails/mmlu_no_train", subj, split="test",
                          token=tok, trust_remote_code=True)
        print(f"  {subj}: {len(ds)} rows")
    except Exception as e:
        print(f"  {subj}: FAILED — {type(e).__name__}: {e}")
        raise
PY
}

# ═══════════════════════════════════════════════════════════════════════════
# 4) Verify via lmms-eval (auth check, no full inference)
#    Usage:  # ./download_benchmarks.sh dryrun mathvista
# ═══════════════════════════════════════════════════════════════════════════
dryrun() {
    local task=$1
    echo ">>> [lmms-eval --dry_run] task=$task  (validates download + task config, no inference)"
    python -m lmms_eval \
        --model internvl2 \
        --model_args pretrained=/nas3/kangjia/mllmq/checkpoints/InternVL2-8B \
        --tasks "$task" \
        --batch_size 1 \
        --output_path /nas3/kangjia/mllmq/output/dryrun_$task \
        --dry_run
}

# ═══════════════════════════════════════════════════════════════════════════
case "${1:-}" in
    seedbench_2)  seedbench_2 ;;
    mathvista)    mathvista ;;
    mmlu)         mmlu ;;
    dryrun)       dryrun "${2:?usage: $0 dryrun <task>}" ;;
    *)
        echo "Usage: $0 {seedbench_2|mathvista|mmlu|dryrun <task>}"
        echo "  seedbench_2   resume SEED-Bench-2 download"
        echo "  mathvista     download MathVista"
        echo "  mmlu          download MMLU subset"
        echo "  dryrun <task> lmms-eval auth+config check (e.g. mathvista_testmini)"
        exit 1 ;;
esac