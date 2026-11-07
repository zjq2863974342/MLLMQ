# MLLMQ minimal migration bundle (2026-11-07)

This tarball contains the **minimal, self-contained set** for:

1. reproducing the Janus-Pro-7B MLLMQ experiments (code + scales + calibration + scripts), and
2. writing the CVPR paper (authoritative result tables + analysis figures + writing knowledge).

Extract from the project root:

```bash
tar -xzf mllmq_minimal_20261107.tar.gz -C /path/to/mllmq
cd /path/to/mllmq
```

## What is included

- `TOKNOW.md`, `TOKNOW_patch.md` — **read these first**. They contain the authoritative result tables, method/config, third_party inventory, host-only pitfalls and paper-writing red lines.
- `MLLMQ/` — code, configs, `tasks_local`, current `scale_cache` (5 dirs), sensitivity/ablation code.
- `scripts/`, `queue/` — evaluation/queue scripts.
- `third_party/lmms-eval/` — the patched lmms-eval fork (custom `janus.py`, patched `internvl2.py`/`mathvista`/`mmmu`), required for understanding evaluation.
- `data/coco/*.json` — calibration JSONs only.
- `output/` selected:
  - `score.json` aggregate
  - DPG main scores
  - GenEval `results.jsonl` (W3A16 + W4A8)
  - SEED summaries
  - T2I-CompBench current score JSONs (7/8 categories; `non_spatial`/`complex` still pending)
  - sensitivity visualizations (`output/sencetive`, selected `output/sensitive/grad_ratio_20261102`)
  - A3 mechanism figures
  - `output/paper_samples` qualitative samples
  - analysis `results.jsonl` for Exp1/Exp5/A8/calib/Exp2/mixed-bit (no generated images)
- `MIGRATION_README.md` / `MANIFEST.txt` — this file and the file list.

## What is NOT included (re-download or rebuild)

- Model checkpoints (`checkpoints/Janus/...`, 22G) — re-download.
- Benchmark data (`data/benchmarks*`, `data/MMMU`, COCO train images) — re-download; see `TOKNOW_patch.md` P3/P4.
- Scorer weights (BLIP/CLIP/UniDet/HPSv2/ImageReward/PickScore) — re-download.
- `venvs/` — rebuild from `requirements.txt`; do **not** copy.
- Raw generated images (GenEval/DPG/T2I PNGs) — regenerate.
- Raw understanding `*_results.json` with per-sample predictions (large) — metrics are in `score.json` and `TOKNOW.md` §4.
- `output/sensitive/grad_ratio_20261102/*/heatmap/` raw per-layer heatmaps (163M) — summary figures/npz are included.
- `backen/`, logs, old backups.

## Environment / install

- Python 3.10, torch `2.4.0+cu124`, transformers `4.45.0` (see `TOKNOW_patch.md` P3.1).
- lmms-eval must be editable-installed from the new path:

```bash
pip install -e third_party/lmms-eval --no-deps
python -c "import lmms_eval; print(lmms_eval.__file__)"
```

- Other Python deps: `pip install -r requirements.txt` (install torch first from the PyTorch index; see README).
- T2I scoring needs the two venvs: `venvs/t2icomp` (BLIP/CLIP) and `venvs/t2idet` (UniDet). Rebuild them on the new machine; do not copy.

## Integrity

Verify the tarball with the accompanying `.sha256` file:

```bash
sha256sum -c mllmq_minimal_20261107.tar.gz.sha256
```

## Notes

- The bundle is a **snapshot**. The T2I-CompBench `non_spatial`/`complex` scores were still being retried at packaging time; check `/nas3/kangjia/mllmq/output/janus-pro-7b/t2i_compbench_current_scores/w3a16/` on the source machine if you need the final values.
- `TOKNOW.md` §4 has the current authoritative tables; `TOKNOW_patch.md` explains the evidence caveats (especially “MLLMQ vs MBQ”).