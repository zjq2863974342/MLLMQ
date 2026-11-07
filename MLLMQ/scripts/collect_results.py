#!/usr/bin/env python3
"""
Collect and summarize MLLMQ experiment results from /nas3/kangjia/mllmq/output/.

Scans replicate_* directories for MMMU evaluation results, and new
mbq_vs_mllmq comparison experiments for quant config + eval output.

Usage:    python scripts/collect_results.py
    python scripts/collect_results.py --output_dir /nas3/kangjia/mllmq/output
    python scripts/collect_results.py --csv
    # Multi-benchmark (evals run via eval_benchmarks.py -> output/{/{model}/eval/<task>/<method>_w<n>a16/)
    python scripts/collect_results.py --tasks mme,textvqa_val,mmmu_val --output_dir output
"""

import os
import sys
import json
import argparse
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# MMMU result collection from replicate_* directories
# ---------------------------------------------------------------------------
def collect_mmmu_results(output_root):
    """Scan replicate_* directories for MMMU evaluation JSON results."""
    root = Path(output_root)
    results = []

    if not root.exists():
        print(f"Error: directory not found: {root}")
        return results

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        name = subdir.name

        # Look for checkpoints__*/ dir inside
        for inner in subdir.iterdir():
            if not inner.is_dir():
                continue

            # Find latest results JSON
            json_files = sorted(inner.glob("*_results.json"))
            if not json_files:
                continue

            latest = json_files[-1]
            try:
                with open(latest) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Extract MMMU accuracy
            mmmu = data.get("results", {}).get("mmmu_val", {})
            acc = mmmu.get("mmmu_acc,none")

            # Derive method from directory name
            method = "unknown"
            lower = name.lower()
            if "fp16" in lower:
                method = "fp16"
            elif "mllmq" in lower:
                method = "mllmq"
            elif "mbq" in lower or "replicate_results" in lower:
                method = "mbq"
            elif "awq" in lower:
                method = "awq"
            elif "rtn" in lower:
                method = "rtn"
            elif "smoothquant" in lower:
                method = "smoothquant"

            # Derive w_bit from config if available, then from name
            w_bit = None
            for cf in ["quant_config.json", "config.json"]:
                cf_path = subdir / cf
                if cf_path.exists():
                    try:
                        with open(cf_path) as f:
                            qc = json.load(f)
                        w_bit = qc.get("w_bit")
                    except Exception:
                        pass

            # Fallback: guess from experiment name
            if w_bit is None:
                if "w4" in name.lower():
                    w_bit = 4
                elif "w8" in name.lower():
                    w_bit = 8
                elif "w16" in name.lower():
                    w_bit = 16
                else:
                    w_bit = 4  # default for MBQ paper experiments

            results.append({
                "experiment": name,
                "method": method,
                "w_bit": w_bit,
                "mmmu_acc": acc,
                "result_file": str(latest),
            })
            break  # Only first inner dir

    return results


# ---------------------------------------------------------------------------
# MBQ vs MLLMQ comparison collection
# ---------------------------------------------------------------------------
def collect_comparison_results(output_root):
    """
    Scan {model}_{method}_w* directories for quantized models and their
    evaluation results.
    """
    root = Path(output_root)
    results = []

    if not root.exists():
        return results

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name

        # Skip replicate_* dirs (handled by collect_mmmu_results)
        if name.startswith("replicate_"):
            continue

        # Parse: {model_type}_{method}_w{bit}a{bit}
        parts = name.split("_")
        if len(parts) < 3:
            continue

        # Find quant_config.json
        config_path = subdir / "quant_config.json"
        w_bit = None
        if config_path.exists():
            try:
                with open(config_path) as f:
                    qc = json.load(f)
                w_bit = qc.get("w_bit")
            except Exception:
                pass

        # Find eval results
        mmmu_acc = None
        eval_results = None
        for inner in subdir.iterdir():
            if not inner.is_dir():
                continue
            for jf in sorted(inner.glob("*_results.json")):
                try:
                    with open(jf) as f:
                        data = json.load(f)
                    mmmu = data.get("results", {}).get("mmmu_val", {})
                    acc = mmmu.get("mmmu_acc,none")
                    if acc is not None:
                        mmmu_acc = acc
                        eval_results = str(jf)
                except Exception:
                    pass

        results.append({
            "experiment": name,
            "w_bit": w_bit,
            "mmmu_acc": mmmu_acc,
            "eval_file": eval_results,
        })

    return results


# ---------------------------------------------------------------------------
# Multi-benchmark collection (eval_benchmarks.py output structure)
#   output/<model>/eval/<task_name>/<method>_w<n>a16/*_results.json
# ---------------------------------------------------------------------------
# Metric keys, matching eval_benchmarks.py TASK_METRICS
MULTI_TASK_METRICS = {
    "mmmu_val":          "mmmu_acc,none",
    "mme":               "mme_percetion_score,none",
    "mme_local":         "mme_percetion_score,none",
    "mme_cognition":     "mme_cognition_score,none",
    "textvqa_val":       "acc,none",
    "textvqa_val_local": "acc,none",
    "seedbench_2":       "seed_all,none",
    "seedbench_2_local": "seed_all,none",
    "mathvista_testmini": "mathvista_acc,none",
    "cmmmu_val":         "cmmmu_acc,none",
    "vqav2_val":         "vqav2_acc,none",
    "pope":              "pope_accuracy,none",
}


def collect_multi_benchmark_results(output_root, tasks):
    """
    Scan output/<model>/eval/<task>/<method>_w<n>a16/*_results.json for the
    requested tasks. Returns list of dicts:
      {experiment, method, w_bit, tag, task, metric, score}
    """
    root = Path(output_root)
    results = []

    # Discover model eval dirs: output/<model>/eval/
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        eval_dir = model_dir / "eval"
        models_dir = model_dir / "models"

        for task_name in tasks:
            # Candidate (tag, search_root, task_key) — prefer eval/<task>/ layout,
            # then models/<tag>/checkpoints__*/ for the older eval_mmmu.py flow.
            candidates = []

            # 1) eval/<task>/<tag>/
            if eval_dir.is_dir():
                task_dir = eval_dir / task_name
                if task_dir.is_dir():
                    for cell_dir in sorted(task_dir.iterdir()):
                        if cell_dir.is_dir() and not list(cell_dir.glob("*_results.json")):
                            # This is likely a packed-model dir (model.safetensors),
                            # but if it has an inner checkpoints__* with results, use that.
                            inner_hits = [p for p in cell_dir.glob("checkpoints__*/*_results.json")]
                            if inner_hits:
                                candidates.append((cell_dir.name, inner_hits, task_name))
                            continue
                        candidates.append((cell_dir.name, list(cell_dir.glob("*_results.json")), task_name))

            # 2) models/<tag>/checkpoints__*/ — older eval_mmmu.py output
            if models_dir.is_dir():
                for cell_dir in sorted(models_dir.iterdir()):
                    if not cell_dir.is_dir():
                        continue
                    inner_hits = sorted(cell_dir.glob("checkpoints__*/*_results.json"))
                    if inner_hits:
                        candidates.append((cell_dir.name, inner_hits, task_name))

            # Dedup by tag (prefer eval/ layout first)
            seen_tags = {}
            for tag, hits, tkey in candidates:
                if tag in seen_tags:
                    continue
                seen_tags[tag] = (hits, tkey)

            for tag, (hits, tkey) in seen_tags.items():
                # Extract method + w_bit from cell tag ("mllmq_w4a16" / "fp16")
                method = "unknown"
                lower = tag.lower()
                if lower.startswith("fp16"):
                    method, w_bit = "fp16", 16
                else:
                    for m in ("mllmq", "mbq", "awq", "rtn", "smoothquant"):
                        if lower.startswith(m):
                            method = m
                            break
                    wb = None
                    for part in lower.split("_"):
                        if part.startswith("w") and part[1:].rstrip("a").isdigit():
                            wb = part[1:].rstrip("a")
                            break
                    w_bit = int(wb) if wb else None

                if not hits:
                    results.append({
                        "experiment": f"{task_name}/{tag}", "method": method,
                        "w_bit": w_bit, "tag": tag, "task": task_name,
                        "metric": MULTI_TASK_METRICS.get(task_name, ""), "score": None, "status": "pending",
                    })
                    continue

                latest = hits[-1]
                try:
                    with open(latest) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

                res = data.get("results", {}).get(tkey, {})
                if "mme" in task_name and task_name != "mme_cognition":
                    perc = res.get("mme_percetion_score,none")
                    cog = res.get("mme_cognition_score,none")
                    score = perc if perc is not None else None
                    metric = "mme_percetion_score,none"
                elif task_name == "mme_cognition":
                    metric = "mme_cognition_score,none"
                    score = res.get(metric)
                else:
                    metric = MULTI_TASK_METRICS.get(task_name, "")
                    score = res.get(metric) if metric in res else _first_metric(res)

                results.append({
                    "experiment": f"{task_name}/{tag}", "method": method,
                    "w_bit": w_bit, "tag": tag, "task": task_name,
                    "metric": metric, "score": score, "status": "done",
                    "result_file": str(latest),
                })
    return results


def _first_metric(res):
    """Fallback: pick first metric-looking key (ends in ,none / ,exact_match)."""
    for k in res.keys():
        if k.endswith(",none") or k.endswith(",exact_match") or "acc" in k.lower():
            return k
    return None


def print_multi_benchmark_summary(results):
    """Print a cross-benchmark table: rows = method  cols = tasks."""
    if not results:
        print("No multi-benchmark results found.")
        return

    tasks = sorted({r["task"] for r in results})
    # Order method cells logically: fp16 first, then by method, then bit
    method_order = {"fp16": 0, "mllmq": 1, "mbq": 2, "awq": 3, "rtn": 4, "smoothquant": 5}
    cells = sorted(
        {r["tag"] for r in results},
        key=lambda t: (method_order.get(t.split("_")[0], 99), t),
    )

    header = f"{'Cell':<20}" + "".join(f"{t:>16}" for t in tasks)
    print("=" * len(header))
    print(f"Multi-Benchmark Results  ({len(results)} cells)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for tag in cells:
        row = f"{tag:<20}"
        for task in tasks:
            match = [r for r in results if r["tag"] == tag and r["task"] == task]
            if match:
                r = match[0]
                if r.get("score") is not None:
                    s = r["score"]
                    # format: print as ratio-style 4-dec, or raw as-is
                    row += f"{s:>16.4f}" if isinstance(s, (int, float)) else f"{s:>16}"
                else:
                    row += f"{'pending':>16}"
            else:
                row += f"{'—':>16}"
        print(row)
    print("-" * len(header))
    print("mme shown as: mme_percetion_score (perception raw sum). "
          "textvqa_val shown as: acc,none.")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_mmmu_table(results):
    """Print MMMU results as a formatted table."""
    if not results:
        print("No MMMU results found.")
        return

    # Sort: fp16 first, then by accuracy descending
    fp16_results = [r for r in results if r["method"] == "fp16"]
    other_results = [r for r in results if r["method"] != "fp16"]
    other_results.sort(key=lambda r: r.get("mmmu_acc") or 0, reverse=True)

    all_sorted = fp16_results + other_results

    # Header
    print(f"\n{'Experiment':<40} {'Method':>12} {'W-bit':>6} {'MMMU Acc':>10} {'Δ vs fp16':>11}")
    print("-" * 82)

    fp16_acc = fp16_results[0]["mmmu_acc"] if fp16_results else None

    for r in all_sorted:
        exp = r["experiment"][:39]
        method = r["method"]
        wb = str(r["w_bit"]) if r["w_bit"] else "?"
        acc = f"{r['mmmu_acc']:.5f}" if r["mmmu_acc"] is not None else "N/A"

        if fp16_acc is not None and r["mmmu_acc"] is not None and r["method"] != "fp16":
            delta = f"{(r['mmmu_acc'] - fp16_acc):+.5f}"
        else:
            delta = "-"

        # Highlight MLLMQ
        marker = " ← MLLMQ" if method == "mllmq" else ""
        print(f"{exp:<40} {method:>12} {wb:>6} {acc:>10} {delta:>11}{marker}")

    print("-" * 82)
    print(f"Total: {len(results)} experiments")

    # Highlight MBQ vs MLLMQ comparison
    mbq = [r for r in results if r["method"] == "mbq" and r["mmmu_acc"] is not None]
    mllmq = [r for r in results if r["method"] == "mllmq" and r["mmmu_acc"] is not None]
    if mbq and mllmq:
        mbq_acc = mbq[0]["mmmu_acc"]
        mllmq_acc = mllmq[0]["mmmu_acc"]
        delta = mllmq_acc - mbq_acc
        print(f"\n  *** MBQ vs MLLMQ: Δ = {delta:+.5f} ***")
        if delta > 0:
            print(f"  MLLMQ outperforms MBQ by {delta:.5f}")
        elif delta < 0:
            print(f"  MBQ outperforms MLLMQ by {-delta:.5f}")
        else:
            print(f"  No difference")


def print_comparison_table(comp_results):
    """Print comparison experiment results."""
    if not comp_results:
        return

    print(f"\n{'Experiment':<50} {'W-bit':>6} {'MMMU Acc':>10}")
    print("-" * 68)

    for r in sorted(comp_results, key=lambda x: x["experiment"]):
        exp = r["experiment"][:49]
        wb = str(r["w_bit"]) if r["w_bit"] else "?"
        acc = f"{r['mmmu_acc']:.5f}" if r["mmmu_acc"] is not None else "pending"
        print(f"{exp:<50} {wb:>6} {acc:>10}")


def print_csv(results):
    """Print results as CSV."""
    print("experiment,method,w_bit,mmmu_acc")
    for r in sorted(results, key=lambda x: (x.get("method", ""), x["experiment"])):
        print(f"{r['experiment']},{r.get('method','')},{r.get('w_bit','')},{r.get('mmmu_acc','')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Collect MLLMQ experiment results")
    parser.add_argument("--output_dir", type=str,
                        default="/nas3/kangjia/mllmq/output",
                        help="Root output directory for experiments")
    parser.add_argument("--csv", action="store_true",
                        help="Output as CSV")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated benchmark tasks to collect "
                             "(e.g. mme,textvqa_val,mmmu_val). If set, scans "
                             "output/<model>/eval/<task>/ structure.")
    args = parser.parse_args()

    # ── Multi-benchmark mode ──
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        multi = collect_multi_benchmark_results(args.output_dir, tasks)
        if args.csv:
            print("task,tag,method,w_bit,metric,score")
            for r in sorted(multi, key=lambda x: (x["task"], x["tag"])):
                print(f"{r['task']},{r['tag']},{r['method']},{r['w_bit']},{r['metric']},{r.get('score')}")
        else:
            print_multi_benchmark_summary(multi)
        return

    mmmu_results = collect_mmmu_results(args.output_dir)
    comp_results = collect_comparison_results(args.output_dir)

    # Merge: add method tag from comparison dirs if not in replicate
    for cr in comp_results:
        name = cr["experiment"]
        if "mllmq" in name.lower():
            cr["method"] = "mllmq"
        elif "mbq" in name.lower():
            cr["method"] = "mbq"
        elif "awq" in name.lower():
            cr["method"] = "awq"
        elif "rtn" in name.lower():
            cr["method"] = "rtn"

    # Merge comparison results into mmmu list for unified display
    all_results = mmmu_results + [
        r for r in comp_results
        if r.get("mmmu_acc") is not None
    ]

    if args.csv:
        print_csv(all_results)
    else:
        if all_results:
            print("=" * 82)
            print("MMMU Evaluation Results")
            print("=" * 82)
            print_mmmu_table(all_results)

        # Also show pending experiments
        pending = [r for r in comp_results if r.get("mmmu_acc") is None]
        if pending:
            print("\n" + "=" * 68)
            print("Experiments Pending Evaluation")
            print("=" * 68)
            print_comparison_table(pending)

        if not all_results and not pending:
            print(f"No results found in {args.output_dir}")


if __name__ == "__main__":
    main()