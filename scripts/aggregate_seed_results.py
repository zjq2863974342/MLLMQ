#!/usr/bin/env python3
"""
Aggregate SEED-Bench-2 chunk results into a single combined results.json.

The chunked eval produces 5 separate results files (one per chunk). This
script sums the per-metric counts across chunks and recomputes the aggregate
scores (which are simple count-based ratios for SEED-Bench-2).

Usage:
    python aggregate_seed_results.py --input_dir output/internvl2-8b/eval/seedbench_2_local/<tag>/
"""
import argparse, glob, json, os, sys
from collections import defaultdict


EXPECTED_METRICS = [
    "seed_Video,none",
    "seed_Multiple_Images,none",
    "seed_Image_&_Text_Generation,none",
    "seed_Single_Image,none",
    "seed_Image_Generation,none",
    "seed_Interleaved_Image,none",
    "seed_all,none",
]


def aggregate_results(json_files):
    """
    Combine SEED-Bench-2 chunk results.

    SEED-Bench-2 metrics are simple accuracy counts (correct / total).
    We sum the numerator and denominator across chunks.
    """
    combined_results = {}
    combined_samples = {}

    agg = defaultdict(lambda: {"correct": 0, "total": 0})

    for fpath in sorted(json_files):
        with open(fpath) as f:
            data = json.load(f)

        results = data.get("results", {})
        samples = data.get("samples", {})

        for task_name, task_results in results.items():
            for metric_name, value in task_results.items():
                if isinstance(value, (int, float)):
                    # For SEED, a single accuracy score → average
                    if metric_name not in agg:
                        agg[metric_name] = {"sum": 0.0, "count": 0}
                    agg[metric_name]["sum"] += value
                    agg[metric_name]["count"] += 1

        # Merge samples (use task name as key prefix to avoid collision)
        for task_name, task_samples in samples.items():
            if task_name not in combined_samples:
                combined_samples[task_name] = []
            combined_samples[task_name].extend(task_samples)

    # Build combined results
    combined_results = {}
    for task_name in set(
        k for fpath in json_files
        for k in json.load(open(fpath)).get("results", {}).keys()
    ):
        # Use the first chunk's task name (strip chunk suffix for display)
        base_task = task_name
        for i in range(5):
            base_task = base_task.replace(f"_chunk{i}", "")
        combined_results[base_task] = {}

        for metric_name, stats in agg.items():
            clean_metric = metric_name
            for i in range(5):
                clean_metric = clean_metric.replace(f"_chunk{i}", "")
            if stats["count"] > 0:
                combined_results[base_task][clean_metric] = stats["sum"] / stats["count"]

    return {
        "results": combined_results,
        "samples": combined_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing chunk result JSON files")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <input_dir>/aggregated_results.json)")
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*_results.json")))
    if not json_files:
        print(f"ERROR: No *_results.json files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} result files:")
    for f in json_files:
        print(f"  {os.path.basename(f)}")

    combined = aggregate_results(json_files)

    out_path = args.output or os.path.join(args.input_dir, "aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nAggregated results saved to: {out_path}")
    print(f"\nAggregate scores:")
    for task_name, metrics in combined["results"].items():
        for metric_name, value in metrics.items():
            print(f"  {task_name} / {metric_name} = {value:.6f}")


if __name__ == "__main__":
    main()