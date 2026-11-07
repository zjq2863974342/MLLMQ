#!/usr/bin/env python3
"""Extract DPG-Bench Overall score from official result files.

Usage:
  python scripts/extract_dpg_scores.py --input output/dpg_scores/awq_w3_single_results.txt

The official compute_dpg_bench.py writes per-image lines followed by a summary block.
We just need the final ``DPG-Bench score: <x>`` line.
"""
import argparse
import re
import sys


def parse_result(path):
    """Return overall score string or None if not found/missing."""
    try:
        with open(path, "r") as f:
            txt = f.read()
    except FileNotFoundError:
        return None
    m = re.search(r"DPG-Bench score:\s*([0-9.]+)", txt)
    if not m:
        return None
    return float(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    args = parser.parse_args()
    score = parse_result(args.input)
    if score is None:
        print("MISSING", file=sys.stdout)
        sys.exit(1)
    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
