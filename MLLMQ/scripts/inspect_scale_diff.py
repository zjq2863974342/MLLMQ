#!/usr/bin/env python3
"""Compare two MBQ/MLLMQ scale files (CPU only).

L0 diagnostic for A13: if MLLMQ_T2I_RW_SCALE does not move any layer's chosen
scale, then changing the T2I weight cannot change generation quality.

Usage:
  python scripts/inspect_scale_diff.py <base.pt> <variant.pt> [--thr 1e-3]
"""
import argparse
import torch


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)["scale"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("variant")
    ap.add_argument("--thr", type=float, default=1e-3,
                    help="relative change threshold")
    a = ap.parse_args()

    A, B = load(a.base), load(a.variant)
    n = min(len(A), len(B))
    changed, diffs = [], []
    for i in range(n):
        name_a, _, ta = A[i]
        name_b, _, tb = B[i]
        if ta.shape != tb.shape:
            changed.append((name_a, float("inf"), float("inf")))
            continue
        ta = ta.float(); tb = tb.float()
        rel = (tb - ta).abs() / ta.abs().clamp_min(1e-8)
        mrel = float(rel.mean())
        mx = float(rel.max())
        diffs.append(mrel)
        if mrel > a.thr:
            changed.append((name_a, mrel, mx))

    print(f"layers compared: {n}")
    print(f"layers with mean relative change > {a.thr}: {len(changed)}")
    print(f"mean relative change across layers: {sum(diffs)/len(diffs):.6f}")
    if changed:
        changed.sort(key=lambda x: -x[1])
        print("top changed (prev_op, mean_rel, max_rel):")
        for nm, m, x in changed[:10]:
            print(f"  {nm}: mean={m:.5f} max={x:.5f}")
    else:
        print("=> scale selection essentially unchanged; weight knob has no lever.")


if __name__ == "__main__":
    main()