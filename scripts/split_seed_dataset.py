#!/usr/bin/env python3
"""
Split SEED-Bench-2 dataset into N chunks to avoid system RAM OOM.

The full dataset (24,371 samples) loads ~70 GB into RAM when decoded.
By splitting into 5 chunks (~4,874 samples each), each chunk consumes
~15 GB RAM + ~20 GB model = ~35 GB total, well within the 200 GB limit.

Usage:
    python split_seed_dataset.py --num_chunks 5
"""
import argparse, os, sys, numpy as np
from datasets import load_from_disk, DatasetDict

SRC = "/nas3/kangjia/mllmq/data/benchmarks_local/lmms-lab/SEED-Bench-2"
DST_BASE = "/nas3/kangjia/mllmq/data/benchmarks_local/lmms-lab"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_chunks", type=int, default=5)
    parser.add_argument("--src", default=SRC)
    parser.add_argument("--dst_base", default=DST_BASE)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffle (default: 42)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.src} ...")
    ds = load_from_disk(args.src)
    print(f"Splits: {list(ds.keys())}")

    for split_name, split_ds in ds.items():
        total = len(split_ds)
        chunk_size = (total + args.num_chunks - 1) // args.num_chunks
        print(f"\nSplit '{split_name}': {total} samples → {args.num_chunks} chunks of ~{chunk_size}")

        # Shuffle indices to balance chunk sizes (natural order groups
        # similar data_types together, causing massive imbalance: chunk0=1.4GB
        # vs chunk3=23GB on disk). Shuffling distributes images/videos evenly.
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(total)

        for i in range(args.num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            chunk_indices = indices[start:end]
            chunk = split_ds.select(chunk_indices)
            dst_path = os.path.join(args.dst_base, f"SEED-Bench-2-chunk{i}")
            chunk_ds = DatasetDict({split_name: chunk})
            chunk_ds.save_to_disk(dst_path)
            print(f"  Chunk {i}: {end-start} samples → {dst_path}")

    print("\nDone. Chunk dataset paths:")
    for i in range(args.num_chunks):
        print(f"  /nas3/kangjia/mllmq/data/benchmarks_local/lmms-lab/SEED-Bench-2-chunk{i}")


if __name__ == "__main__":
    main()