#!/usr/bin/env python3
"""Download preference-scorer model weights into checkpoints/ via hf-mirror.

Repos (HF mirror is reachable on this host; direct hf is not):
  1. HPSv2        xswu/HPSv2                          -> checkpoints/hpsv2/        (HPS_v2.1_compressed.pt)
  2. PickScore    yuvalkirstain/PickScore_v1          -> checkpoints/pickscore_v1/ (whole repo, small)
  3. PickScore CLIP processor laion/CLIP-ViT-H-14-laion2B-s32B-b79K
                                                    -> checkpoints/clip-vit-h-14-laion2b-s32b-b79k/  (skip ~7GB model weights)
  4. ImageReward  zai-org/ImageReward                 -> checkpoints/imagereward/ (ImageReward.pt + med_config.json)
  5. Bert tokenizer google-bert/bert-base-uncased     -> checkpoints/bert-base-uncased/ (tokenizer files only)

Usage:
  HF_ENDPOINT=https://hf-mirror.com python3 scripts/download_pref_weights.py
"""
import os
from huggingface_hub import snapshot_download

CP = "/nas3/kangjia/mllmq/checkpoints"
JOBS = [
    dict(repo="xswu/HPSv2", local=os.path.join(CP, "hpsv2"),
         allow=["HPS_v2.1_compressed.pt"]),
    dict(repo="yuvalkirstain/PickScore_v1", local=os.path.join(CP, "pickscore_v1"),
         allow=["config.json", "merges.txt", "model.safetensors",
                "preprocessor_config.json", "pytorch_model.bin",
                "special_tokens_map.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json", "README.md"]),
    dict(repo="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
         local=os.path.join(CP, "clip-vit-h-14-laion2b-s32b-b79k"),
         allow=["config.json", "merges.txt", "preprocessor_config.json",
                "special_tokens_map.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json", "README.md"]),
    dict(repo="zai-org/ImageReward", local=os.path.join(CP, "imagereward"),
         allow=["ImageReward.pt", "med_config.json"]),
    dict(repo="google-bert/bert-base-uncased",
         local=os.path.join(CP, "bert-base-uncased"),
         allow=["config.json", "special_tokens_map.json", "tokenizer_config.json",
                "vocab.txt"]),
]

def main():
    endpoint = os.environ.get("HF_ENDPOINT", "")
    print(f"[i] HF_ENDPOINT = {endpoint or '(default, may fail)'}")
    for j in JOBS:
        print(f"\n=== {j['repo']} -> {j['local']}")
        try:
            p = snapshot_download(j["repo"], local_dir=j["local"],
                                  allow_patterns=j["allow"])
            print(f"    OK -> {p}")
        except Exception as e:
            print(f"    FAIL: {e!r}")

if __name__ == "__main__":
    main()
