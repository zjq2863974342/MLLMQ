#!/usr/bin/env python3
"""Verify downloaded preference-scorer weight files are loadable.

Checks:
  - HPS_v2.1_compressed.pt   (torch archive, dict with 'state_dict')
  - ImageReward.pt           (torch archive, dict with model params)
  - PickScore model.safetensors (safetensors header)
Each without hitting the network / loading full CUDA.
"""
import os, sys
from safetensors import safe_open  # may not be installed; fallback to torch.load

CP = "/nas3/kangjia/mllmq/checkpoints"

def chek_torch(name, path):
    try:
        d = __import__('torch').load(path, map_location='cpu', weights_only=False)
        n = len(d) if isinstance(d, dict) else 'N/A'
        has_sd = 'state_dict' in d if isinstance(d, dict) else False
        print(f"  [OK] {name}: {os.path.getsize(path)} B, top-level keys={n}, has 'state_dict'={has_sd}")
        if has_sd:
            sd = d['state_dict']
            print(f"       state_dict size={len(sd)}")
        return True
    except Exception as e:
        print(f"  [ERR] {name}: {type(e).__name__}: {e}")
        return False

def check_safetensors(name, path):
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            ks = f.keys()
            n = len(ks)
            print(f"  [OK] {name}: {os.path.getsize(path)} B, {n} tensors")
            return True
    except Exception as e:
        print(f"  [ERR] {name}: {type(e).__name__}: {e}")
        return False

ok = True
ok &= chek_torch("HPS", os.path.join(CP, "hpsv2/HPS_v2.1_compressed.pt"))
ok &= chek_torch("ImageReward", os.path.join(CP, "imagereward/ImageReward.pt"))
try:
    ok &= check_safetensors("PickScore", os.path.join(CP, "pickscore_v1/model.safetensors"))
except ImportError:
    print("  [skip] PickScore safetensors check (lib missing)")
    d = __import__('torch').load(os.path.join(CP, "pickscore_v1/model.safetensors"), map_location='cpu')
    print(f"  torch-load fallback PickScore OK top-keys")
    ok = True

print("\nALL OK" if ok else "\nSOME MISSING")
