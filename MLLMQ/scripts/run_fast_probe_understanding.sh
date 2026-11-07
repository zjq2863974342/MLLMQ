#!/usr/bin/env bash
# ============================================================================
# 20-min understanding-side probe with the FIXED wrapper.
#   Question: does MBQ scalar reweighting still beat MBQ-block on MMMU?
#   Builds two W3 scales with n=64 (quick), evaluates MMMU 900 on two GPUs,
#   and runs a paired bootstrap on the 900 questions.
#
# Run on the remote after pausing RTN/AWQ. Does NOT touch other outputs.
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ
ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
IMG=$ROOT/data/coco/train2017
CALIB=$ROOT/data/coco/janus_mixed_calib_256.json
SC=$ROOT/MLLMQ/scale_cache/fast_probe
OUT=$ROOT/output/fast_probe_20261103
LOG=$ROOT/output/logs
mkdir -p "$SC" "$OUT" "$LOG"

grep -q "attention_mask.bool()" qmllm/models/janus/janus.py || { echo "GATE FAIL: janus.py not fixed"; exit 2; }
echo "[gate] fixed wrapper detected"

build(){ # name gpu extra...
  local name=$1 gpu=$2; shift 2
  local out=$SC/$name.pt
  [ -s "$out" ] && { echo "[SKIP] $name"; return 0; }
  echo "[BUILD] $name GPU$gpu n=64"
  CUDA_VISIBLE_DEVICES=$gpu python main_quant.py \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$CALIB" --image_folder "$IMG" --n_samples 64 \
    --w_bit 3 --a_bit 16 --w_group 128 --run_process \
    --scale_path "$out" "$@" > "$LOG/fast_probe_scale_${name}.log" 2>&1
  [ -s "$out" ] && echo "[OK] $name" || { echo "[FAIL] $name"; tail -6 "$LOG/fast_probe_scale_${name}.log"; return 1; }
}

# two scale builds in parallel
build mbq_block  0 --method mbq --loss_mode mae &
P1=$!
build mbq_scalar 1 --method mbq --reweight --reweight_mode scalar --loss_mode mae &
P2=$!
wait $P1 || exit 3
wait $P2 || exit 3

# MMMU 900, one config per GPU
CUDA_VISIBLE_DEVICES=2 python eval_benchmarks.py --model_type janus \
  --model_path "$MODEL" --method mbq --method_tag fast_mbq_block \
  --w_bit 3 --a_bit 16 --w_group 128 --scale_path "$SC/mbq_block.pt" \
  --tasks mmmu_val --output_base "$OUT/mbq_block" --device cuda:0 --log_samples \
  > "$LOG/fast_probe_mmmu_block.log" 2>&1 &
E1=$!
CUDA_VISIBLE_DEVICES=3 python eval_benchmarks.py --model_type janus \
  --model_path "$MODEL" --method mbq --method_tag fast_mbq_scalar \
  --w_bit 3 --a_bit 16 --w_group 128 --scale_path "$SC/mbq_scalar.pt" \
  --tasks mmmu_val --output_base "$OUT/mbq_scalar" --device cuda:0 --log_samples \
  > "$LOG/fast_probe_mmmu_scalar.log" 2>&1 &
E2=$!
wait $E1 || { echo "MMMU block failed"; tail -6 "$LOG/fast_probe_mmmu_block.log"; }
wait $E2 || { echo "MMMU scalar failed"; tail -6 "$LOG/fast_probe_mmmu_scalar.log"; }

python - <<'PY'
import json,glob,numpy as np,os
ROOT="/nas3/kangjia/mllmq"; OUT=os.path.join(ROOT,"output/fast_probe_20261103")
def load(tag):
    fs=sorted(glob.glob(os.path.join(OUT,tag,"**","*_results.json"),recursive=True))
    if not fs: return None
    s=json.load(open(fs[-1]))["samples"]["mmmu_val"]
    return {x["mmmu_acc"]["id"]: int(x["mmmu_acc"].get("parsed_pred")==x["mmmu_acc"].get("answer")) for x in s}
a=load("mbq_block"); b=load("mbq_scalar")
if a is None or b is None:
    print("MISSING MMMU RESULTS"); raise SystemExit(4)
ks=sorted(set(a)&set(b)); x=np.array([a[k] for k in ks]); y=np.array([b[k] for k in ks]); d=y-x
rng=np.random.default_rng(0); idx=rng.integers(0,len(d),size=(20000,len(d))); boots=d[idx].mean(axis=1)
lo,hi=np.percentile(boots,[2.5,97.5]); p=2*min((boots<=0).mean(),(boots>=0).mean())
print(f"MMMU paired n={len(ks)} block={x.mean():.5f} scalar={y.mean():.5f} delta(scalar-block)={100*d.mean():+.2f}pp CI[{100*lo:+.2f},{100*hi:+.2f}] p={p:.3f}")
json.dump({"n":len(ks),"block":float(x.mean()),"scalar":float(y.mean()),"delta_pp":float(100*d.mean()),"ci95_pp":[float(100*lo),float(100*hi)],"p":float(p)},open(os.path.join(OUT,"fast_probe_mmmu.json"),"w"))
PY
echo "FAST PROBE DONE: $OUT/fast_probe_mmmu.json"