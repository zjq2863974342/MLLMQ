#!/usr/bin/env bash
# ============================================================================
# 优先级最高：W3A16 + greedy(temperature=0) + 固定 seed=42 的生成侧补偿对照
#
#   目的：排除采样噪声后，判断“图像 token 补偿”对生图侧到底有没有效。
#   协议（机制判定口径，不是主榜）：
#     - GenEval **全量 553 prompt × 1 张**（greedy 下 4 张完全相同）
#     - W3A16, w_group=128, CFG=5.0, temperature=0（argmax）, seed=42
#     - **校准量统一 n=256**（完整 128 理解 + 128 生成，消掉“抽哪 128 条”的变量）
#     - 每 config 单卡跑完 553 张；评分用 evaluate_images.py
#
#   8 个 config（4 个外部基线 + 2 个单变量对照对，正好 4 卡两波）：
#     1) RTN                     rtn         无 scale
#     2) AWQ                     awq         mixed calib, n=256
#     3) MBQ-block               mbq_block   mixed calib, n=256
#     4) MBQ-scalar              mbq_scalar  mixed calib, n=256
#     5) MLLMQ 文字 token on      mllmq_mae   mixed calib, n=256, T2I on
#     6) MLLMQ 文字 token off     mllmq_mae   mixed calib, n=256, MLLMQ_DISABLE_T2I=1
#     7) MLLMQ 图像 token on      mllmq_mae   gen_tf calib, n=256, GEN_IMG_RW_SCALE=1
#     8) MLLMQ 图像 token off     mllmq_mae   gen_tf calib, n=256, MLLMQ_DISABLE_GEN_IMG=1
#
#   on/off 对只差一个开关；跨“文字 vs 图像”仍有校准集差异，不能直接说谁更强。
#
#   产出：output/janus-pro-7b/greedy_w3/<tag>/geneval/results.jsonl
#   日志：output/logs/greedy_comp_*.log
# Usage:
#   cd /nas3/kangjia/mllmq/MLLMQ && setsid nohup bash scripts/run_remote_greedy_compensation.sh \
#     > ../output/logs/greedy_comp_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd /nas3/kangjia/mllmq/MLLMQ

ROOT=/nas3/kangjia/mllmq
MODEL=$ROOT/checkpoints/Janus/models/Janus-Pro-7B
CALIB=$ROOT/data/coco/janus_mixed_gen_tf_256.json
IMG=$ROOT/data/coco/train2017
CACHE=$ROOT/MLLMQ/scale_cache
N=256
S_BLOCK=$CACHE/mbq/janus_mbq_block_w3a16_n256.pt
S_SCALAR=$CACHE/mbq/janus_mbq_scalar_w3a16_n256.pt
S_TEXT_ON=$CACHE/mllmq/janus_mae_w3a16_n256_fixed.pt
S_TEXT_OFF=$CACHE/mllmq/janus_mae_w3a16_n256_t2ioff.pt
S_IMG_ON=$CACHE/mllmq/janus_gen_tf_w3a16_n256_gen_tf_x1.pt
S_IMG_OFF=$CACHE/mllmq/janus_gen_tf_w3a16_n256_gen_tf_off.pt
S_AWQ=$CACHE/awq/janus_awq_w3a16_n256.pt
OUT=$ROOT/output/janus-pro-7b/greedy_w3
EV=$ROOT/data/benchmarks_gen/geneval/evaluation
LOG=$ROOT/output/logs
SEED=42

export HF_DATASETS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$EV/.torch_home" MODELSCOPE_CACHE="$EV/.ms_cache"
export PYTHONPATH="$ROOT/pydeps/geneval_eval${PYTHONPATH:+:$PYTHONPATH}"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log(){ echo -e "[$(date '+%F %T')] $1"; }
good(){ echo -e "[$(date '+%F %T')] ${GREEN}✅ $1${NC}"; }
fail(){ echo -e "[$(date '+%F %T')] ${RED}❌ $1${NC}"; }
info(){ echo -e "[$(date '+%F %T')] ${CYAN}📌 $1${NC}"; }
CLIP_BIN=$(find "$EV/.ms_cache" -name 'open_clip_pytorch_model.bin' 2>/dev/null | head -1)
score_of(){ python "$EV/summary_scores.py" "$1" 2>/dev/null | awk '/Overall/{print $NF}' | tail -1; }

mkdir -p "$OUT"

# ---------------------------------------------------------------- scales
gen_scale(){ # kind gpu
  local kind=$1 gpu=$2
  local sp calib method_env=() 
  case "$kind" in
    awq)      sp=$S_AWQ;    calib=$ROOT/data/coco/janus_mixed_calib_256.json ;;
    block)    sp=$S_BLOCK;  calib=$ROOT/data/coco/janus_mixed_calib_256.json ;;
    scalar)   sp=$S_SCALAR; calib=$ROOT/data/coco/janus_mixed_calib_256.json ;;
    text_on)  sp=$S_TEXT_ON;  calib=$ROOT/data/coco/janus_mixed_calib_256.json ;;
    text_off) sp=$S_TEXT_OFF; calib=$ROOT/data/coco/janus_mixed_calib_256.json; method_env=(MLLMQ_DISABLE_T2I=1) ;;
    img_on)   sp=$S_IMG_ON;   calib=$CALIB; method_env=(MLLMQ_GEN_IMG_RW_SCALE=1) ;;
    img_off)  sp=$S_IMG_OFF;  calib=$CALIB; method_env=(MLLMQ_DISABLE_GEN_IMG=1) ;;
    *) echo "unknown scale kind $kind"; return 1 ;;
  esac
  [ -s "$sp" ] && { good "SKIP scale ($kind)"; return 0; }
  info "[GPU$gpu] 生成 W3 scale（$kind, n=$N）"
  local extra=()
  case "$kind" in
    awq)    extra=(--method awq) ;;
    block)  extra=(--method mbq --loss_mode mae) ;;
    scalar) extra=(--method mbq --reweight --reweight_mode scalar --loss_mode mae) ;;
    *)      extra=(--method mllmq --loss_mode mae) ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu env "${method_env[@]}" python main_quant.py \
    --model_path "$MODEL" --device cuda:0 \
    --calib_data coco --data_path "$calib" --image_folder "$IMG" --n_samples "$N" \
    --w_bit 3 --a_bit 16 --w_group 128 --run_process "${extra[@]}" \
    --scale_path "$sp" > "$LOG/greedy_comp_scale_${kind}.log" 2>&1
  [ -s "$sp" ] && good "scale ($kind) OK" || fail "scale ($kind) FAIL"
}

# ---------------------------------------------------------------- eval
run_cfg(){ # tag method scale gpu
  local tag=$1 method=$2 sp=$3 gpu=$4
  local out="$OUT/$tag" gdir="$OUT/$tag/geneval"
  local sp_args=()
  if [ "$sp" != "none" ]; then
    [ -s "$sp" ] || { fail "$tag 缺 scale: $sp"; return 1; }
    sp_args=(--scale_path "$sp")
  fi
  if [ -f "$gdir/results.jsonl" ]; then good "SKIP $tag ($(score_of "$gdir/results.jsonl"))"; return 0; fi
  mkdir -p "$out"
  info "[GPU$gpu] $tag：553 prompt × 1（greedy, temp=0, seed=$SEED）"
  CUDA_VISIBLE_DEVICES=$gpu python eval_generate_quant.py \
    --model_path "$MODEL" --benchmark geneval --output_dir "$out" \
    --num_samples 1 --temperature 0 --seed "$SEED" \
    --quant_method "$method" --w_bit 3 "${sp_args[@]}" \
    --cfg_weight 5.0 > "$LOG/greedy_comp_gen_${tag}.log" 2>&1
  [ $? -ne 0 ] && { fail "$tag 生成失败"; return 1; }
  if [ -n "$CLIP_BIN" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python "$EV/evaluate_images.py" "$gdir" \
      --outfile "$gdir/results.jsonl" --clip-model-path "$CLIP_BIN" \
      > "$LOG/greedy_comp_score_${tag}.log" 2>&1
    [ -f "$gdir/results.jsonl" ] && good "$tag :: $(score_of "$gdir/results.jsonl")" \
      || fail "$tag 评分失败"
  else
    fail "$tag 缺 CLIP 权重，仅保留图像"
  fi
}

log "===== 0/3 生成 7 个 W3 n=256 scale（2 波；RTN 无需 scale）====="
gen_scale awq 0 & A=$!
gen_scale block 1 & B=$!
gen_scale scalar 2 & C=$!
gen_scale text_on 3 & D=$!
wait $A $B $C $D
gen_scale text_off 0 & E=$!
gen_scale img_on 1 & F=$!
gen_scale img_off 2 & G=$!
wait $E $F $G

log "===== 1/3 第一波（优先）：MLLMQ text on/off + img on/off ====="
run_cfg mllmq_img      mllmq_mae "$S_IMG_ON"   0 & E=$!
run_cfg mllmq_img_off  mllmq_mae "$S_IMG_OFF"  1 & F=$!
run_cfg mllmq_text     mllmq_mae "$S_TEXT_ON"  2 & G=$!
run_cfg mllmq_text_off mllmq_mae "$S_TEXT_OFF" 3 & H=$!
wait $E $F $G $H
log "第一波结束（关键对照已出）"

log "===== 2/3 第二波：RTN / AWQ / MBQ-block / MBQ-scalar ====="
run_cfg rtn        rtn        none        0 & A=$!
run_cfg awq        awq        "$S_AWQ"    1 & B=$!
run_cfg mbq_block  mbq_block  "$S_BLOCK"  2 & C=$!
run_cfg mbq_scalar mbq_scalar "$S_SCALAR" 3 & D=$!
wait $A $B $C $D
log "第二波结束"

log "===== 3/3 汇总（greedy 口径，只能组内比较）====="
for tag in rtn awq mbq_block mbq_scalar mllmq_text mllmq_text_off mllmq_img mllmq_img_off; do
  f="$OUT/$tag/geneval/results.jsonl"
  [ -f "$f" ] && good "$tag :: $(score_of "$f")" || fail "$tag ✗"
done
info "ALL DONE"