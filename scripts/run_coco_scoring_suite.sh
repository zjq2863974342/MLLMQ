#!/bin/bash
# COCO-val 评分套件：FID + CLIP Score + HPSv2/PickScore（W4 7 config + W8 + 已完工的 W3）
# 2026-10-16 新建。**默认单卡串行** —— 理由见下"纪律"。
#
# 门禁与口径（每一条都是这两天踩出来的，勿删）:
#  ① 生成必须全停（纪律 #1「生成在跑时禁启新 CUDA 进程」）→ pgrep 到 worker 就 exit 3
#  ② 所有 config 一律 **只用 `*_0000.png`**：fp16 目录含 236 张 `_0001+` 老协议残图，
#     不过滤会让 79 个 prompt 在均值里被算 4 次 → baseline 偏（FID 用软链农场，偏好分用 --only_sample，CLIP 用 --only_sample）
#  ③ FID 的 real = `data/coco/val2017_real`（**不是** `data/coco/val2017`，那是个指向 train2017 的软链，
#     5000 个 val image_id 一张都没有）。real 的 Inception stats **只算一次**存 npz，后续 --load-stats 复用
#  ④ FID 绝对值不可与外部论文对齐：num_samples=1 × 5000 张 ≠ 标准 COCO FID(30k 生成图) → 只横向比同一套口径
#  ⑤ 单卡串行、不开双卡：对端 A100 还在生成 W3，共享面是 NAS IO；评分是大读负载，
#     单卡把 IO 压力减半（实测 4 worker 并发写时 W4 只 +1.4%，但读 5000×N 张是另一种压力）。
#     若对端速率掉 >20% 就停下（项目既定判据）
#  ⑥ ImageReward **不在本套件里**：它需要 pip install diffusers（high-risk，脚本自己写的），单独放最后
#
# Usage: setsid nohup bash scripts/run_coco_scoring_suite.sh \
#          > output/logs/score_suite_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

set -o pipefail
cd /nas3/kangjia/mllmq
ROOT=/nas3/kangjia/mllmq
OUT=$ROOT/output/janus-pro-7b
REAL=$ROOT/data/coco/val2017_real
FARM=$OUT/coco_val_farm0000
STATS=$OUT/fid_stats
FIDLOG=$OUT/fid_results
CLIPLOG=$OUT/clip_scores
PREFLOG=$OUT/pref_scores
GPU=${GPU:-0}
FID_WORKERS=${FID_WORKERS:-4}   # 单线程 num_workers=0 实测只 6 img/s(每侧 ~14min)；两侧已统一 384² → 开 worker 安全
export CUDA_VISIBLE_DEVICES=$GPU
# 🔴🔴 **必须限 OMP 线程**（生成侧脚本早就这么做，评分侧漏了 → 慢 11.6×）：
#    实测同一批 300 张 HPSv2：OMP_NUM_THREADS=128 → 93s(3.2 img/s)；=4 → **8s(37.5 img/s)**。
#    本机 nproc=128，torch 默认起 128 线程，而 CLIP/HPSv2 的小张量逐张算 → 线程 thrashing 主导。
#    配 --batch 16 批量前向后才暴露出来（逐张时前向本来就慢，看不出 OMP 的账）。
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # 结果 json 里有中文：不设则 open() 默认 ascii → UnicodeEncodeError
# 🔴 FID 的 pt_inception 权重必须走 NAS 缓存：python 的 urllib **不认 CURL_CA_BUNDLE**
#    → pytorch_fid 自己去 github release 下权重会 SSL CERTIFICATE_VERIFY_FAILED；
#    就算手动 curl 也常被截断（实测 13.6MB / 26.8MB，且签名 URL 不支持 Range → 不能续传）。
#    可用源：`https://ghfast.top/https://github.com/...`（12s 下完 95,628,359 B）。
export TORCH_HOME=$ROOT/checkpoints/torch_home
[ -f "$TORCH_HOME/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth" ] \
  && echo "[$(date +%H:%M:%S)] ✅ TORCH_HOME=$TORCH_HOME（pt_inception 已在 NAS 缓存）" \
  || echo "[$(date +%H:%M:%S)] ⚠️ TORCH_HOME 里没有 pt_inception，FID 会尝试联网（大概率失败）"

GOOD=$'\033[0;32m'; BAD=$'\033[0;31m'; CYAN=$'\033[0;36m'; NC=$'\033[0m'
log()  { echo "[$(date +%H:%M:%S)] $*"; }
good() { echo "[$(date +%H:%M:%S)] ${GOOD}✅ $*${NC}"; }
fail() { echo "[$(date +%H:%M:%S)] ${BAD}❌ $*${NC}"; }
info() { echo "[$(date +%H:%M:%S)] ${CYAN}📌 $*${NC}"; }

# ── 要评分的 config（默认 W4 全 7 + W8；已完工的 W3 两个可用 --w3 加进来）──
CONFIGS=(fp16 rtn_w4 awq_w4 mbq_block_w4 mbq_scalar_w4 mllmq_mae_w4 mllmq_mse_w4 \
         mllmq_mae_w8 mbq_block_w8 awq_w8 mbq_scalar_w8 mllmq_mse_w8 gamt_k12)
# ↑ 10-17 03:30 补齐 4 个新 W8 tag（BUG-C：原来只有 mllmq_mae_w8，导致 ONLY= 里写 mbq_block_w8 等
#   会被"交集"静默丢掉 → 套件跑 0 个 config 还打印"全部完成"。与 10-16 22:10 的 BUG-A 同类。
#   rtn_w8 按用户指示不跑，故不在列。）
EXTRA_W3=0
for a in "$@"; do [ "$a" = "--w3" ] && EXTRA_W3=1; done
if [ $EXTRA_W3 -eq 1 ]; then
  # W3 全 6 config（10-16 21:50 扩满：原来只有已完工的 mae_w3/mbq_block_w3，而 ONLY= 是与 CONFIGS 求交集，
  # 导致 rtn_w3/awq_w3/mbq_scalar_w3/mse_w3 根本选不上）。尚未到 5000 的会被下面的完整性校验自动剔除，
  # 不会拿半套数据出指标。
  CONFIGS+=(mllmq_mae_w3 mbq_block_w3 rtn_w3 awq_w3 mbq_scalar_w3 mllmq_mse_w3)
fi

# ONLY="tag1 tag2 ..." → 只评这些（用于把 10 个 config 拆到两张卡上并行；
# 不同 config 写不同 json，real 的 npz 是只读共享 → 双实例不冲突）
ONLY=${ONLY:-}
_ALL_VISIBLE=("${CONFIGS[@]}")   # 保留过滤前的完整可见表：护栏报错时要印这个，印"已被交集清空的数组"没有诊断价值
if [ -n "$ONLY" ]; then
  _kept=()
  for c in "${CONFIGS[@]}"; do
    for w in $ONLY; do [ "$c" = "$w" ] && _kept+=("$c"); done
  done
  CONFIGS=("${_kept[@]}")
fi
# 🔴 护栏（BUG-A / BUG-C 同类根因：ONLY 与内置表求交集时会**静默丢弃**认不出的 tag，
#    最后跑 0 个 config 却打印"全部完成" —— 与 10-13 那次 `0/19 complete` 谎报同形）
if [ -n "$ONLY" ]; then
  for w in $ONLY; do
    _hit=0
    for c in "${CONFIGS[@]}"; do [ "$w" = "$c" ] && _hit=1 && break; done
    if [ $_hit -eq 0 ]; then
      echo "[$(date +%H:%M:%S)] ❌ ONLY 里的 '$w' 不在本套件可见的 config 表内 → 退出(exit 10)，不静默丢弃"
      echo "[$(date +%H:%M:%S)]    本套件可见的全部 tag: ${_ALL_VISIBLE[*]}"
      exit 10
    fi
  done
fi
[ ${#CONFIGS[@]} -eq 0 ] && { echo "[$(date +%H:%M:%S)] ❌ ONLY 过滤后为空"; exit 8; }

# ── PLAN 预演模式（PLAN=1）：放在门禁①**之前** → 生成在跑时也能预演 ──
#    动机：10-17 03:30 的 BUG-C/BUG-D（ONLY 与内置表求交集会静默丢 tag、两条 ONLY 并集漏了
#    mbq_scalar_w4）都只能靠人工核对命令才能发现。PLAN 模式把"将要评哪些、哪些会被剔、哪几列缺"
#    直接摊出来：抄命令或交接前先跑一次。**只看不动，不建 CUDA 上下文。**
if [ -n "$PLAN" ]; then
  info "══ PLAN 模式（只看不动）· ONLY='${ONLY:-<未设=全评>}' · --w3=$EXTRA_W3 ══"
  printf "  %-16s %7s %6s %6s %6s %6s  %s\n" config uniq FID CLIP pref IR 结论
  _todo=0
  for t in "${CONFIGS[@]}"; do
    d=$OUT/coco_val/$t/coco_val
    u=$(find "$d" -maxdepth 1 -name '*_0000.png' -printf '%f\n' 2>/dev/null | cut -d_ -f1 | sort -u | wc -l)
    leg=$(find "$d" -maxdepth 1 -name '*_000[1-9].png' 2>/dev/null | wc -l)
    f=$([ -f "$FIDLOG/${t}_fid.json" ] && echo 有 || echo 要跑)
    c=$([ -f "$CLIPLOG/${t}_clip.json" ] && echo 有 || echo 要跑)
    if [ -f "$PREFLOG/$t/hpsv2_scores.json" ] && [ -f "$PREFLOG/$t/pickscore_scores.json" ]; then p=有; else p=要跑; fi
    i=$([ -f "$PREFLOG/$t/imagereward_scores.json" ] && echo 有 || echo 要跑)
    # 🔴 判据必须与真门禁**完全一致**（真门禁只看 uniq==5000；legacy 只作提示不作剔除理由）。
    #    BUG-E(10-17 04:30)：第一版把 legacy>0 也当剔除条件 → 把合法的 fp16 误判成要剔除
    #    （它的 236 张 _0001+ 正是评分时靠 *_0000 过滤掉的，真门禁放行、今晚也评成功了）。
    #    真正危险的老协议残档（如 mae_w8 那 322 张/仅 81 uniq）本来就会被 uniq!=5000 抓住。
    # BUG-F(10-17 04:35)：这里原写 `local _note=""`，但 PLAN 块**不在函数里** →
    #   bash 报 "local: can only be used in a function"（stderr，容易被忽略），_note 从不被重置
    #   → fp16 的"含 236 张 legacy"备注**泄漏**给后面所有 config（W3 实际 legacy=0）。修成普通赋值。
    _note=""
    [ "$leg" -ne 0 ] && _note="（含 $leg 张 legacy，评分时按 *_0000 过滤，非剔除理由）"
    if [ "$u" -ne 5000 ]; then v="🔴剔除(uniq=$u/5000)"
    elif [ "$f$c$p$i" = "有有有有" ]; then v="✅已齐→跳过"
    else v="🟡要跑$_note"; _todo=$((_todo+1)); fi
    printf "  %-16s %7s %6s %6s %6s %6s  %s\n" "$t" "$u" "$f" "$c" "$p" "$i" "$v"
  done
  info "PLAN: 候选 ${#CONFIGS[@]} 个，其中 $_todo 个有活要干（其余已齐或被剔除）"
  info "→ 去掉 PLAN=1 再真跑（真跑时门禁①会拦生成进程，纪律 #1）"
  exit 0
fi

# ── 门禁 ①：生成全停 ──
if pgrep -f '[e]val_generate_quant|[m]ain_quant|[e]val_benchmarks|[e]valuate_images' >/dev/null 2>&1; then
  fail "检测到生成/量化进程在跑 → 拒绝启动评分（纪律 #1）"
  pgrep -af '[e]val_generate_quant|[m]ain_quant|[e]val_benchmarks' | head -3
  exit 3
fi
good "门禁①通过：无生成进程在跑"

# ── 门禁 ②/③：数据齐备性 ──
nreal=$(ls "$REAL"/*.jpg 2>/dev/null | wc -l)
if [ "$nreal" -ne 5000 ]; then fail "FID real 集 $REAL 只有 $nreal 张（应 5000）"; exit 4; fi
good "门禁③通过：FID real = val2017_real $nreal 张"

nfarm=$(ls "$FARM"/fp16/*.png 2>/dev/null | wc -l)
if [ "$nfarm" -ne 5000 ]; then
  info "fp16 软链农场不完整($nfarm) → 重建"
  rm -rf "$FARM/fp16"; mkdir -p "$FARM/fp16"
  for f in $PWD/output/janus-pro-7b/coco_val/fp16/coco_val/*_0000.png; do ln -sf "$f" "$FARM/fp16/"; done
  nfarm=$(ls "$FARM"/fp16/*.png 2>/dev/null | wc -l)
fi
[ "$nfarm" -eq 5000 ] || { fail "fp16 农场重建后仍 $nfarm != 5000"; exit 4; }
good "门禁②通过：fp16 过滤后农场 = $nfarm 张（原目录 5236 含 236 张残图）"

# ── 每个 config 的完整性硬校验：必须 5000 个唯一 prompt（半套数据不评）──
#    (10-16 21:55 重写：原来用 CONFIGS=("${CONFIGS[@]/$t}") 做"删除"，那是**逐元素子串替换**，
#     会把被剔除项变成空串留在数组里 → ${#CONFIGS[@]} 虚高、依赖后面的 [ -z ] 兜底。改成重建数组。
#     同时把 `ls+xargs basename`(5000 次 fork) 换成 find -printf，18 个 config 的体检从分钟级降到秒级。)
_kept=()
for t in "${CONFIGS[@]}"; do
  [ -z "$t" ] && continue
  d=$OUT/coco_val/$t/coco_val
  u=$(find "$d" -maxdepth 1 -name '*_0000.png' -printf '%f\n' 2>/dev/null | cut -d_ -f1 | sort -u | wc -l)
  if [ "$u" -ne 5000 ]; then
    fail "剔除 $t — 唯一 prompt 只有 $u/5000（未跑完或含残档），不评半套数据"
  else
    _kept+=("$t")
  fi
done
CONFIGS=("${_kept[@]}")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  fail "剔除后没有可评的 config → 退出（不是失败，只是都还没到 5000）"; exit 9
fi
info "最终待评 ${#CONFIGS[@]} 个: ${CONFIGS[*]}"

mkdir -p "$STATS" "$FIDLOG" "$CLIPLOG" "$PREFLOG"

# ── FID 的 real stats 由 score_fid_coco.py 自己缓存（real_val2017_384.npz），这里不再预算 ──
# 原因：`python -m pytorch_fid` 把 transform 硬编码成 ToTensor()，遇到变长宽的 COCO 真图
#       (426×640 / 640×586 …) 会在 torch.stack 上炸；worker>0 时表现为难懂的
#       "Trying to resize storage that is not resizable"。→ 一律走 scripts/score_fid_coco.py
[ -f "$STATS/real_val2017_384.npz" ] && good "real stats 已有缓存 $STATS/real_val2017_384.npz（10 config 共用）" \
  || info "real stats 尚无缓存 → 第一个 config 会多花 ~14min 建，之后复用"

# ── 逐 config 评分 ──
T0=$(date +%s)
for t in "${CONFIGS[@]}"; do
  [ -z "$t" ] && continue
  d=$OUT/coco_val/$t/coco_val
  fd=$d; [ "$t" = "fp16" ] && fd=$FARM/fp16
  echo ""
  info "=========== $t ==========="

  # FID（走自己的驱动：两侧统一 Resize(384)+CenterCrop、内部按 _0000 过滤、real stats 自动缓存）
  if [ -f "$FIDLOG/${t}_fid.json" ]; then
    good "FID 已有结果，跳过（$(python3 -c "import json;print(round(json.load(open('$FIDLOG/${t}_fid.json'))['fid'],3))" 2>/dev/null)）"
  else
    log "  [1/3] FID (real=val2017_real, fake=$d, only_sample=0000)"
    if TORCH_HOME=$ROOT/checkpoints/torch_home python3 scripts/score_fid_coco.py --config "$t" \
          --only_sample 0000 --num_workers "$FID_WORKERS" > "$FIDLOG/${t}_fid.log" 2>&1; then
      good "  FID $t = $(python3 -c "import json;print(round(json.load(open('$FIDLOG/${t}_fid.json'))['fid'],4))" 2>/dev/null)"
    else
      fail "  FID 失败（见 $FIDLOG/${t}_fid.log）"; tr '\r' '\n' < "$FIDLOG/${t}_fid.log" | tail -5
    fi
  fi

  # CLIP Score
  if [ -f "$CLIPLOG/${t}_clip.json" ]; then
    good "CLIP 已有结果，跳过（$(python3 -c "import json;print(round(json.load(open('$CLIPLOG/${t}_clip.json'))['clip_score_mean'],4))" 2>/dev/null)）"
  else
    log "  [2/3] CLIP Score (openai ViT-L/14, only_sample=0000)"
    if python3 scripts/score_clip_coco.py --images_dir "$d" --config "$t" --only_sample 0000 \
          --output_dir "$CLIPLOG" > "$CLIPLOG/${t}_run.log" 2>&1; then
      good "  CLIP $t = $(python3 -c "import json;print(round(json.load(open('$CLIPLOG/${t}_clip.json'))['clip_score_mean'],4))" 2>/dev/null)"
    else
      fail "  CLIP 失败（见 $CLIPLOG/${t}_run.log）"; tr '\r' '\n' < "$CLIPLOG/${t}_run.log" | tail -5
    fi
  fi

  # HPSv2 + PickScore
  if [ -f "$PREFLOG/$t/hpsv2_scores.json" ] && [ -f "$PREFLOG/$t/pickscore_scores.json" ]; then
    good "pref 已有结果，跳过"
  else
    log "  [3/3] HPSv2 + PickScore (--only_sample 0000)"
    if python3 MLLMQ/score_preferences.py --images_dir "$d" --config "$t" --only_sample 0000 \
          --models hpsv2,pickscore --output_dir "$PREFLOG" > "$PREFLOG/${t}_run.log" 2>&1; then
      good "  pref $t 完成$(python3 scripts/summarize_scores.py --configs all 2>/dev/null | grep "\`$t\`" | head -1 | sed 's/^/  →/')"
    else
      fail "  pref 失败（见 $PREFLOG/${t}_run.log）"; tr '\r' '\n' < "$PREFLOG/${t}_run.log" | tail -5
    fi
  fi

  # ImageReward —— 可选步骤，只在 `clip` 已装时跑。
  # 🔴 绝不在本套件里 pip install：`image-reward` 会把 huggingface_hub 顶到 1.33.0、transformers 顶到
  #    5.20.0（实测 pip --dry-run），等于打死整个 Janus 栈。安全装法只有
  #    `pip install --no-deps openai-clip`（脚本层已用假包对象绕过 __init__ → 不需要 diffusers）。
  if [ -f "$PREFLOG/$t/imagereward_scores.json" ]; then
    good "ImageReward 已有结果，跳过"
  elif python3 -c "import clip" >/dev/null 2>&1 || [ -f "$ROOT/checkpoints/pywheels/openai_clip_lib/clip/clip.py" ]; then
    # 第二个条件才是常态：`clip`/`fairscale` **不装在 site-packages**，
    # 而是由 score_imagereward_coco.py 从 NAS 的 checkpoints/pywheels/*_lib/ 注入 sys.path（零 pip）
    log "  [4/4] ImageReward（离线本地权重，绕过 __init__ → 不碰 diffusers）"
    if python3 scripts/score_imagereward_coco.py --config "$t" --only_sample 0000 \
          > "$PREFLOG/${t}_imagereward_run.log" 2>&1; then
      good "  ImageReward $t 完成（$(python3 -c "
import json,statistics
d=json.load(open('$PREFLOG/$t/imagereward_scores.json',encoding='utf-8'))
v=[x for x in d.values() if isinstance(x,(int,float))]
print(f'n={len(v)} mean={statistics.fmean(v):.4f}' if v else 'n=0')" 2>/dev/null)）"
    else
      fail "  ImageReward 失败（见 $PREFLOG/${t}_imagereward_run.log）"
      tr '\r' '\n' < "$PREFLOG/${t}_imagereward_run.log" | tail -5
    fi
  else
    warn "ImageReward 跳过：缺 python 包 \`clip\`（安全装法: pip install --no-deps openai-clip；🔴 切勿 pip install image-reward / diffusers）"
  fi
done

EL=$(( ($(date +%s) - T0) / 60 ))
echo ""
echo "================ 汇总 ================"
python3 scripts/summarize_scores.py --configs all --md "$OUT/coco_val_scores.md"
echo "======================================"
good "全部完成，耗时 ${EL} min（单卡 GPU$GPU 串行）"
log "ImageReward 未跑：需 pip install diffusers（high-risk），单独最后再做"
