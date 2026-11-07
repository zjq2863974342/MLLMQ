# TOKNOW_patch.md — 迁移补充：无法随文件迁移的须知

> 版本：2026-11-07 21:45
> **迁移约束：本机除 `TOKNOW.md` 与 `TOKNOW_patch.md` 外的其他文件暂时无法迁移。**
> 因此本 patch 承载所有“如果丢掉本地文件就会失传”的信息：方法/配置、scale cache、校准数据、代码入口、环境、third_party、评分器 quirks、主机坑、论文叙事与写作红线、未完成实验。
> 结果数值表见 `TOKNOW.md §4`，本文件不重复；本文件负责解释“这些数是怎么来的、怎么读、哪些不能用”。

---

## P1. 论文主线、贡献与写作红线（最重要）

### P1.1 用户定稿的主线

- **论文主线 = MLLMQ 量化方法**，按叙事顺序集成三点：
  1. **三模态平衡**（vision / caption / t2i_text / gen_image 的梯度重加权）；
  2. **混合位宽**（原 GAMQ，生成感知的按层 W4/W3 分配；作为工具包/实际贡献）；
  3. **逐通道量化**（channel granularity）。
- 混合位宽：研究实验只需证明有效性；三模态平衡、逐通道量化：作为研究实验的重点分析点。
- **用户明确要求“三者都有效，且相对 MBQ 也有效”**。写作时必须同时保留证据边界（见 P1.2）。

### P1.2 组件-证据-声明边界表（必须写进正文/附录的诚实口径）

| 组件 | 对照 | 现有证据 | 可写强度 |
|---|---|---|---|
| 三模态 MLLMQ | vs RTN/AWQ（W3 生成） | GenEval600：学习型 .554–.584 vs RTN .450；DPG：学习型 82.1–82.7 vs RTN 80.34；T2I：RTN 8 类崩 | **强声称**：学习型三模态校准守住 W3 生成 |
| 三模态 MLLMQ | vs MBQ（用户定稿“也有效”） | DPG combined **+1.21pp CI[+0.12,+2.32]**（唯一 CI 不跨 0；单次、4 变体、缺 mbq_block 基线）；GenEval 干净口径 +1.35pp CI[−0.66,+3.41]；greedy 8-config CI 全跨 0 | **写“一致正趋势 / DPG 组合性改善”**，不写“全榜显著优于 MBQ” |
| 逐通道 | vs scalar / MBQ | channel 与 scalar 的 scale **118/120 张量完全相同**；MMMU +0.33pp p=.758 | 作为方法组件/实现细节；不单独声称显著增量 |
| 混合位宽 k12 | vs 均匀 W3/W4 | GenEval 全量三 seed **0.60796±0.00876**，vs W3 0.57119 **+3.68pp**，vs W4 0.63603 **−2.81pp**，达 W4 的 **96.6%**；DPG 81.992 vs W3 82.348（无收益）；T2I complex .358228 ≈ fp16 .3583 | **强声称 GenEval 有效性**；同段报告 DPG null |
| 机制 | 生成 vs 理解 | 生成 576 步 AR 误差累积；理解单次前向；RTN W3 只在生成崩 | 可写机制分析 |

### P1.3 写作红线

**可以写：**
- W3 学习型校准守住生成；W4/W8 与 fp16 不可分；RTN 仅在 3-bit 生成崩溃。
- 理解侧对 W3 鲁棒，方法间无显著排序。
- mixed-bit k12 在 GenEval 上显著优于均匀 W3，但 DPG 无收益。
- MLLMQ 相对 MBQ 在 DPG 组合性上呈一致正趋势。

**不要写：**
- ❌ “重加权提升理解侧。”——理解侧方法差 ≤2pp、CI 跨 0。
- ❌ “RTN 在全量上也不差。”——RTN 只是理解侧合格的 W3 基线，生成侧 W3 崩溃。
- ❌ “逐通道/生成 token 补偿在 MBQ 之上有显著增量。”——greedy 8-config CI 全跨 0；channel 与 scalar scale 几乎相同。
- ❌ “MLLMQ/GAMQ 在所有榜全面更优。”——DPG 无收益、T2I complex 持平。
- ❌ “A8 可用。”——W3A8 崩、W4A8 掉 14.8pp。

### P1.4 推荐 CVPR 8 页结构

1. Introduction：统一理解-生成 MLLM 的部署痛点；W3 生成侧崩溃、理解侧鲁棒的现象；三点贡献。
2. Related Work：VLM 量化（MBQ/Q-VLM/Bi-VLM/VLM-PTQ/MASQuant）、生成模型量化（Q-Diffusion 等）、统一模型（Janus/Show-o）。
3. Method：三模态 mask + 梯度重加权 scale；混合位宽选层；逐通道实现；统一形式化。
4. Experimental Setup：Janus-Pro-7B、benchmarks、协议、噪声。
5. Results：主表（理解/生成）、mixed-bit Pareto、逐 tag、消融。
6. Discussion：生成敏感/理解鲁棒机制；跨榜 trade-off。
7. Limitations：单模型、A8 不可用、pseudo-quant 无加速、T2I current 未定稿。
8. Conclusion。

---

## P2. 模型、量化方法与配置

### P2.1 模型

- **Janus-Pro-7B**：`checkpoints/Janus/models/Janus-Pro-7B`；三模态 token：
  - 理解：vision token + caption token；
  - 生成：T2I prompt 文字 token + 真实图像 token（gen_tf，576 VQ token）；
  - 生成是 **576 步逐 token 自回归**（这是“生成侧误差累积”的机制基础）。
- 另有 Janus-Pro-1B（`configs/janus1b/model.yaml` 记录 24 层、hidden 2048、16 heads、vocab 102400、576 image tokens），但**本论文主结果未使用**。
- Show-o / Show-o2 只做过 CPU/加载冒烟，**没有进入论文结果**。

### P2.2 PTQ 形式化

- 权重伪量化：`ŵ = s · q(round(w / s))`；`s` 来自 RTN / AWQ / MBQ scale search。
- 统一配置六元组：`Q = (B, G, A, C, P, D)`：
  - `B` bit map（均匀 W3/W4/W8，或逐 block 混合）；
  - `G` granularity（tensor/channel/group=128）；
  - `A` activation（A16/A8）；
  - `C` calibration（理解/生成文字/生成真实图像的混合比例、n、teacher forcing）；
  - `P` compensation（vision/caption/t2i_text/gen_image 的开关、scalar|channel、强度）；
  - `D` decode（CFG/temperature/seed，属于评测协议）。
- 固定超参：`w_group=128`、`zero_point=True`、仅量化 **LM 30 层**；视觉编码器/生成头不量化。

### P2.3 方法映射与实现

| 方法 | `--method` | 关键参数 | 说明 |
|---|---|---|---|
| RTN | `rtn` | `--w_bit 3|4`, `--a_bit 16` | 无 scale search；`qmllm/methods/rtn/entry.py` |
| AWQ | `awq` | `--w_bit`, `--scale_path` | 逐通道 scaling；`qmllm/methods/awq/entry.py` |
| MBQ-block | `mbq` | `--scale_path` | block-wise scale search；`qmllm/methods/mbq/entry.py` |
| MBQ-scalar | `mbq` + scalar reweight | 历史 tag `mbq_scalar` | 旧口径；主表用 `mbq_scalar` scale |
| MLLMQ | `mllmq` | `--reweight --reweight_mode channel|scalar --loss_mode mae|mse|mae_balanced` | MBQ + 三模态梯度重加权 |
| Mixed-bit | `mllmq_mixed`（eval 入口） | `scripts/build_mixed_scale.py --mode spread|top|late --score t2i|ratio --top_k k` | 合并 W4/W3 scale；原 GAMQ |

**关键实现位置：**
- `MLLMQ/main_quant.py`：量化入口；`--run_process` 触发 scale search，`--pack_quant` 打包 int32。
- `MLLMQ/qmllm/methods/mbq/quantize/pre_quant.py`：block 选择、模态 mask、梯度采集、重加权 scale 搜索；关键 group 匹配 `wo/o_proj` 与 `down_/w2`。
- `MLLMQ/qmllm/quantization/quant_wrapper.py` / `qlinear.py`：WALinear 与分组伪量化。
- `MLLMQ/scripts/build_mixed_scale.py`：混合位宽 scale 构建；输出 dict `{scale, w4_blocks, w3_bit, top_k}`。
- `MLLMQ/eval_generate_quant.py`：生成评测入口（GenEval/DPG/T2I/COCO）。
- `MLLMQ/eval_benchmarks.py`：理解评测入口（lmms-eval task）。
- `MLLMQ/eval_mmmu.py`：MMMU 专用入口（历史）。

**关键环境变量（MLLMQ 重加权开关）：**
| 变量 | 作用 |
|---|---|
| `MLLMQ_GEN_ONLY=1` | 只用生成样本建 scale（要求 gen_img_mask + t2i_prompt_mask） |
| `MLLMQ_DISABLE_GEN_IMG=1` | 关闭 gen_image（真实图像 token）补偿 |
| `MLLMQ_DISABLE_T2I=1` | 关闭 T2I prompt 文字 token 补偿 |
| `MLLMQ_MASK_BLOCK=1` | block 选择也走 masked 统计（公平对照） |
| `MLLMQ_GEN_IMG_RW_SCALE` | gen_image 重加权强度（默认 1.0） |
| `MLLMQ_T2I_RW_SCALE` | t2i_text 重加权强度（默认 1.0） |
| `MLLMQ_DEBUG_GEN_RW=1` | 打印生成侧重加权诊断 |

**历史命名坑：**
- 仓库里 tag 叫 `mllmq_mse` 的实验，实际 `loss_mode` 是 **`mae_balanced`**，不是 MSE；正文写 “MAE-balanced”。
- `MLLMQ/channel` 与 `MLLMQ/scalar` 的 scale 在 120 个张量里有 **118 个完全相同**；不要把“逐通道”当独立显著贡献。

### P2.4 Scale cache 映射（按实验选对文件）

| 用途 | W3A16 | W4A8 |
|---|---|---|
| 生成（GenEval/DPG/T2I） | `MLLMQ/scale_cache/unified_384/<method>_w3a16.pt` | `MLLMQ/scale_cache/w4a8/mix384/<method>_w4a8.pt` |
| 理解 scalar/channel | `unified_understanding_128/<method>_w3a16.pt` | `w4a8/und128/<method>_w4a8.pt` |
| T2I 变体（t2i_text/img/both） | `unified_384/<method>_w3a16.pt` | `w4a8/mix384/<method>_w4a8.pt` |
| gen_only | `unified_generation_only/mllmq_gen_only_w3a16.pt` | `w4a8/mix384/mllmq_gen_only_w4a8.pt` |
| AWQ 理解/SEED | `unified_understanding_128/awq_w3a16.pt` | `w4a8/und128/awq_w4a8.pt` |

- 文件名/方法：`mllmq_channel_only`、`mllmq_t2i_text`、`mllmq_t2i_img`、`mllmq_both`、`mbq_scalar`、`mbq_block_masked`、`awq`。
- 每个 `.pt` 旁有 `.timing.json`（model_load/data_prep/grad_collection/scale_search/scale_apply/total），可用于效率/复现。
- **旧 scale 不可混用**：`scale_cache/fixed_ablation/*` 与 `fixed_ablation_unified/*` 是旧版；当前主表只用 `unified_*` 与 `w4a8/*`。

### P2.5 校准数据

| 文件 | 用途 |
|---|---|
| `data/coco/janus_mixed_calib_256.json` | 历史 mixed（vision+caption+t2i text）；n=64/128/256 消融 |
| `data/coco/janus_understanding_128.json` | 理解侧校准（`unified_understanding_128/*`） |
| `data/coco/janus_gen_tf_128.json` | 教师强制生成校准（COCO caption → 576 VQ token；gen_tf） |
| `data/coco/janus_mixed_tri_256.json` / `janus_mixed_tri_384.json` | 三模态 mixed（384 用于生成侧） |
| `data/coco/janus_mixed_calib_balanced_256.json` / `genheavy` / `undheavy` | 模态配比消融 |
| `data/coco/sharegpt4v_coco_calib.json` | 旧 VLM 校准（InternVL2 用） |

- 生成侧 scale 默认 **n=128**；理解侧 **n=128**；`n=256` 仅用于复核/消融。
- 纯文本 calib 无 T2I 信号；生成必须含生成样本。

---

## P3. 环境与实验协议

### P3.1 环境（实测版本，不要照 README 的 cu128 猜）

- base：`/opt/conda/bin/python`，**torch 2.4.0+cu124**，**transformers 4.45.0**。
- `venvs/t2icomp`：torch 2.4.0+cu124，transformers 4.45.0（BLIP/CLIP 评分）。
- `venvs/t2idet`：torch 2.4.0+cu124，**无 transformers**（detectron2/UniDet 评分）。
- `venvs/showo`：torch 2.4.0+cu124，transformers 4.45.0（Show-o 未进论文）。
- 4×A100 80GB；SEED 严格 **1 进程/卡**（~56GB RSS）；DPG/生成 ≤ **2 进程/卡**；T2I 生图 2/卡历史上会慢到 ~68s/图，1/卡约 ~13.6s/图。

### P3.2 Benchmark 与指标

| 面向 | Benchmark | 规模/指标 |
|---|---|---|
| 理解 | MMMU | 900 题，accuracy |
| 理解 | MME | 2374 题；P/C 为子类原始分之和（P max 2000 / C max 800） |
| 理解 | TextVQA | 5000 题，exact_match |
| 理解 | MathVista | 1000 题，accuracy |
| 理解 | POPE | random/popular/adversarial 各 3000；R/P/A 用 F1/accuracy |
| 理解 | SEED-Bench-2 | 5 chunks，24371 样本；`seed_all` |
| 生成 | GenEval | full 553 prompts ×4 = 2212；matched 150 prompts ×4 = 600 |
| 生成 | DPG-Bench | 1065 prompts；实评 1064（`partiprompts97.png` 失败） |
| 生成 | T2I-CompBench++ | 8 类 ×300；每 prompt 1 张（num=1） |
| 生成 | COCO-val 5k | FID / CLIP / HPSv2 / PickScore / ImageReward（旧 19-config 表） |

### P3.3 协议

- 主表 GenEval：标准采样（temperature=1.0，无固定 seed）；机制/配对实验用 **greedy T=0, seed=42**。
- 统计：按 tag 分层、prompt 配对 bootstrap，20k 次；600 图噪声 3.83pp、全量 1.63pp。
- 生成 sharding：`generate_geneval_shard.py` 支持分片；DPG/T2I 也按 shard 跑。
- 评测入口：
  - 理解：`eval_benchmarks.py --model_path ... --quant_method ... --tasks ...`（lmms-eval task 在 `MLLMQ/tasks_local/`）。
  - 生成：`eval_generate_quant.py --benchmark {geneval,dpg_bench,t2i_compbench,coco} ...`。

---

## P4. third_party 清单（路径 / 大小 / 许可 / 是否需重下）

| 组件 | 路径 | 大小 | 许可 | 迁移建议 |
|---|---|---:|---|---|
| Janus-Pro-7B | `checkpoints/Janus/models/Janus-Pro-7B` | 22G | code MIT；model DeepSeek License | **重下**，不要打包 |
| Show-o | `checkpoints/Show-o` | 18G | Apache-2.0 | 未进论文；重下 |
| Show-o2-1.5B | `checkpoints/Show-o2-1.5B` | 5.3G | Apache-2.0（需核实） | 未进论文；重下 |
| T2I-CompBench repo | `data/benchmarks_gen/T2I-CompBench-main` | 1.3G | MIT | 需保留脚本/数据；可重下 |
| GenEval | `data/benchmarks_gen/geneval` | 8G | MIT | 需保留；可重下 |
| HPSv2 | `data/benchmarks_gen/HPSv2-master` | 11M | Apache-2.0 | 可重下 |
| ImageReward | `data/benchmarks_gen/ImageReward-main` | 6.8M | Apache-2.0 | 可重下 |
| PickScore | `data/benchmarks_gen/PickScore-main` | 229K | MIT | 可重下 |
| ELLA/DPG | `data/benchmarks_gen/ELLA-main` | 24M | 需核实 | DPG 评分依赖 |
| T2I scorer weights | `checkpoints/t2i_compbench/{blip,clip,unidet_experts}` | 1.8G | 各自上游许可 | 需重下/保留 |
| HPSv2 weights | `checkpoints/hpsv2` | 1.9G | Apache-2.0 | 需重下 |
| ImageReward weights | `checkpoints/imagereward` | 1.7G | Apache-2.0 | 需重下 |
| PickScore weights | `checkpoints/pickscore_v1` | 3.7G | MIT | 需重下 |
| lmms-eval | `third_party/lmms-eval` | 1,251 | 需核实 | 理解评测依赖（原 `MBQ/3rdparty/lmms-eval`，含本地 patch） |
| InternVL2 | `checkpoints/InternVL2-8B` | — | 需核实 | 只用于旧对照，非 Janus 主结果 |

- COCO 数据：`data/coco` 39G；`data/benchmarks_local` 204G（含 MMMU/SEED/TextVQA 等本地缓存）——**不要整包迁移**，按需重下。
- 评分器 quirks：
  - **GenEval**：实际评分器是 torchvision **Faster R-CNN**，不是官方 Mask2Former（仓库里虽有 Mask2Former 权重，但当前 `results.jsonl` 用的是 Faster R-CNN）→ 绝对值不可与 GenEval 论文对齐。
  - **T2I-CompBench**：官方 complex 用 10 张/prompt 且硬编码 `num=10`；本实验 **num=1**，只能内部横向比较。
  - **DPG**：`data/benchmarks_gen/ELLA-main/dpg_bench`；评分用 mplug 体系；`partiprompts97.png` 固定失败。
  - **MME**：`mme_percetion_score`（代码拼写就是 percetion）与 `mme_cognition_score` 是子类原始分之和。

---

## P5. 结果文件格式与读法（如果以后能拿到结果文件）

### P5.1 DPG
- 图片：`output/fixed_ablation_20261103_unified/<tag>/dpg_bench/*.png`（1065）。
- 分数：`output/fixed_ablation_20261103_unified/dpg_scores/<tag>.txt`：
  - 每行 `abs_png_path, score, score`；
  - **最后一行** `DPG-Bench score: <float>` 才是总分；
  - `_detail.txt` 是逐 VQA 明细。
- W4A8 对应 `output/janus-pro-7b/w4a8_supplement/dpg_scores/<tag>.txt`。

### P5.2 GenEval
- `results.jsonl` 每行一个图：`{"filename","tag","prompt","correct","reason","metadata","details"}`。
- 计分：每个 tag 的 4 张图先取均值；`overall` = 6 个 tag 均值。
- **matched 600**：用 `output/fixed_ablation_20261103/rtn/geneval/results.jsonl` 里的 150 个 `(tag,prompt)` 作为 prompt 集，过滤 unified 2212 结果后重算；W4A8 600 使用同一 prompt 集。

### P5.3 理解
- lmms-eval JSON：`{"results": {task: {"<metric>,none": value}}, "n-samples":..., "samples":...}`。
- 关键 key：`mme_percetion_score,none`、`mme_cognition_score,none`、`mmmu_acc,none`、`exact_match,none`、`mathvista_acc,none`、`pope_f1_score,none`。

### P5.4 SEED
- 每个 chunk 一个 `summary_<tag>_<width>_<ts>.json`，取 `seed_all,none`；
- 5 chunks 按 n_samples 加权（chunk0–3=4875，chunk4=4871，总 24371）。

### P5.5 T2I current
- 评分 JSON：`{"tag","device","rows":[{"category","scorer","n_images","rc","score","source"}],"skipped":[]}`。
- scorer：color/shape/texture=BLIP-VQA；non_spatial=CLIPScore；spatial=UniDet-2D；3d_spatial=UniDet+MiDaS；numeracy=UniDet；complex=3-in-1。
- **当前状态（2026-11-07 21:42）**：W3A16 8 tag 均 7/8 JSON，但 `non_spatial` 的 score 为 null、`complex` 缺失；retry v2 在等评分进程退出后重跑这两个类别。W4A8 生成中，未评分。

---

## P6. 主机专属坑（丢掉文件后最容易踩）

1. **DPG 不是 1065**：每个 tag 有 1065 PNG，但评分只算 1064；`partiprompts97.png` 固定失败。正文/表注写 1064。
2. **GenEval600 有两套**：主表是 matched 150 prompts；`score.json → w3a16_600` 是另一套 prompt 集（rtn .48000 等），**不可混用**。
3. **MME 不是百分比**：P/C 是子类原始分之和（P max 2000 / C max 800）。
4. **SEED 首 chunk 慢**：会生成 test split（cached），不是卡死；5 chunk 必须按 n_samples 加权。
5. **T2I legacy vs current**：旧 `output/janus-pro-7b/t2i_compbench/` 是 13 个旧 config；当前 8 config 在 `t2i_compbench_current/`，路径/tag 不匹配，旧评分脚本硬编码旧 tag，不能直接套。
6. **calib 版本混用**：旧 `fixed_ablation` 与当前 `unified_384`/`w4a8` 的 scale 不同；跨方法绝对值不可直接比（尤其旧 mbq/awq 用 n=256、mllmq 用 n=128）。
7. **`mllmq_mse` 实为 `mae_balanced`**；正文不要写 MSE。
8. **channel 与 scalar scale 118/120 相同**；不要把它当独立显著贡献。
9. **GradCacheHook T2I bug**：修复前 T2I 梯度从未采集；`A13/A14` 结果作废；修复版 W4 −3.33pp CI 跨 0。
10. **`score.json` 部分 section 是 2026-11-02 快照**；当前结果用 `fixed_ablation_20261103` section 或原始文件。
11. **pseudo-quant 无加速**：fp16 11.14 s/img、mae_w4 11.46、mae_w3 11.71、mbq_w4 12.22；显存 ~15.3GiB 持平。不要声称部署加速。
12. **A8 不可用**：W3A8 崩（.005/.0017），W4A8 掉 14.8pp；WALinear 慢 ~4×。

---

## P7. 未完成实验与当前状态

- **T2I-CompBench current**：
  - W3A16 8 tag 生成完成（8×2400）；评分 6/8 类有效，`non_spatial`/`complex` 待修复/重跑。
  - W4A8 8 tag 生成中；评分未开始。
  - 评分完成后才能填 `T2I-comp` 列；当前 `[进行中]`。
- **第二模型**：Janus-Pro-1B / Show-o 未进入结果；论文只能写单模型 limitation。
- **效率/打包量化**：`--pack_quant` 有实现，但没有系统的延迟/显存/吞吐基准；不要声称加速。
- **COCO 1k CLIP/IR/Pick**：可选，未跑。
- **score.json / dialog / TODO / paper_outline 同步**：T2I current 新分数尚未写入这些汇总。

---

## P8. 新机器 Claude 的最低必读顺序

1. `TOKNOW.md §4`：权威结果表与口径。
2. 本文件 `P1`：论文主线、组件-证据表、写作红线。
3. 本文件 `P2/P3`：方法、scale cache、校准、协议。
4. 本文件 `P6`：主机坑。
5. 写正文前，先核对 `P1.2` 的声明强度；任何超出该表的“有效性”表述都需要额外证据。

> 如果以后能把 `paper_outline.md` / `TODO.md` / `dialog.md` / 原始结果文件迁过去，再把它们作为二级参考；在此之前，以 `TOKNOW.md + TOKNOW_patch.md` 为准。