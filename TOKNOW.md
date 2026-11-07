# TOKNOW.md — MLLMQ 项目迁移知识库（CVPR 论文写作）

> 版本：2026-11-07 21:30（第 1 块：结果数据）
> 读者：另一台机器上写 **CVPR 论文**的 Claude。
> 用途：只记录**网上搜不到、只有本机能探查到**的信息：权威结果、口径、路径、配置、third_party、机器坑。
> 标记约定：`[权威]` 当前结论可用；`[过时]` 不可作为当前结论；`[进行中]` 未定稿；`[禁写]` 不得写进论文。
> 数据权威层级：**原始结果文件 > `output/score.json` > `dialog.md`/`TODO.md` > `paper_outline.md`**。`score.json` 的 `understanding`/`generation`/`priority_experiments_2026_11` 仍是 2026-11-02 快照；`fixed_ablation_20261103` 是 2026-11-07 新补的当前口径。

---

## 4. 权威结果数据

### 4.0 先读：当前结论与口径

- **当前主结果**是 `output/fixed_ablation_20261103_unified`（W3A16）和 `output/janus-pro-7b/w4a8_supplement`（W4A8）。
- **旧 `fixed_ablation_20261103` 的 DPG 分数作废**（用的是旧 `scale_cache/fixed_ablation/*`）；旧 GenEval 600 只用于定义 matched 150 prompts，不是主表。
- **DPG 实评 1064 张**：每个 tag 目录有 1065 PNG，但 `partiprompts97.png` 是散落文件，评分器在**所有 tag** 中一致跳过。
- **GenEval600 是 matched 150-prompt 子集**：W3A16 的 matched 值 = unified 2212 结果按 `fixed_ablation_20261103` 的 150 prompts 过滤后重算；W4A8 600 使用完全相同的 150 prompts。另一套 `fixed_ablation_20261103` 的 600 分数是不同 prompt 集，**不可混用**。
- **GenEval 评分器是 torchvision Faster R-CNN**，不是官方 Mask2Former → 绝对值不可与 GenEval 论文直接对齐，只做内部比较。
- **MME-P/MME-C 是子类原始分之和**（P max 2000 / C max 800），不是百分比。
- **SEED 5 chunks 按 n_samples 加权合并**：chunk0–3 = 4875，chunk4 = 4871，总 24371。
- **噪声底线**：同配置全量 GenEval 重跑 **1.63pp**；600 图子集 **3.83pp**。低于这个量级的跨方法差异不解读。
- **T2I current 尚未定稿**：W3A16 生成完成，评分 6/8 类有效；W4A8 生成中、未评分。`[进行中]`。

### 4.1 理解侧

#### 4.1.1 W3A16（7 bench，`[权威]`）

数据源：`output/fixed_ablation_20261103_understanding_5h/<tag>/...`；聚合见 `score.json → fixed_ablation_20261103.understanding.w3a16_5h_7bench`。

| config | MMMU | MME-P | MME-C | TextVQA | MathVista | POPE-R | POPE-P | POPE-A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtn | .35000 | 1409.4 | 245.0 | .48814 | 33.5 | .8677 | .8507 | .8347 |
| mbq_scalar | .33778 | 1400.0 | 270.7 | .49448 | 28.3 | .8383 | .8203 | .8230 |
| mllmq_channel_only | .35111 | 1383.1 | 277.1 | .49942 | 28.5 | .8413 | .8253 | .8257 |
| mllmq_t2i_text | .33333 | 1423.5 | 297.1 | .47830 | 29.9 | .8343 | .8253 | .8127 |
| mllmq_t2i_img | .35111 | 1367.0 | 303.2 | .47388 | 29.0 | .8490 | .8420 | .8233 |
| mllmq_both | .33778 | 1409.9 | 283.6 | .47246 | 28.9 | .8483 | .8397 | .8273 |
| awq | .33333 | 1395.0 | 283.9 | .49260 | 28.5 | .8177 | .8093 | .8080 |
| mllmq_gen_only | .38000 | 1286.9 | 288.6 | .49284 | 30.8 | .8233 | .8180 | .8010 |

- 对照 `mbq_block_masked`（不在主表 8 config 内）：MMMU .34778 / MME-P 1352.4 / MME-C 278.9 / TextVQA .48538 / MathVista 27.1 / POPE-R .7920 / P .7807 / A .7783。
- 结论：W3 理解侧整体鲁棒，RTN 甚至数值最高；**方法间无显著排序**（MMMU 配对 CI 全跨 0）。不能把 MMMU 平局外推成“RTN 全量不差”。

#### 4.1.2 W4A8（4 bench，`[权威]`）

数据源：`output/janus-pro-7b/w4a8_supplement/und/<tag>/...`；聚合见 `score.json → ...understanding.w4a8_4bench`。

| config | MMMU | MME-P | MME-C | TextVQA | MathVista |
|---|---:|---:|---:|---:|---:|
| rtn | .28444 | 1088.6 | 170.7 | .40862 | 25.0 |
| mbq_scalar | .35222 | 1368.9 | 284.3 | .51330 | 33.4 |
| mllmq_channel_only | .34444 | 1383.2 | 261.8 | .50868 | 33.8 |
| mllmq_t2i_text | .30111 | 1379.4 | 241.8 | .50634 | 31.8 |
| mllmq_t2i_img | .30222 | 1384.1 | 238.2 | .50000 | 31.2 |
| mllmq_both | .31000 | 1428.8 | 281.1 | .50542 | 32.6 |
| awq | .28778 | 906.1 | 164.3 | .32636 | 22.9 |
| mllmq_gen_only | .32889 | 1485.6 | 291.1 | .49522 | 33.5 |

- **W4A8 `rtn` / `awq` 在理解侧也明显退化**（MMMU .284/.288、MME-P 1089/906），不要只归因于生成侧。
- 结果 JSON 的 metric key：`mme_percetion_score,none`（代码拼写就是这么写的）、`mme_cognition_score,none`、`mmmu_acc,none`、`exact_match,none`（TextVQA）、`mathvista_acc,none`、`pope_f1_score,none`。

### 4.2 生成侧主表（GenEval600 + DPG）

#### 4.2.1 GenEval matched 600（`[权威]`）

- W3A16：从 `output/fixed_ablation_20261103_unified/<tag>/geneval/results.jsonl` 过滤到 150 个 matched prompts 后重算。
- W4A8：`output/janus-pro-7b/w4a8_supplement/geneval/<tag>/geneval/results.jsonl`（600 图）。
- 计算方式：每个 tag 的 4 张图先取均值，`overall` = 6 个 tag 均值的无权平均。

| config | W3A16 GenEval600 | W4A8 GenEval600 |
|---|---:|---:|
| rtn | .45000 | .02833 |
| mbq_scalar | .54167 | .49667 |
| mllmq_channel_only | .53000 | .48000 |
| mllmq_t2i_text | .55167 | .54833 |
| mllmq_t2i_img | .52833 | .52500 |
| mllmq_both | .55167 | .51667 |
| awq | .55167 | .30833 |
| mllmq_gen_only | .57000 | .53167 |

#### 4.2.2 DPG-Bench（`[权威]`，1064 张）

- W3A16：`output/fixed_ablation_20261103_unified/dpg_scores/<tag>.txt`，取最后一行 `DPG-Bench score:`。
- W4A8：`output/janus-pro-7b/w4a8_supplement/dpg_scores/<tag>.txt`。

| config | W3A16 DPG | W4A8 DPG |
|---|---:|---:|
| rtn | 80.342 | 42.153 |
| mbq_scalar | 82.190 | 80.389 |
| mllmq_channel_only | 81.864 | 80.130 |
| mllmq_t2i_text | 82.065 | 80.967 |
| mllmq_t2i_img | 81.460 | 80.085 |
| mllmq_both | 82.475 | 81.298 |
| awq | 82.472 | 69.877 |
| mllmq_gen_only | 81.551 | 80.756 |

- 对照 `mbq_block_masked` W3A16 DPG = **82.577**（不在主表 8 config 内）。
- `dpg_scores/<tag>_detail.txt` 是逐 VQA 明细；`<tag>.txt` 每行 `path, score, score`，最后一行是总分。

### 4.3 GenEval 全量 2212 与噪声（`[权威]`）

数据源：`output/fixed_ablation_20261103_unified/<tag>/geneval/results.jsonl`（553 prompts × 4）。

| config | full 2212 |
|---|---:|
| rtn | .45550 |
| mbq_scalar | .53346 |
| mllmq_channel_only | .53970 |
| mllmq_t2i_text | .54226 |
| mllmq_t2i_img | .55809 |
| mllmq_both | .54517 |
| awq | .55744 |
| mllmq_gen_only | .55198 |
| *control* mbq_block_masked | .55240 |

- 旧 19-config full-2212 参考：`output/logs/geneval_final_scores.txt`（fp16 .64563、rtn_w3 .46627 等），仅用于旧表；当前 8 tag 用上表。
- **重跑噪声**：同 scale 全量 MBQ-scalar W3 两次 0.57397 / 0.55765（**1.63pp**）；600 图 MBQ-block W3 两次 0.58667 / 0.54833（**3.83pp**）。
- **k12 三 seed**：0.61423 / 0.59796 / 0.61170，**mean 0.60796 ± 0.00876**（range 1.63pp）。对 W3 0.57119 = **+3.68pp**；对 W4 0.63603 = **−2.81pp**；达 W4 的 **96.6%**。

### 4.4 混合位宽扫描（原 GAMQ，`[权威]`）

数据源：`output/janus-pro-7b/ablation_mixed/*`、`ablation_mixed_full/w34_spread_k12_full`、`output/dpg_scores/`、`output/t2i_scores/`；聚合见 `score.json → priority_experiments_2026_11.gamt_mixed_precision`。

| 配置 | 平均 bit | k(W4 层) | GenEval 25/tag | Δ vs W3 | GenEval full 2212 | DPG | T2I complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| W3 均匀 | 3.00 | 0 | .56333 | 0 | .57119 | 82.348 | .3482 |
| mixed k8 | 3.267 | 8 | .59667 | +3.33 | — | — | — |
| mixed k10 | 3.333 | 10 | **.62000** | +5.67 | — | **82.350** | — |
| **mixed k12** | **3.400** | **12** | .61000 | +4.67 | **.61423** | **81.992** | **.358228** |
| mixed k14 | 3.467 | 14 | .60833 | +4.50 | — | — | — |
| mixed k16 | 3.533 | 16 | .60667 | +4.33 | — | — | — |
| W4 均匀 | 4.00 | 30 | .63500 | +7.17 | .63603 | 82.648 | .3583 |

- W3 均匀 full 2212 = .57119；W4 = .63603；k12 = .61423。
- k12 vs W3 全量：**+4.30pp，CI [+2.19, +6.40]**（显著）；vs MBQ-scalar W3 全量：**+4.03pp，CI [+1.79, +6.29]**（显著）。
- **跨榜 trade-off（必须同段报告）**：k12 DPG **81.992**（−0.36pp vs W3 82.348）；k10 DPG **82.350**（≈ W3）；T2I complex .358228 ≈ fp16 .3583。
- 逐 tag（k12 vs W3 全量）：color_attr +8.3、colors +6.7、two_object +5.0、position +4.0、single_object +2.2、counting −0.3。
- k10 vs k12 25/tag：+1.00pp，CI [−2.83, +4.83]（不显著）；k10 vs MBQ-block W3：+3.33pp，CI [−0.33, +7.17]（p=.083，不显著）。

### 4.5 SEED（5 chunks，`[权威]`）

- W3A16：`output/fixed_ablation_20261103_understanding_5h_seed/<tag>/seedbench_2_local_chunk{0..4}/`
- W4A8：`output/janus-pro-7b/w4a8_supplement/seed/<tag>/...`
- 每 chunk 最后一个 `summary_<tag>_<width>_<ts>.json` 记录 `seed_all,none` 分数；5 chunk 按 n_samples 加权。

| config | W3A16 SEED | W4A8 SEED |
|---|---:|---:|
| rtn | .47667 | .15219 |
| mbq_scalar | .46166 | .47959 |
| mllmq_channel_only | .46871 | .48078 |
| mllmq_t2i_text | .46806 | .43490 |
| mllmq_t2i_img | .44811 | .45706 |
| mllmq_both | .44278 | .44303 |
| awq | .44930 | .23725 |
| mllmq_gen_only | .46596 | .46432 |
| *control* mbq_block_masked | **missing 0/5** | n/a |

- `score.json` 的 SEED section 只收录了 W3A16 7 个（不含 awq）和 W4A8 4 个；上表的其余值从同一 raw 目录按 chunk 加权算出。

### 4.6 T2I-CompBench++ current（`[进行中]`，未定稿）

- W3A16 生成：`output/janus-pro-7b/t2i_compbench_current/<tag>/t2i_compbench/`，8 tag × 2400（8 类 × 300）。
- W4A8 生成：`output/janus-pro-7b/t2i_compbench_current/w4a8/<tag>/t2i_compbench/`，进行中。
- 评分：`output/janus-pro-7b/t2i_compbench_current_scores/<w3a16|w4a8>/<tag>/<category>.json`。

**W3A16 已出分的 6/8 类（2026-11-07 21:20 快照，仅供写作准备，不得当作最终表）**

| tag | color | shape | texture | spatial | 3d_spatial | numeracy |
|---|---:|---:|---:|---:|---:|---:|
| rtn | .51369 | .34344 | .48453 | .08583 | .21487 | .24975 |
| mbq_scalar | .56203 | .30467 | .44368 | .16168 | .28824 | .35803 |
| mllmq_channel_only | .58550 | .32331 | .43905 | .16058 | .28917 | .35686 |
| mllmq_t2i_text | .57396 | .31183 | .43483 | .17141 | .29129 | .34272 |
| mllmq_t2i_img | .60941 | .33744 | .46706 | .19538 | .29880 | .36778 |
| mllmq_both | .57703 | .32261 | .47686 | .17298 | .31993 | .36417 |
| awq | .64084 | .35035 | .48965 | .17367 | .30425 | .38847 |
| mllmq_gen_only | .67308 | .33842 | .48423 | .15834 | .30867 | .40247 |

- **缺失**：`non_spatial`（CLIPScore）8 tag 全部 `rc=1, score=null`；`complex` 3-in-1 聚合失败（clip rc=1、unidet rc=120）。已排队重跑/修复，**当前不能写进论文最终表**。
- **W4A8**：只有 `.score.lock` + `work/`，无任何 category JSON；生成完成前评分 lane 会等待 2400。
- **legacy T2I 表**：`output/janus-pro-7b/t2i_compbench/<config>/` + `t2i_compbench_scores/` + `t2i_compbench_scores.md`，是旧 13-config 集（fp16、rtn/awq/mbq_block/mbq_scalar/mllmq_mae/mllmq_mse × w3/w4、mllmq_mixed_w34_k12），**不含当前 channel_only/t2i_text/t2i_img/both/gen_only**；只可用于旧表 F，不能当 current 列。
- 所有 T2I 均为 **num=1**（每 prompt 1 张）；官方 complex 用 10 张并硬编码 num=10 → complex 只能内部横向比较。

### 4.7 消融与负结果（写作时必须进 Limitations/附录）

| 实验 | 结果 | 解读 |
|---|---|---|
| T2I 重加权修复版（W4） | .60167 vs 旧 .63500，**−3.33pp，CI [−7.17,+0.67]，p=.10** | 无证据支持增益；不能写“T2I 有害” |
| T2I 重加权修复版（W3） | .55833 vs 旧 .56333，**−0.50pp，CI 跨 0** | null |
| greedy 8-config T2I ON−OFF | **+1.98pp p=.36** | null |
| greedy 8-config 图像 ON−OFF | **−3.35pp p=.10** | 方向偏负 |
| 教师强制 gen_tf（Exp2） | off .61000 / ×1 .60667 / ×2 .60500；Δ −0.33 / −0.50pp，CI 跨 0 | 更强的生成信号也无增益 |
| prompt 噪声（Exp1） | eps 5/10/20/40% = .65833/.67083/.67083/.58750 | ≤20% 无可靠下降；40% −8.33pp |
| A8 激活量化 | W3A8 mllmq .00500 / mbq .00167 崩；W4A8 .48667/.48833（−14.8pp vs A16） | 当前实现 A8 不可用，不是方法区分点 |
| CFG 重扫（n=240） | fp16 最佳 5.5 .67083；mllmq_w3 最佳 4.0 .62500；scalar_w3 最佳 5.5 .60417 | 最优 CFG 无一致方向 |
| 校准量 n=64/128/256 | 64–128 接近饱和；同 n 重跑差 2.67pp | 共有性质，不能支撑方法优越性 |
| channel vs scalar | scale **118/120 张量完全相同**；MMMU +0.33pp p=.758 | 统计等价；不要单独声称逐通道增量 |
| MBQ block vs scalar | 生成侧全量 ±1.2pp（噪声内）；DPG +0.19pp；MMMU +2.44pp p=.130 | 无稳健赢家 |
| **MLLMQ vs MBQ（用户定稿“相对 MBQ 也有效”）** | DPG combined **+1.21pp CI [+0.12,+2.32]**（唯一 CI 不跨 0，单次、4 变体、缺 mbq_block 基线）；GenEval 干净口径 +1.35pp CI [−0.66,+3.41]；greedy 8-config CI 全跨 0 | 写“一致正趋势/DPG 组合性改善”，不写“全榜显著优于 MBQ” |
| A13/A14 | **作废** | 在 GradCacheHook 修复前采集，T2I 项为空，不可引用 |

### 4.8 结果文件路径速查

| 内容 | 路径 |
|---|---|
| 总聚合 | `output/score.json`（注意部分 section 是 2026-11-02 快照） |
| W3A16 DPG | `output/fixed_ablation_20261103_unified/dpg_scores/<tag>.txt` |
| W4A8 DPG | `output/janus-pro-7b/w4a8_supplement/dpg_scores/<tag>.txt` |
| W3A16 GenEval | `output/fixed_ablation_20261103_unified/<tag>/geneval/results.jsonl` |
| W4A8 GenEval | `output/janus-pro-7b/w4a8_supplement/geneval/<tag>/geneval/results.jsonl` |
| W3A16 理解 | `output/fixed_ablation_20261103_understanding_5h/<tag>/<task>/<tag>_w3a16/*_results.json` |
| W4A8 理解 | `output/janus-pro-7b/w4a8_supplement/und/<tag>/<task>/<tag>_w4a8/*_results.json` |
| W3A16 SEED | `output/fixed_ablation_20261103_understanding_5h_seed/<tag>/seedbench_2_local_chunk{0..4}/` |
| W4A8 SEED | `output/janus-pro-7b/w4a8_supplement/seed/<tag>/seedbench_2_local_chunk{0..4}/` |
| T2I current 图 | `output/janus-pro-7b/t2i_compbench_current/{<tag>|w4a8/<tag>}/t2i_compbench/<cat>/samples/` |
| T2I current 分 | `output/janus-pro-7b/t2i_compbench_current_scores/<width>/<tag>/<cat>.json` |
| 旧 DPG（仅旧表 E） | `output/janus-pro-7b/dpg_bench/<config>/` + `output/dpg_scores/<config>_single_results.txt` |
| 旧 T2I（仅旧表 F） | `output/janus-pro-7b/t2i_compbench/<config>/` + `t2i_compbench_scores/` |

### 4.9 不可作为当前结论的路径（`[过时]`）

- `output/fixed_ablation_20261103/<tag>/dpg_results.txt`：旧 scale 的 DPG，不可用。
- `output/fixed_ablation_20261103_unified_understanding/`：MMMU-only 部分分支，已排除。
- `output/janus-pro-7b/dpg_bench/` + `output/dpg_scores/`：旧 13-config 主表，仅旧表 E。
- `output/janus-pro-7b/t2i_compbench/` + `t2i_compbench_scores/`：旧 13-config T2I，仅旧表 F。
- `output/legacy/`：InternVL2 复现，不是 Janus。
- `ablation_a13/`、`ablation_a14/`：作废。
- `score.json.bak-*`：快照，不引用。
- `output/logs/geneval_final_scores.txt`：旧 19-config full-2212，仅旧表 B。

---

## 待写（按计划逐块追加）

- §0 读法/数据权威层级（本文开头已放最小版，后续可扩充）
- §1 CVPR 论文定位、组件-证据表、写作红线
- §2 模型、量化方法、scale cache、代码入口
- §3 实验协议、benchmark、环境、命令
- §5 结论/机制/Limitations
- §6 相关工作与 third_party（路径/许可证/评分器 quirks）
- §7 文件与数据地图、迁移清单
- §8 主机专属坑
- §9 论文写作素材（section skeleton、图表、引用）
- §10 未完成实验与 TODO
- §11 新 Claude 上手 checklist
