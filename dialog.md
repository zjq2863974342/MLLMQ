# MLLMQ 实验记录（精简版）

> 更新：2026-11-05 15:10
> 完整历史：`dialog.md.bak-20261105_012640`（497KB）；本次精简前：`dialog.md.bak-20261105_150541`（37KB）
> 相关：`TODO.md`（当前队列/重启后指令）、`paper_outline.md`、`copy.txt`（副机 §8）

## 0. 当前状态

- **W3A16**：GenEval full 9/9（2212）、MMMU 900、5h 理解表、SEED full 5/5 完成。
- **W3A16 DPG 旧分数过时**：`output/fixed_ablation_20261103/*/dpg_results.txt` 用 `scale_cache/fixed_ablation/*`（11/03 00:xx），不是 `unified_384/*`（11/03 17:xx）；`fixed_ablation_20261103_unified` 无 dpg 产物。需副机用 `run_dpg_unified_remote.sh` 重跑（8 config，~7–10h）。
- **W3A16 T2I-CompBench 没有当前 config**：只有 `awq_w3/mbq_scalar_w3/mllmq_mae_w3/...` legacy tag，没有 `channel_only/t2i_text/t2i_img/both`。
- **W4A8**：scales 10/10、理解 4 项 24/24、GenEval 600 6/6 完成；**DPG / SEED full 未跑**。
- **队列**：W3A16 DPG 重跑（副机）→ W4A8 DPG（2/GPU 上限）→ W4A8 SEED full（1/GPU）。
- **脚本**：`run_w4a8_supplement.sh`（seed/dpg）、`run_w4a8_backfill_{local,remote}.sh`、`w4a8_backfill_scheduler.sh`；详细指令在 `TODO.md §2` 与 `copy.txt §8`。

## 1. 当前任务表

| 顺序 | 任务 | 机器 | 状态 / 预计 |
|---|---|---|---|
| 0 | W3A16 DPG unified_384（8 config） | 副机 4 卡 | 待跑，~7–10h |
| 1 | W4A8 DPG（6 config） | 副机 4 + local4 2 | 待跑；2/GPU 上限 |
| 2 | W4A8 SEED full（6 config × 5 chunks） | 副机 4 + local4 2 | 待跑；1/GPU |
| 3 | T2I-CompBench current config 子集 | 待定 | 可选 P1 |
| 4 | COCO 1k（CLIP/IR/Pick） | 待定 | 可选 P3；不跑 5k FID |

**W4A8 分工**：副机 DPG `rtn/mbq/channel/t2i_text`、SEED `mbq/channel/t2i_text/t2i_img`；local4 DPG `t2i_img/both`、SEED `rtn/both`。

## 2. 关键结果

### 2.1 W4A8 理解（24/24）

| config | MMMU | MME-P | MME-C | TextVQA | MathVista |
|---|---|---|---|---|---|
| rtn | .28444 | 1088.6 | 170.7 | .40862 | 25.0 |
| mbq_scalar | **.35222** | 1368.9 | 284.3 | **.51330** | 33.4 |
| channel_only | .34444 | 1383.2 | 261.8 | .50868 | **33.8** |
| t2i_text | .30111 | 1379.4 | 241.8 | .50634 | 31.8 |
| t2i_img | .30222 | 1384.1 | 238.2 | .49998 | 31.2 |
| both | .31000 | **1428.8** | 281.1 | .50542 | 32.6 |

### 2.2 W4A8 GenEval 600（6/6）

| t2i_text | t2i_img | both | mbq_scalar | channel | rtn |
|---|---|---|---|---|---|
| **.54833** | .52500 | .51667 | .49667 | .48000 | **.02833** |

- **RTN 生成侧崩盘**；学习型 0.48–0.55；生成侧校准（t2i_*）> 理解侧校准（mbq/channel）。

### 2.3 W3A16 6 config（128 und；GenEval600 为 matched 子集；SEED full 5 chunks）

| config | MMMU | MME-P | MME-C | TextVQA | MathVista | GenEval600 | SEED |
|---|---|---|---|---|---|---|---|
| rtn | **.35000** | 1409.4 | 245.0 | **.48814** | **33.5** | .45000 | **.47667** |
| mbq_scalar | .33778 | 1400.0 | 270.7 | .49448 | 28.3 | .54167 | .46166 |
| channel_only | .35111 | 1383.1 | 277.1 | .49942 | 28.5 | .53000 | .46871 |
| t2i_text | .33333 | 1423.5 | 297.1 | .47830 | 29.9 | **.55167** | .46806 |
| t2i_img | **.35111** | 1367.0 | 303.2 | .47388 | 29.0 | .52833 | .44811 |
| both | .33778 | 1409.9 | 283.6 | .47246 | 28.9 | **.55167** | .44278 |

- W3A16 理解侧 RTN 名义最高，学习型未超过；W4A8 下学习型才明显 > RTN（MMMU +6.8pp）。
- W3A16 DPG/T2I 当前 config 未完成：DPG 旧 `fixed_ablation` 分数不能进最终表；T2I legacy tag 不是当前 config。

### 2.4 Exp3 / Exp4 / Exp5

- **Exp3 k12 全量**：`.61423` vs W4 `.63603` / W3 `.57119`；k12−W4 −2.18pp CI[−4.59,−0.11]；k12−W3 +4.30pp CI[+2.40,+6.64]。
- **Exp4 A8**：W4A8 `mbq .48833 / mllmq .48667`；W3A8 `.00167/.00500`（双方法崩）。
- **Exp5 CFG**：fp16 最佳 5.5 `.67083`；mllmq_w3 最佳 4.0 `.62500`；scalar_w3 最佳 5.5 `.60417`；各 cfg 量化均低于 fp16。

### 2.5 InternVL MBQ vs AWQ（受控复核）

| method | MMMU | MME | TextVQA | MathVista |
|---|---|---|---|---|
| AWQ（重建 W3） | .45444 | 76.76% | 75.17% | 58.1 |
| MBQ（W3） | **.4567** | 73.93% | 74.40% | 56.6 |

- 论文 Table 2：FP16 48.0 / RTN 43.7 / AWQ 44.8 / MBQ 46.9；方向一致但 +2.1pp 未复现。
- 复现缺口：`sharegpt4v_coco_calib.json` 是原始 COCO 短 caption（中位 ~52 字符），不是 ShareGPT4V GPT-4V 详细 caption。

## 3. 作废 / 失败尝试（压缩）

| 项目 | 结果 | 原因 / 处置 |
|---|---|---|
| W3A8（W3 权重 + A8 激活） | 双方法 ≈0–0.5% | 通用失效，生成抽象纹理；WALinear ~4× 慢；不再使用 |
| 9 月 InternVL “MBQ-block” | 标号错误 | `block_w3a16/` 实际用 `--method mbq` + `mbq` scale；真正 block 从未评过 |
| 9 月 InternVL 跨日期对比 | 不可用 | AWQ 9/10–9/19、MBQ 9/27；同配置 MME-P 自漂 12 分；无 log_samples |
| InternVL 受控 MME | 失败 | lmms-eval wrapper `'InternVL2' object has no attribute '_model'`；MMMU 正常 |
| 副机可选 SEED channel/rtn | 2/5 中断 | OOM/Killed（396/400GB；SEED ~56GB RSS/进程，2/卡必 OOM）；已改 1/卡续跑 |
| A8 代码缺口 | 已修 | eval_benchmarks 忽略 a_bit；geneval shard 缺 a_bit；auto_scale_wa 缺 gen-image 重加权；RTN/AWQ A8 分支忽略 a_bit |
| COCO FID 扩 W4A8 | 不做 | FID 与 CLIP/偏好分方向相反；5k 成本 ~60h+；最多 1k CLIP/IR/Pick |
| W3A16 DPG 旧分数 | 过时 | fixed_ablation scale，不是 unified_384；需重跑 |
| T2I legacy tag | 不适用 | 旧 tag 没有当前 6 config；需重新生成/评分 |

## 4. 操作纪律（必读）

- **生成满载时禁止新起 CUDA 进程**（2026-10-09 驱动全局锁死，只能重启实例）；评分/验证排空窗期。
- 大规模前 `--limit 5` 冒烟；改 task yaml/metric/scale apply 后必须重测。
- 限线程：`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`（nproc=128）。
- 并发：SEED **1 进程/卡**（~56GB RSS）；DPG **2 进程/卡上限**（~16GB VRAM/进程）；n=384 scale build 1/卡。
- 平台保活 `_takeup.py` 不能杀；不要自起 gpu_heartbeat/MPS。
- 只有一个调度器；所有队列 skip-existing/幂等；调度脚本不要 `set -e`。
- `nvidia-smi` 查询带 `timeout -s 9`。

## 5. 补跑 / 调度

- **W3A16 DPG**：副机先 `WAIT_GENEVAL=0 bash scripts/run_dpg_unified_remote.sh`（unified_384，8 config，~7–10h）；产物 `output/fixed_ablation_20261103_unified/`。
- **W4A8 DPG/SEED**：local4 `MODE=local`、副机 `MODE=remote` 启动 `w4a8_backfill_scheduler.sh`；60s 检查、driver 内部重试 3 次、外层自动重启；DPG 2/GPU 上限，SEED 1/GPU。
- 6 GPU/6 config 时 DPG 自然 1/GPU；2/GPU 是上限/兜底（掉卡/集中配置时用）。
- 中断后重跑 scheduler 或 backfill 脚本，skip-existing。
- T2I current config 建议 P1 子集（50/类，W3A16+W4A8）；COCO 1k 可选 P3；5k FID 不做。

## 6. 近期里程碑（压缩）

- **01:18** local4 加第二 lane（t2i_img/both），2/GPU 生效。
- **07:38** W4A8 理解 24/24；学习型 > RTN，MMMU +6.8pp。
- **08:35/08:41** SEED chunk2 出现重复 result（stray runner 写共享盘）；chunk3 也被它补齐；最终按 chunk 去重。
- **10:08** W3A16 SEED full 6 config 收尾：rtn .47667 名义最高。
- **12:08** W4A8 GenEval 6/6：t2i_text .54833 最高，rtn .02833 崩盘。
- **14:20/15:00** 确认 W3A16 DPG 旧分数过时（fixed_ablation scale）；T2I current config 未跑。
- **15:10** 队列改为 W3A16 DPG 重跑 → W4A8 DPG → W4A8 SEED；文档/调度器就绪，未启动。
- 全程无 OOM/Traceback（可选 SEED 的 10/30 OOM 已处理）。

## 7. 文献调研（压缩）

| 论文 | 关系 |
|---|---|
| MBQ (2412.19509) | 直接竞品；模态均衡 scale search；复现对象 |
| Q-VLM (NeurIPS 2024) | W4A4 PTQ，VLM 量化 baseline |
| Bi-VLM (AAAI 2026) | 二值 PTQ，超低比特 |
| VLM-PTQ (CVPR 2026) | 1B–72B 单卡压缩，SOTA PTQ |
| MASQuant (2603.04800) | 模态感知平滑，机制不同 |
| C-PTQ (2607.21076) | Fisher 通道敏感度，最接近但静态 |
| OmniQuant (ICLR 2024) | per-token 激活校准 |
| SmoothQuant (ICML 2023) | W8A8 经典 baseline |
| AQLM (ICML 2024) | codebook 极低比特，LLM-only |
| Janus-Pro (2501.17811) | 被量化对象 |
| Q-Diffusion (2023) | 生成侧量化；O-proj/down-proj 敏感 |

> 完整文献调研见 `paper_outline.md` 附录 C 与 `dialog.md.bak-20261105_012640`。

## 8. 归档

- 完整历史：`dialog.md.bak-20261105_012640`（497KB / 6061 行）
- 本次精简前：`dialog.md.bak-20261105_150541`（37KB / 488 行）
- 任务/队列：`TODO.md`；论文：`paper_outline.md`；副机：`copy.txt`

## [Watchdog] 2026-11-05 15:08
- 旧主任务无变化：und 24/24、geneval 6/6、SEED 5/5；无 eval/backfill 进程；GPU 空闲；scheduler 263649 空转。
- 补跑未启动：W3A16 DPG 重跑（副机 §8.0）与 W4A8 DPG/SEED（local/remote scheduler）等用户重启/执行。
- 无 Traceback/OOM/FAIL。

> 口径说明（2026-11-05 19:35）：以下两节 [补跑计划] 属于 **local4 主机**（另一台 4 卡机），不是本会话所在的副机；副机 §8 进度见 [工作记录] / [角色确认]。

## [补跑计划] 2026-11-05 17:45 local4 主机升 4 卡 + AWQ 补齐

- local4 主机将重启为 **4 卡容器**；旧 2 卡 scheduler 不再恢复，改用 `MLLMQ/scripts/local4_supplement_scheduler.sh`（TODO §2A）。
- 第一波 DPG 网格 8 进程（每卡 2）：W4A8 `mllmq_t2i_img`/`mllmq_both`（各 2 shards）+ **W3A16 AWQ**（2 shards，续 140/1065）+ **W4A8 AWQ**（2 shards）。
- DPG 后 4 个 score 在 GPU0 串行；同时 W4A8 AWQ und 4 项 + GenEval600 跑 GPU1–3。
- SEED 严格 1/GPU：W4A8 `rtn`、W4A8 `mllmq_both`(mix384)、W3A16 AWQ(und128 W3A16 scale)、W4A8 AWQ(und128 W4A8 scale)。
- W4A8 AWQ 为完整行：scale(und128 n=128 + mix384 n=384) + und 4 + GenEval600 + DPG1065 + score + SEED5；W3A16 AWQ und/GenEval 已完成，不重跑。
- 启动见 TODO §2A.4；预计 **16–20h**。第二波 `gen_only` 按 TODO §5.3/§5.4，队列排空后另排。

## [补跑计划-修正] 2026-11-05 18:00 local4 主机 4 卡：补集 4 config + gen_only 全部 benchmark

- **口径确认**：local4 主机 = 副机 4 config 的补集。
  - W4A8 DPG local4：`t2i_img / both / awq / mllmq_gen_only`
  - W4A8 SEED local4：`rtn / both / awq / mllmq_gen_only`
  - W3A16：`awq`（DPG 续+score+SEED）、`gen_only`（und 7 项+SEED；DPG 副机）；`t2i_img/both` 已完成，跳过。
- DPG：10 个 shard job / 8 slot（2/卡）；W3A16 AWQ 先跑（短），完成后 W4A8 gen_only 补位。
- score：5 个 DPG score 在 GPU0 串行；同时 aux 队列 2 个 GenEval + 15 个 und（AWQ und4 + W4A8 gen_only und4 + W3A16 gen_only und7），7 slot。
- SEED：6 tag / 2 wave，严格 1/GPU；wave1 `rtn/both/W3A16-AWQ/W3A16-gen_only`，wave2 `W4A8-AWQ/W4A8-gen_only`。
- 驱动：`MLLMQ/scripts/run_local4_supplement.sh` + `local4_supplement_scheduler.sh`；操作见 TODO §2A。
- 表格：§5.1 可全填；§5.2 除 T2I-comp 列外可全填（W3A16 6+gen_only DPG、W4A8 副机 4 config SEED 依赖副机完成）。
- ETA：~24–30h（4 卡）。

## [工作记录] 2026-11-05 18:50 泛化模型配置与 Show-o/Janus-1B 适配准备

### 1. 当前补跑状态（事实）

- 副机 W3A16 DPG unified_384：8 个 tag 生成中，18:49 各 634–645/1065；预计 21:00 前后生成完，随后 8 个 score 在 GPU0 串行。
- 副机调度器：持久化 v2，pid 132605；脚本在 `/nas3/kangjia/mllmq/queue/`；W4A8 DPG 采用每 tag 2 个 shard，GPU1–3 各 2 进程；SEED 仍严格 1/GPU。
- W3A16 AWQ DPG（local4 主机）：16:16 启动，16:32 中断在 140/1065；后续 local4 主机已续跑，18:49 约 297/1065。旧 fixed_ablation 82.01 不采用，最终用 unified_384 score。
- W4A8 DPG 尚未开始；等 W3A16 生成结束、评分开始时由 v2 编排器接管。
- TODO 已补：AWQ 行（W3A16 理解 + GenEval600）、gen_only 行、W4A8 AWQ/gen_only 占位；队列第 5/6 项；5.3/5.4 补跑命令。

### 2. 泛化实验范围（已确认）

- 主实验：Janus-Pro-7B。
- 同家族：Janus-Pro-1B（同 MultiModalityCausalLM、同 AR）。
- 跨家族：原版 Show-o 1.3B（256 个离散 VQ token + masked discrete diffusion；不是 flow，也不是 Janus 的逐 token AR）。
- Show-o 只做 fp16 / W4A8 / W3A16；理解侧 MMMU + MME；生成侧 GenEval600 + T2I-CompBench 50/类。
- 口径：Show-o 结果只用于“量化趋势可迁移”，不声称 AR 误差累积机制可迁移；AWQ 暂不做。

### 3. Show-o 现状与代码

- 没有 Show-o-1B 权重。原版 Show-o 1.3B/1.7B 在 `checkpoints/Show-o/models/show-o-w-clip-vit`，2026-11-02 冒烟通过（加载 1.8s、VQA 21s、256×256 T2I 256 token/50 步约 15.9s）。
- `showlab/Show-o2-1.5B` 已从 ModelScope 下载到 `checkpoints/Show-o2-1.5B`（5.27 GiB，Apache-2.0）；它是连续 latent + flow matching，不作为本次目标。
- 环境：`venvs/showo` 可同时导入 `qmllm` 和 `checkpoints/Show-o/src/models`（2026-11-05 验证）。
- 新增（只新增文件，未改 Janus）：
  - `MLLMQ/qmllm/models/showo/__init__.py`
  - `MLLMQ/qmllm/models/showo/showo.py`
  - `MLLMQ/showo_rtn.py`
- 已做验证：`py_compile` 通过；`showo` 已注册进 MODEL_REGISTRY；CPU dummy 模型上局部 RTN 生效（权重 RTN 6 个 Linear，W4A8 WALinear 替换 6 个 Linear）。
- CPU 真实权重验证（2026-11-05 19:00）：
  - fp16：加载 1.454B/F32，单题 1 token 理解 41.6s，总 82.3s；输出 `output/deploy_smoke/showo_cpu_fp16/`。
  - W3A16：144 个 Linear 权重 RTN，单题 1 token 理解 31.9s，总 110.4s；输出 `output/deploy_smoke/showo_cpu_rtn_w3a16/`。
  - W4A8：144 个 Linear 替换为 WALinear（带 dtype 适配），单题 1 token 理解 71.9s，总 141.4s；输出 `output/deploy_smoke/showo_cpu_rtn_w4a8/`。
- 未做：CUDA 加载/生成/理解冒烟，以及 T2I 生成（CPU 上只跑了理解）。当前 4 卡在跑 W3A16 DPG，按纪律不新起 CUDA 进程。

### 4. Show-o 适配的阻塞点（未改核心）

- 现有 RTN 的 `pseudo_quantize_model_weight` **也依赖 `get_blocks`**，不是完全不依赖；因此先在 Show-o wrapper 里实现局部 RTN，直接遍历 Phi 的 24 层 Linear。
- MBQ/MLLMQ 还需要 Phi 分支：
  - `get_blocks` / `move_embed` 不认 `PhiForCausalLM`；
  - `GradCacheHook.register_hooks` 的名称过滤只匹配 `v_proj`，漏掉 `dense` / `fc1` / `fc2`；
  - `run_mbq` 分组只认 `down_/w2` 和 `wo/o_proj`，Phi 的 `dense` / `fc2` 需要映射；
  - `auto_scale.py` / `auto_scale_wa.py` 没有 `PhiDecoderLayer` 分支；Phi 的 attention 与 MLP 共用同一个 `input_layernorm` 输出（并行残差），不能照抄 Llama。
- MLLMQ 三模态 mask 最后接；`dense/fc1/fc2` 分组或并行残差若有歧义，停下来报告，不自行猜。
- 校准数据：现有 `janus_gen_tf_128.json` 是 Janus 576 token；Show-o 需要 256 个 MAGVIT token，需另建 gen_tf 校准集。

### 5. Janus-Pro-1B wrapper 配置

- 架构与 7B 相同（`MultiModalityCausalLM`），复用 `qmllm.models.janus.JanusModel`，不需要新 wrapper 类。
- 配置：24 层、hidden 2048、intermediate 5632、16 heads、vocab 102400、bf16、384 分辨率、576 image tokens。
- 新增 `MLLMQ/configs/janus1b/model.yaml`，记录 model_path、scale_cache/eval 输出路径和上述维度。
- 注意：`main_quant.py` 默认 `--scale_path` 是 `scale_cache/<method>/janus_w<bit>.pt`，会和 7B 冲突；1B 必须显式传 `--scale_path scale_cache/janus1b/<method>_w<bit>_a<abit>.pt`。混合精度的 k 要按 24 层重新折算。

### 6. 下一步

1. 等 GPU 窗口，先跑 Show-o CUDA fp16 冒烟（`MLLMQ/showo_rtn.py --mode fp16`），再跑 W3A16 / W4A8 RTN 的 CUDA 理解与 T2I 生成。
2. RTN 通过后，评估 MBQ 的 Phi 分支；先在独立模块/局部 patch 做，确认不影响 Janus 再考虑核心修改。
3. 最后接 MLLMQ 三模态 mask。

## [角色确认] 2026-11-05 19:30

- 本会话所在机器确认为**副机**。132605（`queue/w4a8_remote_queue_v2.sh` 编排器）和 15481 + 8 个 `eval_generate_quant.py`（W3A16 DPG §8.0）是副机自己的正常进程，**不停止、不迁移**。
- local4（§2A）在另一台 4 卡机，不在本机；本机 GPU 上的 8 个 DPG python 属于副机 §8.0，不是 local4。
- 之前“不要在本机运行 start_remote_queue.sh”的告警基于角色误判；副机上可以运行，但当前编排器已在跑，无需重复启动。

## [队列追加] 2026-11-06 12:42 · T2I-CompBench++ P1（副机 W3A16）

- 任务：副机 8 个 W3A16 config（rtn、mbq_scalar、mllmq_channel_only、mllmq_t2i_text、mllmq_t2i_img、mllmq_both、awq、mllmq_gen_only），8 类 × 300 × 1 张 = 2400 图/config，共 19,200 图；输出到 `output/janus-pro-7b/t2i_compbench_current/<tag>/`，日志到 `output/logs/t2i_compbench_current/`。
- 明确不跑 W4A8 T2I（主机负责）、不跑 mbq_block_masked、不覆盖旧 `output/janus-pro-7b/t2i_compbench/`、生成后不套旧评分脚本。
- 新增 `queue/t2i_compbench_w3a16_tail_queue.sh`（只新增队列脚本，未改实验脚本）：
  - 启动后先等待 §8 ALL DONE：无 remote 编排器/scheduler/driver 进程，W3A16 DPG 8 tags 1065+score，W4A8 DPG 4 tags 1065+score，W4A8 SEED 4 tags 5/5 chunks；
  - 满足后检查 GPU，若单卡显存 >10GB 连续 30 分钟则写 `ABORT_GPU_BUSY` 退出，不抢卡；
  - 采用 **8 进程并发 = 2 进程/GPU**（GPU0: rtn+mbq_scalar；GPU1: channel_only+t2i_text；GPU2: t2i_img+both；GPU3: awq+gen_only），按用户最新要求覆盖原文档的 1/卡；
  - skip-existing（PNG≥2400 跳过）；启动即失败且 0 新增则写 `<tag>.FAILED` 并继续其他 tag；不评分。
- 已于 12:42 以 `setsid nohup` 启动，PID 311525，当前在等 §8；另加 durable cron `349b6d03`（每小时 :13）做跨重启存活检查/重启。
- 当前 §8：W4 `mllmq_t2i_text` 788/1065（12:42），W4 SEED 4 tags 0/5；预计 §8 完成约 21:00–22:30，P1 随后开始。

## [调度调整] 2026-11-06 15:01 · §8 未完成时提前启动 P1 前 4 个 tag

- 用户确认：§8 剩余较久，可以“先运行下一批”。§8 剩余 = t2i_text 评分（~15:15 完成）+ SEED 5 波（约 6–7.5h），预计 **21:15–22:45 ALL DONE**。
- 新增 `queue/t2i_compbench_early_overlap.sh`：等 t2i_text 评分进程结束后，启动 4 个 P1 tag，**1 进程/GPU**，与 W4A8 SEED（仍严格 1 进程/GPU）同卡共存：
  - GPU0 rtn、GPU1 mllmq_channel_only、GPU2 mllmq_t2i_img、GPU3 awq。
  - 启动前检查 `free -g` available ≥100GB，避免与 SEED 的 ~56GB RSS/进程叠加导致主机 OOM。
- `rtn` 已提前在 GPU0 启动（15:00 左右，已开始出图）；其余 3 个在评分结束后由 early launcher 启动。
- 修改 `queue/t2i_compbench_w3a16_tail_queue.sh`：增加 `tag_running` 检查，§8 完成后只补跑剩余 4 个 tag（mbq_scalar、mllmq_t2i_text、mllmq_both、mllmq_gen_only），避免与 early 4 个重复/双进程；最终仍为 2 P1/GPU。
- 风险与口径：SEED 保持 1 进程/GPU；同卡共存的 P1 为额外非 SEED 进程，GPU 约 17GB/进程、主机 RSS 约 15–20GB/进程，预计总 RAM ~300GB < 400GB；若出现 RAM/OOM 压力，优先暂停 P1，不杀 SEED。
- 未修改任何实验/评分脚本；P1 仍未评分。

## [调度回退] 2026-11-06 15:15 · 为保证 SEED，停止提前启动的 P1

- 用户担心提前跑的 rtn 影响 SEED。实测 rtn 单进程 ~13.6s/图；按 2400 张需约 9h（无争用），与 SEED 共存或后续 2/GPU 会更久，预计次日 00:30–03:00 才能跑完。
- 决定优先保护 §8：已于 15:15 停止 early launcher 和 rtn 进程，GPU0/1/3 恢复空闲，SEED 将独占 4 卡。
- rtn 已生成 74 张，全部 PNG 校验通过（PIL verify，0 损坏）；skip-existing 会在之后续跑，只损失当前未写完的 1 张。
- P1 回到原计划：由 `t2i_compbench_w3a16_tail_queue.sh` 在 §8 ALL DONE 后启动全部 8 个 tag，2 进程/GPU。
- 当前 t2i_text 评分 1006/1065（~15:19 完成），随后 SEED 启动；§8 预计 21:20–22:50 完成。

## [结果] 2026-11-07 20:50 · 主结果收尾；只剩 T2I-comp

- 副机 remote 队列 v2 于 2026-11-07 01:33 正常结束（`orchestrator v2 done`），无 Traceback / CUDA OOM / Killed。
- **W3A16 DPG unified_384**：8 config + `mbq_block_masked` 对照均 1065/1065 + 评分完成：
  rtn 80.34 / mbq_block_masked 82.58 / mbq_scalar 82.19 / channel 81.86 / t2i_text 82.07 / t2i_img 81.46 / both 82.47 / awq 82.47 / gen_only 81.55。
- **W4A8 DPG**：8 config 均 1065/1065 + 评分完成：
  rtn 42.15 / mbq_scalar 80.39 / channel 80.13 / t2i_text 80.97 / t2i_img 80.09 / both 81.30 / awq 69.88 / gen_only 80.76。
- **SEED**：W3A16 `awq` .4493、`gen_only` .4660（均 5/5）；W4A8 8 config 均 5/5：rtn .1522、mbq_scalar .4796、channel .4808、t2i_text .4349、t2i_img .4571、both .4430、awq .2372、gen_only .4643。
- **理解补齐**：W3A16 `gen_only`（MMMU .380 / MME-P 1286.9 / MME-C 288.6 / TextVQA .4928 / MathVista 30.8 / POPE×3 齐）；W4A8 `awq`（MMMU .2878 / MME-P 906.1 / MME-C 164.3 / TextVQA .3264 / MathVista 22.9）；W4A8 `gen_only`（MMMU .3289 / MME-P 1485.6 / MME-C 291.1 / TextVQA .4952 / MathVista 33.5）。
- **GenEval600**：W4A8 `awq` .30833、`gen_only` .53167；其余 6 config 已在表中。
- **T2I-CompBench**：
  - W3A16 P1：8 config × 2400（8 类 × 300）全部生成完（18:45 `TAIL QUEUE DONE`），**未评分**。
  - W4A8：local4 在跑，20:50 进度 rtn/mbq_scalar 971、channel 1148、t2i_text 1147、awq 824、gen_only 826、t2i_img/both 728（各 /2400）；~68s/图，ETA 11-08 20:30 ~ 11-09 04:15；**未评分**。
- 已知小问题：每个 DPG score 均有 1 张 `partiprompts97.png` 评分失败（1/1065），最终分按 1064 张计算。
- 文档：`TODO.md` 已精简并填表（备份 `TODO.md.bak-20261107_2050`）。
- 当前本机（副机）无队列/GPU 任务；剩余工作只有 T2I 生成与 16 config 评分。
