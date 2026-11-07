# MLLMQ 论文大纲（paper_outline）

> 生成：2026-10-29 · 更新：2026-11-02（重定位为**可配置量化工具包**）
> 方法：Janus-Pro-7B 三模态（理解 vision+caption / 生成 T2I 文字 token / 真实图像 token）**训练后量化（PTQ）**；
> **论文定位（2026-11-02）**：主贡献 = **可配置 PTQ 工具包 MLLMQ-QKit**；主线仍是**生成侧补偿**，
> 按层混合精度（原 GAMQ）降为该工具包的一个**可配置位宽轴**，不再作为独立新方法宣传。
> ⚠️ 标注图例：**[已证实]** / **[待验证]** / **[负面/中性]** / **[待实现]** —— 正文写法必须区分

---

## 0. 论文重定位：可配置量化工具包 MLLMQ-QKit（2026-11-02）

### 0.1 一句话定位

> 把过去几个月探索过的量化轴——**逐通道量化 / T2I 文字 token 补偿 / 真实图像 token 补偿 / 按层混合精度**——收进一个统一、可复现、可配置的 PTQ 工具包；用户按显存/位宽预算选择开关，工具包给出可用配置，并明确告诉用户**哪些开关真的有用、哪些是 null**。

工具包是载体，不是贡献本身。要避免“工程包装”的质疑，必须有三个科学支点：

1. **统一配置空间**：RTN / AWQ / MBQ / MLLMQ / 混合精度都是同一 config 的特例；
2. **有效性地图**：在同一协议下量化每个开关的真实收益（包括大量 null）；
3. **预算感知的自动配置**：给定平均位宽/目标 benchmark，自动选位宽分配和补偿开关。

### 0.2 统一形式化

权重伪量化：`ŵ = s · q(round(w / s))`，其中 `s` 由 base 方法（RTN / AWQ / MBQ scale search）给出。
一个量化配置写成六元组：

```
Q = (B, G, A, C, P, D)
B: bit map        均匀 W3/W4，或逐 block 混合（k 个 block 用 W4）
G: granularity    per-tensor / per-channel / per-group(g)
A: activation     A16 / A8
C: calibration    {理解, 生成文字, 生成真实图像} 的混合比例 + n + teacher forcing
P: compensation   vision / caption / t2i_text / gen_image 的开关、scalar|channel、强度
D: decode         CFG / temperature / seed（评测协议，不是量化器本身）
```

补偿项写在 scale 上：`s_{m,c} ← s_{m,c} · r_{m,c}`，`m` 为模态，`c` 为通道；scalar 模式对整层共享一个 `r`，channel 模式逐通道。混合精度只改 `B`，不改 `s`；二者正交，可组合。

### 0.3 配置 schema（拟实现的 YAML）

```yaml
# configs/janus_w34_balanced.yaml
model:
  name: Janus-Pro-7B
  path: /nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B

calibration:
  data: coco_mixed              # vision + caption + t2i text
  path: data/coco/janus_mixed_calib_256.json
  n: 128
  teacher_forcing: false        # true: 真实图像 -> 576 VQ token（gen_tf）
  seed: 42

weights:
  base: mbq                     # rtn | awq | mbq | mllmq
  bits: 3
  group_size: 128
  zero_point: true
  granularity: channel          # tensor | channel | group
  loss: mae                     # mae | mse(实为 mae_balanced)
  mixed_precision:
    enabled: true
    k_w4: 12                    # 平均 3.40 bit
    select: ratio               # spread | ratio | raw | auto

activations:
  bits: 16                      # A8 当前不可用

compensation:
  vision: true                  # 理解侧图像 token
  caption: true                 # 理解侧文字
  t2i_text: false               # 生成侧 prompt 文字 token
  gen_image: false              # 生成侧真实图像 token（gen_tf）
  mode: channel                 # scalar | channel
  strength:
    t2i_text: 1.0
    gen_image: 1.0

decode:
  cfg: 5.0
  temperature: 0                # <=0 => greedy
  seed: 42
  num_samples: 1
```

### 0.4 开关 → 已有方法映射

| 方法 | `base` | `granularity` | 补偿开关 | `B` | 备注 |
|---|---|---|---|---|---|
| RTN | `rtn` | group | 全关 | 均匀 | 无 scale search；**W3 仅在生成侧崩溃**（GenEval −9~−12pp、DPG −4.3pp、T2I 8 类崩），理解侧不劣；W4/W8 追平 |
| AWQ | `awq` | channel | 全关 | 均匀 | 逐通道 scaling |
| MBQ-block | `mbq` | group | 全关 | 均匀 | block-wise scale search |
| MBQ-scalar | `mbq` | scalar | vision + caption | 均匀 | 以前叫 scalar reweight |
| MLLMQ | `mllmq` | channel | vision + caption (+t2i_text) | 均匀 | 逐通道相对标量的增量 |
| MLLMQ-T2I | `mllmq` | channel | + `t2i_text` | 均匀 | 修复 hook 后的版本 |
| MLLMQ-Img | `mllmq` | channel | + `gen_image`（gen_tf calib） | 均匀 | **无增益**：greedy ON−OFF −3.35pp p=.10 |
| QKit-Mixed-k12 | 任意 | 任意 | 任意 | `k_w4=12` | 原 GAMQ |
| **QKit-Auto** | 搜索 | 搜索 | 搜索 | 自动 | 本文方法贡献 |

### 0.5 有效性地图（当前证据，正文的核心表之一）

| 轴 | 选项 | 结论 | 证据（统一口径） | 默认建议 |
|---|---|---|---|---|
| base | RTN / AWQ / MBQ | W3 RTN **仅在生成侧**崩；AWQ/MBQ 守住 | GenEval 全量 W3：RTN 0.46627 vs 学习型 0.554–0.584；W4 RTN 0.60961（追至 −2~−3pp）、W8 0.64822（追平）；DPG RTN 79.14 vs 其余 82.1–82.7；T2I spatial/numeracy 崩 | `mbq` |
| **评测面** | **理解 / 生成** | **W3 理解侧鲁棒、生成侧崩** | 理解：MMMU RTN 0.3500（n=256）、MathVista 33.5、TextVQA 0.4881 均为 W3 最好/并列，方法差 ≤2pp 且 CI 跨 0；生成：RTN 掉 17.9pp、学习型仅掉 6–10pp | 生成必须学习型 scale；理解 RTN 可用 |
| 粒度 | scalar / channel | **无显著差异** | W4 +0.04pp、W3 −0.28pp（旧全量，CI 跨 0）；n=256 MMMU scalar−block **+2.44pp，CI[−0.67,+5.56]，p=.130**；**mllmq-channel 与 scalar 的 scale 118/120 张量完全相同**，实测 MMMU +0.33pp p=.758 | `scalar`（更省） |
| T2I 文字补偿 | on / off | **GenEval 无增益；DPG 弱正（待复核）** | GenEval：W4 −3.33pp（CI 跨 0）、W3 −0.50pp、greedy ON−OFF +1.98pp p=.36；DPG t2i_text−scalar **+0.97pp CI[−0.08,+2.02]**（单次） | `off`（DPG 专项除外） |
| 真实图像补偿 | on / off | **GenEval 方向偏负；combined DPG 弱正（待复核）** | GenEval greedy ON−OFF −3.35pp p=.10；DPG both−scalar **+1.21pp CI[+0.12,+2.32]**（单次、4 变体多重比较、缺 mbq_block DPG 基线） | `off`（DPG 专项除外） |
| 混合精度 | k=8..16 | **GenEval 显著正，DPG 无收益** | k12 3.40bit：GenEval +4.30pp（CI 2.19–6.40）；DPG −0.36pp；k10 DPG +0.002 | 预算 3.3–3.5bit 时 `on` |
| 激活位宽 | A16 / A8 | **A8 不可用** | W3A8 两方法均崩（0.005 / 0.0017） | `A16` |
| 校准量 | 64 / 128 / 256 | 64–128 接近饱和 | n=64 已接近；差异被噪声淹没 | `n=128` |
| 校准组成 | 纯文本 / mixed | 生成必须含生成样本 | 纯文本 calib 无 T2I 信号 | `mixed` |
| 选层准则 | spread / ratio / raw | spread/ratio 待测 | raw 被深度主导（=前 12 层） | `spread` 或 `auto` |
| 评测协议 | sampling / greedy | 全量重跑噪声 **1.63pp** | 600 图子集 **3.83pp**；评分无漂移 | 机制实验 greedy，主榜多种子 |

> 这张表就是工具包的“说明书”：用户不需要自己试 20 个 config，直接按预算和 benchmark 选默认值。

### 0.6 自动配置策略 QKit-Auto（科学核心）

**输入**：模型、平均位宽预算 `b_avg`（如 3.4 bit）、目标 benchmark、校准集。
**输出**：一个完整 `Q`（bit map + 补偿开关 + 粒度）。

流程：

1. **敏感度画像（1 GPU，≤30min）**
   - 逐 block：`t2i/caption` 梯度比、A3 重建误差、激活量化误差；
   - 逐模态：vision / caption / t2i_text / gen_image 的梯度尺度；
   - 逐位置（可选）：图像 token 的前/中/后 1/3。
2. **位宽分配**
   - 预算固定时，选 k 个 block 用 W4；若允许逐层 3/4/8，则解多选择背包（MCKP）；
   - 目标函数：`max Σ u_i(x_i)  s.t.  Σ bits_i ≤ b_avg · 30`；
   - `u_i` 用画像分数归一化，校准集与验证集按 prompt 切分，避免过拟合。
3. **补偿开关**
   - 按 0.5 的有效性地图初始化；只有在**目标 benchmark 上曾显示正向**且画像显示有 headroom 时才打开；
   - 文字/图像补偿默认关闭——这是当前证据下的正确默认，不是代码缺陷。
4. **噪声感知的验证级联**
   - 25/tag 子集（600 图）粗筛 → 全量 553 × greedy（seed=42）复核 → 需要主榜时再跑标准采样多种子；
   - 明确记录：子集噪声 3.83pp，子集上 ≤3pp 的差异不作为结论。
5. **成功判据（先写死，避免事后挑结果）**
   - 3.40 bit：Auto 在 GenEval 全量 ≥ 手工最佳选层 −1pp，且 ≥ 均匀 W3 +3pp；
   - 3.00 bit：Auto ≥ 均匀 W3 +1.5pp（若有）；
   - 不满足就如实报告“自动策略未超过手工/均匀”，不硬造方法贡献。

### 0.7 预设（Presets）

| preset | 平均 bit | 关键配置 | 适用 |
|---|--:|---|---|
| `debug-fp16` | 16 | 不量化 | 复现上界 |
| `quality-w4` | 4.00 | `mbq` + A16 | 精度优先 |
| `balanced-w34` | 3.40 | `mbq` + `k_w4=12` + `spread` | 默认推荐；GenEval 最优点 |
| `fast-w3` | 3.00 | `mbq-block` 或 `mbq-scalar`，**不要 RTN** | 显存最紧 |
| `gen-comp` | 3.00/3.40 | 打开 `t2i_text` / `gen_image` | 实验性；等 greedy 结论 |
| `auto` | 用户给定 | QKit-Auto 输出 | 新模型/新预算 |

### 0.8 工具包结构（拟）

```
mllmq_kit/
  configs/                 # YAML presets
  mllmq_kit/config.py      # schema + 校验 + manifest
  mllmq_kit/quant/         # rtn, awq, mbq, mixed
  mllmq_kit/compensation/  # modality masks, reweight
  mllmq_kit/calib/         # datasets, sensitivity profiler
  mllmq_kit/policy/        # knapsack / auto config
  mllmq_kit/eval/          # deterministic generation + scoring
  mllmq_kit/cli.py
```

当前 `MLLMQ/` 目录保留为 backend，不做大迁移；工具包层只负责 config、policy、manifest 和统一评测，避免破坏正在跑的实验。

CLI 草案：

```bash
# 1) 用预设量化并导出 scale
python -m mllmq_kit quantize --config configs/janus_w34_balanced.yaml

# 2) 统一评测（greedy 机制口径）
python -m mllmq_kit eval --config configs/janus_w34_balanced.yaml \
  --benchmark geneval --temperature 0 --seed 42

# 3) 自动搜索预算内配置
python -m mllmq_kit search --budget 3.4 --target geneval \
  --out configs/janus_auto_3.4.yaml
```

### 0.9 论文实验矩阵（工具包版）

| 表 | 内容 | 目的 |
|---|---|---|
| T1 主表 | fp16 / W4 / W3 / RTN / AWQ / MBQ-block / MBQ-scalar / MLLMQ / Mixed-k12 / **QKit-Auto**，理解 7 项 + GenEval + DPG + T2I | 统一评测，证明工具包可复现全部方法 |
| T2 开关消融 | 0.5 的每一行 on/off，W4/W3 两档 | 有效性地图；null 也进表 |
| T3 策略对比 | 等 3.4 bit 下：均匀 W3、线性插值参考、手工 spread-k12、手工 ratio-k12、**Auto** | 自动策略是否真的有用 |
| T4 稳健性 | Auto/手工配置 × 3 seeds；greedy vs sampling；n=64/128/256 | 结论是否超过 1.63pp 噪声 |
| T5 效率 | 平均 bit、packed 权重体积、端到端延迟/显存 | 工具包的“实际贡献” |
| T6 跨模型 | 至少一个第二模型（当前只有 Janus-Pro-7B） | 泛化性；没有就写 limitation |

### 0.10 贡献与“不声称”

**可声称（按证据强度）**

1. 统一配置空间：RTN / AWQ / MBQ / MLLMQ / 混合精度都是同一 config 的特例（概念 + 代码）；
2. 统一协议下的**有效性地图**：逐通道 vs 标量、T2I 文字/图像补偿、混合精度、校准量、激活位宽的真实收益与 null；
3. **预算感知的自动配置**（待 T3 验证）：在等平均位宽下接近/超过手工选层，并给出可复现的配置；
4. 开源工具包 + 确定性评测 harness（greedy/seed/shard/评分），解决该领域小样本结论不可复现的问题。

**不声称**

- 不声称提出新的量化算子或新的 rounding 数学；
- 不声称生成侧补偿一定有效：**当前证据是 null**（greedy 8-config 与 n=256 MMMU 均无显著差异；T2I 修复版 W4 −3.33pp CI 跨 0）；也不声称逐通道/生成 token 补偿在 MBQ 之上有增量；
- 不声称“重加权提升理解侧”：理解侧 W3 鲁棒、方法间无显著差异（RTN 与各方法差 ≤2pp、CI 跨 0）；
- 不声称“RTN 是合格的 W3 基线”：RTN 只在理解侧合格，生成侧 W3 崩溃（GenEval −9~−12pp vs 学习型），W4/W8 才追平；
- 不声称 MBQ-scalar 优于 MBQ-block：两者无稳健差异（生成侧 CI 跨 0，MMMU +2.4pp p=.13 不显著）；
- 不声称工具包在所有模型/所有榜上全面更优（DPG 无收益必须写）；
- 不把混合精度当作新方法（大模型已普遍使用），只作为工具包的一个轴。

### 0.11 实现状态与工期

| 项 | 状态 | 备注 |
|---|---|---|
| RTN / AWQ / MBQ / MLLMQ / mixed | **[已有]** | 当前 CLI + scale cache |
| 模态 mask / 补偿开关 | **[已有]** | 目前用 `MLLMQ_*` 环境变量，需改为 config 字段 |
| 混合精度 scale builder | **[已有]** | 支持 spread / top(t2i) / top(ratio) |
| 分片生成 + greedy + seed | **[已有]** | `generate_geneval_shard.py` 已支持 |
| packed 权重导出 | **[已有]** | `--pack_quant`，但无延迟/体积基准 |
| YAML config + manifest | **[待实现]** | 1–2 天 |
| QKit-Auto（画像 + 背包） | **[待实现]** | 2–3 天 |
| 延迟/显存基准 | **[待实现]** | 1 天 |
| 预设 + README + 单测 | **[待实现]** | 1–2 天 |

总工期约 **5–8 个工作日**（不含补跨模型实验）。

### 0.12 审稿风险与应对

| 质疑 | 应对 |
|---|---|
| “只是把已知方法包了一层” | 核心不是开关，而是 0.5 的有效性地图 + 0.6 的自动策略 + 0.9 的等预算对比；没有这些就只是工程 |
| “混合精度不新” | 明确降为工具包的一个轴；贡献在“何时该开、开多少层” |
| “生成侧补偿没效果” | 如实报告 null；把 greedy 8-config 作为决定性证据；若仍 null，主线转“推理时轨迹级补偿” |
| “只有一个模型” | 写 limitation；有条件时补 Janus-Pro-1B 或仅理解侧的第二模型 |
| “只有 pseudo-quant，没有真实加速” | 报 packed 体积；补端到端延迟；明确工具包定位是研究/选型，不是 kernel |
| “小样本结论不可靠” | 我们已有 1.63/3.83pp 噪声实测 + greedy 协议，这是方法学优势 |

### 0.13 与现有数据章节的关系

- 5.3（T2I 修复重测）、5.10 表 A–N 继续作为**证据来源**，不改数据；
- 原 **GAMQ** 全文改称 **mixed-bit allocation** 或 **QKit 的 bit-map 轴**；
- 主线仍写生成侧补偿：当前补偿是**校准期权重缩放**，若 greedy 证明无效，下一版方法改为**推理时轨迹级补偿**，工具包为它预留 `compensation` 配置段。

### 0.14 第二模型选型（泛化性，2026-11-02 讨论稿）

**目的**：回答审稿人“是否只对 Janus-Pro-7B 有效”。最低要求是**一个不同架构/不同家族**的统一理解-生成模型，能定位并量化中间 LM 层；不必追 SOTA，只看相对趋势（W3 vs W4 vs 工具包）。

| 模型 | 生成方式 | LM 主干 | 生成 token 量（成本） | 理解能力 | 权重可得性 | 移植成本 | 推荐度 |
|---|---|---|---|---|--:|--:|---|
| **Janus-Pro-1B** | AR + VQ token | Llama 系 1B | 576（与 7B 同） | 与 7B 同 | 开放（ModelScope/HF） | 低，0.5–1d | ★★★★★ 必做：**同家族 scale 泛化**，代码零改动 |
| **Chameleon-7B** | early-fusion AR + VQ | Llama-2 7B | ≈1024（约 2×） | 中 | **gated，需申请** | 中高，3–5d | ★★★★ 最理想的“不同家族、同为纯 AR” |
| **Emu3-8B** | 纯 next-token AR | Llama-2 风格 8B | **≈4096（约 7×，最慢）** | 中高 | 开放 | 高，4–6d | ★★★★ 概念最干净；全量评测很贵，建议 25/tag |
| **Show-o2-7B** | AR 语言头 + flow matching 生成头 | **Qwen2.5-7B-Instruct** | 27×27 latent + flow steps（中） | 中高 | 开放，Apache 2.0 | 中，3–5d | ★★★★★ **首选第二模型**：7B、开放、LM 主干是标准 Qwen2.5，量化对象清晰 |
| **Show-o2-1.5B** | 同上 | Qwen2.5-1.5B-Instruct | 快 | 中 | 开放 | 中，2–4d | ★★★☆ 小模型/边缘压力测试 |
| **Show-o-1.3B（原版）** | **256 个离散 VQ token + masked discrete diffusion**（非 flow，也非逐 token AR） | Phi-1.5 1.3B | 256（快） | 中 | 开放 | 中，2–4d | ★★★ 离散 token 可测“图像 token 补偿”，但 1.3B 且是 masked 解码，不能替 7B/AR |
| **VILA-U-7B** | 统一 AR 视觉 token | Llama/Vicuna 系 | ≈256（较快） | 中 | 开放（需核实） | 中，3–5d | ★★☆ **不首选**：知名度低、维护/权重不确定性高，审稿人可能也不熟 |
| **Lumina-mGPT-7B** | 纯 AR + VQ token | Llama-2 7B | ≈1024 | 弱（生成强） | 开放 | 中高，3–5d | ★★★ 只能证明生成侧泛化 |
| **BAGEL-7B-MoT** | 双专家 MoT（理解/生成，生成侧为 diffusion/flow，非 AR token） | 两个 transformer 栈 | 取决于生成专家（1024px，多步） | 强 | 开放 | **很高，10–13d** | ★★☆ 名气大，但“中间 LM 层”不是单一栈：只量化理解专家不影响生图，量化生成专家又不是 LM-PTQ；与本文主线不一致 |
| **AnyGPT-7B** | any-to-any AR | Llama-2 7B | ≈1024 | 弱 | 开放 | 中，3–5d | ★★☆ 预算方案；图像质量低 |
| InternVL2-8B | 仅理解 | InternLM2 8B | — | 强 | 本地已有 | 低，1d | ★★ 只能做“配置层适配”广度，不能替 T2I |

> 权重可得性只做了初步检查（HF mirror：Emu3/Lumina 可访问；Chameleon 在当前环境无 token，需申请）。**正式排期前必须核实 repo/许可证/生成脚本**。

**成本估算（4 卡，4 个 config：fp16 / W4 / W3 / W3+工具包）**

| 模型 | GenEval 全量 553 × 4 config | 理解 2–3 项 | 备注 |
|---|---|--:|---|
| Janus-Pro-1B | ~5–6h | ~2h | 必做 |
| Show-o2-1.5B | ~4–6h | ~2h | 小模型 |
| **Show-o2-7B** | ~8–16h | ~2–3h | 7B + flow head；按 432/512 分辨率控制成本 |
| VILA-U / Lumina / Chameleon | ~8–12h | ~2–3h | 1024 token 量级 |
| Emu3-8B | **~2–4 天** | ~3h | 4096 token；建议 25/tag 先筛，再对 W3 关键对跑全量 |

**推荐方案（按时间预算）**

- **A. 1–2 天**：只加 **Janus-Pro-1B**，全量 W3 四配置。定位写成“同家族 scale 泛化 + 工具包可复现”，不声称跨家族。
- **B. 5–7 天（推荐，性价比最高）**：Janus-Pro-1B 全量 + **Show-o2-7B** 的 W3 四配置。7B、Apache 2.0、LM 主干是标准 Qwen2.5，量化对象清晰；跨家族证据成立。注意生成是 flow matching，不是 AR VQ，所以“图像 token 补偿”不直接迁移，但 **LM 层量化 + T2I 文字 token 补偿**可以测。
- **C. 1 周（纯 AR 补充）**：Janus-Pro-1B 全量 + **Emu3-8B**（25/tag 筛 + W3 关键对全量）。纯 AR、不同家族、BAAI 出品；如果审稿人强调“你的补偿是 AR token 机制”，它比 Show-o2 更贴合。
- **D. Chameleon 路线**：只有拿到 gated 权重才考虑；它是最接近 Janus 的纯 AR 统一模型，但申请/移植的不确定性最高。
- **E. BAGEL 路线（不推荐用于本文）**：知名度高、能力强，但双专家 MoT 结构意味着“量化中间 LM 层”要先重新定义量化对象——只量化理解专家不影响生图，量化生成专家又不是本文的 LM-PTQ。移植预计 **10–13 个工作日**（代码/两套 expert 适配 5–6d、校准 mask 2d、生成评测 2–3d、调试 2d），且 W3 评测更贵（1024px、多步）。除非你本来就熟悉 BAGEL 代码，否则性价比低于 Show-o。
- **VILA-U 不推荐**：虽然架构合适，但知名度低、权重/生成脚本维护状态不确定；作为“挡泛化质疑”的第二模型，审稿人认可度不如 Emu3/Chameleon/Show-o。

**关于 Show-o / Show-o2 的定位（重要）**

- **两个 Show-o 的生成机制不同，不要混用**：
  - **Show-o2-7B / 1.5B**：3D causal VAE 连续 latent + **flow matching**（config：`image_latent_dim=16`、`num_diffusion_layers=10`，无 codebook）；
  - **原版 Show-o-1.3B**：**256 个离散 VQ token** + **masked discrete diffusion**（config：`codebook_size=8192`、`num_vq_tokens=256`、`mask_token_id=58497`），既不是 flow，也不是逐 token AR；
  - **Janus**：576 个 VQ token + 逐 token AR next-token。
- **Show-o2-7B 解决了原版 1.3B 的 scale 问题**：LM 主干是 `Qwen2.5-7B-Instruct`（config 已核实），隐藏维 3584，10 层 diffusion/flow head；Apache 2.0，权重开放。它是最适合本文的第二模型：7B、不同家族、LM 层定义清晰。
- **对“补偿”的可迁移性**：
  - Show-o2：**可以测**中间 LM 层量化、T2I 文字 token 补偿、工具包 W3/W4；**不能**测 Janus 式 576 图像 token 补偿（没有离散图像 token）；
  - 原版 Show-o：有 256 个离散图像 token，**可以**测 token 级补偿，但解码是 masked diffusion，不是 AR 累积，不能替 Janus 的 576 步误差累积；
  - 若论文要主张“AR 生成 token 补偿机制跨家族”，必须用纯 AR 模型（Emu3 / Chameleon / Lumina-mGPT）。
- **Show-o2-1.5B** 只作为小模型/边缘压力测试，不能单独承担 7B 跨家族结论。
- **原版 Show-o-1.3B** 已被 Show-o2 取代，除非已有现成代码，否则不建议再投入。

**移植清单（每个新模型都要过一遍）**

1. 找到 transformer block 列表与 block 内 linear 层（对应 `get_blocks` / `get_named_linears`）；
2. 在 token 流里定义模态 mask：vision / caption / t2i_text / gen_image；
3. 校准前向：输入、labels、mask 三项齐全；
4. 生成函数：AR 解码或 diffusion head，能固定 temperature/seed；
5. 评测 prompt 模板：GenEval/DPG 是模型无关的，理解侧需要 per-model 模板；
6. **先跑 fp16 对齐**：同 prompt 的输出/分数与官方脚本一致，再开量化；
7. 记录 manifest：模型 revision、config hash、scale hash、seed。

**报告口径**

- 跨模型只比**相对趋势**（W3 vs W4、均匀 vs 工具包），不比绝对分；
- 第二模型若用 25/tag，必须标注 600 图噪声 3.83pp，且只对 >3pp 的差异下结论；
- 关键结论（W3 是否守得住、工具包是否 ≥ 均匀 W3）尽量用全量 greedy 复核。

---

### 0.15 收敛策略与消融协议（2026-11-02 晚更新）

**GAMQ 冻结**：不再做大规模功能扫描/重跑。以现有 V1 base scale 的结果作为结论性结果：GenEval 全量 k12 **+4.30pp CI[2.19,6.40]**；DPG k12 **−0.36pp**；T2I complex ≈ fp16。GAMQ 仅作为“固定量化器上的 bit-map 轴”，不与新校准 V2 的数字混用；正文注明 scale 版本与评测口径。

**Janus wrapper 门禁（2026-11-02 实测）**：
- 修正 mask/labels 后，中位层 **98–100%** 的 mask 位置有非零梯度，padding 不再淹没梯度；
- 但 per-channel 比值仍然小：**vis/cap=0.016、gen/cap=0.025、t2i/cap=0.016**，低于 InternVL2 的 **vis/cap=0.041**；通道 spread 也更窄（Janus p10–p90 约 3–4×，InternVL2 约 6.5×）。
- 未修复 vs 修复对照（同 4 条理解样本、per-channel）：A 未修复 **vis/cap=0.0305**；B 只修 padding **0.0300**；C 修 labels/caption **0.0163**。→ 理解侧 padding 只占 0.2%，vis/cap 的变化主要来自 **caption 定义（all-text → answer-only）**；修复后比值反而**降低**约 1.9×，T2I 也从 0.052 降到 0.016。
- **有效梯度权重**（token 数 × 每 token 平均 |grad|，相对 answer caption；4 样本、全层/通道）：Janus **vis=0.69×、gen_img=1.29×、t2i_text=0.16×**；InternVL2 **vis=0.90×**。→ per-token 比值 0.016 会误导；按有效权重，Janus vis 与 InternVL2 只差约 1.3×，不是 2.5×；T2I 文字项确实弱（0.16×），但 gen_img 不弱（1.29×），其 on/off null（−3.35pp）因此是有效负结果。
- 结论：修复解决“测量正确性”，**不会放大**重加权 headroom；T2I 文字补偿不值得再追，gen_img/vision 的有效权重不低但 on/off 已 null。**不重建全矩阵**；若继续，只做下面的最小 on/off。

**消融协议（最小权威集）**：
- 理解侧 **1 个**：MMMU（900 题）；
- 生成侧 **2 个**：GenEval（全量 553；小效应先用 25/tag 600 图筛）+ DPG-Bench（1065 prompt）；
- 每个 on/off 对照报告 paired bootstrap CI；全量 <2pp、子集 <4pp 不解读；
- 不做 GAMQ 大批量功能测试，不做 K8–K16 位图扫描（K14 也不做）。

## 候选标题
1. *MLLMQ-QKit: A Configurable Post-Training Quantization Toolkit for Unified Understanding-and-Generation MLLMs*（工具包版首选）
2. *Which Quantization Knobs Matter? A Configurable PTQ Toolkit for Autoregressive Image Generation*
3. *MLLMQ: Modality-Balanced Post-Training Quantization for Tri-Modal Large Models*（旧方法名，可作子方法）
4. *Balancing What Matters: Tri-Modal Calibration for Low-Bit Quantization of Unified Understanding-and-Generation Models*
5. *When Does W3 Survive? Tri-Modal Quantization of Janus-Pro*

---

## Abstract（要点）
> 注：以下是按旧主线（MLLMQ 重加权）写的摘要草稿；2026-11-02 重定位后，**贡献表述以 §0.10 为准**（可配置工具包 + 有效性地图 + 预算感知自动配置）。数据与负结果仍有效。
1. 统一"理解+生成"多模态模型（Janus-Pro 类）落地受限于显存/成本 → 低比特 PTQ 是刚需；
2. 挑战：三模态（视觉、理解文本、生成 T2I 文本）梯度尺度差异大，均一/仅双模态校准会失衡；
3. 提出 **MLLMQ**：block-wise 量化 + **三模态逐通道梯度重加权** + 可选 MAE/MAE-balanced 目标；
4. 结果（诚实版）：
   - **[已证实]** W3 下 **RTN 在生成侧崩溃、学习型校准守住**（GenEval 0.4663 vs 0.554–0.584；DPG 79.14 vs 82.1–82.7；T2I 8 类崩）；**理解侧不受影响**（MMMU 0.3500 并列最高，方法差 ≤2pp、CI 跨 0）→ **崩溃是生成侧特异的**，RTN 在 W4/W8 追平；
   - **[已证实]** W8/W4 与 fp16 不可分；
   - **[负面/中性]** 三模态均衡相对 MBQ(scalar) 的生成侧增益：全量 GenEval 三档 n × W4/W3 均无显著差异；
   - **[负面/中性]** T2I 生成梯度修复后无收益：W4 **−3.33pp（95% CI [−7.17,+0.67]，不显著）**、W3 **−0.50pp（CI 跨 0）**；prompt 噪声 ≤20% 无影响；教师强制生成校准亦无增益；
   - **[已证实]** **生成感知的按层混合精度（GAMQ）**：3.40 bit 达 W4 精度的 96.6%，比均匀 W3 高 **+4.30pp**（95% CI [+2.19, +6.40]），比等平均位宽线性插值高 **1.71pp**；
   - **[负面/中性]** GAMQ 的 GenEval 增益**不迁移到 DPG**：k12 81.99（−0.36pp）、**k10 82.35（≈ 均匀 W3）**；T2I complex 与 fp16 持平（0.3582 vs 0.3583）；
   - **[负面/中性]** pseudo-quant 下无推理效率收益；A8 激活量化无方法区分度（W3A8 两方法均崩）。

---

## 1. Introduction
- 1.1 背景：统一多模态模型（理解+生成）兴起；部署瓶颈
- 1.2 问题：低比特 PTQ 在统一模型上的**模态失衡**——视觉 token / 理解文本 / 生成文本梯度量级不同
- 1.3 现有方法：RTN/AWQ/GPTQ/SmoothQuant（语言）/ MBQ（block-wise）未显式处理生成 T2I 文本模态
- 1.4 贡献（**按证据强度排序，避免过度声称**）：
  1. 三模态校准 + 逐通道梯度重加权框架；
  2. 在 Janus-Pro-7B 上系统评测理解 + 生成（6 方法 × 3 位宽 × 多 benchmark）；
  3. **鲁棒性发现（生成侧特异）**：W3 下 RTN 在生成侧崩溃（GenEval −9~−12pp、DPG −4.3pp、T2I 8 类崩），学习型校准守住；**理解侧对 W3 鲁棒、方法间无显著差异**（MMMU 0.3500 并列最高）→ 重加权保护的是生成路径，不是理解；
  4. **[负面/中性]** 生成侧 T2I 模态均衡无显著增益；修复后 W4 点估计 −3.33pp（95% CI [−7.17,+0.67]，不显著）、W3 −0.50pp；机制解释：t2i/cap≈0.052（⚠️ 该值受 padding/labels bug 影响，见 §0.15；修复后 per-channel 中位 vis/cap≈0.017、t2i/cap≈0.016）、scale 仅移动 5%、prompt 表示噪声 ≤20% 无影响、教师强制生成校准亦无增益；逐 tag 呈 colors/counting/position 之间的**再分配迹象**（子集噪声内，仅作提示）；
  5. **[已证实]** **GAMQ：按层混合精度**，在等平均位宽下显著优于均匀量化（3.40 bit = 96.6% W4 精度，vs W3 +4.30pp，vs 线性插值 +1.71pp），但在 DPG 上不迁移。

## 2. Related Work
- 2.1 语言模型 PTQ（RTN/AWQ/GPTQ/SmoothQuant/QuaRot）
- 2.2 多模态/MLLM 量化（含 MBQ）
- 2.3 统一理解-生成模型（Janus）

## 3. Method: MLLMQ
- 3.1 预备：block-wise 量化、scale 搜索、MBQ Fisher 目标
- 3.2 三模态校准数据：understanding(COCO) + generation(T2I prompt)，`task_type` 标记
- 3.3 **梯度采集与模态 mask**：vision_mask / caption_mask / **t2i_prompt_mask**
- 3.4 逐通道重加权：以 caption 为基准，`t2i_grad / cap_grad` 中位数调制生成相关层
- 3.5 损失模式：MAE / MAE-balanced（**注意：本仓库 `mse` 实为 mae_balanced，须写准**）
- 3.6 与 MBQ 的关系：scalar（vision+caption）→ channel（+T2I）为本文增量

## 4. Experimental Setup
- 4.1 模型：Janus-Pro-7B（三模态）
- 4.2 位宽：W8 / W4 / W3（A16）
- 4.3 基线：RTN / AWQ / MBQ-block / MBQ-scalar / (SmoothQuant、GPTQ 若补)
- 4.4 校准集：`janus_mixed_calib_256`（128 理解 + 128 生成），seed=42 嵌套子集 n∈{64,128,256}
- 4.5 评测：
  - 理解：MME / TextVQA / MMMU / MathVista / POPE(R/P/A)
  - 生成：GenEval；COCO-val（FID / CLIP / HPSv2 / PickScore / ImageReward）；DPG-Bench；T2I-CompBench++
  - **GenEval 两档口径**：子集 25/tag = 6 tag × 25 prompt × 4 = **600 图**；全量 = **553 prompt × 4 = 2212 图**
  - **显著性**：按 tag 分层、prompt 配对、4 图先取均值的 bootstrap（20000 次，95% CI）；overall = 6 tag 均值的均值
  - T2I complex 为每 prompt 1 张、num=1 的自适配口径，**不可与论文绝对值对齐**
- 4.6 实现细节：w_group=128，A100/A800，pseudo-quant 口径声明
- 4.7 噪声声明（必写）：**同一 scale 重跑实测：全量 1.63pp、600 图 3.83pp**；任何 <2pp 的全量差异、<4pp 的子集差异不得作为方法结论，必须用多 seed + 全量

## 5. Results

> ⚠️ **2026-11-01 审计口径警告（必须遵守）**
> 主表中 **mllmq 的 scale 用 10-02 新 calib、n=128**，而 **mbq/awq 的 scale 用 10-01 旧 calib、n=256** → 跨方法数字混入了「校准内容 + 校准量」两个变量，**不可直接跨方法引用**。
> 已决定：**以 current calib + n=128 为唯一干净口径**，正在重跑（`run_clean_comparison.sh`）；完成后 5.1/5.2 的跨方法结论需用新数字替换。组内纵向（如 W3 vs W4 的同一方法）不受影响。
> 另：仓库中 `mllmq_mse` 实际为 **`loss_mode=mae_balanced`**，不是 MSE，正文须写 "MAE-balanced"。

### 5.1 主结果：理解侧 **[已证实]**
- 完整表见 **表 A**（20 config × 7 benchmark，140/140 格）
- **W8/W4 与 fp16 不可分**：MME-P 1543–1595 vs fp16 1587；TextVQA 0.543–0.560 vs 0.558；各方法互差多在 ±1pp
- **W3 整体退化但方法间无分离**：MME-P 1365–1445、TextVQA 0.457–0.488；RTN W3 在理解侧并不比学习型方法差（TextVQA 0.4881 最高）→ **RTN 的崩溃是生成侧特异的**
- **n=256 固定 scale 的 8-config MMMU 复核（2026-11-03，fixed wrapper）**：RTN **0.35000** / mllmq_t2i_img 0.34667 / mllmq_channel_only 0.34444 / mbq_scalar 0.34111 / AWQ 0.33667 / mllmq_t2i_text 0.33333 / mllmq_both 0.32667 / mbq_block 0.31556
  - 配对 bootstrap：rtn−scalar **−1.00pp p=.574**（RTN 仍数值最高）、scalar−block **+2.44pp CI[−0.67,+5.56] p=.130**、mllmq_channel−scalar **+0.33pp p=.758**
  - ⇒ **理解侧没有方法排序，RTN 与任何学习型方法都无显著差异**；scalar 对 block 的 ~2.4pp 方向稳定（n=64 为 +2.00pp p=.245）但两次都 underpowered
- **机制**：理解是**单次前向 + 一个短答案**，没有自回归误差累积，W3 的权重误差可被容忍；因此重加权在理解侧没有可发挥的空间。**不能把“MMMU 平局”外推成“RTN 在全量上也不差”**——RTN 只是理解侧合格的 W3 基线，生成侧不合格（见 5.2）
- **GAMQ k12（3.40bit）理解侧不崩**：MME-P 1477.9 / MME-C 278.6（合计 1756.5）、TextVQA 0.5033、MMMU 0.3411、MathVista 30.7、POPE-R/P/A 0.8830/0.8643/0.8407 → 多数指标介于均匀 W3 与 W4 之间（单次评测，未做 bootstrap）
- **结论**：理解侧不支持"MLLMQ 更优"，只能支持"W4/W8 安全、W3 理解侧整体鲁棒、方法间无显著差异"；**任何“重加权提升理解”的表述都不成立**

### 5.2 主结果：生成侧 **[已证实，但有校准口径警告]**
- GenEval 主表见 **表 B**（19 config，原主表口径）；干净口径/全量见 **表 C、D**
- **W3 RTN 崩溃**：GenEval 0.46627 vs fp16 0.64563（**−17.94pp，保 72.2%**）；学习型方法 0.554–0.584（保 86–90%）
- **W4/W8 与 fp16 不可分**：W4 0.628–0.639、W8 0.639–0.650 vs fp16 0.646
- **COCO-val（表 G）**：W3 RTN FID 34.18 / ImageReward 0.190，其余 W3 27.4–27.8 / 0.57–0.62；W4/W8 与 fp16 的 FID 差异 ≤1
- **DPG-Bench（表 E）**：W3 RTN 79.14 vs fp16 83.41（−4.28pp）；其余 W3 82.1–82.7；W4 82.6–83.7
- **T2I-CompBench++（表 F）**：W3 RTN 全 8 类崩（spatial 0.0869 vs fp16 0.2087、numeracy 0.2398 vs 0.4373）；检测类（UniDet）区分度远大于 BLIP/CLIP 软指标
- **跨榜结论**：三 benchmark 互证"W3 RTN 崩溃、学习型校准守住"；但**学习方法之间无一致排名**（MLLMQ 在 DPG 排 5/6、T2I numeracy 排 5/6）
- **RTN 的资格（必须写进正文，防止误读）**：RTN 是**理解侧合格的 W3 基线**，但**不是生成侧合格的 W3 基线**。W4 时 RTN 已追至 0.60961（比学习型低 2–3pp）、W8 时 0.64822 与学习型（0.639–0.650）持平 → **学习型 scale 的价值集中在 3-bit 生成，而不是“任何位宽 / 任何评测面”**
- **机制（生成侧为何崩）**：生成是 **576 步自回归轨迹**，每一步的权重误差都会累积；W3 group-128 的误差在图像 token 路径上会直接崩坏。MBQ/MLLMQ 的重加权正是把**图像/生成 token 路径的激活尺度**保护起来，所以收益出现在生成侧而理解侧为零。
  - 逐 tag 也吻合：k12 找回的正是 W3 最痛的 color_attr/colors/position/two_object（组合类、长轨迹），counting 不动
- ⚠️ 主表 mllmq 用 current calib/n=128，mbq/awq 用旧 calib/n=256 → **跨方法绝对值不可直接比**；干净口径的全量结论（表 C/N）：
  - MLLMQ W3 0.57119 vs 干净 scalar W3 0.55765（**+1.35pp，CI [−0.66, +3.41]，不显著**）
  - MLLMQ W3 0.57119 vs 干净 block W3 0.56012（**+1.11pp，CI [−0.90, +3.14]，不显著**）
  - 同一 scalar W3 scale 两次全量跑为 **0.57397 / 0.55765（1.63pp 重跑噪声）** → 上述差异均在噪声内
  - **k12 第二次全量复测（s2，2026-11-03）**：**0.59796** vs 首跑 **0.61423 = −1.63pp**（同配置、同协议、无 seed）→ 独立复现 1.6pp 生成噪声；k12 对 W3 的 +4.30pp 约为噪声的 **2.6×**，是当前唯一稳健正结果，三 seed 已完成：mean 0.60796 ± 0.00876
- **不支持的增量**：**“逐通道 / 生成 token 补偿在 MBQ 之上还有增量”不成立**——greedy 8-config（553 图、同 calib、T=0、seed=42）中 MLLMQ 与 MBQ-block/scalar 的配对 Δ 均 CI 跨 0；n=256 MMMU 上 mllmq_channel−mbq_scalar = +0.33pp p=.758；且两者 scale 118/120 张量相同。该负结果应作为有效性地图的一格明确报告，而不是省略。

### 5.3 三模态重加权的生成侧增益 **[无增益：W4 点估计负但 CI 跨 0，W3 无差异]**
> 🔴 **2026-11-01 18:01 发现**：`GradCacheHook` 的早退条件只检查 vis/cap mask，纯文本生成样本（vis=cap=空）被整条跳过 → **T2I 梯度从未被采集**，`rw_t2i` 恒为 None，scale 搜索中的 T2I 项从未生效。
> ⇒ 之前所有 `mllmq_*` scale 实际上等于"vision+caption 的逐通道重加权"（无 T2I）。
> **已修复并完成重测（2026-11-02 03:14）**：修复后旧 calib 的 T2I 梯度首次出现（t2i/cap≈0.052），W4 76/120 层、W3 78/120 层 scale 移动（mean 5.3%/5.5%）。

**修复版结果（GenEval 25/tag = 600 图，同一 calib、同一 150 prompt）**

| 宽度 | 旧 scale（无 T2I） | 修复版（含 T2I） | Δ | 逐 tag Δ（single / two / counting / colors / position / color_attr） |
|---|--:|--:|--:|---|
| W4 | 0.63500 | **0.60167** | **−3.33pp** | −4 / +1 / −7 / **+4** / **−9** / −5 |
| W3 | 0.56333 | **0.55833** | **−0.50pp** | +3 / −8 / −4 / −6 / +5 / +7 |

- **W4 点估计 −3.33pp（CI [−7.17,+0.67]，p=0.10，不显著）；W3 −0.50pp（CI 跨 0）** → 未触发 ≥1pp 条件，DPG/T2I/全量链条**未执行**（driver 03:14 自动判定）
- ⚠️ **不能说“T2I 有害”**：600 图子集的 CI 跨 0，且同配置子集重跑噪声已达 3.83pp；只能说“无证据支持增益”。greedy 8-config 全量 553 已最终判定：T2I ON−OFF **+1.98pp p=.36**（不显著）、图像 ON−OFF **−3.35pp p=.10**；不再单独跑 fixed 全量
- 逐 tag 迹象：colors +4，但 counting −7、position −9 → 像"换偏好"而非"加信息"；但每 tag 仅 100 图（SE≈4–5pp），**只能作为假说，不能作为结论**
- 结合 Exp2（gen_tf 17% scale 变化也无增益）与 **greedy W3 8-config 全量 553（T=0、seed=42、同 calib）：MLLMQ 各变体 vs MBQ-block/scalar 的配对 CI 全部跨 0** → **GenEval 上没有证据支持生成侧梯度补偿在 MBQ 之上有增量**；T2I ON−OFF **+1.98pp p=.36**（不显著）、图像 ON−OFF **−3.35pp p=.10**（方向偏负）。全量 fixed W4/W3 不再单独跑（结论已被 greedy 8-config 覆盖）
- **DPG 新证据（2026-11-03，n=256 同一 calib，1065 图/1064 配对）**：mbq_scalar 81.827；MLLMQ 四变体 82.163–83.041。逐 prompt 配对 vs scalar：both **+1.21pp CI[+0.12,+2.32]**、t2i_text +0.97pp CI[−0.08,+2.02]、channel +0.64pp CI[−0.51,+1.78]、img +0.34pp CI[−0.75,+1.42]。**仅 combined 的 CI 不跨 0，但为单次生成、4 变体多重比较、且缺 mbq_block DPG 基线** → 只能写“DPG 上出现弱正信号，需重复验证”，不能改写 GenEval 的 null 结论
- 下面保留旧数据（无 T2I）作为 **channel vs scalar（两模态）** 的对照：

- **旧结果（bug 期间，等价于 channel-without-T2I vs scalar）**：GenEval 全量 2212 张，配对分层 bootstrap
  - **W4**：n64 **−1.08pp**（CI −2.83~+0.66）/ n128 **+0.04pp**（CI −1.93~+1.95）/ n256 **+0.78pp**（CI −1.11~+2.69）
  - **W3**：n128 **−0.28pp**（CI −2.31~+1.78）/ n256 **−0.65pp**（CI −2.60~+1.32）
  - ⇒ 全部 CI 跨 0：**逐通道 vs 标量重加权统计等价**（此时 T2I 尚未生效）
- **修复版诊断**（仅说明信号强度，不代表效果）：
  - 旧 calib：t2i/cap = 0.052（attn）/ 0.053（mlp），76/120 层移动
  - 教师强制 gen_tf：gen/cap = 0.101，120/120 层移动，scale mean 17.3%
  - Exp2 用 gen_tf 校准：off 0.61000 / ×1 0.60667 / ×2 0.60500（Δ −0.33 / −0.50pp，CI 跨 0）→ **更强的生成信号也没有增益**
- **Exp1（prompt 注噪，独立有效）**：eps 5% 0.65833 / 10% 0.67083 / 20% 0.67083 / 40% 0.58750（n=240）
  - 20% vs 5%：+1.25pp（CI −4.17~+6.67，不显著）；40% vs 5%：**−7.08pp**（CI −13.33~−1.25，显著）
  - ⇒ prompt 表示在 ≤20% 扰动下不是量化瓶颈；40% 才影响生成
- 旧"三重否定"中 **A13/A14 失效**（它们操作的 T2I 项本来就没生效），已作废；Exp1 结论仍成立。

### 5.3b ★按层混合精度（GAMQ，2026-11-01）**[已证实，但存在跨榜 trade-off]**
- **动机**：W3 的退化集中在组合类（position/color_attr/counting 掉 10–13pp），而均匀 W4 无退化 → 用"部分层 W4、其余 W3"换取更优的 accuracy–average-bit 前沿
- **实现**：分别取 W4 / W3 校准得到的 per-layer smooth scale，按层合并成 mixed scale（`build_mixed_scale.py`）；推理时逐 block 用不同位宽伪量化（`pseudo_quantize_model_weight_mixed`）
- **选层**：spread 模式（30 层中均匀间隔选 k 层保 W4），k=12 → 平均 **3.40 bit**
- **GenEval 扫描（25/tag=600 图）**：

| 配置 | 平均位宽 | W4 层数 k | GenEval 25/tag | Δ vs W3 | GenEval 全量 | DPG | T2I complex |
|---|--:|--:|--:|--:|--:|--:|--:|
| W3 均匀 | 3.00 | 0 | 0.56333 | 0 | 0.57119 | 82.348 | 0.3482 |
| mixed k8 | 3.27 | 8 | 0.59667 | +3.33 | — | — | — |
| **mixed k10** | **3.33** | **10** | **0.62000** | **+5.67** | — | **82.350** | — |
| **mixed k12** | **3.40** | **12** | **0.61000** | **+4.67** | **0.61423** | **81.992** | **0.3582** |
| mixed k14 | 3.47 | 14 | 0.60833 | +4.50 | — | — | — |
| mixed k16 | 3.53 | 16 | 0.60667 | +4.33 | — | — | — |
| W4 均匀 | 4.00 | 30 | 0.63500 | +7.17 | 0.63603 | 82.648 | 0.3583 |

- **GenEval 全量（2212 图）**：mixed k12 = **0.61423**，W3 = 0.57119，W4 = 0.63603
  - vs W3：**+4.30pp，95% CI [+2.19, +6.40]（显著）**
  - vs W4：−2.18pp（少 0.6 bit，符合预期）
  - 等平均位宽线性插值预期 0.59713 → **实测高出 +1.71pp**；达到 W4 精度的 **96.6%**（W3 仅 89.8%）
  - k12 vs MBQ-scalar W3 全量：**+4.03pp，CI [+1.79, +6.29]（显著）**
- **逐 tag（k12 vs W3，全量）**：color_attr **+8.3**、colors **+6.7**、two_object **+5.0**、position **+4.0**、single_object +2.2、counting **−0.3**
  → 找回的正是 W3 最痛的属性/颜色/位置组合类，机制上与"生成路径误差累积"一致
- **k10 vs k12**：25/tag **+1.00pp，CI [−2.83, +4.83]（不显著）**；k10 vs MBQ-block W3 **+3.33pp，CI [−0.33, +7.17]**（p=0.083，不显著）→ 子集无法区分 k10/k12
- **跨榜 trade-off（必须写）**：
  - DPG-Bench k12 = **81.992**（−0.36pp vs 均匀 W3 82.348）；**k10 = 82.350，与 W3 完全持平（+0.002pp）**
    → **DPG 上 GAMQ 无任何收益，且与 k 选择无关**；换 k 不能救 DPG
  - T2I complex k12 = **0.358228**，与 fp16 **0.3583** 持平，高于均匀 W3 **0.3482**（+1.00pp；单次、num=1，噪声未量化）
  - COCO 5k：已生成 **5000/5000**，评分待本机补（缺 hpsv2/ImageReward）
  - 理解侧 7 benchmark：副机因缺少 `--model_path` 误载 InternVL2 而失败；**本机 2 卡已修正并重跑完成（2026-11-02 17:08，7/7）**：MME 1756.47（P 1477.90 / C 278.57）、TextVQA 0.50328、MMMU 0.34111、MathVista 30.7、POPE R/P/A = 0.8830/0.8643/0.8407；数据源 `output/janus-pro-7b/eval_mixed/`，已并入 score.json 与表 A
  → 论文只能声称 **"GenEval 组合性收益 + T2I 不退化 + DPG 无收益"**，不能声称全面更优
- **统计稳健性**：全量 GenEval 的同配置重跑噪声实测 **1.63pp**、600 图子集 **3.83pp**（表 N）；k12 的 +4.30pp 约为全量重跑噪声的 2.6×，是当前唯一稳健的正结果；**s2 复测 0.59796（−1.63pp）、s3 在跑**，最终按 3 seed 报告 mean±sd
- **[已完成]** COCO 5k 评分（k12：FID 28.327 / CLIP 0.6629 / HPSv2 0.2648 / PickScore 21.605 / ImageReward 0.691）；理解侧 7 项 7/7
- **[待验证]** top/late 选层；选层准则改为生成敏感度（A3 误差 × T2I 梯度）
- **写作定位**：作为当前唯一在 GenEval 上统计显著的正向方法贡献（accuracy–average-bit Pareto），但需在同一节报告 DPG 回退；MLLMQ 的 T2I 重加权作为机制分析章节

### 5.3c 相关负结果（必须进 Limitations 或补充材料）
- **激活量化 A8（W4A8/W3A8）**：W4A8 = mllmq 0.48667 / scalar 0.48833（等价，但比 W4A16 掉 **14.8pp**）；W3A8 **两者都崩**（0.00500 / 0.00167，生成抽象纹理）→ 通用实现失效，非方法差异；WALinear 慢 ~4×
- **解码时补偿（CFG 重扫，n=240）**：fp16 5.0→5.5 **+2.08pp**；mllmq_w3 最优在 4.0（5.0→4.0 +0.83pp）；scalar_w3 最优在 5.5（+1.25pp）→ **最优 CFG 无一致方向，不支持"量化模型更受益于解码补偿"**
- **教师强制生成校准（Exp2）**：gen_tf 替代纯文本生成样本，×1/×2 相对 off **−0.33 / −0.50pp**（CI 均跨 0）→ 更强的生成侧信号也无收益
- **A13（T2I 剂量 ×2/×3）与 A14（去 T2I）作废**：均在 hook 修复前采集，T2I 项为空，不可引用

### 5.4 校准效率 **[已证实（弱）]**
- n∈{64,128,256} 完整表见 **表 L**：n=64 即接近饱和；MBQ/LMMQ 同饱和
- 但同一 n=128 的 mllmq_w3 两次评测为 **0.5650 / 0.5917（差 2.67pp）** → 600 图口径下校准量差异被噪声淹没
- **同一 scale 重跑**（表 N）：scalar W3 全量 **0.57397 / 0.55765（1.63pp）**；block W3 25/tag **0.58667 / 0.54833（3.83pp）** → 噪声独立于校准量
- 结论：校准量饱和是**共有性质**，非 MLLMQ 独有；不能用来支撑方法优越性

### 5.5 消融
- 5.5.1 重加权模式：channel / scalar / none → 全量 GenEval 无显著差异（表 I）；**n=256 MMMU 上 channel−scalar = +0.33pp p=.758**，且两者 scale 118/120 张量完全相同 → 当前实现下“逐通道”与 scalar 实为同一方法（待审计）
- 5.5.1b MBQ 内部 block vs scalar：**无稳健赢家**。生成侧全量 block 名义 +1.21/+1.03/+0.03pp（W3/W4/W8，均 ≤1.63pp 噪声）；DPG +0.19pp；MMMU scalar 名义 +2.44pp p=.130（n=256）/ +2.00pp p=.245（n=64）→ 只能报告“无显著差异”，不能选边
- 5.5.2 重加权模态：none → vision+caption → +T2I（**无增益：W4 −3.33pp 但 CI 跨 0；W3 −0.50pp**）
- 5.5.3 loss_mode：mae / mae_balanced（仓库误名 `mse`）
- 5.5.4 group size / 激活量化：A8 失效（见 5.3c）
- 5.5.5 W2 超低比特（未做）

### 5.6 效率与部署 **[负面/中性]**
- pseudo-quant：fp16 11.14 / mae_w4 11.46 / mae_w3 11.71 / mbq_w4 12.22 s/img，显存 ~15.3GiB 持平
- **诚实声明**：无推理收益；真实部署收益需打包 int32（见 5.7）

### 5.7 真实打包量化（若补）
- pack_quant 后延迟/显存/吞吐

### 5.8 分析

#### 5.8.1 生成侧敏感、理解侧鲁棒：机制与证据（正文建议单独成节）
**论点**：W3 的损害是**生成侧特异**的；理解侧对 W3 鲁棒，重加权在那里没有可发挥空间。

**证据（相对 fp16）**
| 评测面 | W3 RTN | W3 学习型 | W4 RTN | W8 RTN | 结论 |
|---|---|---|---|---|---|
| 理解 MMMU | 0.3500（最高/并列） | 0.3156–0.3444 | 0.3756 | 0.3878 | 方法差 ≤2.5pp，无排序 |
| 理解 MME-P / MathVista / TextVQA | 1409 / 33.5 / 0.4881 | 1401–1445 / 28.8–30.3 / 0.474–0.480 | — | — | RTN 不劣，甚至部分最高 |
| 生成 GenEval 全量 | **0.46627** | 0.554–0.584 | 0.60961 | 0.64822 | W3 RTN 掉 17.9pp，学习型只掉 6–10pp；W4/W8 追平 |
| 生成 DPG | **79.14** | 82.35–82.74 | 83.55 | — | RTN −4.28pp，学习型 −0.7~−1.1pp |
| 生成 T2I spatial/numeracy | 崩 | 守住 | — | — | RTN 全 8 类崩 |

**机制解释（写作模板）**
1. **生成是 576 步自回归轨迹**：每一步的权重误差都会累积；W3 group-128 的误差在**图像 token 路径**上会直接崩坏。MBQ/MLLMQ 的重加权把图像/生成 token 路径的激活尺度保护起来，所以收益出现在生成侧。
2. **理解是单次前向 + 一个短答案**：没有误差累积，W3 本身可承受；重加权能带来的增益自然消失。
3. **逐 tag 互证**：k12 找回的恰是 W3 最痛的组合类（color_attr/colors/position/two_object），counting 不动 → 与“长轨迹误差累积”一致。

**写作红线**
- ✅ 写：“W3 学习型校准守住生成；W4/W8 与 fp16 不可分；RTN 仅在 3-bit 生成崩溃。”
- ✅ 写：“理解侧对 W3 鲁棒，方法间无显著差异。”
- ❌ 不写：“重加权提升理解侧。”
- ❌ 不写：“RTN 在全量上也不差。”——MMMU 平局不能外推；RTN 只是理解侧合格的 W3 基线。
- ❌ 不写：“逐通道 / 生成 token 补偿在 MBQ 之上有增量。”——GenEval greedy 8-config 与 n=256 MMMU 均无显著差异；DPG 仅有 MLLMQ-both 的 **+1.21pp 单次弱正**（CI 下界 +0.12，4 变体多重比较、缺 mbq_block 基线），只能写“待复核”。

#### 5.8.2 模态/层敏感度（若补）
#### 5.8.3 逐层量化误差分解（若补）
#### 5.8.4 配比鲁棒性（若补）

### 5.9 定性结果
- 同 prompt 的 fp16/W4/W3 对比网格；失败案例

### 5.10 完整数据表（论文正文/补充材料直接可用）

> 数据来源：`output/score.json`（2026-11-02 19:11，已并入 GAMQ k12 理解侧 7 项），全部从原始 `results.jsonl` / `*_results.txt` 重算，不读二手汇总。
> 使用建议：正文放 **表 B（GenEval 主表）、表 E（DPG）、表 F（T2I）、表 H（GAMQ）、表 I/J（显著性与负结果）、表 N（噪声）**；
> 表 A（理解侧全表）、C/D（全量 GenEval）、G（COCO）、K/L/M 放补充材料。
> 所有 600 图子集结果必须带 "±4pp 噪声" 的脚注；全量结果带 "±1.6pp"；bootstrap 为按 tag 分层、prompt 配对的 20000 次重采样，**不覆盖生成随机性**。

### 表 A. 理解侧主表（20 config × 7 benchmark，140/140 格）

| config | MME-P | MME-C | TextVQA | MMMU | MathVista | POPE-R | POPE-P | POPE-A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `awq_w3` | 1406.3 | 263.9 | 0.4835 | 0.3378 | 29.6 | 0.8523 | 0.8397 | 0.8263 |
| `mbq_block_w3` | 1444.6 | 286.1 | 0.4795 | 0.3100 | 29.9 | 0.8730 | 0.8590 | 0.8380 |
| `mbq_scalar_w3` | 1400.9 | 272.5 | 0.4768 | 0.3467 | 28.8 | 0.8713 | 0.8617 | 0.8403 |
| `mllmq_mae_w3` | 1429.4 | 277.5 | 0.4739 | 0.3322 | 30.3 | 0.8473 | 0.8400 | 0.8290 |
| `mllmq_mse_w3` | 1365.1 | 277.1 | 0.4566 | 0.3189 | 29.3 | 0.8523 | 0.8453 | 0.8323 |
| `rtn_w3` | 1409.4 | 245.0 | 0.4881 | 0.3500 | 33.5 | 0.8677 | 0.8507 | 0.8347 |
| `gamt_k12_w3` (3.40bit) | 1477.9 | 278.6 | 0.5033 | 0.3411 | 30.7 | 0.8830 | 0.8643 | 0.8407 |
| `awq_w4` | 1547.8 | 285.0 | 0.5464 | 0.3578 | 34.8 | 0.8507 | 0.8400 | 0.8290 |
| `mbq_block_w4` | 1574.6 | 297.9 | 0.5458 | 0.3567 | 33.7 | 0.8477 | 0.8407 | 0.8313 |
| `mbq_scalar_w4` | 1543.5 | 263.9 | 0.5436 | 0.3700 | 35.3 | 0.8503 | 0.8417 | 0.8303 |
| `mllmq_mae_w4` | 1584.7 | 282.5 | 0.5483 | 0.3622 | 38.4 | 0.8450 | 0.8377 | 0.8270 |
| `mllmq_mse_w4` | 1575.1 | 281.1 | 0.5468 | 0.3656 | 35.0 | 0.8483 | 0.8383 | 0.8273 |
| `rtn_w4` | 1558.6 | 267.1 | 0.5419 | 0.3756 | 34.9 | 0.8420 | 0.8343 | 0.8247 |
| `awq_w8` | 1595.3 | 267.5 | 0.5582 | 0.3856 | 35.9 | 0.8560 | 0.8487 | 0.8387 |
| `mbq_block_w8` | 1579.6 | 261.4 | 0.5572 | 0.3833 | 35.7 | 0.8560 | 0.8483 | 0.8387 |
| `mbq_scalar_w8` | 1589.7 | 274.3 | 0.5568 | 0.3833 | 36.2 | 0.8560 | 0.8480 | 0.8380 |
| `mllmq_mae_w8` | 1592.8 | 269.3 | 0.5603 | 0.3933 | 35.4 | 0.8550 | 0.8470 | 0.8373 |
| `mllmq_mse_w8` | 1588.3 | 259.3 | 0.5586 | 0.3922 | 35.8 | 0.8563 | 0.8487 | 0.8390 |
| `rtn_w8` | 1588.8 | 269.3 | 0.5579 | 0.3878 | 35.8 | 0.8570 | 0.8490 | 0.8387 |
| `fp16` | 1587.2 | 267.5 | 0.5577 | 0.3856 | 35.9 | 0.8563 | 0.8483 | 0.8387 |

> `gamt_k12_w3` 为 GAMQ k12（`janus_mixed_w34_spread_k12`，3.40bit）理解侧重跑，2026-11-02 17:08 完成；
> 与表 A 其余行同协议（MME 两分项、TextVQA exact_match、POPE accuracy），数据源 `output/janus-pro-7b/eval_mixed/`。

### 表 B. GenEval 主表（原主表口径，注意 5.2 的校准警告）

| config | n | single | two | counting | colors | position | color_attr | **Overall** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `awq_w3` | 2212 | 90.6 | 63.6 | 31.9 | 70.2 | 53.8 | 28.0 | **0.56350** |
| `mbq_block_w3` | 2212 | 88.8 | 65.2 | 34.7 | 72.6 | 57.5 | 31.5 | **0.58366** |
| `mbq_scalar_w3` | 2212 | 90.3 | 67.2 | 30.0 | 69.9 | 55.8 | 29.8 | **0.57155** |
| `mllmq_mae_w3` | 2212 | 89.7 | 66.7 | 35.0 | 75.3 | 54.0 | 30.2 | **0.58478** |
| `mllmq_mse_w3` | 2212 | 89.4 | 66.2 | 30.6 | 70.7 | 50.2 | 25.2 | **0.55401** |
| `rtn_w3` | 2212 | 85.6 | 54.8 | 20.9 | 60.9 | 35.8 | 21.8 | **0.46627** |
| `awq_w4` | 2212 | 89.7 | 73.0 | 37.8 | 73.4 | 63.0 | 41.0 | **0.62981** |
| `mbq_block_w4` | 2212 | 91.2 | 74.0 | 40.6 | 75.3 | 62.3 | 39.8 | **0.63855** |
| `mbq_scalar_w4` | 2212 | 91.2 | 72.0 | 38.4 | 75.3 | 63.7 | 36.2 | **0.62821** |
| `mllmq_mae_w4` | 2212 | 91.2 | 74.0 | 34.1 | 72.9 | 64.5 | 40.8 | **0.62904** |
| `mllmq_mse_w4` | 2212 | 91.2 | 72.7 | 37.2 | 72.6 | 63.0 | 40.2 | **0.62837** |
| `rtn_w4` | 2212 | 90.0 | 69.2 | 35.9 | 73.1 | 60.8 | 36.8 | **0.60961** |
| `awq_w8` | 2212 | 91.9 | 74.2 | 40.9 | 76.6 | 66.0 | 39.2 | **0.64817** |
| `mbq_block_w8` | 2212 | 90.6 | 74.2 | 40.6 | 73.9 | 66.0 | 38.2 | **0.63946** |
| `mbq_scalar_w8` | 2212 | 91.2 | 72.0 | 42.5 | 75.0 | 65.0 | 37.8 | **0.63912** |
| `mllmq_mae_w8` | 2212 | 90.0 | 74.2 | 44.7 | 74.2 | 68.0 | 39.0 | **0.65022** |
| `mllmq_mse_w8` | 2212 | 92.5 | 73.7 | 41.2 | 74.7 | 63.7 | 37.8 | **0.63954** |
| `rtn_w8` | 2212 | 90.3 | 73.2 | 45.9 | 74.2 | 67.0 | 38.2 | **0.64822** |
| `fp16` | 2212 | 90.3 | 76.8 | 40.6 | 73.7 | 66.8 | 39.2 | **0.64563** |

### 表 C. GenEval 干净口径 / 全量（2212 图；同一 553 prompt × 4）

| run | n | Overall | 备注 |
|---|--:|--:|---|
| mllmq W4 n64 | 2212 | **0.62224** | `mllmq_mae_w4_n64_full` |
| mllmq W4 n128 | 2212 | **0.63603** | `mllmq_mae_w4_n128_full` |
| mllmq W4 n256 | 2212 | **0.64560** | `mllmq_mae_w4_n256_full` |
| scalar W4 n64 | 2212 | **0.63302** | `mbq_scalar_w4_n64` |
| scalar W4 n128 | 2212 | **0.63565** | `mbq_scalar_w4_n128` |
| scalar W4 n256 | 2212 | **0.63776** | `mbq_scalar_w4_n256` |
| block W4 n64 | 2212 | **0.62164** | `mbq_block_w4_n64_full` |
| block W4 n128 | 2212 | **0.64142** | `mbq_block_w4_n128_full` |
| block W4 n256 | 2212 | **0.62476** | `mbq_block_w4_n256_full` |
| mllmq W3 n128 | 2212 | **0.57119** | `mllmq_mae_w3_n128_full` |
| mllmq W3 n256 | 2212 | **0.55157** | `mllmq_mae_w3_n256_full` |
| scalar W3 n128 | 2212 | **0.57397** | `mbq_scalar_w3_n128_full` |
| scalar W3 n256 | 2212 | **0.55803** | `mbq_scalar_w3_n256_full` |
| **scalar W3 n128（干净新跑）** | 2212 | **0.55765** | `ablation_clean25_full/scalar_w3_full` |
| **block W3 n128（干净新跑）** | 2212 | **0.56012** | `ablation_clean25_full/block_w3_full` |
| GAMQ k12 (3.40bit) | 2212 | **0.61423** | `w34_spread_k12_full` |

> 注 1：**只有 n=128 是当前干净口径**（current calib + n=128）；n64/n256 行为同一 calib 的校准量变体，用于 5.4 的饱和曲线。
> 注 2：同一 scalar W3 n128 scale 的两次全量跑为 **0.57397 / 0.55765（差 1.63pp）** → 全量 GenEval 也有 ~1.6pp 重跑噪声；因此
> MLLMQ W3（0.57119）vs 干净 scalar/block 的 +1.35/+1.11pp **不显著**（CI 分别 [−0.66, +3.41] / [−0.90, +3.14]）。

### 表 D. 全量 GenEval 逐 tag（关键 7 个 run）

| run | single | two | counting | colors | position | color_attr | Overall |
|---|--:|--:|--:|--:|--:|--:|--:|
| mllmq W4 n128 | 90.9 | 69.7 | 37.8 | 73.7 | 64.0 | 45.5 | **0.63603** |
| scalar W4 n128 | 90.3 | 76.5 | 38.8 | 76.1 | 64.8 | 35.0 | **0.63565** |
| block W4 n128 | 90.6 | 76.5 | 42.2 | 71.3 | 64.5 | 39.8 | **0.64142** |
| mllmq W3 n128 | 92.2 | 65.7 | 34.4 | 70.7 | 53.5 | 26.2 | **0.57119** |
| scalar W3 n128 | 86.6 | 66.7 | 36.9 | 71.3 | 54.5 | 28.5 | **0.57397** |
| GAMQ k12 | 94.4 | 70.7 | 34.1 | 77.4 | 57.5 | 34.5 | **0.61423** |

### 表 E. DPG-Bench（single，1065 张，mplug；越高越好）

| config | DPG score | vs fp16 (pp) |
|---|--:|--:|
| `awq_w3` | 82.460 | -0.952 |
| `mbq_block_w3` | 82.738 | -0.674 |
| `mbq_scalar_w3` | 82.546 | -0.866 |
| `mllmq_mae_w3` | 82.348 | -1.064 |
| `mllmq_mse_w3` | 82.075 | -1.337 |
| `rtn_w3` | 79.136 | -4.276 |
| `awq_w4` | 83.478 | +0.066 |
| `mbq_block_w4` | 83.737 | +0.325 |
| `mbq_scalar_w4` | 83.749 | +0.337 |
| `mllmq_mae_w4` | 82.648 | -0.764 |
| `mllmq_mse_w4` | 83.583 | +0.171 |
| `rtn_w4` | 83.549 | +0.137 |
| `fp16` | 83.412 | +0.000 |
| `gamt_k12` (3.40bit) | 81.992 | -1.420 |
| `gamt_k10` (3.33bit) | 待跑 | — |

> 注：DPG 历史结果分三段采集，W3 段混用本机 A800 与对端机器，W4 段全来自对端；同段内自洽，跨段/跨硬件只讲趋势。fp16 仅 1 次评测，噪声带未量化；W4 多点略高于 fp16 属噪声量级。

### 表 F. T2I-CompBench++（13 config × 8 类，每类 300 张）

| config | color | shape | texture | spatial | 3d_spatial | non_spatial | numeracy | complex* |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `awq_w3` | 0.6646 | 0.3502 | 0.5494 | 0.1802 | 0.3149 | 0.3061 | 0.3980 | 0.3470 |
| `mbq_block_w3` | 0.6530 | 0.3492 | 0.5475 | 0.2099 | 0.3162 | 0.3068 | 0.4251 | 0.3554 |
| `mbq_scalar_w3` | 0.6762 | 0.3705 | 0.5375 | 0.1880 | 0.3092 | 0.3070 | 0.3815 | 0.3520 |
| `mllmq_mae_w3` | 0.6651 | 0.3540 | 0.5020 | 0.2138 | 0.3140 | 0.3066 | 0.3646 | 0.3482 |
| `mllmq_mse_w3` | 0.6380 | 0.3605 | 0.4966 | 0.1685 | 0.3074 | 0.3055 | 0.3502 | 0.3515 |
| `rtn_w3` | 0.5160 | 0.3304 | 0.4718 | 0.0869 | 0.2349 | 0.2951 | 0.2398 | 0.2941 |
| `awq_w4` | 0.6674 | 0.3543 | 0.5221 | 0.2118 | 0.3245 | 0.3081 | 0.4542 | 0.3580 |
| `mbq_block_w4` | 0.7012 | 0.3568 | 0.5289 | 0.2158 | 0.3289 | 0.3071 | 0.4441 | 0.3567 |
| `mbq_scalar_w4` | 0.6441 | 0.3366 | 0.4933 | 0.2469 | 0.3161 | 0.3088 | 0.4527 | 0.3622 |
| `mllmq_mae_w4` | 0.6634 | 0.3730 | 0.5327 | 0.2018 | 0.3419 | 0.3088 | 0.4523 | 0.3583 |
| `mllmq_mse_w4` | 0.6707 | 0.3637 | 0.5084 | 0.1967 | 0.3315 | 0.3083 | 0.4455 | 0.3583 |
| `rtn_w4` | 0.5344 | 0.3220 | 0.4630 | 0.1504 | 0.3116 | 0.3050 | 0.3986 | 0.3316 |
| `fp16` | 0.6378 | 0.3571 | 0.4910 | 0.2087 | 0.3272 | 0.3077 | 0.4373 | 0.3583 |
| `gamt_k12` | — | — | — | — | — | — | — | 0.3582 |

\* complex 为每 prompt 1 张、脚本内 num=1 的自适配口径；官方为 10 张/num=10，绝对值不可对齐。

### 表 G. COCO-val（每 config 5000 张；FID 越低越好，其余越高越好）

| config | FID ↓ | CLIP ↑ | HPSv2 ↑ | PickScore ↑ | ImageReward ↑ |
|---|--:|--:|--:|--:|--:|
| `awq_w3` | 27.77 | 0.6618 | 0.2590 | 21.4802 | 0.6165 |
| `mbq_block_w3` | 27.36 | 0.6626 | 0.2582 | 21.4830 | 0.6012 |
| `mbq_scalar_w3` | 27.50 | 0.6617 | 0.2595 | 21.4582 | 0.6176 |
| `mllmq_mae_w3` | 27.71 | 0.6625 | 0.2584 | 21.4459 | 0.5985 |
| `mllmq_mse_w3` | 27.78 | 0.6614 | 0.2576 | 21.4024 | 0.5739 |
| `rtn_w3` | 34.18 | 0.6349 | 0.2407 | 20.8293 | 0.1900 |
| `awq_w4` | 29.45 | 0.6670 | 0.2673 | 21.7377 | 0.7583 |
| `mbq_block_w4` | 30.03 | 0.6676 | 0.2673 | 21.7408 | 0.7434 |
| `mbq_scalar_w4` | 29.93 | 0.6666 | 0.2676 | 21.7500 | 0.7572 |
| `mllmq_mae_w4` | 29.30 | 0.6675 | 0.2671 | 21.7583 | 0.7594 |
| `mllmq_mse_w4` | 29.39 | 0.6684 | 0.2670 | 21.7611 | 0.7466 |
| `rtn_w4` | 29.21 | 0.6611 | 0.2652 | 21.6712 | 0.6750 |
| `awq_w8` | 29.10 | 0.6679 | 0.2688 | 21.7970 | 0.7739 |
| `mbq_block_w8` | 29.10 | 0.6688 | 0.2687 | 21.8016 | 0.7729 |
| `mbq_scalar_w8` | 29.05 | 0.6685 | 0.2683 | 21.7927 | 0.7734 |
| `mllmq_mae_w8` | 29.01 | 0.6679 | 0.2685 | 21.7916 | 0.7829 |
| `mllmq_mse_w8` | 29.16 | 0.6681 | 0.2685 | 21.7890 | 0.7729 |
| `fp16` | 29.09 | 0.6679 | 0.2685 | 21.7895 | 0.7767 |

> 注：W3 的 FID 反而低于 fp16/W4，但 CLIP/HPSv2/PickScore/ImageReward 全面更低 → 低比特图像更平滑，**FID 不能单独作为生成质量结论**。FID 为 5k 张、num_samples=1 的口径，不可与标准 30k FID 对齐；W3 图来自对端机器，跨档只讲趋势。

### 表 H. GAMQ 按层混合精度（W4/W3 分配，spread 选层）

| config | 平均 bit | W4 blocks k | GenEval 25/tag (600) | Δ vs W3 | Δ vs W4 | GenEval 全量 | DPG | T2I complex |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `w34_spread_k8` | 3.2667 | 8 | 0.59667 | +3.33 | -3.83 | — | — | — |
| `w34_spread_k10` | 3.3333 | 10 | 0.62000 | +5.67 | -1.50 | — | 82.350 | — |
| `w34_spread_k12` | 3.4000 | 12 | 0.61000 | +4.67 | -2.50 | 0.61423 | 81.992 | 0.3582 |
| `w34_spread_k14` | 3.4667 | 14 | 0.60833 | +4.50 | -2.67 | — | — | — |
| `w34_spread_k16` | 3.5333 | 16 | 0.60667 | +4.33 | -2.83 | — | — | — |
| `W3 uniform` | 3.0000 | 0 | 0.56333 | 0 | −7.17 | 0.57119 | 82.348 | 0.3482 |
| `W4 uniform` | 4.0000 | 30 | 0.63500 | +7.17 | 0 | 0.63603 | 82.648 | 0.3583 |

线性插值（3.40 bit，W3→W4 全量）= **0.59713**；k12 实测 **0.61423**，高出 **+1.71pp**，达 W4 的 **96.6%**。

### 表 I. 配对分层 bootstrap（prompt 配对，20000 次，95% CI）

| 对比 | scope | Δ (pp) | 95% CI | p | 显著 |
|---|---|--:|---|--:|:--:|
| mllmq_mae_w4_n128_full − mbq_scalar_w4_n128 | full | +0.04 | [-1.93, +1.95] | 0.9681 | — |
| mllmq_mae_w4_n128_full − mbq_block_w4_n128_full | full | -0.54 | [-2.57, +1.45] | 0.6044 | — |
| mllmq_mae_w3_n128_full − mbq_scalar_w3_n128_full | full | -0.28 | [-2.31, +1.78] | 0.8004 | — |
| w34_spread_k12_full − mllmq_mae_w3_n128_full | full | +4.30 | [+2.19, +6.40] | 0.0000 | ✅ |
| w34_spread_k12_full − mbq_scalar_w3_n128_full | full | +4.03 | [+1.79, +6.29] | 0.0006 | ✅ |
| w34_spread_k10 − w34_spread_k12 | 25tag | +1.00 | [-2.83, +4.83] | 0.6015 | — |
| w34_spread_k10 − mllmq_mae_w3a16_ctl | 25tag | +5.67 | [+1.67, +9.83] | 0.0075 | ✅ |
| w34_spread_k10 − mbq_block_w3_n128 | 25tag | +3.33 | [-0.33, +7.17] | 0.0826 | — |
| w34_spread_k12 − mllmq_mae_w3a16_ctl | 25tag | +4.67 | [+0.83, +8.50] | 0.0159 | ✅ |
| mllmq_mae_w4_128 − mbq_scalar_w4_n128 | 25tag | +2.83 | [-1.33, +6.83] | 0.1821 | — |
| mllmq_mae_w4_128 − mbq_block_w4_n128 | 25tag | +0.17 | [-3.67, +3.83] | 0.9360 | — |
| mllmq_mae_w3a16_ctl − mbq_scalar_w3a16_ctl | 25tag | -4.33 | [-8.00, -0.67] | 0.0221 | ✅ |
| gen_tf_x1 − gen_tf_off | 25tag | -0.33 | [-4.17, +3.50] | 0.8705 | — |
| gen_tf_x2 − gen_tf_off | 25tag | -0.50 | [-4.33, +3.33] | 0.8031 | — |
| noise_eps020 − noise_eps005 | noise240 | +1.25 | [-4.17, +6.67] | 0.6451 | — |
| noise_eps040 − noise_eps005 | noise240 | -7.08 | [-13.33, -1.25] | 0.0216 | ✅ |
| mllmq_w3_n128_full − 干净 scalar_w3_full | full | +1.35 | [-0.66, +3.41] | 0.1873 | — |
| mllmq_w3_n128_full − 干净 block_w3_full | full | +1.11 | [-0.90, +3.14] | 0.2722 | — |
| scalar_w3_n128_full（旧）− scalar_w3_full（新） | full | +1.63 | [-0.36, +3.60] | 0.1137 | —（同配置，噪声）|
| mbq_block_w3_n128（旧）− block_w3（新） | 25tag | +3.83 | [+0.33, +7.33] | 0.0344 | ✅（同配置，噪声）|

### 表 J. 不显著 / 负结果（必须写进正文或附录）

| 结论 | Δ (pp) | 95% CI | p | 说明 |
|---|--:|---|--:|---|
| MLLMQ W4 vs MBQ-scalar W4（全量 GenEval） | +0.038 | [-1.93, +1.95] | 0.968 | CI 跨 0 → MLLMQ 在 W4 无优势 |
| MLLMQ W4 vs MBQ-block W4（全量） | -0.540 | [-2.57, +1.45] | 0.604 | block W4 全量 0.64142，MLLMQ 0.63603 |
| MLLMQ W3 vs MBQ-scalar W3（全量） | -0.278 | [-2.31, +1.78] | 0.800 | CI 跨 0；W3 无优势 |
| GAMQ k10 vs k12（25/tag） | +1.000 | [-2.83, +4.83] | 0.602 | k10 点估计最高，但 600 图无法区分 |
| GAMQ k10 vs MBQ-block W3（25/tag） | +3.333 | [-0.33, +7.17] | 0.083 | p≈0.083 |
| MLLMQ W4 vs MBQ-scalar W4（25/tag） | +2.833 | [-1.33, +6.83] | 0.182 | 600 图子集的假阳性示例：全量 +0.04pp |
| MLLMQ W3 vs MBQ-scalar W3（25/tag） | -4.333 | [-8.00, -0.67] | 0.022 | 子集 −4.33pp（CI −8.00~−0.67），全量 −0.28pp（CI 跨 0）→ 子集排名不可靠 |
| Exp2 gen_tf ×1 vs off | -0.333 | [-4.17, +3.50] | 0.871 | 教师强制生成校准未带来增益 |
| Exp2 gen_tf ×2 vs off | -0.500 | [-4.33, +3.33] | 0.803 | 权重加大后仍无增益 |
| Exp1 prompt 噪声 20% vs 5% | +1.250 | [-4.17, +6.67] | 0.645 | ≤20% 无可靠下降；40% 才明显（−7.08pp，CI −13.33~−1.25） |
| DPG GAMQ k12 vs 均匀 W3 | -0.355 | — | — | 81.992 vs 82.348；单次评测，无误差棒 → 只能记作 trade-off |
| **DPG GAMQ k10 vs 均匀 W3** | **+0.002** | — | — | **82.350 vs 82.348；换 k 不解决 DPG 回退** |
| **MLLMQ W3 vs 干净 scalar W3（全量新跑）** | **+1.354** | **[-0.66, +3.41]** | **0.187** | 新跑 scalar=0.55765；不显著 |
| **MLLMQ W3 vs 干净 block W3（全量新跑）** | **+1.107** | **[-0.90, +3.14]** | **0.272** | 新跑 block=0.56012；不显著 |
| **同配置重跑：scalar W3 n128 全量** | **-1.632** | — | — | 同一 scale 两次跑：0.57397 / 0.55765 → 全量噪声 1.6pp |
| **同配置重跑：GAMQ k12 全量（s1/s2）** | **-1.632** | — | — | 0.61423 / 0.59796（2026-11-03，独立复现 1.6pp 生成噪声） |
| **MMMU：MBQ-scalar vs MBQ-block（n=256）** | **+2.444** | **[-0.67, +5.56]** | **0.130** | 0.34111 vs 0.31556；n=64 快探 +2.00pp p=.245，方向两次复现但不显著 |
| **MMMU：MLLMQ-channel vs MBQ-scalar（n=256）** | **+0.333** | **[-1.44, +2.22]** | **0.758** | 0.34444 vs 0.34111；两者 scale 118/120 张量完全相同 |
| **MMMU：RTN vs MBQ-scalar（n=256）** | **-1.000** | **[-4.22, +2.22]** | **0.574** | 0.35000 vs 0.34111；RTN 数值最高但无显著差异 → 理解侧无方法排序 |
| **DPG：MLLMQ-both vs MBQ-scalar（1064 配对）** | **+1.214** | **[+0.12, +2.32]** | — | 83.041 vs 81.827；单次生成 + 4 变体多重比较 + 缺 block 基线 → 弱正，待复核 |
| **DPG：MLLMQ-t2i_text vs MBQ-scalar（1064 配对）** | **+0.973** | **[-0.08, +2.02]** | — | 82.800 vs 81.827；CI 跨 0（边界） |
| **DPG：MLLMQ-channel vs MBQ-scalar（1064 配对）** | **+0.637** | **[-0.51, +1.78]** | — | 82.464 vs 81.827；不显著 |
| **DPG：MLLMQ-t2i_img vs MBQ-scalar（1064 配对）** | **+0.335** | **[-0.75, +1.42]** | — | 82.163 vs 81.827；不显著 |
| **同配置重跑：block W3 n128 25/tag** | **-3.834** | — | — | 同一 scale 两次跑：0.58667 / 0.54833 → 子集噪声 3.8pp |
| **T2I 修复版 W4** | **-3.333** | **[-7.17, +0.67]** | **0.103** | 0.60167 vs 旧 0.63500；**CI 跨 0，不能判定有害** |
| **T2I 修复版 W3** | **-0.500** | — | — | 0.55833 vs 旧 0.56333；无增益 |
| A8 激活量化 | — | — | — | W3A8 0.005 & 0.0017；W4A8 ~0.487（A16 基线 0.563/0.635）；不构成方法结论 |
| A13 / A14 T2I 剂量与去 T2I | — | — | — | 在 hook 修复前采集，T2I 项为空；不可引用 |

### 表 K. 探索性实验（含负结果）

| 实验 | 配置 | Overall | n | 结论 |
|---|---|--:|--:|---|
| Exp2 教师强制生成 | `off` | 0.61000 | 600 | ×1/×2 相对 off −0.33/−0.50pp，不显著 |
| Exp2 教师强制生成 | `x1` | 0.60667 | 600 | ×1/×2 相对 off −0.33/−0.50pp，不显著 |
| Exp2 教师强制生成 | `x2` | 0.60500 | 600 | ×1/×2 相对 off −0.33/−0.50pp，不显著 |
| Exp1 prompt 噪声 | `eps005` | 0.65833 | 240 | ≤20% 无可靠下降；40% −7.08pp |
| Exp1 prompt 噪声 | `eps010` | 0.67083 | 240 | ≤20% 无可靠下降；40% −7.08pp |
| Exp1 prompt 噪声 | `eps020` | 0.67083 | 240 | ≤20% 无可靠下降；40% −7.08pp |
| Exp1 prompt 噪声 | `eps040` | 0.58750 | 240 | ≤20% 无可靠下降；40% −7.08pp |
| A8 激活量化 | `mllmq_w3a8` | 0.00500 | 600 | W3A8 崩、W4A8 −14.8pp |
| A8 激活量化 | `mbq_scalar_w3a8` | 0.00167 | 600 | W3A8 崩、W4A8 −14.8pp |
| A8 激活量化 | `mllmq_w4a8` | 0.48667 | 600 | W3A8 崩、W4A8 −14.8pp |
| A8 激活量化 | `mbq_scalar_w4a8` | 0.48833 | 600 | W3A8 崩、W4A8 −14.8pp |
| A8 激活量化 | `mllmq_w3a16_ctl` | 0.56333 | 600 | W3A8 崩、W4A8 −14.8pp |
| A8 激活量化 | `mllmq_w4a16_ctl` | 0.63500 | 600 | W3A8 崩、W4A8 −14.8pp |
| CFG 重扫 | `fp16_cfg4.0` | 0.65833 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `fp16_cfg4.5` | 0.65417 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `fp16_cfg5.0` | 0.65000 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `fp16_cfg5.5` | 0.67083 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `mllmq_w3_cfg4.0` | 0.62500 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `mllmq_w3_cfg4.5` | 0.61250 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `mllmq_w3_cfg5.0` | 0.61667 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `mllmq_w3_cfg5.5` | 0.60833 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `scalar_w3_cfg4.0` | 0.56667 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `scalar_w3_cfg4.5` | 0.57083 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `scalar_w3_cfg5.0` | 0.59167 | 240 | 最优 CFG 无一致方向 |
| CFG 重扫 | `scalar_w3_cfg5.5` | 0.60417 | 240 | 最优 CFG 无一致方向 |

### 表 L. 校准量消融（25/tag，600 图）

| run | Overall | n |
|---|--:|--:|
| `awq_w4_n128` | 0.63500 | 600 |
| `awq_w4_n64` | 0.61167 | 600 |
| `mbq_block_w3_n128` | 0.58667 | 600 |
| `mbq_block_w3_n256` | 0.58500 | 600 |
| `mbq_block_w3_n64` | 0.57500 | 600 |
| `mbq_block_w3_n64_r2` | 0.57667 | 600 |
| `mbq_block_w4_n128` | 0.63333 | 600 |
| `mbq_block_w4_n64` | 0.63167 | 600 |
| `mbq_scalar_w4_n128` | 0.60667 | 600 |
| `mllmq_mae_w3_n128` | 0.56500 | 600 |
| `mllmq_mae_w3_n128_r2` | 0.59167 | 600 |
| `mllmq_mae_w3_n256` | 0.57500 | 600 |
| `mllmq_mae_w3_n64` | 0.57000 | 600 |
| `mllmq_mae_w4_128` | 0.63500 | 600 |
| `mllmq_mae_w4_256` | 0.61500 | 600 |
| `mllmq_mae_w4_64` | 0.63500 | 600 |
| `mllmq_mae_w4_n128_r2` | 0.64000 | 600 |
| `mllmq_mae_w4_n64_r2` | 0.61167 | 600 |

### 表 M. T2I hook 修复版重测（已完成，2026-11-02 03:14）

| 宽度 | 旧 scale（无 T2I） | 修复版（含 T2I） | Δ | 逐 tag Δ（single/two/counting/colors/position/color_attr） |
|---|--:|--:|--:|---|
| W4 | 0.63500 | **0.60167** | **−3.33pp** | −4 / +1 / −7 / +4 / −9 / −5 |
| W3 | 0.56333 | **0.55833** | **−0.50pp** | +3 / −8 / −4 / −6 / +5 / +7 |

scale 变化：W4 5.3%（76/120 层）；W3 5.5%（78/120 层）。
判定：最高提升 −0.50pp < +1.0pp → **不跑 DPG/T2I/全量**（`no_gain_skip_cross_bench_and_full`）。
结论：T2I 重加权**无证据支持增益**；W4 点估计 −3.33pp（CI [−7.17,+0.67]，不显著），W3 −0.50pp。逐 tag 呈 colors +4 / counting −7 / position −9 的“换偏好”模式，但子集 n=100/tag，方向性只能算提示，不能算结论。后续 greedy 8-config 全量 553 已确认无增益（T2I ON−OFF +1.98pp p=.36；图像 −3.35pp p=.10）。

### 表 N. 干净 W3 全量基线 + 同配置重跑噪声（2026-11-02）

| run | scope | Overall | 说明 |
|---|---|--:|---|
| mllmq W3 n128 | full 2212 | 0.57119 | 旧跑（10-31），单次 |
| scalar W3 n128（新跑） | full 2212 | 0.55765 | 与旧跑同一 scale；**旧跑为 0.57397** |
| block W3 n128（新跑） | full 2212 | 0.56012 | 与 MLLMQ 差 +1.11pp（CI −0.90~+3.14，不显著） |
| scalar W3 n128（旧跑） | full 2212 | 0.57397 | 同配置重跑差 **−1.63pp** |
| block W3 n128（旧，25/tag） | 600 | 0.58667 | 同 scale、同 prompt |
| block W3 n128（副机，25/tag） | 600 | 0.54833 | 同配置重跑差 **−3.83pp** |
| scalar W3（干净 25/tag） | 600 | 0.61667 | 与 scalar W4 0.60167 倒挂 → 子集噪声的直观证据 |
| block W4（干净 25/tag） | 600 | 0.64167 | — |
| awq W4 / W3（干净 25/tag） | 600 | 0.63500 / 0.54333 | — |

> **核心噪声结论**：全量 GenEval 同配置重跑 **1.63pp**，600 图子集最高 **3.83pp**；因此
> (a) MLLMQ W3 vs 干净 scalar/block 全量的 +1.35/+1.11pp **不显著**；
> (b) 任何 <2pp 的全量差异、<4pp 的子集差异都不能作为方法结论；
> (c) GAMQ 的 +4.30pp 约为全量噪声的 2.6×，是目前唯一可能稳健的正结果。

---

## 6. Discussion
- **为何 W3 是分水岭**：GenEval 0.571（mllmq）vs 0.636（W4），主要损失在 color_attr/colors/position/counting；W4 与 fp16 的差距仅 ~1pp，W3 扩大到 ~6pp
- **哪些模态最敏感**：生成侧计数/属性组合 > 检测类 T2I（spatial/numeracy）> 理解侧；理解侧 W3 仅降 4–8%，生成侧降 10–25% → **生成路径误差累积更致命**
- **为什么是生成侧（机制主论点）**：生成是 **576 步自回归轨迹**，每一步的权重误差都会累积，W3 group-128 的误差在图像 token 路径上直接崩坏；理解是**单次前向 + 一个短答案**，没有累积，W3 可承受。MBQ/MLLMQ 的重加权保护的是**图像/生成 token 路径的激活尺度**，因此收益只可能出现在生成侧；理解侧出现平局是机制上可预期的，而不是“实验没测出来”。
- **RTN 的正确资格**：RTN 是**理解侧合格的 W3 基线**（MMMU 0.3500 并列最高、MathVista/TextVQA 不劣），但**不是生成侧合格的 W3 基线**（GenEval 0.46627，比学习型低 9–12pp）。且这一崩溃是 3-bit 特异的：W4 RTN 0.60961、W8 RTN 0.64822，已追至与学习型不可分 → **学习型 scale 的价值集中在 3-bit 生成**。绝不能把 MMMU 平局外推为“RTN 在全量上也不差”。
- **三模态均衡在何种条件下有效**：实测 t2i/cap≈0.052、scale 仅 5% 移动；生成侧信号（gen_tf 17% 移动）也无增益；修复 T2I 后 W4 点估计 −3.33pp（CI 跨 0）、W3 −0.50pp；逐 tag 呈 colors +4 / counting −7 / position −9 的再分配迹象 → **提示“换偏好”，但子集不足以定论**
- **与 MBQ 的本质关系**：全量 GenEval 上 MLLMQ ≈ MBQ-scalar（+0.04 / −0.28pp，CI 跨 0），逐通道相对标量没有带来可测收益；**n=256 MMMU：mllmq-channel − scalar = +0.33pp p=.758，且两者 scale 118/120 张量完全相同** → 当前实现的“逐通道”在理解侧就是 scalar，须审计或合并叙述；真正的正收益来自 **bit allocation（GAMQ）**，而不是模态重加权。
- **MBQ 内部也无稳健赢家**：scalar 对 block 的 MMMU 优势两次复现（n=64 +2.00pp p=.245；n=256 +2.44pp p=.130）但均不显著；生成侧 block 名义高 1.2/1.0/0.0pp（W3/W4/W8 全量），而全量重跑噪声 1.63pp → **block 与 scalar 在生成侧不可区分**。不能写“scalar 优于 block”。
- **GAMQ 的机制假说**：把 W4 容量集中到生成敏感层，等价于在 3.4 bit 下重分配误差；但 DPG 回退说明当前 spread 选层并非对所有生成指标都最优 → 需要按 benchmark 或按误差敏感度选层

## 7. Limitations
- **MLLMQ 未超越 MBQ 的绝对精度**（全量 GenEval CI 跨 0）；主表跨方法数字受校准集/校准量混淆，干净口径已统一为 current calib + n=128
- **GAMQ 跨榜不一致**：GenEval +4.30pp（显著），DPG k12 −0.36pp、**k10 +0.002pp（无收益）**；T2I 仅测了 complex 一项且为 num=1 口径
- **T2I 重加权无增益**：修复 hook 后 W4 −3.33pp（CI [−7.17,+0.67]，不显著）、W3 −0.50pp（CI 跨 0）；不能说“有害”，只能说“测了没有收益”
- **理解侧没有任何方法优势**：W3 七个理解 benchmark 上 RTN 与各学习型方法差 ≤2pp、CI 全跨 0；**不能把 MMMU 平局外推为“RTN 在全量上也不差”**——RTN 的崩溃是生成侧特异的
- **逐通道实现待审计**：`mllmq_channel_only` 与 `mbq_scalar` 的 scale 118/120 张量完全相同（平均相对差 0.07%–0.16%），MLLMQ 在理解侧实际等价于 scalar；若论文坚持“逐通道”粒度，需先确认 reweight 是否沿输入通道真正生效，否则应合并 channel/scalar 叙述
- **MBQ block vs scalar 无稳健赢家**：scalar 的 MMMU 优势两次复现（+2.00pp p=.245；+2.44pp p=.130）但都不显著，900 题下要检出 ~2.4pp 需 >2500 题；生成侧 block 名义高 ≤1.2pp，落在 1.63pp 重跑噪声内 → 只能写“无显著差异”
- **DPG 弱正信号未定论**：MLLMQ-both 对 scalar +1.21pp（CI 下界 +0.12），但为单次生成、4 个变体多重比较、缺 mbq_block DPG 基线；DPG 自身重跑噪声未测。不得据此改写“GenEval 上生成侧补偿无增量”的主结论，需重复验证后再决定是否作为补充正向结果
- **噪声与重复性（实测）**：同 scale 全量重跑 **1.63pp**（scalar 两次；k12 s1/s2 = 0.61423/0.59796 独立复现）、600 图子集 **3.83pp**；GAMQ 已有 3 次生成（s1/s2/s3 = 0.61423/0.59796/0.61170，mean 0.60796 ± 0.00876）；bootstrap 只覆盖 prompt 采样，**不覆盖生成随机性**（同配置重跑在 prompt bootstrap 下仍可"显著"）
- **激活量化不可用**：A8 在当前实现下 W3 崩溃、W4 掉 ~15pp；pseudo-quant 无真实加速
- 单一模型（Janus-Pro-7B）、单一校准集（COCO 系）、无 W2、无打包 int 实现

## 8. Conclusion
- W3 可用的学习型量化；三模态校准框架；未来：打包实现、跨模型、W2

---

## 图表清单（建议）
- Fig.1 三模态校准与重加权示意
- Fig.2 模态梯度分布（可视化）
- Fig.3 W8/W4/W3 × 方法 的 GenEval/理解 折线
- Fig.4 W3 崩坏定性网格
- Fig.5 校准量饱和曲线（mllmq vs mbq vs awq）
- Fig.6（若补）逐层误差热力图
- **Fig.7 GAMQ：accuracy–average-bit 前沿**（W3 均匀 / mixed k8·k10·k12·k14·k16 / W4 均匀 + 线性插值参考线；**建议加一个 DPG 小面板显示回退**）
- **Fig.8 逐 tag 增益柱状图**（k12 vs W3：color_attr +8.3 / colors +6.7 / two_object +5.0 / position +4.0 / counting −0.3）
- **Tab.1 理解主表** = 表 A（20×7，140 格，含 GAMQ k12 行）
- **Tab.2 GenEval 主表** = 表 B（19 config）+ 表 C/D（干净口径全量）
- **Tab.3 COCO** = 表 G；**Tab.4 DPG** = 表 E；**Tab.5 T2I** = 表 F
- **Tab.6 消融** = 表 K/L（Exp1/Exp2/A8/CFG/校准量）
- **Tab.7 效率**（pseudo-quant 延迟/显存）
- **Tab.8 GAMQ 主表** = 表 H（五点扫描 + 全量 + DPG + T2I）
- **Tab.9 显著性与负结果** = 表 I/J（16 组 bootstrap + 13 条 null/negative）
- **Tab.10 T2I hook 修复重测** = 表 M（出分后填）

---

## 写作策略（2026-11-02 15:35 更新：A 已证伪，B 已确认，C 为主线但 DPG 无收益）

### 情形 A：正确对照显示稳定 +2~3pp ❌ **已证伪（2026-11-02 03:14）**
- 修复版 25/tag：W4 **0.60167（−3.33pp，CI [−7.17,+0.67]，不显著）**、W3 **0.55833（−0.50pp，CI 跨 0）** → 未触发 ≥1pp，跨榜/全量未跑；后续 greedy 8-config 全量 553 已判定无增益（T2I +1.98pp p=.36；图像 −3.35pp p=.10）。
- 逐 tag 呈"换偏好"迹象（W4 colors +4、counting −7、position −9；W3 此消彼长），但每 tag n=100，只能作提示；后续 greedy 8-config 全量 553 已判定无增益。
- ⇒ **T2I 生成梯度重加权不作为方法贡献**，只能作为负结果/机制分析。

### 情形 B：正确对照无显著差异 ✅ 已确认（作为机制分析章节）
- **不要**写"MLLMQ 最优"。
- 全量 GenEval：W4 vs scalar +0.04pp、W3 vs scalar −0.28pp（旧跑）；干净 W3 新跑 +1.35/+1.11pp 也**不显著**，且同 scale 全量重跑噪声 1.63pp。
- 主贡献改为：**W3 极端量化下的鲁棒性框架** + **三模态敏感度分析**；
- 机制解释必须用实测值：**t2i/cap≈0.052（不是 ≈1）**、scale 仅移动 5%、prompt 噪声 ≤20% 无影响、教师强制生成（17% scale 变化）也无增益、修复 T2I 后 W4 点估计 −3.33pp（CI 跨 0，不能判定有害）。

### 情形 C：GAMQ 按层混合精度（当前推荐主线）✅（GenEval 正、DPG 无收益）
- 标题/摘要主打 **"生成感知的混合精度量化"**：在等平均位宽下显著优于均匀量化；
- 核心数字：3.40 bit 达 W4 的 **96.6%**，比均匀 W3 **+4.30pp（95% CI +2.19~+6.40）**，比线性插值 **+1.71pp**；
- **必须同段报告**：DPG k12 81.99（−0.36pp）、**k10 82.35（≈ W3）→ 跨榜无收益**；T2I complex 0.3582 ≈ fp16 0.3583；
- 逐 tag 故事：专门找回 W3 最痛的 color_attr/colors/position/two_object（组合类），counting 不动；
- k10 是 25/tag 最高点（0.62000）但与 k12 差 1.00pp（CI 跨 0）；**k10 DPG 已完成，证明换 k 不能救 DPG**；
- 统计上：k12 三 seed 全量 **s1/s2/s3 = 0.61423 / 0.59796 / 0.61170，mean 0.60796 ± 0.00876（range 1.63pp）**；对 W3 0.57119 = **+3.68pp**，对 W4 −2.81pp → 正文用 mean±sd 报告，而不是单次。
- MLLMQ 三模态重加权与 A8/CFG/T2I 负结果作为 **机制分析与 Limitations**，增强可信度；
- 待补：生成敏感度选层；k12 三 seed 已完成（0.60796 ± 0.00876），理解侧 7 项与 COCO 5k 评分均已完成。

### 情形 D：生成侧敏感 / 理解侧鲁棒（机制叙事，与 C 并列写入 Discussion）
**一句话论点**：W3 的损害是生成侧特异的；理解侧对 W3 鲁棒，重加权在那里没有可发挥空间。

**机制（三句话，正文可直接用）**
1. 生成是 **576 步自回归轨迹**：每一步的权重误差会累积，W3 group-128 的误差在**图像 token 路径**上会直接崩坏；MBQ/MLLMQ 的重加权正是把图像/生成 token 路径的激活尺度保护起来。
2. 理解是**单次前向 + 一个短答案**：没有误差累积，W3 本身可承受；重加权能带来的增益自然就消失了。
3. 所以：**RTN 是理解侧合格的 W3 基线，但不是生成侧合格的 W3 基线**；之前的 MMMU 平局不能外推成“RTN 在全量上也不差”。

**证据（全部已实测，勿再引用旧口径）**
- RTN W3：GenEval 全量 **0.46627**（学习型 0.554–0.584）、DPG **79.14**（学习型 82.1–82.7）、T2I 8 类崩；W4 RTN 0.60961、W8 RTN 0.64822 → **追平在 W4/W8**
- 理解侧 W3（n=256 固定 scale，MMMU 900）：RTN **0.35000**、mllmq_t2i_img 0.34667、mllmq_channel 0.34444、mbq_scalar 0.34111、AWQ 0.33667、mllmq_t2i_text 0.33333、mllmq_both 0.32667、mbq_block 0.31556；**所有配对 CI 跨 0**
- 噪声：全量同配置重跑 **1.63pp**（scalar 两次；k12 s1/s2 = 0.61423/0.59796）→ 生成侧 ≤1.2pp 的 MBQ block/scalar 差异不可解读
- 负结果：greedy 8-config 中 MLLMQ vs MBQ 全部 CI 跨 0；n=256 MMMU mllmq_channel−scalar +0.33pp p=.758；**逐通道/生成 token 补偿在 MBQ 之上无增量**

**写法红线**
- ✅ “W3 学习型校准守住生成；W4/W8 与 fp16 不可分；RTN 仅在 3-bit 生成崩溃。”
- ✅ “理解侧对 W3 鲁棒，方法间无显著差异。”
- ❌ “重加权提升理解侧。”
- ❌ “RTN 在全量上也不差。”
- ❌ “逐通道 / 生成 token 补偿在 MBQ 之上有增量。”（DPG 的 +1.21pp 单次弱正只能写“待复核”）

> 一句话：**主线走 C（GenEval 显著、DPG 无收益，必须双榜报告），D 作为机制解释，B/A 作为负结果，A 已证伪。**
---

# 附录 A：MBQ 原文拆解（行文思路 + 实验思路）

> 原文：*MBQ: Modality-Balanced Quantization for Large Vision-Language Models*（arXiv 2412.19509, THU-NICS）
> 仓库：原 MBQ 仓库已删除（源备份在 `backen/MBQ`）；lmms-eval 已迁至 `third_party/lmms-eval`；MLLMQ 代码在 `MLLMQ/`

## A.1 一句话定位
MBQ 的论文 = **"观察到一个现象（不同模态 token 的敏感度差异巨大）→ 一个极简修正（校准损失按敏感度重加权）→ 大范围验证（6 模型 × 2 量化范式 × 6 benchmark）→ 真实加速（自研 kernel）"**。
它没有提新网络结构，卖点是**"发现 + 校准策略修正"**，靠**广覆盖实验**和**真实部署加速**立住。

## A.2 行文结构（逐节作用）

| 章节 | 内容 | 行文作用 |
|---|---|---|
| 1 Introduction | VLM 部署开销；现有 PTQ 只管 LLM | 提出 gap：跨模态差异被忽略 |
| 2 Preliminaries | VLM 推理两阶段、量化格式、channel-wise equalization | 铺垫符号与统一框架 |
| 3 Method | **3.1 敏感度跨模态不同**（Fig.1 展示第 13 层梯度：视觉/语言 token 量级差异）；3.2 MBQ（按模态重加权重建损失）| 核心：先给证据，再给方法 |
| 4 End-to-End Acceleration | 自研 fused W3 GEMV + W4A8 kernel | 把"量化"落到"真加速" |
| 5 Experiments | 见下 | 主体 |
| 6 Conclusion | — | — |
| 7 Related Work | LLM 量化 / 高效 VLM | 放在后面（附录式） |
| 8 Additional | W4/W8A8 结果、case study | 补全覆盖 |

> **行文套路**：Fig.1（现象）→ Sec.3.1（诊断）→ Sec.3.2（一个公式级修正）→ Sec.5（海量实验）→ Sec.4+5.4（真加速）。**先证明"问题真实存在"，再证明"我的修正有效"，最后证明"能落地"。**

## A.3 实验思路（这是重点）

### Setup（5.1）
- **校准**：ShareGPT4V 的 COCO caption，**128 对**图像-描述（同时含视觉+语言建模能力）
- **评测**：OCRBench、TextVQA（文字识别）；VizWiz、SEED-Bench（视觉感知）；ScienceQA、MMMU（视觉推理）；另 LLaVA-bench 做 case study
- **模型**：**3 个家族 × 2 个规模 = 6 个** —— LLaVA-onevision 7B/72B、InternVL2 8B/26B、Qwen2-VL 7B/72B
- **基线**：weight-only(W3) 对比 RTN/GPTQ/AWQ；weight-activation(W4A8) 对比 RTN/SmoothQuant

### Main Results（5.2，Table 1/2）
- 主表 = **6 模型 × {W3, W4A8} × {fp16, 各基线, MBQ} × 6 benchmark**，直接铺满
- 关键数字：W3 最高 **+4.4%**、W4A8 最高 **+11.6%**；LLaVA-ov-7B 平均 60.4(AWQ)→**64.8(MBQ)**；72B 上 AWQ 甚至比 RTN 还差 17%，MBQ 反超
- **叙事**：不只报平均，还报"小模型更敏感（7/8B RTN 掉 9.6% vs 26B+ 只掉 1.5%）"

### Ablation（5.3，Table 3/4/5）—— **四刀，每刀一个设计点**
1. **校准数据**：Pile(纯语言) vs COCO(图文) → 证明"用图文校准 + 模态平衡"才有效
2. **模态平衡**：on/off + **MAE vs MSE** → 明确"MAE 优于 MSE"（W3/W4A8 均 >1%）
3. **ViT 也量化**：ViT(SQ) + VLM(MBQ) → 证明视觉编码器不敏感、可一起量化
4. **纯语言 benchmark（MMLU）**：证明不牺牲语言能力

### Efficiency（5.4，Table 6/7）
- **单 kernel**：fused W3 GEMV 在 4 种矩阵形状上 1.2–5.0× 加速
- **端到端**：ViT(W4A8) 1.15×；VLM prefill/decode 分别用 W4A8/W3 kernel
- **意义**：让"量化"从"精度不掉"升级为"真实更快"

### Additional（8）
- W4 / W8A8 全覆盖；case study

## A.4 这套设计为什么有说服力（可复用的四点）
1. **每个 claim 都有对应实验**：现象（Fig.1）→ 方法（Tab.3 on/off）→ 泛化（6 模型）→ 落地（kernel）。
2. **对照强且全**：RTN/GPTQ/AWQ/SmoothQuant 都在；缺 GPTQ 会被质疑。
3. **消融"一刀一变量"**：校准数据、损失形式、量化范围、语言能力，四个独立维度。
4. **精度 + 效率双证**：只报精度会被问"那到底快没快"。

## A.5 对我们（MLLMQ）的映射与差距

| 维度 | MBQ | MLLMQ 现状 | 差距/启示 |
|---|---|---|---|
| 核心卖点 | 视觉 vs 语言敏感度差异 | **再加生成 T2I 文本模态**（三模态）| 结构上是 MBQ 的自然扩展，逻辑成立 |
| 现象证据 | Fig.1 梯度分布 | 无 | **必须补**：三模态梯度/敏感度可视化（对应 experiment_more 2.1/2.4） |
| 模型覆盖 | 3 家族 × 2 规模 = 6 | **仅 Janus-Pro-7B** | 需补第二/第三模型（至少 InternVL2-8B），否则"单模型"硬伤 |
| 量化范式 | W3（weight-only）+ W4A8（weight-act）| 只有 A16（纯 weight-only）| 需补 **W4A8/W3A8**（experiment_more 3.3） |
| 基线 | RTN/GPTQ/AWQ/SQ | RTN/AWQ/MBQ/（SQ 配置有）| **缺 GPTQ**，需补 |
| 消融 | 校准数据 / MAE-vs-MSE / ViT / 纯语言 | 只有 reweight 模式 + 校准量 | 需补：模态消融（vision→+caption→+T2I）、校准配比、纯语言能力 |
| 效率 | 真 kernel + 端到端 | **pseudo-quant，无加速** | **最大短板**：需真打包量化/加速，否则"量化"论文缺一条腿 |
| Benchmark | 6 个理解类 | 理解 7 + **生成 4**（GenEval/COCO/DPG/T2I）| **这是我们的差异优势**：MBQ 不涉及生成，MLLMQ 覆盖生成侧 |
| MAE vs MSE | 明确 MAE 更优 | **混合**（GenEval mae 好，DPG W4 mse 好）| 不能照抄"MAE 更优"，需如实分层报告 |

## A.6 结论：MLLMQ 要按 MBQ 的"论文骨架"补齐什么

MBQ 之所以能发，是因为 **"现象证据 + 方法极简 + 6 模型 + 4 刀消融 + 真加速 + 生成侧空白"**。
MLLMQ 目前占了 **"生成侧空白"**（差异优势），但缺：
1. ❌ 现象证据（三模态梯度/敏感度图）
2. ❌ 多模型泛化（只有 Janus）
3. ❌ 真实加速（pseudo-quant）
4. ❌ GPTQ 基线
5. ❌ 完整消融（模态/配比/纯语言）

**最小可发表补齐**（按优先级）：
① 三模态敏感度图 + 误差分解（便宜、撑现象）→ ② 第二模型 InternVL2-8B（撑泛化）→ ③ GPTQ 基线 → ④ 打包量化加速（撑效率）→ ⑤ 模态消融 + 配比鲁棒性。

> 其中 **★正确对照（生成侧 mllmq vs mbq_scalar）若成立，则是 MBQ 框架里"新增一个模态带来增益"的直接证据**，正好补上 MBQ 没有的生成维度——这也是我们相对 MBQ 最可能的增量。


---

# 附录 B：MBQ 论文结构与重点细节（2026-11-04 补充）

> 论文：*MBQ: Modality-Balanced Quantization for Large Vision-Language Models*，arXiv 2412.19509，THU-NICS。
> 仓库：原 MBQ 仓库已删除（源备份在 `backen/MBQ`）；lmms-eval 已迁至 `third_party/lmms-eval`；MLLMQ 代码在 `MLLMQ/`。以下为结构化拆解，供写作/对照使用。

## B.1 一句话定位与贡献

> **“发现一个现象（不同模态 token 的敏感度差异巨大）→ 一个极简修正（校准损失按敏感度重加权）→ 大范围验证（6 模型 × 2 量化范式 × 6 benchmark）→ 真实加速（自研 fused kernel）”**。

- 不提出新网络结构；卖点是 **“现象 + 校准策略修正 + 广覆盖实验 + 真实部署加速”**。
- 核心贡献：(1) 揭示 VLM 视觉/语言 token 的量化敏感度差异；(2) 模态平衡重加权（MBQ）；(3) W3 GEMV / W4A8 kernel；(4) 6 模型 × W3/W4A8 的系统评测。

## B.2 章节大纲（逐节作用）

| 章节 | 内容 | 论证作用 |
|---|---|---|
| Abstract | VLM 部署瓶颈；模态敏感度差异；MBQ + 加速 | 一句话讲清“发现 + 方法 + 落地” |
| 1 Introduction | VLM 推理两阶段；现有 PTQ 只针对 LLM；跨模态差异被忽略 | 提出 gap，定义问题 |
| 2 Preliminaries | VLM 推理图、量化格式（W/A、group、zero-point）、channel-wise equalization | 统一符号，铺垫 scale search |
| 3 Method | **3.1 跨模态敏感度差异**（Fig.1：第 13 层视觉 vs 语言 token 梯度量级）；**3.2 MBQ**（按模态重加权重建损失） | 先给现象证据，再给公式级修正 |
| 4 End-to-End Acceleration | fused W3 GEMV、W4A8 kernel、ViT/VLM 端到端 | 把“精度”落到“真实加速” |
| 5 Experiments | Setup / Main / Ablation / Efficiency | 主体：覆盖 + 消融 + 双证 |
| 6 Conclusion | 总结与展望 | — |
| 7 Related Work | LLM 量化 / 高效 VLM | 后置（附录式） |
| 8 Additional | W4/W8A8、case study | 补全覆盖 |

> 行文套路：**Fig.1（现象）→ Sec.3.1（诊断）→ Sec.3.2（一个公式级修正）→ Sec.5（海量实验）→ Sec.4+5.4（真加速）**。

## B.3 方法细节

### B.3.1 现象：模态敏感度不同
- 同一层中，视觉 token 与语言 token 的梯度量级差异大；统一重建损失会被数量多/量级大的模态主导。
- Fig.1 用第 13 层的梯度分布做可视化；这是全文的“现象证据”，也是后续重加权的动机。

### B.3.2 重加权损失（MBQ 的核心公式）
- 以语言 token（caption）为基准，计算视觉/语言梯度比：
  `r = mean(|g_vision|) / mean(|g_caption|)`
- 校准重建损失改为：
  `L = L_ans + r · L_vis`（原始 MBQ 为 **scalar** 重加权；逐层共享一个 r）
- 目的：让视觉 token 的重建误差与语言 token **按敏感度平衡**，而不是被 token 数量或原始量级淹没。
- MLLMQ 的扩展：把 scalar 扩展为 **per-channel**，并加入 T2I 文字 / 真实图像 token 的第三、第四模态项（`t2i/cap`、`gen/cap`）。

### B.3.3 Block-wise scale search
- 逐 block 搜索 scale：候选 `s = x_max^α`（20 个 grid），在重建损失（MAE/MSE）下选最优。
- 量化配置：W3/W4、group-128、zero-point=True；只量化 LM，视觉编码器默认 fp16（消融中再讨论 ViT）。
- 校准：ShareGPT4V 的 COCO caption，**128 对图像-描述**；同时覆盖视觉与语言建模。

### B.3.4 量化范式（关键）
- **W3 weight-only**：权重 3-bit，激活 16-bit；对比 RTN/GPTQ/AWQ。
- **W4A8 weight-activation**：权重 4-bit、激活 8-bit；对比 RTN/SmoothQuant；这是 MBQ 收益最大的 regime（最高 **+11.6%**）。
- 真实加速：fused W3 GEMV（1.2–5.0×）、ViT W4A8（1.15×）、VLM prefill/decode 分别用 W4A8/W3 kernel。

## B.4 实验矩阵

### 模型（3 家族 × 2 规模 = 6）
- LLaVA-OneVision 7B / 72B
- InternVL2 8B / 26B
- Qwen2-VL 7B / 72B

### Benchmark（理解为主）
- 文字识别：OCRBench、TextVQA
- 视觉感知：VizWiz、SEED-Bench
- 视觉推理：ScienceQA、MMMU
- 语言能力：MMLU（证明不牺牲纯语言）
- Case study：LLaVA-bench

### 基线
- W3：RTN / GPTQ / AWQ
- W4A8：RTN / SmoothQuant
- 对照：fp16

### 主结果（论文 Table 1/2）
- W3 最高 **+4.4%**；W4A8 最高 **+11.6%**。
- LLaVA-ov-7B 平均 60.4（AWQ）→ **64.8（MBQ）**。
- 小模型更敏感：7/8B RTN 平均掉 **9.6%**，26B+ 只掉 **1.5%**。
- 72B 上 AWQ 甚至比 RTN 差 **17%**，MBQ 反超。
- 叙事：不只报平均，还报“规模 × 量化范式”的交互。

### 消融（四刀，一刀一变量）
1. **校准数据**：Pile（纯语言）vs COCO（图文）→ 证明图文校准 + 模态平衡才有效。
2. **模态平衡**：on/off；**MAE vs MSE** → 明确 MAE 更优（W3/W4A8 均 >1%）。
3. **ViT 也量化**：ViT(SQ) + VLM(MBQ) → 视觉编码器可一起量化。
4. **纯语言 benchmark（MMLU）**：证明语言能力不牺牲。

### 效率
- 单 kernel：fused W3 GEMV 在 4 种矩阵形状 1.2–5.0×。
- 端到端：ViT(W4A8) 1.15×；VLM prefill/decode 分别用 W4A8/W3 kernel。
- 意义：量化不只是“精度不掉”，还要“真实更快”。

## B.5 这套设计为什么有说服力

1. **每个 claim 都有对应实验**：现象（Fig.1）→ 方法（Tab.3 on/off）→ 泛化（6 模型）→ 落地（kernel）。
2. **对照强且全**：RTN/GPTQ/AWQ/SmoothQuant 都在；缺 GPTQ 会被质疑。
3. **消融“一刀一变量”**：校准数据、损失形式、量化范围、语言能力，四个独立维度。
4. **精度 + 效率双证**：只报精度会被问“那到底快没快”。

## B.6 对 MLLMQ 的直接启示

### 为什么 W4A8 值得补
- MBQ 的最大收益出现在 **W4A8（+11.6%）**，而不是 W3（+4.4%）；我们当前全部是 W3A16，**没有测到 MBQ 最有利的 regime**。
- W4A8 下 RTN/SmoothQuant 更容易崩，方法间差异更可能被放大；这是检验“模态重加权是否真的有用”的更敏感实验。
- 我们已有的 W3A8（Janus）是 0.005（所有方法都崩），说明 A8 的实现/设置很关键；W4A8 是更现实的部署档位。

### 我们与 MBQ 的差异（写作时明确）
| 维度 | MBQ | MLLMQ 现状 |
|---|---|---|
| 重加权粒度 | **scalar**（逐层一个 r） | per-channel（但实现上 channel≈scalar） |
| 模态 | 视觉 + 语言 | 视觉 + 语言 + **T2I 文字 + 真实图像** |
| 生成侧 | 不涉及 | **GenEval/DPG/T2I/COCO**（差异优势） |
| 量化范式 | W3 + W4A8 + kernel | 目前 W3A16；W4A8 待补 |
| 模型覆盖 | 3 家族 × 2 规模 | 仅 Janus-Pro-7B |
| 效率 | 真 kernel | pseudo-quant，无加速 |
| 基线 | RTN/GPTQ/AWQ/SQ | RTN/AWQ/MBQ；**缺 GPTQ** |

### 最小补齐顺序（沿用 A.6）
① 三模态敏感度图 + 误差分解 → ② W4A8/W3A8 范式 → ③ 第二模型（InternVL2-8B）→ ④ GPTQ 基线 → ⑤ 打包量化/真实加速。

> **一句话**：MBQ 的骨架是“现象 → 极简修正 → 广覆盖 → 真加速”；MLLMQ 的差异优势是**生成侧**，但当前缺 W4A8、第二模型、GPTQ 和真实加速。W4A8 是下一个最该补的实验，因为它正是 MBQ 收益最大的 regime。

---

> 以下由 `dialog.md`（2026-11-05 精简归档）迁移，供 Related Work / 竞品定位使用。

# 附录 C：文献调研（2024–2026 多模态 / 生成模型量化）

> 部分条目基于领域知识整理，标注 ⚠️ 的需提交前用 arxiv 正式检索确认。

---

## 9.1 多模态 VLM 量化（VLM / MLLM → LLM Backbone）

**核心问题**：LLM backbone 做量化，ViT 部分通常保留 fp16/INT8。关键难题是校准数据——纯文本 calib 下 LLM head 表现尚可，但视觉路径的 scale 被系统性低估。

| 论文 | 年份 | 出处 | 核心方法 | 与 MLLMQ 关系 |
|------|------|------|---------|---------------|
| **MBQ** (Modality-Balanced Quantization) | 2024 | arxiv 2412.19509 | 模态均衡 scale search，视觉/文本梯度分别收集，取中位数比作为 reweight coefficient | **直接竞品**；MLLMQ = MBQ + per-channel reweight。MBQ 未做 per-channel 粒度，且仅验证理解任务 |
| **Q-VLM** | 2024 | NeurIPS 2024 (arxiv 2410.08119) | W4A4 PTQ，视觉编码器量化误差补偿 | **标杆 baseline**，MLLMQ 必比方法 |
| **Bi-VLM** | 2025 | AAAI 2026 | 二值 PTQ，saliency-aware 混合量化算法 | 超低比特方向，方法交叉度低 |
| **VLM-PTQ** | 2026 | CVPR 2026 | 1B–72B 单 GPU 压缩，SOTA PTQ | 规模覆盖广，方法论可对比 |
| **MASQuant** | 2026 | arxiv 2603.04800 | 模态感知平滑因子学习，解决 Smoothing Misalignment | 思路相关但机制不同（平滑 vs reweight），可引用对比 |
| **Breaking Modality Heterogeneity** | 2026 | arxiv 2605.19929 | 拆解视觉/语言模态低比特量化异质性 | 提出「模态异质性」概念，可引用作为理论背景 |
| **C-PTQ** | 2026 | arxiv 2607.21076 | Fisher 信息矩阵通道敏感度 PTQ | **最接近的方法**，但 Fisher（静态、间接）≠ MLLMQ Gradient Reweight（任务驱动、直接优化量化损失），详见 §9.6.2 |
| **OmniQuant** | 2023 | ICLR 2024 | Per-layer per-token 激活 calibration，交叉熵扰动搜索最优 calib 配置 | 讨论 calib data diversity，可支撑 MLLMQ 的 calib size ablation（256 vs 512） |
| **SmoothQuant** | 2022 | ICML 2023 | Per-channel 平滑因子迁移量化难度至权重侧 | W8A8 经典方法，MLLMQ 可对比其表现 |
| **AQLM** | 2023 | ICML 2024 | Codebook 级极低比特量化（1-2 bit），LLM-only | 推理延迟高，不适用于 VLM + 部署场景 |

**关键共识**：VLM 量化必须用 **multimodal calib**（含图文对），纯文本 calib 导致 ViT 路径退化。MLLMQ 使用的 COCO + ShareGPT4V 恰满足此要求。

---

## 9.2 统一理解 + 生成模型量化（Janus / Emu 等）

**目前空白最大、最有发表价值的方向**。Janus-Pro-7B 同时做理解（VQA / Captioning）和生成（T2I via VQ decoder），两分支梯度敏感度差距极大，且双编码器架构（SigLIP + VQ Tokenizer）为量化带来独特挑战。

| 论文 | 年份 | 出处 | 核心 | 与 MLLMQ 关系 |
|------|------|------|------|---------------|
| **MLLMQ** (本项目) | 2025 | — | 首个对 Janus-Pro-7B 做统一理解+生成的 4-bit 量化；per-channel gradient reweight 平衡 IQA vs T2I 梯度 | ⭐ **核心贡献：Janus 量化是空白，MLLMQ 是第一篇** |
| **Janus-Pro** (DeepSeek) | 2025 | arxiv 2501.17811 | 解耦理解/生成编码器 | 被量化对象，原文不涉及量化 |
| **Janus / JanusFlow** (DeepSeek) | 2024-25 | arxiv / GitHub | 统一多模态理解+生成 | 同类模型，可作扩展实验 |
| **Emu / Emu2** (BAAI) | 2023-24 | arxiv | LLM+SD 扩散头统一模型 | SD 分支可借 Diffusion quantization 方法 + LLM head 用 MLLMQ |
| ⚠️ 暂无 | — | — | **尚未发表 Janus-Pro 量化论文** | MLLMQ 占先发优势 |

---

## 9.3 生成模型 / Diffusion 量化

Janus 的 T2I 分支本质是 diffusion-based generator，有独立且丰富的量化文献。

| 论文 | 年份 | 核心思路 | 精度 | 与 MLLMQ 的关联 |
|------|------|---------|------|----------------|
| **Q-Diffusion** (Li et al.) | 2023 | U-Net per-layer per-channel weight quant + row-wise truncated activation；SD/SDXL → near-zero FID loss | W8W8 | 证明 SD 可用轻量级量化方案；启发 Janus T2I branch 可用轻量 weight-only |
| ⚠️ **DiT-W4A16** | 2024-25 | DiT (Diffusion Transformer) W4A16 量化 | W4A16 | 待确认具体论文，方向相关 |
| ⚠️ **DiffuseQuant** | 2024 | Diffusion weight-only quant | W8/W4 | **待确认**：声称发现 attention O-proj 和 MLP down-proj 最敏感 |
| **SmoothQuant → Diffusion** | 2024 | Per-channel weight tied quant + smoothquant balance ratio for SDXL | W8 | 可对比 Janus T2I branch 用 SQ vs MLLMQ |

**关键交叉洞察**：Diffusion quantization literature 的核心发现——**attention O-projection 和 MLP down-projection 是对量化最敏感的层**——与 MLLMQ 的 per-channel gradient reweight 实验结论收敛在同一组层上，形成跨领域佐证。


---


## 9.4 Survey 论文

| 论文 | 年份 | 出处 | 与 MLLMQ 关系 |
|------|------|------|---------------|
| ⚠️ How to Quantize LLMs (Dong et al.) | 2024 | arxiv（待确认 ID） | LLM 量化综述，VLM 仅 2-3 段，引为背景 |
| **A Comprehensive Survey and Guide to MLLMs** (Liang et al.) | 2026 | MDPI 14(6):125 | 通用 MLLM 综述，非量化专项 |
| **A Survey of Quantization in LLM** (Chen et al.) | 2026 | SciOpen | LLM 量化综述，MLLM 覆盖少 |
| ⚠️ Compression of MLLM | 2024-25 | ICLR/NeurIPS Workshop | 标签后补 |


---

## 9.5 MLLMQ 在文献中的定位

**难度递增排列**：

- **Text-only LLM** [Easy]
  - GPTQ / AWQ / SmoothQuant  ...  AQLM (1-bit) [Hard]

- **VLM (Vision+Text)**
  - MBQ / Q-VLM / **MLLMQ**  ...  C-PTQ / MASQuant

- **Unified (IQA+T2I)** ← **MLLMQ 首个突破**
  - **★ MLLMQ ★** | .... (完全空白)

- **Diffusion**
  - Q-Diffusion / SmoothQuant → SD

**关键定位**：

- MLLMQ 对 Janus-Pro-7B 的 4-bit 量化是领域首次：截至 2026.09，尚无论文对 Janus 系列做量化
- MLLMQ 对 InternVL2-8B 的 W4A16 = **0.4900** (+1.39% vs fp16，09-17 最终复跑)：唯一超 baseline 的 4-bit 方法
- MLLMQ reweight 在统一理解+生成模型上理论收益最大：双任务梯度场不同，reweight 机制最合理
- Diffusion quantization 结论与 MLLMQ hypothesis 一致（O-proj / MLP down-proj 最敏感），跨领域交叉验证

---

## 9.6 TODO 文献验证

### 9.6.1 需重点区分的竞品

#### MBQ (arxiv 2412.19509)

| 维度 | MBQ | MLLMQ |
|------|-----|-------|
| 粒度 | 模态级平衡（vision vs text 两模态） | Per-channel 级 reweight |
| 方法 | 视觉/文本梯度分别收集，取中位数比作为系数 | 反向传播梯度直接学习 channel-wise reweight |
| 任务域 | 仅理解任务 | 理解 + 生成统一评测 |
| 文本 token 敏感度差异 | ❌ 未涉及 | ✅ 理解 vs 生成任务中文本 token 敏感度差异是 MLLMQ 独有论述维度 |

**结论**：MBQ 是模态平衡量化直接竞品，但（1）未涉及 per-channel 粒度；（2）仅针对理解任务，未覆盖统一生成模型中「文本 token 在理解 vs 生成任务间敏感度差异」这一空缺维度，MLLMQ 可填补。

#### C-PTQ (arxiv 2607.21076)

C-PTQ: ∂/∂θ log p(y|x,θ) → 对「模型输出似然」求导 → 问"扰动参数让概率变化多大？" MLLMQ: ∂Loss_quant/∂w_ch → 对「量化损失」求导 → 问"调整 reweight 减少多少量化误差？"

一个是"模型本身的敏感度"，一个是"量化引入的误差敏感度"——两个不同的物理量。


| 维度 | C-PTQ (Fisher) | MLLMQ (Gradient Reweight) |
|------|---------------|---------------------------|
| 评估目标 | 参数对**模型输出似然**的重要性 | 参数对**量化误差**的重要性 |
| 逻辑链条 | 保护重要参数 → 间接减少量化损失 | 直接优化量化损失 |
| 量化针对性 | ❌ 通用敏感度，量化仅为适配 | ✅ 专门为量化设计 |
| Outlier 感知 | ❌ Fisher 不反映数值分布特性 | ✅ 量化 outlier 直接导致 loss 飙升 |
| 与 Diffusion 量化结论一致性 | — | ✅ 与「O-proj / MLP down-proj 最敏感」一致 |

**具体例子**：

通道 A: Fisher 很高（模型很依赖它） 通道 B: Fisher 中等，但数值分布特殊（outlier 多）

C-PTQ: → Fisher(A) > Fisher(B) → 给 A 更多 bit → 但 B 的 outlier 导致量化误差极大！← Fisher 捕捉不到

MLLMQ: → 量化 B 后 loss 飙升 → ∂Loss/∂w_B >> ∂Loss/∂w_A → 给 B 更多 reweight 保护 → 直接针对量化痛点


**结论**：C-PTQ 使用 Fisher Information（∂/∂θ log p(y|x,θ)）评估通道对「模型输出似然」的敏感度，属于通用敏感度度量，间接适配量化；MLLMQ 使用 ∂Loss_quant/∂w_ch 直接优化「量化损失」，本质不同——一个测"参数对模型有多重要"，一个测"参数对量化误差有多敏感"。且 MLLMQ 的敏感层结论（O-proj、MLP down-proj）与 Diffusion quantization 文献一致，进一步佐证合理性。需在 Related Work 中明确对比 Fisher 静态估计 vs 可学习 Gradient Reweight。

#### Breaking Modality Heterogeneity (arxiv 2605.19929)

提出了「模态异质性」概念（vision/language 路径量化难度差异），可引用其定义和定量分析，作为 MLLMQ 处理多模态量化失衡的理论背景支撑。

### 9.6.2 必引 Baseline & 可参考方法

| 论文 | 定位 | 说明 |
|------|------|------|
| **Q-VLM** (NeurIPS 2024) | 📖 必引 baseline | VLM 量化标杆，MLLMQ 需对比 |
| **MASQuant** (arxiv 2603.04800) | 📖 可参考对比 | 模态感知平滑因子学习，与 reweight 思路平行但机制不同 |
| **Bi-VLM** (AAAI 2026) | 📖 可引用 | 二值量化方向，方法交叉度低 |

### 9.6.3 可突出贡献点

| 贡献点 | 状态 | 详情 |
|--------|------|------|
| Janus-Pro 量化 | ✅ 新领域 | **暂无任何 Janus 系列量化论文** |
| per-channel reweight | ✅ 新方法论 | 需与 C-PTQ Fisher 明确区分（§9.6.2） |
| 统一视觉生成模型量化 | ✅ 空白 | 非仅理解任务，涵盖生成维度 |
| Modality-Balanced 术语 | ❌ 已被 MBQ 占用 | 建议避免作为核心 claim 使用 |

### 9.6.4 更新 TODO 汇总

| TODO | 状态 | 建议 |
|------|------|------|
| **per-channel reweight 同期检查** | ⚠️ C-PTQ 最接近 | Related Work 中正面引用并区分：Fisher（静态、间接）vs Gradient Reweight（任务驱动、直接优化量化损失）。强调 MLLMQ 结论与 Diffusion quantization 文献收敛于同一组敏感层 |
| **Janus-Pro quantization 空白** | ✅ 确认空白 | 核心 novelty 亮点 |
| **MBQ 模态平衡竞品** | ⚠️ 需区分 | MBQ 未做 per-channel 粒度；MLLMQ 独有「统一生成模型中文本 token 在理解 vs 生成任务间敏感度差异」论述维度 |
| **Survey 收录** | ✅ 暂无缝综 | 机会窗口：MLLM 量化专项综述 |
| **MMLMQ / MMLLMQ 同名** | ✅ 无冲突 | 名称可用 |
| **Modality-Balanced 同名** | ❌ 已被 MBQ 占用 | 避免使用 |
| **Breaking Modality Heterogeneity** | 📖 可引用 | 引用「模态异质性」概念作为背景 |
| **Q-VLM** | 📖 必引 | 核心 baseline |
| **MASQuant** | 📖 可对比 | 模态感知平滑 vs Reweight |
| **⚠️ 标注论文** | 🔴 需确认 | 文中标记 ⚠️ 的条目提交前务必用 arxiv 检索确认存在性及详细信息 |


---


