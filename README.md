# MLLMQ 复现与迁移指南

> 📖 **三文档联动**：`experiment_status.md`（重启快照+进度）| `notification.md`（本文——命令+FAQ）| `dialog.md`（诊断+坑）
>
> 用途：在新的容器 / 机器上从零复现 **InternVL2-8B W3/W4/W8 × RTN/AWQ/MBQ/MLLMQ** 的量化 + MMMU 评估流程，**完全 paper-equal** 复现。
> 适用范围：1× A100 80GB · Linux · Python 3.10 · CUDA 12.x
> 对账基线：0.4422 / 0.4456 / 0.4589 / 0.4867 (W4A16 MMMU)；0.4833 fp16 reference；MME RTN W4 perc=1652.07 / cog=584.64

---

## 0. 快速 TL;DR

```bash
# 一行依赖
pip install -r /nas3/kj/mllmq/MLLMQ/requirements.txt

# MMMU 评估
bash scripts/run_remaining_evals.sh    # 容错版：单 cell 失败不中断，已完成跳过

# 多 benchmark 扩展（MME / TextVQA），详见 §4.3
bash scripts/run_core4_w4a16.sh        # MME + TextVQA W4A16 四方法串行评测
```

跑完全部量化（scale search + pack）约 3h，MMMU 评估 8 cell 约 6h，MME 每 method ~2.5h。完成后到 §9 查看汇总方法。

---

## 1. 目录与文件全景

```
/nas3/kj/mllmq/                                    ← 项目根
├── MLLMQ/                                         ← 本仓库（量化 + 评估）
│   ├── main_quant_vl.py                            ← ★ 量化入口（4 方法统一）
│   ├── eval_mmmu.py                                ← ★ MMMU 评估入口
│   ├── eval_benchmarks.py                          ← ★ 通用多 benchmark 评估入口（MME/TextVQA/...）
│   ├── tasks_local/                                 ← ★ MME / TextVQA 本地 task yaml（无 token）
│   ├── data/benchmarks_local/                       ← ★ 离线可用的 MME / TextVQA dataset cache
│   ├── qmllm/methods/{rtn,awq,mbq,smoothquant}/    ← 各方法实现
│   │   ├── awq/quantize/pre_quant.py               ← ★ 已新增 run_awq_vl()
│   │   ├── awq/quantize/auto_scale.py              ← ★ 修改：n_bit→n_bits
│   │   ├── awq/quantize/quantizer.py               ← ★ 修改：n_bit→n_bits（同 typo，2026-09-11 补齐修复）
│   │   └── mbq/quantize/pre_quant.py               ← 已加 InternVLChatModel 分支
│   ├── scripts/run_full_matrix.sh                  ★ 一键跑全部 4×2 评估
│   ├── scripts/run_experiment_vl.sh                ★ 单组量化+评估流水
│   ├── scale_cache/{awq,mbq,mllmq,rtn}/            ← 已保存的 scale (.pt)，避免重复 search
│   └── requirements.txt                            ★ 完整 pip 列表（§2）
│
├── MBQ/                                            ← MBQ 原仓库（含修改项）
│   ├── qmllm/                                       ← 与 MLLMQ/ 同源，验证用
│   └── 3rdparty/lmms-eval/                          ← ★ 内含评估框架 + 补丁
│       └── lmms_eval/models/internvl2.py            ← ★ 第 142 行起打补丁（§5）
│
├── checkpoints/InternVL2-8B/                       ← 模型权重（OpenGVLab/InternVL2-8B，bf16，~17 GB）
├── data/
│   ├── coco/{sharegpt4v_coco_calib.json, train2017/}  ← COCO 校准集
│   └── MMMU/data/validation-00000-of-00001.parquet   ← MMMU 评估集（900 题）
│
├── output/                                          ← 量化输出 + 评估输出
│   ├── scale_cache/{awq,mbq,rtn,mllmq}/             ← 老位置（与 MLLMQ/scale_cache/ 互为 symlink 或副本）
│   ├── internvl2_<method>_w<n>a16/                  ← 打包模型（部署用；评估时不用，仅作记录）
│   └── eval_internvl2_<method>_w<n>a16/             ← 评估结果目录
│       ├── <ts>_<method>_w<n>_results.json
│       └── run.log
│
├── dialog.md                                        ← 当前文档的诊断 + 心得（章节化记录）
├── notification.md                                  ← ← ← 你正在读
└── run.md                                           ← 历史运行指南（旧 w4-only 版本）
```

---

## 2. 环境（硬性 — 别改版本）

### 2.1 系统包
```bash
apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential ninja-build cmake \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg
```

### 2.2 Python（建议 conda 隔离）
```bash
conda create -n mllmq python=3.10 -y
conda activate mllmq
```

### 2.3 PyTorch（必须与 CUDA 匹配，A100 用 cu121 / cu128 都行）
```bash
# cu128 推荐（与已验证的镜像一致）
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 2.4 transformers

当前环境为 **4.57.6**（升级原因见 `dialog.md` §Q11b）。

```bash
pip install transformers==4.57.6 tokenizers==0.20.3 safetensors==0.7.0
```

| 场景 | 推荐版本 |
|------|----------|
| 仅 InternVL2-8B MMMU 复现 | 4.45.0（旧 API 兼容） |
| 加 Qwen2.5-VL-7B | **≥4.46**，推荐 4.57.6 |

InternVL2-8B 在 4.57.6 下会报 `GenerationMixin` 等 warning，但 `eval_mmmu.py` / `eval_benchmarks.py` 已内置补丁，不影响功能。

### 2.5 其余依赖（一次装齐，复现脚本验证版）
```bash
pip install \
    accelerate==1.15.0 datasets==2.16.1 hf_transfer==0.1.9 huggingface_hub==0.36.2 \
    sentencepiece==0.1.99 timm==1.0.29 einops==0.8.2 \
    decord==0.6.0 av==17.1.0 imageio opencv-python-headless==4.11.0.86 pillow==10.4.0 \
    qwen-vl-utils==0.0.14 \
    sacrebleu==2.6.0 evaluate==0.4.6 peft==0.20.0 \
    ftfy numexpr pycocotools matplotlib jsonschema markdown-it-py \
    safetensors scipy regex requests tqdm pyyaml packaging protobuf==3.20.0
```

### 2.6 lmms-eval（必须从 MBQ 下 editable 安装）
```bash
cd /nas3/kj/mllmq/MBQ/3rdparty/lmms-eval
pip install -e .
```
该分支是基于 lmms-eval v0.3 改的，MBQ 项目固定用这个版本；如果装新版（>=0.4）会破坏 wrapper 接口。

### 2.7（可选）模型下载
HF 直连在国内常失败，推荐 ModelScope：
```bash
pip install modelscope
python -c "from modelscope.hub import snapshot_download; snapshot_download('OpenGVLab/InternVL2-8B', local_dir='/nas3/kj/mllmq/checkpoints/InternVL2-8B')"
```
如走 HF：
```bash
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface_hub.snapshot_download(
    repo_id='OpenGVLab/InternVL2-8B',
    local_dir='/nas3/kj/mllmq/checkpoints/InternVL2-8B')
```
若 transformers 装了 5.x，`snapshot_download` 可能报 `typer` 错 — 用 `python -c` 一行 import 即可避免 CLI 工具链。

---

## 3. 量化（4 方法统一入口）

`MLLMQ/main_quant_vl.py` 已支持 4 方法分发（`--method`）。统一参数：

| 参数 | 取值 / 备注 |
|------|-------------|
| `--model_type` | 固定 `internvl2`（也可 qwen2_vl / llava，但本实验只跑 vl）|
| `--model_path` | `/nas3/kj/mllmq/checkpoints/InternVL2-8B` |
| `--calib_data` | `coco` |
| `--data_path` | `/nas3/kj/mllmq/data/coco/sharegpt4v_coco_calib.json` |
| `--image_folder` | `/nas3/kj/mllmq/data/coco/train2017` |
| `--n_samples` | `128`（COCO 校准样本数；MLLMQ 用全部）|
| `--device` | `cuda:0` |
| `--w_bit` | `4` / `3` / `8` |
| `--a_bit` | `16`（激活保持 fp16）|
| `--w_group` | `128` |

### 3.1 RTN（无 scale search，纯 round-to-nearest）
```bash
cd /nas3/kj/mllmq/MLLMQ
python main_quant_vl.py \
    --method rtn --model_type internvl2 --model_path /nas3/kj/mllmq/checkpoints/InternVL2-8B \
    --calib_data coco --data_path /nas3/kj/mllmq/data/coco/sharegpt4v_coco_calib.json \
    --image_folder /nas3/kj/mllmq/data/coco/train2017 --n_samples 128 \
    --w_bit 4 --a_bit 16 --w_group 128 \
    --pack_quant --output_path /nas3/kj/mllmq/output/internvl2_rtn_w4a16
```
仅 `--pack_quant` 触发 fake-quant → saved；不调 scale search。

### 3.2 AWQ（Activation-aware Weight Quantization）
```bash
python main_quant_vl.py \
    --method awq --model_type internvl2 --model_path /nas3/kj/mllmq/checkpoints/InternVL2-8B \
    --calib_data coco --data_path /nas3/kj/mllmq/data/coco/sharegpt4v_coco_calib.json \
    --image_folder /nas3/kj/mllmq/data/coco/train2017 --n_samples 128 \
    --w_bit 4 --a_bit 16 --w_group 128 --run_process --pack_quant \
    --scale_path /nas3/kj/mllmq/MLLMQ/scale_cache/awq/internvl2_InternVL2-8B_w4a16.pt \
    --output_path /nas3/kj/mllmq/output/internvl2_awq_w4a16
```
注意 scale cache 文件名包含模型名 `InternVL2-8B`：`internvl2_InternVL2-8B_w{a}a16.pt`。

### 3.3 MBQ
```bash
python main_quant_vl.py \
    --method mbq --model_type internvl2 --model_path /nas3/kj/mllmq/checkpoints/InternVL2-8B \
    --calib_data coco --data_path /nas3/kj/mllmq/data/coco/sharegpt4v_coco_calib.json \
    --image_folder /nas3/kj/mllmq/data/coco/train2017 --n_samples 128 \
    --w_bit 4 --a_bit 16 --w_group 128 --run_process --pack_quant \
    --scale_path /nas3/kj/mllmq/MLLMQ/scale_cache/mbq/internvl2_w4a16.pt \
    --output_path /nas3/kj/mllmq/output/internvl2_mbq_w4a16
```

### 3.4 MLLMQ = MBQ + `--reweight`
```bash
python main_quant_vl.py \
    --method mllmq --model_type internvl2 --model_path /nas3/kj/mllmq/checkpoints/InternVL2-8B \
    --calib_data coco --data_path /nas3/kj/mllmq/data/coco/sharegpt4v_coco_calib.json \
    --image_folder /nas3/kj/mllmq/data/coco/train2017 --n_samples 128 \
    --w_bit 4 --a_bit 16 --w_group 128 --reweight --run_process --pack_quant \
    --scale_path /nas3/kj/mllmq/MLLMQ/scale_cache/mllmq/internvl2_w4a16.pt \
    --output_path /nas3/kj/mllmq/output/internvl2_mllmq_w4a16
```
注意：`--method mllmq` 已隐式启用 reweight，无需写 `--reweight`；两者可同时显式写，含义一致。

### 3.5 已缓存 scale（避免重做 30 min 校准）
scale cache 已存在，则 `--run_process` 也不会重做，跳过加载：
```
scale_cache/{rtn,awq,mbq,mllmq}/internvl2_InternVL2-8B_w{a}a16.pt  ← AWQ 老格式
scale_cache/{mbq,mllmq}/internvl2_w{a}a16.pt                       ← MBQ / MLLMQ
```
若有 cache 就把 `--run_process` 留给首次，续跑时省略即可（脚本会自动跳过 scale search）。

---

## 4. 评估（同一入口 4 方法）

`MLLMQ/eval_mmmu.py` 统一接口：
```bash
cd /nas3/kj/mllmq/MLLMQ
python eval_mmmu.py \
    --method <rtn|awq|mbq|mllmq> --w_bit <3|4|8> \
    [--scale_path <path-to-scale.pt>] \
    --output_dir /nas3/kj/mllmq/output/eval_internvl2_<method>_w<n>a16
```
关键事实：
- 评估不再依赖保存的打包模型，直接 load **fp16 InternVL2-8B** → 手动 `apply_<method>(scale_path)` → `pseudo_quantize_model_weight(w_bit)` → 走 lmms-eval。
- RTN 不需要 `--scale_path`。
- 时间：每 cell 约 45 min（MMMU 900 题，greedy decode）。
- 输出：`<ts>_<method>_w<n>_results.json` + `run.log`。

MMMU 评估自动用 lmms-eval 的 mmmu_val 任务；预下载数据：
```bash
python -m lmms_eval --tasks mmmu_val --batch_size 1 --device cuda:0 --dry_run 2  # 触发 lmms-eval 自动下载
# 或手动放到 /nas3/kj/mllmq/data/MMMU/data/validation-00000-of-00001.parquet
```

### 4.3 多 benchmark 扩展（MME / TextVQA）

> 前提：无 HF token 时 lmms-eval 无法联网下载 gated dataset。已通过 `tasks_local/*.yaml` + `data/benchmarks_local/` 离线重建，详见 `dialog.md` §8。

命令（当前 A100 串行跑）：

```bash
cd /nas3/kj/mllmq/MLLMQ
export HF_DATASETS_OFFLINE=1

# 单 task
python eval_benchmarks.py --method rtn --w_bit 4 --tasks mme_local ...
python eval_benchmarks.py --method mllmq --w_bit 4 --scale_path scale_cache/mllmq/internvl2_w4a16.pt --tasks mme_local ...
python eval_benchmarks.py --method rtn --w_bit 4 --tasks textvqa_val_local ...

# --log_samples 保存每条样本的预测输出（JSON 会大很多）
python eval_benchmarks.py --method mllmq --w_bit 4 --scale_path ... --tasks textvqa_val_local --log_samples ...
```

关键参数：

| 参数 | 说明 |
|------|------|
| `--tasks` | 任务名，用逗号分隔；支持 `mme_local`、`textvqa_val_local`、`mmmu_val` 等；也支持 group `core5` |
| `--task_include_path` | 默认 `tasks_local/`，加载本地 yaml |
| `--log_samples` | 在结果 JSON 增加 `samples` 字段（per-sample prediction） |

MME  metric：`mme_percetion_score,none` 和 `mme_cognition_score,none`（原始分总和）。
TextVQA metric：`exact_match,none`（即 `acc,none`）。

### 4.1 最终结果（更新至 2026-09-11 全部完成）

| Method | W3A16 | W4A16 | W8A16 |
|---|---|---|---|
| RTN | ✅ 0.4422 | ✅ 0.4422 | ✅ 0.4778 |
| AWQ | ✅ 0.4511 | ✅ 0.4456 | ✅ 0.4833 |
| MBQ  | ✅ 0.4611 | ✅ 0.4589 | ✅ 0.4756 |
| MLLMQ ★ | ✅ 0.4611 | **0.4867** (+0.34%) | ✅ 0.4822 |
| fp16 baseline | — | — | 0.4833 |

> 全部 12 cell 已完成。MLLMQ W4A16 是唯一超过 fp16 baseline 的方法。W8 列 AWQ (0.4833) 与 fp16 持平；MLLMQ W3 = MBQ W3 (0.4611)，reweight 在低精度无额外收益。

### 4.2 批量评估（推荐）

**已弃用脚本**：`run_all_evals.sh` 用了 `set -euo pipefail`，单 cell 崩溃会终止后续所有 cell。**已替换为容错版**。

```bash
# 续跑剩余 cell（自动跳过已完成的，单 cell 失败不中断）
bash /nas3/kj/mllmq/MLLMQ/scripts/run_remaining_evals.sh

# 单 cell 自己跑：
python eval_mmmu.py --method rtn --w_bit 4 --output_dir /tmp/eval_rtn_w4  # RTN 不需 scale
python eval_mmmu.py --method mllmq --w_bit 4 \
    --scale_path scale_cache/mllmq/internvl2_w4a16.pt \
    --output_dir /tmp/eval_mllmq_w4
```

---

## 5. 代码补丁（diff 级别）

### 5.1 ★ `/nas3/kj/mllmq/MLLMQ/qmllm/methods/awq/quantize/pre_quant.py`

**问题**：原 `run_awq()` 只支持纯 LLM forward，对 InternVL2 的 `InternVLChatModel.model.language_model.model.layers` 路径无效。

**修改**：新增 `run_awq_vl()` — 用 MBQ 的 `process_input / Catcher / move_embed / get_blocks` 把视觉前缀也拼进 calibration forward，但用 AWQ 自己的 `auto_scale_block` / `apply_scale` 计算 scale。

```diff
+ from qmllm.methods.mbq.quantize.pre_quant import (
+     move_embed as m_move_embed, get_blocks as mbq_get_blocks,
+     process_input,
+ )

+ def run_awq_vl(process_model, prompt_inputs, prompt_kwargs, *, w_bit, q_config, auto_scale=True):
+     """AWQ scale search for Vision-Language models."""
+     model = process_model.model
+     layers = mbq_get_blocks(model)               # 自动选 InternVLChatModel 分支
+     # ... 与 MBQ 相同的 Catcher 收集 inps/attention_mask
+     # 但调用的是 AWQ 的 auto_scale_block + apply_scale
+     ...
+     return awq_results
```

并在 `main_quant_vl.py` 调用：
```python
from qmllm.methods.awq.quantize.pre_quant import run_awq_vl
awq_results = run_awq_vl(process_model, prompt_inputs, prompt_kwargs,
                          w_bit=args.w_bit, q_config=q_config, auto_scale=True)
```

### 5.2 ★ `/nas3/kj/mllmq/MLLMQ/qmllm/methods/awq/quantize/auto_scale.py`

**问题**：第 95-97 行传入 `n_bit=w_bit`，但 `pseudo_quantize_tensor` 形参是 `n_bits=`（多一个 s），TypeError。

**修复（1 行）**：
```diff
-                n_bit = w_bit,
+                n_bits = w_bit,
```

### 5.2.1 ★ `/nas3/kj/mllmq/MLLMQ/qmllm/methods/awq/quantize/quantizer.py`（2026-09-11 发现，与 5.2 同 typo 第 2 处）

**问题**：`auto_scale.py:97` 修了 `n_bit=` → `n_bits=`，但 `quantizer.py:74`（`pseudo_quantize_model_weight` 里一行循环调用 `pseudo_quantize_tensor(n_bit=w_bit, **q_config)`）漏修。**只在评估阶段触发**（量化阶段用 `auto_scale_block` 走的是另一条路径不经过 `quantizer.py`，所以量化不崩，但 eval `eval_mmmu.py` 里的 `pq_awq(model, w_bit=8, q_config)` 会崩）。

**现象**：`TypeError: pseudo_quantize_tensor() got an unexpected keyword argument 'n_bit'`，发生在 `[eval_mmmu] applying awq w8` → `pseudo weight quantization...` 跑完后。

**影响**：`scripts/run_all_evals.sh` 用了 `set -euo pipefail`，AWQ W8 崩 → 脚本 exit 1 → AWQ W3 / MBQ W8 / MBQ W3 / MLLMQ W8 / MLLMQ W3 **全部被跳过**。这是工程量最大的教训——`set -e` 可靠性不如逐个 try/catch。

**修复（1 行）**：
```diff
  m.weight.data = pseudo_quantize_tensor(
-     m.weight.data, n_bit=w_bit, **q_config
+     m.weight.data, n_bits=w_bit, **q_config
  )
```

确认：全仓库搜索 `n_bit[^s]` 在 awq/mbq/rtn/mllmq 子目录确认无残留。

### 5.4.1 ★ `/nas3/kj/mllmq/MLLMQ/eval_benchmarks.py`（通用多 benchmark 评估入口）

新增于 2026-09-15，支持 MME / TextVQA / MMMU 等多任务：

```python
# 关键改动 1：支持本地 task include_path
TaskManager(include_path=args.task_include_path)

# 关键改动 2：log_samples 可配置
kwargs = dict(
    ...,
    log_samples=args.log_samples,
    ...,
)

# 关键改动 3：统一 apply_quant 复用 eval_mmmu.py 逻辑
apply_quant(lm._model, args.method, args.w_bit, args.w_group, args.scale_path)
```

默认 `--task_include_path` 指向 `MLLMQ/tasks_local/`，无 HF token 也能跑 `mme_local` / `textvqa_val_local`。

#### 2026-09-15 新增（多 benchmark 续跑基建）

| 文件 | 作用 |
|------|------|
| `scripts/run_pending_eval.sh` | 串行排队评测：GPU 门控（等其它 eval 进程结束）+ 幂等（重复跑自动跳过已完成 cell）。队列 = MME(MBQ/AWQ) → TextVQA(5 方法) |
| `scripts/continue_seed_download.py` | ModelScope 忽略 TLS 续传 SEED-Bench-2 缺失 134 shard（已有 284/418），落到 `data/seed_patch/` |
| `scripts/setup_bench.sh` | 合并 `downloads/` + `seed_patch/` 重建 `benchmarks_local/lmms-lab/SEED-Bench-2` DatasetDict |
| `tasks_local/seedbench_2_local.yaml` + `utils.py` seed 函数 | SEED-Bench-2 本地 task 配置（无 token），供 `eval_benchmarks.py --tasks seedbench_2_local` |

**metric key 后缀修复**：`eval_benchmarks.py` 和 `collect_results.py` 中 MME 双指标读取时 key 需带 `,none` 后缀（lmms-eval 实际产出 `mme_percetion_score,none` / `mme_cognition_score,none`）。`collect_results.py` 的 `MULTI_TASK_METRICS` 已同步补 `mme_local` / `textvqa_val_local` / `seedbench_2_local` 别名。

**SEED-Bench-2 使用顺序**：
```bash
# 1) 补全下载（~134 shard，后台，不占 GPU）
python scripts/continue_seed_download.py
# 2) 重建本地 cache
bash scripts/setup_bench.sh seedbench2
# 3) 评测（等 TextVQA 队列完成后）
python eval_benchmarks.py --method mllmq --w_bit 4 \
    --scale_path scale_cache/mllmq/internvl2_w4a16.pt \
    --tasks seedbench_2_local --device cuda:0 --batch_size 1 \
    --output_base /nas3/kj/mllmq/output/internvl2-8b/eval
```

### 5.3 ★ `/nas3/kj/mllmq/MLLMQ/main_quant_vl.py`

多处改动：
1. 增加 method 分发（`rtn / awq / mbq / mllmq`）：
```python
if method in ("mbq", "mllmq"):
    ...
elif method == "awq":
    from qmllm.methods.awq.quantize.pre_quant import run_awq_vl
    ...
else:  # rtn
    print("[RTN] pure round-to-nearest, no scale search")
```

2. `load_vl_model` 给 `torch.linspace` 套 monkey patch：
```python
_orig_linspace = torch.linspace
torch.linspace = lambda *a, **kw: _orig_linspace(*a, **{**kw, "device": kw.get("device", "cpu")})
try:
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
finally:
    torch.linspace = _orig_linspace
```

3. transformers ≥5 时打 `mark_tied_weights_as_initialized` 补丁（4.45.0 不打）：
```python
import transformers
if transformers.__version__.split(".")[0] >= "5":
    from transformers.modeling_utils import PreTrainedModel
    _orig = PreTrainedModel.mark_tied_weights_as_initialized
    def _patched(self, info_dict):
        if not hasattr(self, "all_tied_weights_keys") and hasattr(self, "_tied_weights_keys"):
            self.all_tied_weights_keys = self._tied_weights_keys or {}
        return _orig(self, info_dict)
    PreTrainedModel.mark_tied_weights_as_initialized = _patched
```

### 5.4 ★ `/nas3/kj/mllmq/MLLMQ/eval_mmmu.py`（MMMU 统一评估入口）

统一评估入口：
```python
from qmllm.methods.mbq.quantize.pre_quant import apply_mbq
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight as pq_mbq
from qmllm.methods.awq.quantize.pre_quant import apply_awq
from qmllm.methods.awq.quantize.quantizer import pseudo_quantize_model_weight as pq_awq
from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight as pq_rtn

def apply_quant(model, method, w_bit, w_group, scale_path):
    q_config = {"zero_point": True, "q_group_size": w_group}
    if method in ("mbq", "mllmq"):
        scales = torch.load(scale_path, map_location="cpu")
        apply_mbq(model, scales)
        pq_mbq(model, w_bit=w_bit, q_config=q_config)
    elif method == "awq":
        scales = torch.load(scale_path, map_location="cpu")
        apply_awq(model, scales)
        pq_awq(model, w_bit=w_bit, q_config=q_config)
    elif method == "rtn":
        pq_rtn(model, w_bit=w_bit, q_config=q_config)
    model.cuda()             # apply_* 可能留下 CPU 参数

def main():
    args = parse_args()
    from lmms_eval.models.internvl2 import InternVL2
    lm = InternVL2.create_from_arg_string(
        f"pretrained={args.model_path}",
        {"batch_size": str(args.batch_size), "device": args.device},
    )
    apply_quant(lm._model, args.method, args.w_bit, args.w_group, args.scale_path)
    lm._model.cuda().eval()

    from lmms_eval import evaluator
    from lmms_eval.tasks import TaskManager
    from lmms_eval.loggers import EvaluationTracker
    eval_tracker = EvaluationTracker(output_path=args.output_dir)
    results = evaluator.simple_evaluate(
        model="internvl2", lm=lm,
        model_args=f"pretrained={args.model_path}",
        tasks=[args.tasks], batch_size=str(args.batch_size), device=args.device,
        log_samples=False, evaluation_tracker=eval_tracker, task_manager=TaskManager(),
    )
    json.dump(results, open(...,"w"), indent=2, default=str)
```

### 5.5 ★ `/nas3/kj/mllmq/MBQ/3rdparty/lmms-eval/lmms_eval/models/internvl2.py`

第 142 行附近，把 `AutoModel.from_pretrained` 调用包成 `torch.linspace force-CPU` 的 try/finally，并把 `low_cpu_mem_usage=True` 去掉（避免 transformers 5.x meta-device 路径）：

```python
_orig_linspace = torch.linspace
torch.linspace = lambda *a, **kw: _orig_linspace(*a, **{**kw, "device": kw.get("device", "cpu")})
try:
    self._model = AutoModel.from_pretrained(
        self.path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval().cuda()
finally:
    torch.linspace = _orig_linspace
```
**注意**：这是 3rdparty 改动，submodule 重新 checkout 会丢失。`dialog.md` 顶部有这个提醒。

### 5.6 ★ `/nas3/kj/mllmq/MLLMQ/qmllm/methods/mbq/quantize/pre_quant.py`

`mbq_get_blocks()` 已加 `InternVLChatModel` 分支：
```python
elif model.__class__.__name__ == "InternVLChatModel":
    layers = model.language_model.model.layers
```
并加了 `move_embed` 对 `model.model.tok_embeddings`（注：AWQ 的 `move_embed` 仍使用 `model.model.embed_tokens` 旧路径，已替换为 MBQ 的 `m_move_embed`）。

---

## 6. scale cache 复用策略

scale cache 文件命名约定：
- MBQ / MLLMQ：`scale_cache/{mbq,mllmq}/internvl2_w{a}a16.pt`
- AWQ：`scale_cache/awq/internvl2_InternVL2-8B_w{a}a16.pt`（**包含模型目录名**）
- RTN：无 cache（量化即 quantize，无搜索过程）

每个 .pt 大致 50–200 MB，包含 `{scale: [...], zero: [...], mix_layers: [...], ...}`。**已生成的 cache 复用，不重跑**，对结果无影响（cache 由 calibration data + method 决定，与任务无关）。

---

## 7. 量级、延迟与显存

| 量 | 值 |
|----|----|
| 单次量化（scale search + pack）耗时 | MBQ/MLLMQ W4 ≈ 30 min、A WQ ≈ 25 min、RTN ≈ 2 min |
| 单次 MMMU 评估 (900 题) | ≈ 45 min（RTN 略快）|
| 单次 MME 评估 (2374 题) | ≈ 2.5h（单样本 ~3-4s，图像大）|
| 单次 TextVQA 评估 (val 5000 题) | ≈ 2-2.5h |
| 显存峰值（calibration + quantize）| ≈ 30 GB |
| 显存峰值（eval, bf16）| ≈ 20 GB |
| 硬盘：fp16 InternVL2-8B | 17 GB |
| 硬盘：scale cache（W4 一份）| ≈ 80 MB |
| 硬盘：完整 4×3 输出（不含 checkpoint）| < 60 GB，eval json + log ≈ 5 MB |

整组 6 cell 串行跑 ≈ **6 小时**（不含已完成）。机器若有 1 张 A100 80G 已够；≥ 2 张可并行评估（量化阶段必须单卡，因为 calibration 缓存是 process_model 状态）。

---

## 8. 复现 sanity 检查清单

✅ **必跑**（每条失败说明环境没装对）
```bash
[ -f /nas3/kj/mllmq/checkpoints/InternVL2-8B/config.json ]             || echo "missing model"
[ -d /nas3/kj/mllmq/MBQ/3rdparty/lmms-eval ]                          || echo "missing sub"
python -c "import lmms_eval; print(lmms_eval.__version__)"             || pip install -e MBQ/3rdparty/lmms-eval
python -c "import transformers; print(transformers.__version__)"       || \  # 4.45.0-4.57.6；含 Qwen2.5-VL 用 4.57.6
python -c "from qmllm.methods.awq.quantize.pre_quant import run_awq_vl" || echo "missing my patch"
python -c "from qmllm.methods.mbq.quantize.pre_quant import apply_mbq" || echo "missing mbq"
# n_bits 拼写修复 — 两处都要确认，q_bit 不合法性只能靠搜 `n_bit[^s]` 确认
grep -rn "n_bit[^s]" MLLMQ/qmllm/methods/awq/ | grep -v "n_bits" && echo "n_bit typo still present!" || echo "n_bits OK"
ls /nas3/kj/mllmq/MLLMQ/scale_cache/mbq/internvl2_w4a16.pt            || echo "need to run calibration"
ls /nas3/kj/mllmq/data/MMMU/data/validation-*.parquet                  || python -m lmms_eval --tasks mmmu_val --dry_run
```

✅ **结果预期**：跑完后
- MMMU：`output/eval_internvl2_<method>_w<n>a16/<ts>_<method>_w<n>_results.json` 存在，查 `results.mmmu_val.mmmu_acc,none` 与 §4.1 一致（±0.005）
- MME：`output/internvl2-8b/eval/mme_local/<method>_w4a16/<ts>_<method>_w4_mme_local_results.json`，查 `results.mme_local.mme_percetion_score,none` 和 `mme_cognition_score,none`
- TextVQA：同上路径 `textvqa_val_local/<method>_w4a16/...`，查 `results.textvqa_val_local.exact_match,none`
- W4A16 列：RTN≈0.442、AWQ≈0.446、MBQ≈0.459、MLLMQ≈0.487

若结果偏离 >0.01，先 §9 自检。

---

## 9. 自检（结果偏离时按顺序检查）

1. **n_bits 拼写修复**：`grep -rn "n_bit[^s]" MLLMQ/qmllm/methods/awq/ | grep -v "n_bits"` 必须返回空。两处位置：`auto_scale.py:97` + `quantizer.py:74`。只要有一处还是 `n_bit=`，AWQ eval 必崩。
2. **transformers 版本**：`pip show transformers | head -2` 必须在 4.45.0–4.57.6 区间。`dialog.md` 详细说明不同模型该用哪个版本。
2. **lmms-eval 版本**：必须是 `MBQ/3rdparty/lmms-eval` editable 安装（`pip show lmms_eval | grep Location` 应该指向该路径）。`pip install -e .` 后重试。
3. **scale cache**：确保 `--scale_path` 用的是同一个量化方法产生的 cache。MBQ 的 cache 不能给 AWQ 用（数据结构不同）。
4. **apply_<method>**：检查 eval_mmmu 日志里有 `[eval_mmmu] applying <method> w<n>`，并确认无 `device mismatch` 报错。
5. **calib data**：`--data_path` 必须是 128-样本 COCO；用 pileval 或 text-only 数据会让 AWQ/MBQ 的 scale 退化。
6. **检查 zeros**：MBQ `zero_point=True`，AWQ 默认 `zero_point=True`。RTN 也有 zero_point（`pq_rtn` 配置同）。
7. **hash check**：输出文件 md5 不强制（量化有浮点随机性），但 precision 应在 1% 误差内。
8. **看 `dialog.md` 的 §Q1-Q10**：9 类坑都列了。

---

## 10. 数据流图（详见 `dialog.md` §4，此处简版）

```
main_quant_vl.py (量化)
  1) load fp16 InternVL2-8B
  2) load COCO calib (128)
  3) run_<method> → scale_cache/<method>/*.pt
  4) apply_<method> → model weights *= scale
  5) pseudo_quantize_model_weight → round-to-nearest
  6) save_packed_model → output/internvl2_<method>_w<n>a16/

eval_mmmu.py (评估)
  1) load fp16 InternVL2-8B via lmms_eval's InternVL2 wrapper
  2) apply_<method>(scale_cache) + pseudo_quantize
  3) lmms_eval simple_evaluate(task=mmmu_val)
  4) dump <ts>_<method>_w<n>_results.json
```

---

## 11. FAQ（从 dialog.md 抽出的高频坑）

**Q: 我有 A1 / H100 / V100，跑得动吗？**
A: A100 80G 验证过；H100 同显存没问题。A1 同系列（low-end）可能 batch_size=1 都 OOM，需要 `--device cpu` 慢跑。

**Q: `huggingface-cli login` 给我 `TypeError: 'type' object is not subscriptable`。**
A: typer/click 与 py3.10 冲突。直接 `from huggingface_hub import snapshot_download` 或用 ModelScope。

**Q: `apply_<method>` 后生成一塌糊涂。**
A: (a) 看 scale cache 是否真是该方法的；(b) `model.cuda()` 是否调了；(c) 是否对 lm_head 也 pseudo-quant（**绝对不要**，会崩坏）。

**Q: 评估报 `lm_head shape mismatch`。**
A: 同上，幻量化要排除 `lm_head` 和 `model.language_model.model.embed_tokens`。

**Q: AWQ 跑 quantization 时 `pseudo_quantize_tensor() got an unexpected keyword argument 'n_bit'`。**
A: 两处 `n_bit→n_bits` 都要修：① `auto_scale.py:97`（scale search 阶段）② `quantizer.py:74`（评估阶段，**这一步跑完 passthrough 量就告崩**）。详见 notification.md §5.2 和 §5.2.1。如果只修了一处漏了另一处，量化阶段过但 eval 崩。

**Q: 评估跑完 mmlu-val 不是 mmmu_val。**
A: `--tasks` 加 `mmmu_val`；默认就是。检查 json 里的 `configs` 字段。

**Q: 跨容器共享 scale cache 到 Mac/Windows。**
A: `tar czf scale_cache_<method>_<bit>.tgz MLLMQ/scale_cache/<method>/internvl2_*.pt`；注意文件 50–200 MB / 份。

**Q: 跑 MME / TextVQA 没有 HF token 怎么办？**
A: lmms-lab/MME、lmms-lab/textvqa 是 gated dataset，`datasets.load_dataset` 联网验证会报 `LocalTokenNotFoundError`。解决方案：用 `HF_DATASETS_OFFLINE=1` + 本地 `tasks_local/` yaml + `data/benchmarks_local/` 重建的 cache。`eval_benchmarks.py --tasks mme_local --task_include_path MLLMQ/tasks_local` 会自动走本地路径。详情见 `dialog.md` §8.2-8.3。

**Q: eval 结果 JSON 里只有评分，没有模型生成的回答。**
A: 默认 `log_samples=False`，lmms-eval 只算聚合分数。加 `--log_samples` 参数可在 `samples` 字段保存每条样本的 `question_id / pred`。数据量大时 JSON 会很大（MME 2374 条 × 每条推理输出 16 token），按需使用。

---

## 12. 数据流图（多 benchmark 版）

```
main_quant_vl.py (量化)
  1) load fp16 InternVL2-8B
  2) load COCO calib (128)
  3) run_<method> → scale_cache/<method>/*.pt
  4) apply_<method> → model weights *= scale
  5) pseudo_quantize_model_weight → round-to-nearest
  6) (可选) save_packed_model → output/internvl2-<method>-w<n>a16/

eval_mmmu.py (MMMU 评估)
  1) load fp16 InternVL2-8B via lmms_eval InternVL2 wrapper
  2) apply_<method>(scale_cache) + pseudo_quantize
  3) lmms_eval simple_evaluate(task=mmmu_val)
  4) dump output/eval_internvl2_<method>_w<n>a16/*_results.json

eval_benchmarks.py (通用多 benchmark 评估)
  1) load fp16 InternVL2-8B via lmms_eval InternVL2 wrapper
  2) apply_<method>(scale_cache) + pseudo_quantize
  3) TaskManager(include_path=tasks_local/) 加载 mme_local / textvqa_val_local
  4) lmms_eval simple_evaluate(task=mme_local/textvqa_val_local/...)
  5) dump output/internvl2-8b/eval/<task>/<method>_w<n>a16/*_results.json
```

---

## 13. 联系方式 / 后续 TODO

- ✅ MMMU 4×3 矩阵已完成（2026-09-11）。
- ✅ MME 9/12 cell 已完成（2026-09-16），3 cell 缺口 + TextVQA 13 cell + SEED-Bench-2 全待跑。
- 📋 **当前进度详见 [`experiment_status.md`](../experiment_status.md)**，重启前必读。
- 完成后用 `scripts/collect_results.py --tasks ...` 汇总跨 benchmark 结果。
- 当前结果已证明 MLLMQ W4A16 **优于 fp16 baseline +0.34%**（MMMU），是论文最大亮点；多 benchmark 验证其是否普遍优于其他 4-bit 方法。

---

— MLLMQ 复现指南 end · 三文档联动：`experiment_status.md`（快照）| `notification.md`（how）| `dialog.md`（why）—