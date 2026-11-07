# Janus-Pro-1B / Show-o (1.7B) 部署记录

> 部署日期：2026-11-02 · 环境：1× A800 80GB（本机 2 卡，冒烟测试用 GPU1）· 权重来自 ModelScope（国内直连，~25–100 MB/s）

## 1. 权重与代码位置

| 组件 | 路径 | 大小 | 校验 |
|---|---|---|---|
| Janus-Pro-1B | `checkpoints/Janus/models/Janus-Pro-1B/` | 3.9 GiB | `pytorch_model.bin` sha256 = `ea7cf164…0cc3f` ✅ |
| Show-o 主权重 | `checkpoints/Show-o/models/show-o-w-clip-vit/` | 5.42 GiB | `pytorch_model.safetensors` sha256 = `4dcf09cc…6ffc` ✅ |
| Phi-1.5（LLM 初始化/config） | `checkpoints/Show-o/models/phi-1_5/` | 2.64 GiB | ✅ |
| MAGVIT-v2（VQ tokenizer） | `checkpoints/Show-o/models/magvitv2/` | 364 MiB | ✅ |
| CLIP ViT-L/14-336 | `checkpoints/Show-o/models/clip-vit-large-patch14-336/` | 3.2 GiB（含多余的 tf_model.h5） | ✅ |
| Janus 官方代码 | `checkpoints/Janus/janus/` | — | 已有 |
| Show-o 官方代码 | `checkpoints/Show-o/src/` | — | GitHub `showlab/Show-o` main 分支，经 ghfast 镜像获取 |

说明：Show-o 官方称为 **1.3B LLM**（其自身权重实测 1.454B），加上 CLIP-ViT + MAGVIT-v2 后总计约 **1.8B**，即通常说的 “Show-o 1.7B”；本次部署的是
`show-o-w-clip-vit`（256×256 生成）。若需 512×512，请另下
`showlab/show-o-w-clip-vit-512x512` 并把 `--num-vq-tokens` 设为 1024、`--resolution` 设为 512。

## 2. 运行环境

Janus-Pro-1B 直接复用项目主环境（`/opt/conda/bin/python`，torch 2.4 / transformers 4.45）。

Show-o 使用独立 venv，**不污染正在跑实验的主环境**：

```bash
# 已创建：--system-site-packages，继承 torch/transformers，仅额外安装以下包
/nas3/kangjia/mllmq/venvs/showo/bin/python -c "import diffusers, omegaconf; print(diffusers.__version__)"
# diffusers 0.30.1, omegaconf 2.3.0, jaxtyping 0.2.28, typeguard 2.13.3
```

## 3. 冒烟测试（已验证通过）

```bash
cd /nas3/kangjia/mllmq

# Janus-Pro-1B：理解 + 文生图，约 45s（首次加载 30s）
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python3 scripts/deploy/janus_pro_1b_smoke.py \
    --device cuda:0 \
    --out-dir output/deploy_smoke/janus_pro_1b

# Show-o：理解 + 文生图，约 50s
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 venvs/showo/bin/python scripts/deploy/showo_smoke.py \
    --device cuda:0 \
    --image checkpoints/Show-o/src/mmu_validation/dog.png \
    --out-dir output/deploy_smoke/showo
```

实测结果（2026-11-02）：

- Janus-Pro-1B：加载 29.4s（bf16，2.089B 参数）；VQA 3.2s；384×384 T2I 576 token / 11.3s。
- Show-o：加载 1.8s（fp32，1.454B 参数，另有 CLIP/MAGVIT）；VQA 21s；256×256 T2I 256 token / 15.9s。
- 两者生成的 “red apple on wooden table” 图像均正常（保存于 `output/deploy_smoke/`）。

## 4. 常用参数

Janus-Pro-1B（`janus_pro_1b_smoke.py`）：

- `--model-path` 模型目录；`--image` 理解输入图；`--gen-prompt` 生成提示词
- `--parallel-size` 一次生成张数；`--image-token-num 576`；`--img-size 384`
- `--skip-understanding` / `--skip-generation` 只跑单侧

Show-o（`showo_smoke.py`）：

- `--num-vq-tokens 256`（256×256）/ `1024`（512×512）
- `--timesteps` 扩散步数（默认 50，越小越快）
- `--guidance-scale 5.0` CFG
- `--src` 官方代码目录；`--phi-model` / `--magvit-model` / `--clip-model` 均可指向本地目录

## 5. 后续接入量化评测

- **Janus-Pro-1B**：与 7B 同架构（`MultiModalityCausalLM`），`MLLMQ/main_quant.py` 只需把
  `--model_path` 指向 `checkpoints/Janus/models/Janus-Pro-1B`，scale cache / eval 输出目录换名即可。
  注意 1B 的 `language_config.hidden_size=2048`、`intermediate_size=5632`，与 7B 不同。
- **Show-o**：Phi-1.5 + MAGVIT-v2 架构，`qmllm` 尚无对应 wrapper，需要新增
  `qmllm/models/showo`（LLM backbone 为 `models/phi.py::PhiForCausalLM`），不能直接套 Janus 的适配。

## 6. 备注

- 失败重试的 HF 分片残留 `.cache/huggingface` 约 9.4 GB，可安全删除（当前磁盘 4.5 TB 空闲，未清理）：
  ```bash
  rm -rf checkpoints/Janus/models/Janus-Pro-1B/.cache \
         checkpoints/Show-o/models/*/.cache
  ```
- CLIP 目录中的 `tf_model.h5`（1.6 GB）用不到，可删除。
- 未改动任何正在运行的 GenEval 实验；冒烟测试仅占用 GPU1 的少量空闲显存/算力。