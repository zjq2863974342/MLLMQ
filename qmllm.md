# qmllm 项目结构分析

## 目录概览

```
qmllm/
├── calibration/           # 校准数据集加载器
│   ├── pileval.py         # 纯文本校准数据（Pile 数据集）
│   └── coco_vl.py         # 多模态校准数据（COCO 图文对）
├── datasets/              # 数据集工具
│   └── multimodal_dataset.py  # LLaVA / InternVL 的 DataLoader 构建器
├── methods/               # 四种量化方法实现
│   ├── awq/               # AWQ（激活感知权重量化）
│   ├── mbq/               # MBQ（模态均衡量化）★ 核心方法
│   ├── rtn/               # RTN（四舍五入量化，基线方法）
│   └── smoothquant/       # SmoothQuant（平滑量化）
├── models/                # MLLM 模型包装器（统一接口）
│   ├── base.py            # 抽象基类 BaseModel
│   ├── janus/             # Janus（DeepSeek）
│   ├── internvl2/         # InternVL2
│   ├── llava_onevision/   # LLaVA-OneVision
│   ├── llava_v15/         # LLaVA-v1.5
│   ├── qwen2_vl/          # Qwen2-VL
│   └── vila/              # VILA（NVIDIA）
├── quantization/          # 量化原语
│   ├── quant_funcs.py     # 伪量化核心函数
│   ├── qlinear.py         # WALinear 模块（权重+激活量化）
│   ├── quant_wrapper.py   # 方法调度器（总入口）
│   └── pack.py            # 真实量化 + int32 打包 + safetensors 导出
├── sensitive/             # 梯度敏感度分析（占位）
├── utils/                 # 工具函数
│   ├── registry.py        # 注册器模式
│   └── search.py          # 模块查找与操作工具
└── visualization/         # 可视化（占位）
```

---

## 1. 量化原语层 (`qmllm/quantization/`)

本层是量化框架的基础，所有量化方法都依赖此层的函数。

### 1.1 `quant_funcs.py` — 核心量化函数

**依赖**: `torch`

**功能**: 提供伪量化（pseudo-quantization）的基础函数。伪量化指将张量量化到低位整数再反量化回浮点，从而在 fp16 精度下模拟量化效果。

| 函数 | 功能 |
|------|------|
| `pseudo_quantize_tensor(tensor, n_bits, zero_point, q_group_size, per_tensor, inplace)` | **核心量化函数**。支持对称/非对称、per-group/per-tensor 量化。公式: `clamp(round(w/scale)+zero, 0, 2^n-1) - zero) * scale` |
| `quantize_weight_per_channel_absmax(w, n_bits, zero_point)` | Per-channel 权重量化（按输出通道） |
| `quantize_weight_per_tensor_absmax(w, n_bits, zero_point)` | Per-tensor 权重量化（整个张量共享 scale） |
| `quantize_activation_per_token_absmax(t, n_bits, zero_point)` | Per-token 激活量化（每个 token 独立 scale） |
| `quantize_activation_per_tensor_absmax(t, n_bits, zero_point)` | Per-tensor 激活量化 |

**被依赖**: `qlinear.py`, 所有方法的 `quantizer.py`, `auto_scale.py`

### 1.2 `qlinear.py` — WALinear 模块

**依赖**: `torch`, `torch.nn`, `quant_funcs.*`

**功能**: 定义 `WALinear(nn.Module)`，替代原始 `nn.Linear` 实现权重+激活（WA）量化推理。

**关键方法**:
- `from_float(module, ...)` — 静态工厂：从 `nn.Linear` 创建 WALinear，支持 per-channel / per-tensor / per-group 权重量化
- `forward(x)` — 量化输入激活 → fp16 线性计算 → 可选量化输出

**被依赖**: `pseudo_quantize_model_weight_act()`（RTN/MBQ/SmoothQuant 的 WA 路径）

### 1.3 `quant_wrapper.py` — 方法调度器

**依赖**: `os`, 四种方法的 `entry.py`

**功能**: 量化流程的总入口，根据 `args.method` 分发到对应方法。

```python
qwrapper(model, prompt_inputs, prompt_kwargs, args):
    if method == "awq":      → awq_entry(...)
    if method == "smoothquant": → smoothquant_entry(...)
    if method == "mbq":      → mbq_entry(...)
    if method == "rtn":      → rtn_entry(...)
```

**WA 判定**: 当 `w_bit < 16` 且 `a_bit < 16` 时启用 WA 量化。

### 1.4 `pack.py` — 真实量化打包

**依赖**: `json`, `os`, `torch`, `torch.nn`, `tqdm`, `safetensors`

**功能**: 不同于伪量化，此模块真正将权重转换为低位整数、打包为 int32 并导出 safetensors。这是模型部署前的最后一步。

| 函数 | 功能 |
|------|------|
| `real_quantize_weight(weight, n_bits, group_size)` | 对单个权重矩阵进行非对称 per-group 量化，返回 `(qweight, scales, qzeros)` |
| `pack_model_linears(model_llm, w_bit, group_size)` | 遍历 LLM 骨干的所有 `nn.Linear`，量化并替换为 packed buffer |
| `save_packed_model(model, output_path, w_bit, group_size)` | 导出完整 safetensors，包含 `quant_config.json` |

**打包格式**（兼容 GPTQ/AutoGPTQ）:
- `*.qweight`: int32 张量，每 8 个 4-bit 值打包为 1 个 int32
- `*.scales`: float16 张量，每个 group 一个 scale
- `*.qzeros`: int32 张量，打包的零点值

---

## 2. 校准数据层 (`qmllm/calibration/`)

### 2.1 `pileval.py` — 纯文本校准

**依赖**: `torch`, `datasets.load_dataset`

**功能**: 从 HuggingFace 的 `mit-han-lab/pile-val-backup` 数据集加载纯文本校准数据。

**流程**: 分词 → 过滤过长样本 → 拼接 → 切分为 `block_size=512` 的块 → 返回 `(prompt_inputs, prompt_kwargs)`

**被依赖**: AWQ / MBQ 的 `run_awq()` / `run_mbq()`

### 2.2 `coco_vl.py` — 多模态校准

**依赖**: `os`, `json`, `torch`, `numpy`, `PIL`, `datasets`

**功能**: 加载 COCO 图文对作为多模态校准数据。需要模型 wrapper 实现 `preprocess_data()`, `data_collator()`, `generate_input()` 等方法。

**被依赖**: AWQ / MBQ 的多模态校准路径

---

## 3. 数据集工具层 (`qmllm/datasets/`)

### 3.1 `multimodal_dataset.py` — DataLoader 构建器

**依赖**: `torch`, `transformers`, `PIL`，可选 `llava`, `internvl`（try/except 保护）

**功能**: 为 LLaVA 和 InternVL 模型构建专用 DataLoader。

| 函数 | 功能 |
|------|------|
| `build_llava_dataloader(n_samples, tokenizer, ...)` | 构建 LLaVA 的 `LazySupervisedDataset` + DataLoader |
| `build_internvl_dataloader(n_samples, model, tokenizer, ...)` | 调用 InternVL 的 `build_datasets` 构建 DataLoader |

---

## 4. 量化方法层 (`qmllm/methods/`)

### 4.1 AWQ — 激活感知权重量化

**入口**: `awq/entry.py::awq_entry(model, prompt_inputs, prompt_kwargs, ...)`

**文件依赖关系**:
```
awq_entry()
  ├── run_awq()                    # pre_quant.py — 逐层 scale 搜索
  │     ├── auto_scale_block()     # auto_scale.py — 20 点网格搜索最优 scale
  │     └── apply_scale()          # auto_scale.py — 应用 scale（reparameterization）
  ├── apply_awq()                  # pre_quant.py — 加载缓存 scale 并应用
  └── pseudo_quantize_model_weight()  # quantizer.py — 最终伪量化
```

**核心算法** (`auto_scale.py`):
1. 对于每个 Transformer 层的每个线性层对（LN→QKV, V→O, LN→Gate+Up, Up→Down）
2. 在 `[0.0, 1.0]` 范围内 20 个候选 ratio 上网格搜索
3. 计算 `scales = x_max^ratio`，将 scale 乘到权重上，模拟量化前向传播
4. 计算重构误差（MSE/MAE），选最小误差的 ratio
5. 通过 `scale_ln_fcs()` / `scale_fc_fc()` 将最优 scale 融回权重

**支持架构**: `OPTDecoderLayer`, `LlamaDecoderLayer`, `BloomBlock`, MPT, Falcon, BigCode, NeoX, `Qwen2DecoderLayer`, `InternLM2DecoderLayer`

### 4.2 MBQ — 模态均衡量化 ★

**入口**: `mbq/entry.py::mbq_entry(model, prompt_inputs, prompt_kwargs, ...)`

**与 AWQ 的核心差异**:

| 特性 | AWQ | MBQ |
|------|:---:|:---:|
| 基础 scale 搜索 | ✓ | ✓ |
| 梯度重加权（`--reweight`） | ✗ | ✓ |
| 失真输入（`--distort`） | ✗ | ✓ |
| WA 量化 scale 搜索 | ✗ | ✓（`auto_scale_wa.py`） |
| WA 量化 + 失真 | ✗ | ✓（`auto_scale_wa_distort.py`） |
| 损失模式（MAE/MSE） | 仅 MSE | MAE + MSE |

**文件依赖关系**:
```
mbq_entry()
  ├── run_mbq()                       # pre_quant.py — 逐层 MBQ scale 搜索
  │     ├── GradCacheHook             # 反向 hook 收集视觉/文本 token 梯度
  │     ├── auto_scale_block()        # auto_scale.py — 基础 scale 搜索（同 AWQ）
  │     ├── auto_scale_block_distort() # auto_scale_distort.py — 失真输入版
  │     ├── auto_scale_block_wa()     # auto_scale_wa.py — WA 量化 scale 搜索
  │     ├── auto_scale_block_wa_distort() # auto_scale_wa_distort.py — WA+失真
  │     └── apply_scale()             # auto_scale.py — 应用 scale
  ├── apply_mbq()                     # pre_quant.py — 加载缓存 scale 并应用
  └── pseudo_quantize_model_weight()  # quantizer.py — 权重伪量化
       or pseudo_quantize_model_weight_act()  # quantizer.py — WA 伪量化（替换为 WALinear）
```

**梯度重加权机制** (`--reweight`):
1. 在前向传播时为每个 Linear 层注册反向 hook
2. 分别计算视觉 token 和文本 token 的梯度绝对值均值
3. 计算 `vis_grad / cap_grad` 比值
4. 取 self-attention 输出投影和 MLP 下投影的中位数作为重加权系数
5. 在 scale 搜索的损失计算中：`loss = L_caption + ratio * L_vision`

**失真输入机制** (`--distort`):
1. 对当前层应用量化后，用其量化输出（而非干净输出）作为下一层的输入
2. 模拟级联量化误差，使 scale 搜索更接近实际推理场景

### 4.3 RTN — 四舍五入量化（基线）

**入口**: `rtn/entry.py::rtn_entry(model, prompt_inputs, prompt_kwargs, ...)`

**功能**: 最简单的量化方法。不做 scale 搜索，直接将权重四舍五入到目标位宽。
- 仅权重: `pseudo_quantize_tensor(weight, n_bits=w_bit)`
- WA: 替换所有 `nn.Linear` 为 `WALinear`

**用途**: 作为其他方法的精度下界基线。

### 4.4 SmoothQuant — 平滑量化

**入口**: `smoothquant/entry.py::smoothquant_entry(model, prompt_inputs, prompt_kwargs, ...)`

**文件依赖关系**:
```
smoothquant_entry()
  ├── get_act_scales()          # gen_act_scales.py — 通过前向传播收集每通道最大激活值
  ├── smooth_lm()               # smooth.py — 将激活 scale 融进 LN 和 Linear 权重
  │     └── smooth_vit()        # smooth.py — ViT 的平滑处理
  └── pseudo_quantize_model_weight_act()  # quantizer.py — 替换为 WALinear
```

**平滑公式**: `scales = act_scales^alpha / weight_scales^(1-alpha)`

---

## 5. 模型包装器层 (`qmllm/models/`)

### 5.1 `base.py` — 抽象基类

**依赖**: `abc`

**抽象方法**（所有 wrapper 必须实现）:

| 方法 | 返回 | 功能 |
|------|------|------|
| `fetch_vit()` | Module | 获取视觉编码器 |
| `fetch_llm()` | Module | 获取语言模型骨干 |
| `fetch_proj()` | Module | 获取视觉-语言投影层 |
| `vision_preprocess(image)` | Tensor | 图像预处理 |
| `language_preprocess(text)` | Tensor | 文本分词 |
| `forward(*args, **kwargs)` | Output | 前向传播 |

### 5.2 已支持的模型

| 模型 | 注册键 | 包装器类 | LLM 骨干 | 图像理解 | 图像生成 |
|------|--------|---------|---------|:---:|:---:|
| **Janus** (DeepSeek) | `"janus"` | `JanusModel` | Llama | ✓ | ✓ |
| **InternVL2** | `"internvl2"` | `InternVL2` | InternLM2 | ✓ | ✗ |
| **LLaVA-OneVision** | `"llava_onevision"` | `LLaVA_onevision` | Llama/Qwen | ✓ | ✗ |
| **LLaVA-v1.5** | `"llava"` | `LLaVA_v15` | Llama | ✓ | ✗ |
| **Qwen2-VL** | `"qwen2_vl"` | `Qwen2_VL` | Qwen2 | ✓ | ✗ |
| **VILA** (NVIDIA) | `"vila"` | `VILA` | Llama | ✓ | ✗ |

### 5.3 注册机制

```python
# utils/registry.py
MODEL_REGISTRY = Register()

# 每个模型通过装饰器注册
@MODEL_REGISTRY.register("janus")
class JanusModel(BaseModel):
    ...

# 使用
from qmllm.models import get_process_model
WrapperClass = get_process_model("janus")  # → JanusModel
```

### 5.4 Janus 包装器设计要点

```python
class JanusModel(BaseModel):
    def __init__(self, model, tokenizer, processor=None):
        self._model = model              # 完整 MultiModalityCausalLM
        self.model = model.language_model  # LlamaForCausalLM（量化目标）

    def forward(self, **inputs):
        # 自动将输入移到 model 所在设备
        # 有多模态输入（含 pixel_values）→ prepare_inputs_embeds + language_model
        # 纯文本 → language_model 直通
```

---

## 6. 工具函数层 (`qmllm/utils/`)

### 6.1 `registry.py` — 注册器模式

**功能**: 实现装饰器风格的注册器，支持 `@REGISTRY("name")` 和 `@REGISTRY.register("name")` 两种用法。

**三个全局实例**:
- `MODEL_REGISTRY` — 模型包装器注册
- `DATASET_REGISTRY` — 数据集注册
- `METHOD_REGISTRY` — 量化方法注册

### 6.2 `search.py` — 模型操作工具

| 函数 | 功能 |
|------|------|
| `get_op_by_name(module, op_name)` | 按点分隔名称查找子模块 |
| `set_op_by_name(layer, name, new_module)` | 按名称替换子模块 |
| `get_op_name(module, op)` | 反向查找模块的名称字符串 |
| `append_str_prefix(x, prefix)` | 为 scale 列表条目添加前缀 |
| `get_module_by_name_suffix(model, module_name)` | 按后缀查找模块 |
| `simple_dispatch_model(model, device_map)` | 跨设备模型分派（用 accelerate） |

---

## 7. 整体调用流程图

```
main_quant.py / lmms_eval
        │
        ▼
quant_wrapper.py::qwrapper()          ← 方法调度器
        │
        ├─ method=="awq" ──────────► awq_entry()
        │                               ├─ run_awq()       → auto_scale_block()
        │                               ├─ apply_awq()      → apply_scale()
        │                               └─ pseudo_quantize_model_weight()
        │
        ├─ method=="mbq" ──────────► mbq_entry()
        │    ★ 默认方法                  ├─ run_mbq()       → auto_scale_block[_wa][_distort]()
        │                               ├─ apply_mbq()      → apply_scale()
        │                               └─ pseudo_quantize_model_weight[_act]()
    │
        ├─ method=="smoothquant" ──► smoothquant_entry()
        │                               ├─ get_act_scales()
        │                               ├─ smooth_lm() + smooth_vit()
        │                               └─ pseudo_quantize_model_weight_act()
        │
        └─ method=="rtn" ──────────► rtn_entry()
                                        └─ pseudo_quantize_model_weight[_act]()

        ★ 真实量化部署：pack.py::save_packed_model()
           └─ real_quantize_weight() → safetensors 导出
```

---

## 8. 关键数据流

### 8.1 校准数据流

```
caleval (纯文本):
  get_calib_dataset() → prompt_inputs + prompt_kwargs
    prompt_inputs  = {"input_ids": [N, 512]}
    prompt_kwargs  = {"labels": ..., "attention_mask": None,
                      "vision_mask": None, "caption_mask": None}

COCO (多模态):
  get_multimodal_calib_dataset() → model.preprocess_data() → data_collator() →
  model.generate_input() → prompt_inputs + prompt_kwargs
    prompt_inputs  = {"input_ids": ..., "pixel_values": ...}
    prompt_kwargs  = {"vision_mask": ..., "caption_mask": ...}
```

### 8.2 Scale 搜索结果格式

```python
mbq_results = {
    "scale": [
        (prev_op_name, [layer_name1, layer_name2], scales_tensor),
        # 例: ("model.layers.0.input_layernorm",
        #      ["model.layers.0.self_attn.q_proj",
        #       "model.layers.0.self_attn.k_proj",
        #       "model.layers.0.self_attn.v_proj"],
        #      tensor([4096]))
        ...
    ]
}
```

### 8.3 打包模型输出格式

```
Janus-Pro-7B-mbq/
├── model.safetensors     # 量化权重 + 未量化张量
│   ├── language_model.model.layers.0.self_attn.q_proj.qweight  (int32)
│   ├── language_model.model.layers.0.self_attn.q_proj.scales   (fp16)
│   ├── language_model.model.layers.0.self_attn.q_proj.qzeros   (int32)
│   ├── ... (共 210 个量化层 × 3 = 630 个量化键)
│   └── vision_model.* / gen_vision_model.* / embed_tokens.* 等 (fp16)
├── mbq_scales_w4.pt      # 缓存的 scale 搜索结果
└── quant_config.json     # 量化配置
```

---

## 9. 注意事项

1. **伪量化 vs 真实量化**: 所有方法的 `auto_scale` 搜索和 `pseudo_quantize_*` 都只做伪量化（fp16 模拟）。实际 int 打包仅在 `pack.py` 中，且当前仅有 `main_quant.py` 调用它。

2. **模型包装器的灵活设计**: `InternVL2`/`LLaVA` 等复杂包装器实现了完整的数据预处理管线，而 `JanusModel` 是最简实现（仅桥接 LLM 骨干），因为 Janus 的校准数据加载走独立路径。

3. **WA 量化**: 激活量化是运行时的（`WALinear.forward()` 中 per-token 量化），不需要打包。仅权重需要打包和保存。

4. **`sensitive/gen_grad.py`**: 空文件，占位符。梯度敏感度分析的实现位于 `MLLMQ/sensitive/` 目录下，独立于 qmllm 框架。
