# Exp2 数据准备：教师强制 T2I 校准集（gen_tf）

本目录只做**数据**，不改共享代码（`eval_generate_quant.py` / `qmllm/**` 未动）。
目标：把「生成任务」从只有 prompt 文本，升级为真正的 **(caption → 目标图像 VQ token)** 样本，
供后续在主机侧实现 gen_image 模态的 collator / loss。

## 产物

| 文件 | 说明 |
|---|---|
| `build_gen_tf_calib.py` | 构建脚本（可重跑，确定性 seed=42） |
| `build.log` | 本次运行日志 |
| `recon_check.png` | 4 张原图 vs VQ 重建图（左原图 / 右重建） |
| `../../data/coco/janus_gen_tf_calib_128.json` | **最终数据，128 条，0.48 MB** |

## 数据字段

```json
{
  "image": "train2017/000000333724.jpg",
  "prompt": "Some people on surf boards riding a wave.",
  "image_token_ids": [576 个 int, 取值 0..16383],
  "task_type": "gen_tf"
}
```

- 图片池：`data/coco/train2017` 中 **12 位数字命名的 COCO 原图**（已过滤掉 flux 等非 COCO 文件），
  `numpy.default_rng(42).shuffle` 后取前 128 张。
- prompt：`data/coco/annotations/captions_train2017.json` 中该图的**第一条** caption。
- 图像 token：Janus 冻结生成 tokenizer（`gen_vision_model`，codebook=16384，384²→24×24=576）。

## 图像 token 化 API（已验证）

```python
sys.path.insert(0, "/nas3/kangjia/mllmq/checkpoints/Janus")
from janus.models import MultiModalityCausalLM, VLChatProcessor

model = MultiModalityCausalLM.from_pretrained(MODEL, local_files_only=True,
                                              torch_dtype=torch.bfloat16).cuda().eval()
x = transform(pil_img)                       # [3,384,384]
quant, loss, info = model.gen_vision_model.encode(x.unsqueeze(0))
ids = info[2].reshape(576)                   # info[2]: min_encoding_indices, shape [B*576]
dec = model.gen_vision_model.decode_code(ids.view(1, -1),
                                         shape=[1, 8, 24, 24])   # -> [1,3,384,384] in [-1,1]
```

预处理必须与生成一致：
`Resize((384,384)) + ToTensor() + Normalize([0.5]*3, [0.5]*3)`。

## 自检结果（2026-10-31，seed=42，n=128）

- token id 范围：**0 .. 16383**；全局唯一 15793；
  单样本唯一数 mean/min/max = **545.9 / 483 / 564**（共 576）。
- 重建 PSNR（4 张）：25.55 / 26.91 / 20.54 / 21.32 dB，**均值 23.58 dB**，重建图可辨
  （见 `recon_check.png`）。
- 教师强制前向（2 条）：
  - `inputs_embeds = text_emb(prompt) ⊕ prepare_gen_img_embeds(image_token_ids) ⊕ emb(image_end_id)`
  - `seq_len = 594 / 597`，
  - `gen_head` 的 CE loss = **7.1063 / 6.6667**，均 finite。
- 输出：`data/coco/janus_gen_tf_calib_128.json`，128 条，0.48 MB。

## 已知坑

1. **CPU 上不能用 bf16 跑 VQ 卷积**：会报 `Input type (float) and bias type (c10::BFloat16) should be the same`。
   脚本已按设备选 dtype：CPU→fp32、CUDA→bf16，并把图像 cast 到模型 dtype。
2. **train2017 目录混有非 COCO 图片**（如 `flux_dev_...jpg`），必须只保留 12 位数字文件名，
   否则 `int(stem)` 失败 / 取不到 caption。
3. **token id 是生成词表（0..16383），不是文本词表**。`image_start_id=100016`、
   `image_end_id=100593` 属于文本词表；拼接时图像 token 要走 `prepare_gen_img_embeds`，
   预测要走 `gen_head`（不是 `lm_head`）。
4. **decode 必须给 reshape 形状**：`decode_code(ids_flat, shape=[B,8,24,24])`；
   只传 `ids.view(1,24,24)` 且不给 shape 时返回的是 `[B,24,24,C]`，直接送 decoder 会维度错误。
5. 训练侧 loss 的标签应是 `image_token_ids` 本身（教师强制），文本部分不参与 gen loss；
   这样才能把「图像 token 的量化误差」真正纳入校准。

## 复核（2026-10-31 独立复跑，`verify_gen_tf_calib.py`）

1. **条数/长度/范围**：128 条；`image_token_ids` 长度全为 **576**；id 范围 **[0, 16383]**。
2. **与理解图像重叠**：`janus_mixed_calib_256.json` 中 128 张 understanding 图 vs 本数据 128 张，
   **交集 = 0**（basename 比对）。
3. **教师强制前向（2 条随机样本）**：
   `input_ids = prompt + image_start + 576 tokens + image_end`，labels 仅在 576 个图像 token 上。
   - sample 0：`seq_len=597`，`labeled_tokens=576`，**loss=11.6606**，finite
   - sample 1：`seq_len=597`，`labeled_tokens=576`，**loss=11.7374**，finite
   - 无 shape 错；说明该输入/标签排布可被 LM 正常前向。
4. **recon_check.png**：776×1536 RGB，4 行「原图 | VQ 重建」；
   左右半边 PSNR **22.76 dB**、std=0.227（非空白）；目视重建与原因高度一致
   （冲浪者、马桶、滑板人、披萨均清晰可辨）。

> 注：第 3 步按主机要求用 `input_ids`+`labels` 做形状/loss sanity check；
> 真正的 gen_tf collator 应把图像位置替换为 `prepare_gen_img_embeds(image_token_ids)`、
> 用 `gen_head` 预测（见上文已知坑 3）。