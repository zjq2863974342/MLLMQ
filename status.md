
依旧是定一个 10 分钟的监控任务，输出实验进展矩阵并更新dialog.md 对应的进展块内容。当检测到目前的任务完成之后就开启新任务并继续监控，我要出去一趟，之后我希望资源能充分利用，实验能一直跑，靠你了 









# MLLMQ 实验运行状态

> 📅 2026-09-16 17:40 | 🛑 所有进程已停 | 准备双卡重启
>
> **配套文档**：[`dialog.md`](dialog.md)（why——诊断与坑记录）、[`notification.md`](notification.md)（how——复现指南）、[`run.md`](run.md)（历史计划）
> 三个文档互补：status=快照/action，notification=流程/命令，dialog=坑/决策

---

## 当前结论：已完成得分

### MME（12 cell 目标，9 done / 3 缺口）

| Method | W4 | W8 | W3 |
|--------|----|----|----|
| **RTN** | ✅ perc=**1652** cog=**585** | ✅ perc=**1628** cog=**573** | ✅ perc=**1532** cog=**544** |
| **AWQ** | ✅ perc=**1636** cog=**561** | ❌ | ✅ perc=**1605** cog=**521** |
| **MBQ** | ✅ perc=**1597** cog=**559** | ❌ | ✅ perc=**1576** cog=**501** |
| **MLLMQ** | ✅ perc=**1641** cog=**562** | ❌ | ✅ perc=**1578** cog=**543** |

**关键观察**：
- MLLMQ W4 perc=1641 仅次于 RTN W4 perc=1652，但 cogn=562 接近 AWQ W4
- MLLMQ W3 = MBQ W3（perc 1578 vs 1576），结论与 MMMU 一致：**低精度下 reweight 无额外收益**
- W8 RNT > W4 所有方法，量化损失在 W8 级不显著

### TextVQA（13 cell 目标，0 done）

> 全部待跑，含 fp16 baseline。

### SEED-Bench-2（离线 cache ✅，未开始评测）

---

## 已知数据状态

| 数据集 | 路径 | 状态 |
|--------|------|------|
| MME | `data/benchmarks_local/lmms-lab/MME/` | ✅ load_from_disk 正常，2374 题 |
| TextVQA | `data/benchmarks_local/lmms-lab/textvqa/` | ✅ 3 splits (train/val/test)，validation=5000 题 |
| SEED-Bench-2 | `data/benchmarks_local/lmms-lab/SEED-Bench-2/` | ✅ 重建完成，24371 行，5 个 IPC stream 分片 |
| SEED 原始 parquet | `data/seed_parquet/` | ✅ 418/418 完整（已验证可读） |
| SEED 损坏旧版 | `data/benchmarks_local/lmms-lab/seed_broken_backup/` | ⚠️ 备份，勿用 |
| MMMU | `data/MMMU/data/validation-00000-of-00001.parquet` | ✅ 900 题 |
| COCO calib | `data/coco/sharegpt4v_coco_calib.json` | ✅ 128 samples |
| COCO 图片 | `data/coco/train2017/` | ✅ |

### 后续 benchmark 未下载（需 token）

| Benchmark | 大小 | lmms-eval task | 下载方式 |
|-----------|------|----------------|----------|
| MathVista | ~1GB | `mathvista_testmini` | `scripts/download_benchmarks.sh mathvista`（需 HF_TOKEN） |
| MMLU | ~500MB | `mmlu` | `scripts/download_benchmarks.sh mmlu`（需 HF_TOKEN） |
| CMMMU | TBD | `cmmmu_val` | ModelScope |

> ModelScope 下载端点已验证可达（`/api/v1/datasets/<org>/<repo>/repo?FilePath=...`），HF 超时不可达。

---

## Scale cache（全部就绪，无需重跑校准）

| 方法 | 文件 | 大小 |
|------|------|------|
| AWQ W4 | `MLLMQ/scale_cache/awq/internvl2_InternVL2-8B_w4a16.pt` | 1.4 MB |
| AWQ W8 | `MLLMQ/scale_cache/awq/internvl2_InternVL2-8B_w8a16.pt` | 1.4 MB |
| AWQ W3 | `MLLMQ/scale_cache/awq/internvl2_InternVL2-8B_w3a16.pt` | 1.4 MB |
| MBQ W4 | `MLLMQ/scale_cache/mbq/internvl2_w4a16.pt` | 1.4 MB |
| MBQ W8 | `MLLMQ/scale_cache/mbq/internvl2_w8a16.pt` | 1.4 MB |
| MBQ W3 | `MLLMQ/scale_cache/mbq/internvl2_w3a16.pt` | 1.4 MB |
| MLLMQ W4 | `MLLMQ/scale_cache/mllmq/internvl2_w4a16.pt` | 1.4 MB |
| MLLMQ W8 | `MLLMQ/scale_cache/mllmq/internvl2_w8a16.pt` | 1.4 MB |
| MLLMQ W3 | `MLLMQ/scale_cache/mllmq/internvl2_w3a16.pt` | 1.4 MB |
| RTN | 无需 scale | — |

---

## 代码修复清单（已应用，重启后自动生效）

| # | 文件 | 修复内容 | 原因 |
|---|------|----------|------|
| 1 | `MLLMQ/tasks_local/textvqa_utils.py:61-68` | `output_path=None` → try/except fallback 到 `/tmp` | fp16 TextVQA 崩溃 |
| 2 | `scripts/run_repair_queue.sh` | GPU 门控 ≤2 并发（`eval_count()` 检查） | ≥3 进程同卡 OOM |
| 3 | `q-awq-quantizer.py:74` (已修) | `n_bit=` → `n_bits=` | AWQ eval 崩（两处 typo 之一） |
| 4 | `q-awq-auto-scale.py:97` (已修) | 同上 | AWQ scale search 崩（已应用，notification.md §5.2） |
| 5 | `MBQ/3rdparty/lmms-eval/lmms_eval/models/internvl2.py:142` | `torch.linspace` force-CPU monkey patch | transformers ≥5 兼容 |
| 6 | `MLLMQ/main_quant_vl.py` | `mark_tied_weights_as_initialized` 补丁 | transformers ≥5 兼容 |

---

## 关键文件位置速查

```
/nas3/kj/mllmq/
├── experiment_status.md          ← 你正在读（重启快照）
├── dialog.md                     ← 诊断记录（Q1-Q11、坑）
├── notification.md               ← 复现指南（命令 + 补丁 + FAQ）
├── run.md                        ← 原始计划（参考用）
│
├── MLLMQ/                        ← 主仓库
│   ├── eval_benchmarks.py        ★ 通用多 benchmark 评测
│   ├── eval_mmmu.py              ★ MMMU 评测
│   ├── main_quant_vl.py          ★ 量化入口（4 方法）
│   ├── tasks_local/              ★ 本地 task yaml（无 token）
│   │   ├── mme_local.yaml
│   │   ├── textvqa_val_local.yaml
│   │   └── seedbench_2_local.yaml
│   ├── scale_cache/{awq,mbq,mllmq}/  ★ 已缓存的 scale
│   └── scripts/collect_results.py    ★ 跨 benchmark 汇总
│
├── MBQ/3rdparty/lmms-eval/       ← 评估框架（已 editable 安装）
│
├── scripts/
│   ├── run_repair_queue.sh       ★ 单卡修复队列（GPU 门控 ≤2）
│   ├── download_benchmarks.sh    ★ MathVista/MMLU 下载
│   ├── setup_bench.sh            ★ SEED-Bench-2 重建
│   └── continue_seed_download.py ★ ModelScope 补全下载
│
├── checkpoints/
│   ├── InternVL2-8B/             ★ 主实验模型（~17GB bf16）
│   └── Qwen2.5-VL-7B-Instruct/   ★ 跨模型（16GB，待实验）
│
├── data/
│   ├── benchmarks_local/lmms-lab/
│   │   ├── MME/                  ✅
│   │   ├── textvqa/              ✅
│   │   └── SEED-Bench-2/         ✅（已重建）
│   └── seed_parquet/             ✅ 418 原生 parquet
│
└── output/
    └── internvl2-8b/eval/
        ├── mme_local/            ← MME 结果
        └── textvqa_val_local/    ← TextVQA 结果
```

---

## 待补 cell（15 cell）

```
MME (3):         awq_w8  mbq_w8  mllmq_w8
TextVQA (13):    fp16  +  rtn/awq/mbq/mllmq × w4/w8/w3
```

> RTN W4 TextVQA 之前跑到 2290/5000（46%）被 kill，无 checkpoint，需重跑。
> 每个 cell ~2.5h（MME 2374 题）或 ~2-2.5h（TextVQA 5000 题）

---

## 重启命令

### 前提：确认 GPU

```bash
nvidia-smi
# 应有 cuda:0 和 cuda:1 两张 A100 80GB
```

### 单卡方案（默认）

```bash
cd /nas3/kj/mllmq
export HF_DATASETS_OFFLINE=1
bash scripts/run_repair_queue.sh   # GPU 门控 ≤2，设备 cuda:0
```

### 双卡方案（推荐，~22h 完）

```bash
cd /nas3/kj/mllmq
export HF_DATASETS_OFFLINE=1

# 卡0: MME W8 缺口 (3) + TextVQA W4 (4) + fp16 (1) = 8 cell
# run_repair_queue.sh 已设 --device cuda:0

# 卡1: TextVQA W8 (4) + W3 (4) = 8 cell
# 需要先修改脚本中所有 --device cuda:0 为 --device cuda:1
# 然后：
nohup bash scripts/run_repair_queue.sh > output/logs/repair_gpu0.log 2>&1 &
nohup bash scripts/run_repair_queue_gpu1.sh > output/logs/repair_gpu1.log 2>&1 &
```

### 门控机制（已内置）

`run_repair_queue.sh` 每轮先检查 `pgrep -fc eval_benchmarks.py`，≥2 时 sleep 60s 等待。双卡需各自独立门控（分别检查 cuda:0 和 cuda:1 上的进程数）。

### 监控命令

```bash
# 实时看精度进度
tail -f /nas3/kj/mllmq/output/logs/repair_*_$(date +%Y%m%d)*.log | grep -oP '\d+/\d+'

# 看 GPU
watch -n 10 nvidia-smi

# 看已完成 cell
ls -lt /nas3/kj/mllmq/output/internvl2-8b/eval/mme_local/*/2026*_results.json
ls -lt /nas3/kj/mllmq/output/internvl2-8b/eval/textvqa_val_local/*/2026*_results.json
```

---

## 后续扩展计划

### Phase 2：跨模型（Qwen2.5-VL-7B / Janus-Pro-7B）

> 模型都在 `checkpoints/`，待 InternVL2 多 benchmark 跑完再启用。
> 命令参考：`notification.md §3` (Qwen) 和 `notification.md §4` (Janus)。

### Phase 3：SEED-Bench-2 四方法评测

```bash
cd /nas3/kj/mllmq/MLLMQ
export HF_DATASETS_OFFLINE=1

# 每个 method + w_bit 组合，用 --tasks seedbench_2_local
python eval_benchmarks.py --method mllmq --w_bit 4 \
    --scale_path scale_cache/mllmq/internvl2_w4a16.pt \
    --tasks seedbench_2_local --device cuda:0 --batch_size 1 \
    --output_base /nas3/kj/mllmq/output/internvl2-8b/eval
```

### Phase 4：MathVista / MMLU（需 token + 联网下载）

```bash
# 下载
bash scripts/download_benchmarks.sh mathvista  # 需 HF_TOKEN
bash scripts/download_benchmarks.sh mmlu       # 需 HF_TOKEN

# 或通过 ModelScope（HF 超时情况下的备选，已验证可达）
```

### 完成后汇总

```bash
cd /nas3/kj/mllmq/MLLMQ
python scripts/collect_results.py \
    --output_dir /nas3/kj/mllmq/output/internvl2-8b/eval \
    --tasks mme_local,textvqa_val_local,seedbench_2_local \
    --csv
```

---

## 已完成 cell 详细文件列表（对账用）

```
MME ✅ (10 cell，W4 全 + W8 RTN + W3 全):
  rtn_w4a16/20260914_230802_rtn_w4_mme_local_results.json   ← 有效
  rtn_w4a16/20260914_203704_*.json                           ← 废弃 (cogn=25)
  rtn_w4a16/20260915_153134_*.json                           ← 废弃 (cogn=25)
  rtn_w4a16/20260915_153333_*.json                           ← 废弃 (cogn=N/A)
  rtn_w8a16/20260916_035716_rtn_w8_mme_local_results.json    perc=1628 cogn=573
  rtn_w3a16/20260916_035552_rtn_w3_mme_local_results.json    perc=1532 cogn=544
  awq_w4a16/20260916_014704_awq_w4_mme_local_results.json    perc=1636 cogn=561
  awq_w3a16/20260916_082920_awq_w3_mme_local_results.json    perc=1605 cogn=521
  mbq_w4a16/20260915_191810_mbq_w4_mme_local_results.json    perc=1597 cogn=559
  mbq_w3a16/20260916_130627_mbq_w3_mme_local_results.json    perc=1576 cogn=501
  mllmq_w4a16/20260915_163623_mllmq_w4_mme_local_results.json perc=1641 cogn=562
  mllmq_w3a16/20260916_173441_mllmq_w3_mme_local_results.json perc=1578 cogn=543

TextVQA ❌ (0 cell，全部待跑)
```