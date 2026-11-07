# 给「另一台机（4 GPU）」Claude 的指令

把下面整段直接粘贴给另一台机上的 Claude：

---

你在共享 NFS 仓库 `/nas3/kangjia/mllmq` 上工作（Janus-Pro 7B 量化论文实验）。本机有 **4 张 GPU**。本机任务是跑 TODO.md「重点实验」里分配给远端的一批，**不要碰本机正在写的目录**：
- 禁止修改/清理：`output/janus-pro-7b/mllmq_vs_mbqscalar_full/mllmq_mae_w4_n128_full/`、`.../mllmq_mae_w4_n256_full/`（另一台机正在生成）。
- 你只写：`output/janus-pro-7b/ablation_a1_full/`、`output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3/`、`output/janus-pro-7b/ablation_a13/`、`output/janus-pro-7b/ablation_a2/`。

## 任务（已写成一个可续跑的 4-lane 队列）
1. **A1 全量最后一格**：`mbq_block_w4_n256`（已生成 696/2212，续跑）
2. **W3 全量正确对照 n=256**：`mllmq_mae_w3_n256`（scale 已存在，无需校准）
3. **W3 全量正确对照 n=256**：`mbq_scalar_w3`（scale 已存在）
4. **A13 k=2**：强制放大 T2I 重加权 ×2（n=128 W4，GenEval 25/tag 子集）
5. **A13 k=3**：同上 ×3
6. 可选（`WITH_A2=1`）：A2 校准配比鲁棒性（6 scale + 6 子集 eval，4 lane 跑）

预计：前 5 项约 7–8h（W3 全量每项 ~7h，A1 剩余 ~5h，A13 每项 ~2.5h），A2 再加 ~5h。

## 执行步骤
1) 确认环境：
```bash
cd /nas3/kangjia/mllmq/MLLMQ
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
ls -la scale_cache/mbq/janus_mbq_block_w4a16.pt \
       scale_cache/mllmq/janus_mae_w3a16_n256.pt \
       scale_cache/mbq/janus_mbq_scalar_w3a16.pt
test -f ../data/benchmarks_gen/geneval/evaluation/.torch_home/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth && echo FRCNN_OK
find ../data/benchmarks_gen/geneval/evaluation/.ms_cache -name open_clip_pytorch_model.bin | head -1
```
若只有部分卡空闲，用 `GPUS="0 1 2"` 之类缩小 lane 数（默认 `"0 1 2 3"`）。

2) 预览队列（不占 GPU）：
```bash
PLAN=1 bash scripts/run_remote_priority_queue.sh
```
应看到 5 行（A1 / W3×2 / A13×2）。

3) 后台启动：
```bash
cd /nas3/kangjia/mllmq/MLLMQ
setsid nohup bash scripts/run_remote_priority_queue.sh \
  > /nas3/kangjia/mllmq/output/logs/remote_q_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```
（连跑 A2：`WITH_A2=1 setsid nohup bash scripts/run_remote_priority_queue.sh ... &`）

4) 监控：
```bash
ls -t /nas3/kangjia/mllmq/output/logs/remote_q_*.log | head -1 | xargs tail -50
for d in ../output/janus-pro-7b/ablation_a1_full/mbq_block_w4_n256_full \
         ../output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3/mllmq_mae_w3_n256_full \
         ../output/janus-pro-7b/mllmq_vs_mbqscalar_full_w3/mbq_scalar_w3_n256_full \
         ../output/janus-pro-7b/ablation_a13/t2ix2 ../output/janus-pro-7b/ablation_a13/t2ix3; do
  echo -n "$(basename $d): "; find $d -name '*.png' 2>/dev/null | wc -l
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

## 完成标准 / 回报
- `mbq_block_w4_n256` 的 Overall（对照全量：n64=0.62164、n128=0.64142）
- `mllmq_mae_w3_n256` / `mbq_scalar_w3_n256` 的 Overall（W3 全量，方法区分度最大的档位）
- `t2ix2` / `t2ix3` 的 Overall（对照 150 子集：k=1=0.63500、mbq_scalar=0.60667、mbq_block=0.63333）
- 任何 FAIL 的 tag 及日志末尾报错

注意：
- 4 个进程同时跑，CPU/内存占用高（每个约 27GB RSS）；若机器内存 <160GB，改成 `GPUS="0 1"` 两 lane 串行，避免 OOM。
- 不要同时起第二份相同脚本；重启前先 `pkill -f run_remote_priority_queue.sh`。
- 容器若重启，重新执行第 3 步即可续跑，已完成项自动跳过。