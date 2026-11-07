# TODO — Janus-Pro MLLMQ（2026-11-07 精简版）

> 本次更新前备份：`TODO.md.bak-20261107_2050`；更早归档：`TODO.md.bak-20261105_150038`。
> 结果快照：`dialog.md`（待同步）；论文结构：`paper_outline.md`。
> **结论：理解侧全部完成；生成侧只剩 T2I-CompBench 一列**（W3A16 已生成待评分，W4A8 生成中）。

## 0. 状态总览（2026-11-07 20:50）

| 项 | 状态 |
|---|---|
| W3A16 GenEval full / MMMU / 5h 理解 / SEED full | ✅ 完成 |
| W3A16 DPG unified_384（8 config，另含 mbq_block_masked 对照） | ✅ 1065/1065 + 评分完成 |
| W3A16 awq / gen_only 补齐（und + SEED） | ✅ 完成 |
| W3A16 T2I-CompBench P1（8 config × 2400） | ✅ 生成完成；❌ 未评分 |
| W4A8 scales / und / GenEval600 | ✅ 完成 |
| W4A8 DPG（8 config） | ✅ 1065/1065 + 评分完成 |
| W4A8 SEED full（8 config × 5 chunks） | ✅ 完成 |
| W4A8 T2I-CompBench（8 config × 2400） | 🔄 local4 在跑，各 728–1148/2400；未评分 |
| COCO / Show-o | ⏸ 暂不跑 / 另议 |

## 1. 剩余工作（只保留未完成项）

1. 等 local4 主机跑完 W4A8 T2I-CompBench 生成（8 config × 2400）。
2. 给 W3A16 + W4A8 共 16 个 config 的 T2I-CompBench 结果评分，填 §3.2 的 `T2I-comp` 列。
   - P1 生成时口径是“生成后不套旧评分脚本”，评分前需确认当前 8 类 × 300 的评分方式。
3. 生成/评分完成后同步 `dialog.md` 与 `paper_outline.md`。

**运行信息（只读）**
- W4A8 T2I lane：`queue/t2i_w4a8_lane_queue.sh`；日志 `output/logs/local4_supplement/t2i_w4a8/<tag>.log`；产物 `output/janus-pro-7b/t2i_compbench_current/w4a8/<tag>/`。
- W3A16 T2I P1：产物 `output/janus-pro-7b/t2i_compbench_current/<tag>/`；日志 `output/logs/t2i_compbench_current/<tag>.log`。
- 定时 watchdog 仍在 `.claude/scheduled_tasks.json`；**不要在副机执行 `queue/start_remote_queue.sh`**（本机 GPU 当前空闲）。
- 本机（副机）已无队列任务；W4A8 T2I 由 local4 主机运行，写共享盘 `/nas3`。

## 2. 口径（必须遵守）

- 模型：Janus-Pro-7B；w_group=128，zero_point=True，仅 LM 30 层。
- 环境：base `/opt/conda/bin/python`（torch 2.4.0+cu124 / transformers 4.45.0）；`venvs/` 只用于 Show-o/T2I 评分。
- W3A16：生成侧用 `unified_384/*`；理解侧 scalar/channel 用 `unified_understanding_128/*`，t2i 变体用 `unified_384/*`；gen_only 用 `unified_generation_only/*`。
- W4A8：生成侧用 `scale_cache/w4a8/mix384/*`；理解侧 scalar/channel 用 `w4a8/und128/*`，t2i 变体用 `w4a8/mix384/*`。
- AWQ：W3A16 生成用 `unified_384/awq_w3a16.pt`、理解/SEED 用 `unified_understanding_128/awq_w3a16.pt`；W4A8 生成用 `w4a8/mix384/awq_w4a8.pt`、理解/SEED 用 `w4a8/und128/awq_w4a8.pt`。
- DPG 分数取 `dpg_scores/<tag>.txt` 最后一行 `DPG-Bench score`；每个 tag 有 1 张 `partiprompts97.png` 评分失败，实际按 1064 张计算。
- SEED 为 5 chunks 去重后均值；SEED 严格 1 进程/卡，DPG ≤2 进程/卡。
- AWQ / gen_only 的 W3A16 GenEval600 为 matched 600 子集；全量 2212 仅作参考（awq .55744、gen_only .55198）。

## 3. 结果表

> `—` = 未测。MME 为子类原始分之和（P max 2000 / C max 800）。

### 3.1 理解侧

| 位宽 | config | MMMU | MME-P | MME-C | TextVQA | MathVista | SEED |
|---|---|---|---|---|---|---|---|
| W3A16 | rtn | .35000 | 1409.4 | 245.0 | .48814 | 33.5 | .4767 |
| W3A16 | mbq_scalar | .33778 | 1400.0 | 270.7 | .49448 | 28.3 | .4617 |
| W3A16 | channel_only | .35111 | 1383.1 | 277.1 | .49942 | 28.5 | .4687 |
| W3A16 | t2i_text | .33333 | 1423.5 | 297.1 | .47830 | 29.9 | .4681 |
| W3A16 | t2i_img | .35111 | 1367.0 | 303.2 | .47388 | 29.0 | .4481 |
| W3A16 | both | .33778 | 1409.9 | 283.6 | .47246 | 28.9 | .4428 |
| W3A16 | awq | .33333 | 1395.0 | 283.9 | .49260 | 28.5 | .4493 |
| W3A16 | gen_only | .38000 | 1286.9 | 288.6 | .49284 | 30.8 | .4660 |
| W4A8 | rtn | .28444 | 1088.6 | 170.7 | .40862 | 25.0 | .1522 |
| W4A8 | mbq_scalar | .35222 | 1368.9 | 284.3 | .51330 | 33.4 | .4796 |
| W4A8 | channel_only | .34444 | 1383.2 | 261.8 | .50868 | 33.8 | .4808 |
| W4A8 | t2i_text | .30111 | 1379.4 | 241.8 | .50634 | 31.8 | .4349 |
| W4A8 | t2i_img | .30222 | 1384.1 | 238.2 | .49998 | 31.2 | .4571 |
| W4A8 | both | .31000 | 1428.8 | 281.1 | .50542 | 32.6 | .4430 |
| W4A8 | awq | .28778 | 906.1 | 164.3 | .32636 | 22.9 | .2372 |
| W4A8 | gen_only | .32889 | 1485.6 | 291.1 | .49522 | 33.5 | .4643 |

- W3A16 gen_only 的理解/SEED 使用“仅生成校准”的 `unified_generation_only/mllmq_gen_only_w3a16.pt`，用于检验跨侧迁移。
- W4A8 awq 理解/SEED 明显偏低（MMMU .2878、SEED .2372），与 DPG 69.88 一致，是 AWQ 在 W4A8 的真实退化。

### 3.2 生成侧

| 位宽 | config | GenEval600 | DPG | T2I-comp |
|---|---|---|---|---|
| W3A16 | rtn | .45000 | 80.34 | 已生成 / 待评分 |
| W3A16 | mbq_scalar | .54167 | 82.19 | 已生成 / 待评分 |
| W3A16 | channel_only | .53000 | 81.86 | 已生成 / 待评分 |
| W3A16 | t2i_text | .55167 | 82.07 | 已生成 / 待评分 |
| W3A16 | t2i_img | .52833 | 81.46 | 已生成 / 待评分 |
| W3A16 | both | .55167 | 82.47 | 已生成 / 待评分 |
| W3A16 | awq | .55167 | 82.47 | 已生成 / 待评分 |
| W3A16 | gen_only | .57000 | 81.55 | 已生成 / 待评分 |
| W4A8 | rtn | .02833 | 42.15 | 生成中 |
| W4A8 | mbq_scalar | .49667 | 80.39 | 生成中 |
| W4A8 | channel_only | .48000 | 80.13 | 生成中 |
| W4A8 | t2i_text | .54833 | 80.97 | 生成中 |
| W4A8 | t2i_img | .52500 | 80.09 | 生成中 |
| W4A8 | both | .51667 | 81.30 | 生成中 |
| W4A8 | awq | .30833 | 69.88 | 生成中 |
| W4A8 | gen_only | .53167 | 80.76 | 生成中 |

- W3A16 T2I-comp：8 config × 2400 张（8 类 × 300）已生成完，尚未评分。
- W4A8 T2I-comp：截至 2026-11-07 20:50 各 tag 728–1148/2400；按 ~68s/图，ETA 约 11-08 20:30 至 11-09 04:15。
- W3A16 DPG 另有 `mbq_block_masked` 对照 **82.58**（不在主表 8 config 内）。
- W4A8 `rtn` GenEval600 .02833、DPG 42.15 为生成侧崩盘；学习型方法在 DPG 80–81、GenEval600 .48–.55。

## 4. 归档

- 本次更新前 TODO：`TODO.md.bak-20261107_2050`
- 旧 TODO 全文：`TODO.md.bak-20261105_150038`
- 旧 dialog 全文：`dialog.md.bak-20261105_012640`、`dialog.md.bak-20261102_2035`