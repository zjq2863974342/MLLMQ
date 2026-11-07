#!/usr/bin/env python3
"""Generate GenEval significance-aware HTML report."""
import json
import collections
import numpy as np

BASE = "/nas3/kangjia/mllmq/output/janus-pro-7b/geneval"

configs = ['fp16','rtn_w8','rtn_w4','rtn_w3','awq_w8','awq_w4','awq_w3',
           'mbq_block_w8','mbq_block_w4','mbq_block_w3','mbq_scalar_w8','mbq_scalar_w4','mbq_scalar_w3',
           'mllmq_mae_w8','mllmq_mae_w4','mllmq_mae_w3','mllmq_mse_w8','mllmq_mse_w4','mllmq_mse_w3']
TAGS = ['single_object','two_object','counting','colors','position','color_attr']
TAGS_CN = {'single_object':'单物体','two_object':'双物体','counting':'计数','colors':'颜色','position':'位置','color_attr':'颜色-属性'}

data = {}
for c in configs:
    by_tag = collections.defaultdict(list)
    for l in open(f'{BASE}/{c}/results.jsonl'):
        d = json.loads(l)
        by_tag[d['tag']].append(1 if d['correct'] else 0)
    data[c] = {t: np.array(by_tag[t], dtype=float) for t in TAGS}

def overall(c):
    return float(np.mean([data[c][t].mean() for t in TAGS]))

def paired_boot(c1, c2, seed=7, NITER=9999):
    rng = np.random.default_rng(seed)
    de = np.empty(NITER)
    for i in range(NITER):
        ds = []
        for t in TAGS:
            a1, a2 = data[c1][t], data[c2][t]
            n = min(len(a1), len(a2))
            idx = rng.integers(0, n, n)
            ds.append(a1[idx].mean() - a2[idx].mean())
        de[i] = np.mean(ds)
    p2 = 2 * min(np.mean(de <= 0), np.mean(de >= 0))
    lo, hi = np.percentile(de, [2.5, 97.5])
    return float(np.mean(de)), float(lo), float(hi), float(p2)

def prop_z(a, b):
    n1, k1 = len(a), a.sum()
    n2, k2 = len(b), b.sum()
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    from math import erf, sqrt
    return 1 - erf(abs(p1 - p2) / (sqrt(2) * se))

def sig(p):
    if p < 0.001:
        return '<span class="sig sig3">***</span>'
    if p < 0.01:
        return '<span class="sig sig2">**</span>'
    if p < 0.05:
        return '<span class="sig sig1">*</span>'
    return '<span class="ns">ns</span>'

def ci_str(lo, hi):
    return f'[{lo:+.3f},{hi:+.3f}]'

METHODS = ['rtn','awq','mbq_block','mbq_scalar','mllmq_mae','mllmq_mse']

# mae vs each method per width
mae_vs = ''
for w in [8, 4, 3]:
    for m in METHODS:
        if m == 'mllmq_mae':
            continue
        d, lo, hi, p = paired_boot(f'mllmq_mae_w{w}', f'{m}_w{w}')
        mae_vs += (f'<tr><td><b>mllmq_mae_w{w}</b></td><td>{m}_w{w}</td>'
                   f'<td class="num">{d:+.4f}</td><td class="num small">{ci_str(lo, hi)}</td><td>{sig(p)}</td></tr>\n')
    d, lo, hi, p = paired_boot(f'mllmq_mae_w{w}', 'fp16')
    mae_vs += (f'<tr><td><b>mllmq_mae_w{w}</b></td><td>fp16</td>'
               f'<td class="num">{d:+.4f}</td><td class="num small">{ci_str(lo, hi)}</td><td>{sig(p)}</td></tr>\n')

# W3 all-pairs
w3_pairs = ''
for i, m1 in enumerate(METHODS):
    for m2 in METHODS[i + 1:]:
        d, lo, hi, p = paired_boot(f'{m1}_w3', f'{m2}_w3')
        w3_pairs += (f'<tr><td class="mname">{m1}_w3</td><td class="mname">{m2}_w3</td>'
                     f'<td class="num">{d:+.4f}</td><td class="num small">{ci_str(lo, hi)}</td><td>{sig(p)}</td></tr>\n')

# W3 per-task
task_rows = ''
for t in TAGS:
    mae = data['mllmq_mae_w3'][t]; rtn = data['rtn_w3'][t]; mse = data['mllmq_mse_w3'][t]
    n = len(mae)
    dr = mae.mean() - rtn.mean()
    pr = prop_z(mae, rtn)
    dm = mae.mean() - mse.mean()
    pm = prop_z(mae, mse)
    task_rows += f'''<tr>
      <td>{TAGS_CN[t]}<div class="tag-en">{t}</div></td>
      <td class="num">{data['fp16'][t].mean():.3f}</td>
      <td class="num rtn">{rtn.mean():.3f}</td>
      <td class="num mae">{mae.mean():.3f}</td>
      <td class="num">{dr * 100:+.1f}pp</td><td>{sig(pr)}</td>
      <td class="num">{dm * 100:+.1f}pp</td><td>{sig(pm)}</td>
    </tr>\n'''

sens = []
for t in TAGS:
    sens.append((data['fp16'][t].mean() - data['rtn_w3'][t].mean(), t))
sens.sort(reverse=True)

def pct(c):
    return 100 * overall(c) / overall('fp16')

cells = {}
for m in METHODS:
    for w in [8, 4, 3]:
        c = f'{m}_w{w}'
        cells[c] = f'{overall(c):.4f}<div class="ret">({pct(c):+.1f}%)</div>'

html = f'''<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<title>GenEval 量化评测报告 — Janus-Pro-7B（含显著性）</title>
<style>
body {{ font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; margin:0; padding:24px 40px; background:#0f172a; color:#e2e8f0; line-height:1.55; }}
.wrap{{max-width:1120px;margin:0 auto}}
h1{{font-size:26px;margin:8px 0 4px}}
h2{{font-size:19px;margin:36px 0 12px;padding-bottom:6px;border-bottom:2px solid #1e293b;color:#f1f5f9}}
h3{{font-size:15px;margin:24px 0 8px;color:#94a3b8}}
.sub{{color:#64748b;font-size:13px;margin-bottom:18px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}}
.card{{flex:1;min-width:180px;background:#1e293b;border-radius:10px;padding:16px 18px;border:1px solid #334155}}
.card .num{{font-size:26px;font-weight:700}}
.card .lbl{{font-size:12px;color:#94a3b8;margin-top:4px}}
table{{border-collapse:collapse;width:100%;margin:12px 0 8px;font-size:13.5px}}
th{{background:#1e293b;color:#94a3b8;font-weight:600;padding:7px 9px;text-align:center;border:1px solid #334155}}
td{{padding:7px 9px;text-align:center;border:1px solid #334155;background:#16202e}}
td.mname{{text-align:left;font-weight:600;color:#e2e8f0}}
td.num{{font-variant-numeric:tabular-nums}}
.ov{{font-weight:700;font-size:14.5px}}
.ret{{font-size:11.5px;color:#94a3b8}}
tr:hover td{{background:#1c2a3d}}
.sig{{font-weight:700;font-size:12px}}
.sig3{{color:#4ade80}}.sig2{{color:#a3e635}}.sig1{{color:#facc15}}.ns{{color:#64748b}}
.rtn{{color:#f87171}}.mae{{color:#6ee7b7}}.awq{{color:#fbbf24}}
.note{{background:#1c2536;border-left:4px solid #4ade80;padding:12px 16px;border-radius:6px;font-size:13px;color:#cbd5e1;margin:18px 0}}
.findings li{{margin:8px 0}}
.tag-en{{color:#475569;font-size:11px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600;margin-right:6px}}
.b-gold{{background:#6d28d9;color:#e9d5ff}}.b-good{{background:#065f46;color:#a7f3d0}}
.b-warn{{background:#92400e;color:#fde68a}}.b-bad{{background:#991b1b;color:#fecaca}}
.small{{font-size:12px;color:#94a3b8}}
.legend{{font-size:12.5px;color:#94a3b8;margin:6px 0}}
</style></head><body><div class="wrap">

<h1>GenEval 量化效果评测报告 <span style="font-size:14px;color:#64748b">含显著性分析</span></h1>
<div class="sub">模型 Janus-Pro-7B · 评分器 torchvision Faster R-CNN（非论文 Mask2Former，绝对值仅横向可比）· 每配置 6 子任务各 320–400 张 · 生成于 2026-10-12</div>

<div class="cards">
  <div class="card"><div class="num">0.6456</div><div class="lbl">fp16 baseline</div></div>
  <div class="card"><div class="num">0.6502</div><div class="lbl">W8 最佳 mllmq_mae</div></div>
  <div class="card"><div class="num">0.6386</div><div class="lbl">W4 最佳 mbq_block</div></div>
  <div class="card"><div class="num">0.5848</div><div class="lbl">W3 最佳 mllmq_mae</div></div>
  <div class="card"><div class="num" style="color:#fca5a5">0.4663</div><div class="lbl">W3 最差 rtn</div></div>
</div>

<h2>方法学：这份报告怎么判断「可信」</h2>
<div class="note">不是只看点估计排名。两个估计相差 0.0011 / 0.5pp / 1pp 在很多配置间落在噪声内。以下全部判断基于：
<ul class="findings" style="margin:8px 0 0">
  <li><b>paired bootstrap delta</b>（9999 次，seed 固定）对任意两配置的差值分布 → 双尾 p 值。样本为同一 553 组 prompt，天然配对，比独立样本检验更有力</li>
  <li><b>two-proportion z-test</b> 对子任务级（n=320–400）</li>
  <li>图例：<span class="sig3">***</span> p&lt;0.001 <span class="sig2">**</span> p&lt;0.01 <span class="sig1">*</span> p&lt;0.05 <span class="ns">ns</span> 不显著</li>
</ul>
</div>

<h2>Overall 主表（方法 × 位宽）</h2>
<div class="small">括号内为相对 fp16 留存率。点估计差异是否可信见下方两两显著性表。</div>
<table>
  <tr><th>方法</th><th>W8</th><th>W4</th><th>W3</th></tr>
  <tr><td class="mname" style="color:#a5b4fc">fp16</td>
    <td style="background:#1e293b">{cells['rtn_w8'].split('<div')[0]}<div class="ret">baseline</div></td>
    <td style="background:#1e293b">0.6456<div class="ret">baseline</div></td>
    <td style="background:#1e293b">0.6456<div class="ret">baseline</div></td></tr>
  <tr><td class="mname">rtn</td><td>{cells['rtn_w8']}</td><td>{cells['rtn_w4']}</td><td class="rtn">{cells['rtn_w3']}</td></tr>
  <tr><td class="mname">awq</td><td>{cells['awq_w8']}</td><td>{cells['awq_w4']}</td><td class="awq">{cells['awq_w3']}</td></tr>
  <tr><td class="mname">mbq_block</td><td>{cells['mbq_block_w8']}</td><td>{cells['mbq_block_w4']}</td><td>{cells['mbq_block_w3']}</td></tr>
  <tr><td class="mname">mbq_scalar</td><td>{cells['mbq_scalar_w8']}</td><td>{cells['mbq_scalar_w4']}</td><td>{cells['mbq_scalar_w3']}</td></tr>
  <tr><td class="mname" style="color:#6ee7b7">mllmq_mae</td><td>{cells['mllmq_mae_w8']}</td><td>{cells['mllmq_mae_w4']}</td><td class="mae">{cells['mllmq_mae_w3']}</td></tr>
  <tr><td class="mname" style="color:#6ee7b7">mllmq_mse</td><td>{cells['mllmq_mse_w8']}</td><td>{cells['mllmq_mse_w4']}</td><td>{cells['mllmq_mse_w3']}</td></tr>
</table>

<h2>显著性检验：两两对比（paired bootstrap delta）</h2>
<div class="small">Δ = 左方法 − 右方法（Overall）。CI 不跨 0 才显著。注：W8 行中 fp16 格显示的值是 rtn_w8，仅作占位——fp16 本身每宽三列相同。 </div>

<h3>关键：mllmq_mae vs 各方法（按位宽）</h3>
<table>
  <tr><th>左</th><th>右</th><th>Δ Overall</th><th>95% CI</th><th>显著性</th></tr>
  {mae_vs}
</table>
<p class="legend">结论预告：<b>W3 档 mae 对 rtn/awq/mse 均显著（rtn p&lt;1e-3，mse p=0.004）；但 mae vs mbq_block/scalar 均不显著。</b>W4 档仅对 rtn 显著（p=0.041），对 awq/mbq 系全不显著。W8 档无一对显著——W8 全是噪声。</p>

<h3>W3 档内方法间全矩阵</h3>
<table>
  <tr><th>左</th><th>右</th><th>Δ Overall</th><th>95% CI</th><th>显著性</th></tr>
  {w3_pairs}
</table>
<p class="legend">W3 档只有两类显著：① 领先方法（mae/block） vs rtn（p&lt;0.001）② mae vs mse（p=0.004）。其余（mae vs block、mae vs scalar、block vs scalar…）均为噪声内。</p>

<h2>W3 子任务级检验（two-proportion z-test）</h2>
<div class="small">这是信号最强、最能讲故事的维度：每个子任务 n=320–400</div>
<table>
  <tr><th>子任务</th><th>fp16</th><th>rtn_w3</th><th>mllmq_mae_w3</th><th>mae−rtn</th><th>p</th><th>mae−mse</th><th>p</th></tr>
  {task_rows}
</table>

<h3>量化敏感度排名（fp16 − rtn_w3 退化幅度）</h3>
<p>{' ＞ '.join(f'<b>{TAGS_CN[t]}</b>(−{d*100:.0f}pp)' for d, t in sens)}</p>

<h2>主结论（只声明有显著性的）</h2>
<div class="note"><b>先划范围</b>：本实验每配置仅 2212 张图、子任务 n=320–400，检测到的效应量下限约 1.5–2pp（CI 宽度 ±1.8pp）。<b>低于此的差异不能仅凭本次数据判定为「有」或「无」。</b></div>
<ol class="findings">
  <li><span class="badge b-gold">核心且显著</span> <b>W3 档 rtn 崩溃、mllmq_mae 保住 90.6%，差距真实巨大</b>
      — Δ=+0.1183，95% CI [0.0968, 0.1398]，p&lt;0.001。这个 11.8pp 跨越 5 个 CI 宽度，不是噪声。这是全实验最强、最干净的信号。</li>
  <li><span class="badge b-good">显著</span> <b>W3 档 mllmq_mae &gt; mllmq_mse</b> — Δ=+0.0307 [0.0104, 0.0514]，p=0.0038。同族对比中 mae 目标优于 mse 目标。</li>
  <li><span class="badge b-good">显著</span> <b>W3 mae &gt; awq</b>（Δ=+0.0211，p=0.044）；<b>W4 mae &gt; rtn</b>（Δ=+0.0195，p=0.041）——均勉强过 0.05，属于「存在但较弱」的信号。</li>
  <li><span class="badge b-warn">不显著</span> <b>W3 档 mae vs mbq_block（Δ=0.0011）和 vs mbq_scalar（Δ=0.0130）均不显著</b>
      — 「W3 并列最佳」最多表述为「与 MBQ 家族打平」，不能写成「显著优于 MBQ」。这与您对接近数值的担忧一致。</li>
  <li><span class="badge b-warn">不显著</span> <b>W4/W8 档方法间差异几乎全部 ns</b>      — W4 档「mbq_block 最佳」「mllmq 无优势」均为点估计倾向（14/16 对比不显著）；W8 档 0.639–0.650 区间全是噪声，只有「W8 近似无损」这个弱结论可下可下。</li>
  <li><span class="badge b-warn">子任务证据强</span> <b>mae 相对 rtn 在 5/6 子任务显著</b>      — two_object / counting / colors / position p&lt;0.001，color_attr p=0.006，仅 single_object 不显著（+4.1pp，p=0.12）。跨任务一致性支持结论 1，position +18.3pp / two_object +11.9pp 是量化的真实代价代价。</li>
  <li><span class="badge b-warn">子任务证据弱</span> <b>mae vs mse 在全部 6 个子任务上无一对显著</b>（max p=0.11）
      — mae-vs-mse 的唯一证据是 Overall 的 p=0.004。不能声称每任务都赢 mse。需要 COCO-val 交叉验证。</li>
</ol>

<h2>⚠️ 显著性视角的待验证项</h2>
<table>
  <tr><th>#</th><th>事项</th><th>为什么重要</th></tr>
  <tr><td>1</td><td>多 seed（尤其 mae vs block/scalar、mae vs mse）</td><td>0.5–1.3pp 的差距在本次统计力下不可判定；3–5 seed 才能把「打平」界运维成「有底气」或彻底否定</td></tr>
  <tr><td>2</td><td>Mask2Former 复评</td><td>当前 Faster R-CNN；若换原版评分器后 11.8pp 的 mae-vs-rtn 与 5/6 显著子任务保持，主 story 才稳</td></tr>
  <tr><td>3</td><td>n=256 MLLMQ 消融</td><td>当前 mae n=128 vs 对手 n=256；「样本减半 + 打平」若在 n=256 下变成「超越」，story 更强</td></tr>
</table>

<h2>实验配置</h2>
<div class="small">
<ul>
  <li>每配置 553 prompt × 4 samples = 2212 张；6 子任务 n=320–400</li>
  <li>评分器：torchvision Faster R-CNN（COCO 检测）→ 绝对值不可对齐论文，仅横向有效</li>
  <li>统计：paired bootstrap delta（9999×, seed=7）、two-proportion z-test</li>
  <li>量化校准：AWQ/MBQ n_smp=256，MLLMQ n_smp=128</li>
</ul>
</div>
</div></body></html>'''

with open(f'{BASE}/geneval_report.html', 'w') as f:
    f.write(html)
print('written:', f'{BASE}/geneval_report.html', len(html), 'bytes')