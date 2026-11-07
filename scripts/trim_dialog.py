#!/usr/bin/env python3
"""Trim redundant MONITOR snapshot blocks from dialog.md.

Deletes:
  1. All `[MONITOR-ISSUE]`  (2026-09-20/21 OOM polling, 41 blocks)
  2. Pure `[MONITOR]` snapshot blocks whose body is only GPU/RAM/png
     numbers with no analysis, AND whose title carries no milestone marker.

Keeps:
  - `[恢复会话]` / `[补记]` / `[实测]` blocks (substantive session recovery docs)
  - `[MONITOR]` blocks that contain ETA/rate handoff analysis or milestone events
  - Everything non-MONITOR (BUG records, experiment matrices, literature, etc.)

Backed up to backen/dialog.md.bak before trimming (caller responsibility).
"""
import re
import sys

PATH = '/nas3/kangjia/mllmq/dialog.md'
lines = open(PATH, encoding='utf-8').readlines()

blocks = []
cur = None
start = 0
pat_h = re.compile(r'^#{1,3} ')
pat_mon = re.compile(r'^### \[(MONITOR-ISSUE|MONITOR)\]|^### \[(恢复会话|补记|实测)\]')
for i, l in enumerate(lines):
    m = pat_mon.match(l)
    if m:
        if cur:
            blocks.append((cur, start, i))
        cur = l.strip()
        start = i
        continue
    if cur is not None and pat_h.match(l):
        blocks.append((cur, start, i))
        cur = None
if cur:
    blocks.append((cur, start, len(lines)))

# Milestone / analysis markers in title
TITLE_KEEP = re.compile(r'🎉|🔧|🔻|🟡|🟢|🔴|GPU0 彻底|订正|判别|DONE|重置|排查|根治|慢档|双界|首个|第\s*[0-9]+\s*次|速率|GPU0 慢|恢复|订正')

# ANALYSIS tokens — any of these in a [MONITOR] body means it's substantive
ANALYSIS_TOKENS = re.compile(
    r'ETA|交接|速率|验证|完成|DONE|里程碑|判别|订正|慢档|双界|自查|自愈|stale|'
    r'排查|根治|订正|GPU0 慢|看门狗|错误|报错|异常|D=2|恢复会话|补记|实测|'
    r'第\s*[0-9]+\s*次|首个|减速|加速|诊断|翻|dump|辛苦|打过|剩余量|节奏|出关|'
    r'▸|▶|重新|重试|失败|成功|切换|过会|夹|爬坡|下探|回升|回升|持平|回落|validation'
)

# numbers like "2/19 configs complete" / "7/2212 images" are pure progress
PROGRESS_ONLY = re.compile(
    r'^(?:📈\s*)?GenEval:\s*[0-9]+/[0-9]+\s*configs?.{0,40}$'
    r'|^(?:📈\s*)?\d+\.\d+|^\d+\/\d+\s*(?:images|configs|prompts).{0,30}$'
)

# pure snapshot test: block has <=1 non-empty body line that is NOT
#   a pure GPU/RAM/png/status line
PURE_LINE = re.compile(
    r'(?:📊|💾|📌|📈|📝|No\s+.*\s+running|⚠️\s+GPU.*LOW.*|GPU\d*[=:]\s*\d+.*|^\s*[-*]\s*(?:GPU|RAM|进程|All|Driver|Status|No|⚠).*|^\s*\d+$|^\s*$)'
)

def analysis_line_count(body_lines):
    """Count body lines that carry analysis beyond raw numbers/progress."""
    c = 0
    for l in body_lines:
        l = l.strip()
        if not l or l.startswith('###'):
            continue
        if PROGRESS_ONLY.search(l):
            continue  # pure progress line, not analysis
        if ANALYSIS_TOKENS.search(l):
            c += 1
    return c

delete = []
keep = []
for head, s, e in blocks:
    body_lines = [l.rstrip() for l in lines[s:e]]
    body = '\n'.join(body_lines)
    if re.match(r'^### \[MONITOR-ISSUE\]', head):
        delete.append((head, s, e))
    elif re.match(r'^### \[(恢复会话|补记|实测)\]', head):
        keep.append((head, s, e))  # always keep substantive
    else:
        # [MONITOR]
        alc = analysis_line_count(body_lines[1:])
        if TITLE_KEEP.search(head) or alc >= 1:
            keep.append((head, s, e))
        else:
            # no analysis line at all → pure snapshot
            delete.append((head, s, e))

# Drop zero-length false spans (should not happen)
delete = [d for d in delete if d[2] - d[1] > 0]

del_rows = sum(e - s for _, s, e in delete)
report = []
report.append(f"Total blocks scanned: {len(blocks)}")
report.append(f"DELETE {len(delete)} blocks / {del_rows} rows ({del_rows/len(lines)*100:.1f}% of file)")
report.append(f"KEEP   {len(keep)} blocks")
report.append("")
report.append("---- DELETED (first 50) ----")
for head, s, e in delete[:50]:
    report.append(f"  L{s+1}-{e} [{e-s:2d}行] {head.strip()[:70]}")
if len(delete) > 50:
    report.append(f"  ... and {len(delete)-50} more")

out = '\n'.join(report)
if '--preview' in sys.argv:
    print(out)
else:
    # remove spans from highest line to lowest
    idx = sorted((s, e) for _, s, e in delete)
    new = []
    last = 0
    for s, e in idx:
        new.extend(lines[last:s])
        last = e
    new.extend(lines[last:])
    with open(PATH, 'w', encoding='utf-8') as f:
        f.write(''.join(new))
    print(f"[exec] WROTE {PATH}: {len(new)} lines (was {len(lines)}), -{len(lines)-len(new)} rows")
    print(out)
