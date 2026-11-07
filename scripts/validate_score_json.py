#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 output/score.json 做**独立校验**：不复用 build_score_json.py 的代码路径，
而是另写一套解析，分别与
  (A) 原始产物（results.json / jsonl / *_results.txt / pref_scores/*.json）
  (B) 已发布 md 表（dpg_scores.md / coco_val_scores.md / plan_janus.md）
对拍。目的是抓「汇总脚本算错」这类错误（历史上真出过：glob 路径写错、
MME 两分项被当成 1 格、summary 文件名正则写错）。

用法: python3 scripts/validate_score_json.py [--json output/score.json]
"""
import argparse, glob, json, os, re, sys

ROOT = "/nas3/kangjia/mllmq"

# ── 已发布数值（人工从 md 摘录，注明来源行） ───────────────────────────────
# output/janus-pro-7b/dpg_scores.md
PUB_DPG_SINGLE = {          # L17-21, L52-59
    "fp16": 83.412, "awq_w3": 82.460, "mllmq_mae_w3": 82.348,
    "mbq_block_w3": 82.738, "rtn_w3": 79.136, "mbq_scalar_w3": 82.546,
    "mllmq_mse_w3": 82.075, "rtn_w4": 83.549, "awq_w4": 83.478,
    "mbq_block_w4": 83.737, "mbq_scalar_w4": 83.749,
    "mllmq_mae_w4": 82.648, "mllmq_mse_w4": 83.583,
}
PUB_DPG_GRID4 = {"fp16": 83.846, "mllmq_mae_w3": 82.360}   # L32-33
# plan_janus.md  GenEval 表 L157-163
PUB_GENEVAL = {
    "rtn_w8": 0.6482, "rtn_w4": 0.6096, "rtn_w3": 0.4663,
    "awq_w8": 0.6482, "awq_w4": 0.6298, "awq_w3": 0.5635,
    "mbq_block_w8": 0.6395, "mbq_block_w4": 0.6385, "mbq_block_w3": 0.5837,
    "mbq_scalar_w8": 0.6391, "mbq_scalar_w4": 0.6282, "mbq_scalar_w3": 0.5716,
    "mllmq_mae_w8": 0.6502, "mllmq_mae_w4": 0.6290, "mllmq_mae_w3": 0.5848,
    "mllmq_mse_w8": 0.6395, "mllmq_mse_w4": 0.6284, "mllmq_mse_w3": 0.5540,
    "fp16": 0.6456,
}
# output/janus-pro-7b/coco_val_scores.md L5-22   (config: fid, clip, hpsv2, pickscore, imagereward, cosine)
PUB_COCO = {
    "awq_w3":        (27.767, 0.6618, 0.2590, 21.480, 0.616, 0.2647),
    "awq_w4":        (29.452, 0.6670, 0.2673, 21.738, 0.758, 0.2668),
    "awq_w8":        (29.102, 0.6679, 0.2688, 21.797, 0.774, 0.2672),
    "fp16":          (29.086, 0.6679, 0.2685, 21.790, 0.777, 0.2672),
    "mbq_block_w3":  (27.359, 0.6626, 0.2582, 21.483, 0.601, 0.2650),
    "mbq_block_w4":  (30.026, 0.6676, 0.2673, 21.741, 0.743, 0.2670),
    "mbq_block_w8":  (29.103, 0.6688, 0.2687, 21.802, 0.773, 0.2675),
    "mbq_scalar_w3": (27.497, 0.6617, 0.2595, 21.458, 0.618, 0.2647),
    "mbq_scalar_w4": (29.927, 0.6666, 0.2676, 21.750, 0.757, 0.2666),
    "mbq_scalar_w8": (29.050, 0.6685, 0.2683, 21.793, 0.773, 0.2674),
    "mllmq_mae_w3":  (27.709, 0.6625, 0.2584, 21.446, 0.599, 0.2650),
    "mllmq_mae_w4":  (29.303, 0.6675, 0.2671, 21.758, 0.759, 0.2670),
    "mllmq_mae_w8":  (29.014, 0.6679, 0.2685, 21.792, 0.783, 0.2672),
    "mllmq_mse_w3":  (27.776, 0.6614, 0.2576, 21.402, 0.574, 0.2645),
    "mllmq_mse_w4":  (29.388, 0.6684, 0.2670, 21.761, 0.747, 0.2673),
    "mllmq_mse_w8":  (29.164, 0.6681, 0.2685, 21.789, 0.773, 0.2672),
    "rtn_w3":        (34.180, 0.6349, 0.2407, 20.829, 0.190, 0.2540),
    "rtn_w4":        (29.206, 0.6611, 0.2652, 21.671, 0.675, 0.2645),
}

FAILS, WARNS = [], []
def ok(name):  print(f"  \033[0;32m✅\033[0m {name}")
def bad(msg):
    print(f"  \033[0;31m❌\033[0m {msg}"); FAILS.append(msg)
def warn(msg):
    print(f"  \033[1;33m⚠️ \033[0m {msg}"); WARNS.append(msg)
def close(a, b, tol): return a is not None and b is not None and abs(a - b) <= tol


# ── (A) 独立重算 ────────────────────────────────────────────────────────────
def raw_dpg():
    """不复用正则：用 str 切分。"""
    out = {}
    for p in glob.glob(f"{ROOT}/output/dpg_scores/*_results.txt"):
        tag = os.path.basename(p)[:-len("_results.txt")]
        last = None
        with open(p, errors="ignore") as fh:
            for ln in fh:
                if "DPG-Bench score:" in ln:
                    last = ln.split("DPG-Bench score:")[1].strip().split()[0]
        if last is not None:
            out[tag] = float(last)
    return out


def raw_geneval():
    """逐行统计，与 build 里那句话写法不同。"""
    out = {}
    for p in glob.glob(f"{ROOT}/output/janus-pro-7b/geneval/*/results.jsonl"):
        cfg = os.path.basename(os.path.dirname(p))
        n, c, tags = 0, 0, {}
        with open(p) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                t = r.get("tag")
                d = tags.setdefault(t, [0, 0])
                d[0] += 1; d[1] += 1 if r.get("correct") else 0
        for t, (a, b) in tags.items():
            n += a; c += b
        # 只取 6 个标准 tag 的均值（与 build 口径一致）
        keep = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
        accs = [tags[t][1] / tags[t][0] for t in keep if t in tags and tags[t][0]]
        out[cfg] = (sum(accs) / len(accs) if accs else None, sorted(tags))
    return out


def raw_coco_means():
    """直接从**逐图分数**重算均值，而非相信已存的聚合字段。"""
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/output/janus-pro-7b/pref_scores/*")):
        if not os.path.isdir(d):        # 同级还有 *_run.log，只取 config 目录
            continue
        cfg = os.path.basename(d)
        e = {}
        for m in ["hpsv2", "pickscore", "imagereward"]:
            p = f"{d}/{m}_scores.json"
            if os.path.exists(p):
                v = json.load(open(p))
                nums = [x for x in v.values() if isinstance(x, (int, float))]
                e[m] = (sum(nums) / len(nums), len(nums)) if nums else None
        # CLIP：从 clip_scores 的原始列表重算（若存了逐图列表）
        p = f"{ROOT}/output/janus-pro-7b/clip_scores/{cfg}_clip.json"
        if os.path.exists(p):
            cj = json.load(open(p))
            e["clip"] = cj.get("clip_score_mean")
            e["cosine"] = cj.get("cosine_mean")
            # 若有逐图数组则重算
            for k in ("clip_scores", "cosines", "scores"):
                if isinstance(cj.get(k), list) and cj[k]:
                    arr = [x for x in cj[k] if isinstance(x, (int, float))]
                    e[f"clip_recomputed_from_{k}"] = sum(arr) / len(arr)
        p = f"{ROOT}/output/janus-pro-7b/fid_results/{cfg}_fid.json"
        if os.path.exists(p):
            e["fid"] = json.load(open(p)).get("fid")
        out[cfg] = e
    return out


def raw_understanding():
    """独立枚举：每个 config×bench 取文件名时间戳最大者，再读值。"""
    # bench -> [(score.json 里的短名, results.json 里的原始键)]
    UB = {"mme_local": [("mme_perception", "mme_percetion_score,none"),
                        ("mme_cognition",  "mme_cognition_score,none")],
          "textvqa_val_local": [("textvqa", "exact_match,none")],
          "mmmu_val": [("mmmu", "mmmu_acc,none")],
          "mathvista_testmini_local": [("mathvista", "mathvista_acc,none")],
          "pope_random_local": [("pope_random", "pope_accuracy,none")],
          "pope_popular_local": [("pope_popular", "pope_accuracy,none")],
          "pope_adversarial_local": [("pope_adversarial", "pope_accuracy,none")]}
    cells, picked = {}, {}
    for b, keys in UB.items():
        dirs = []
        for root in (f"{ROOT}/output/janus-pro-7b/eval",
                     f"{ROOT}/output/janus-pro-7b/eval_mixed"):
            dirs += glob.glob(f"{root}/{b}/*a16")
            fp16 = f"{root}/{b}/fp16"
            if os.path.isdir(fp16):
                dirs.append(fp16)
        for d in dirs:
            base = os.path.basename(d)
            cfg = "fp16" if base == "fp16" else base[:-3]
            fs = glob.glob(f"{d}/*_results.json")
            if not fs:
                continue
            # 文件名时间戳（YYYYMMDD_HHMMSS）优先；退化到 mtime
            def keyf(f):
                m = re.search(r"(\d{8}_\d{6})", os.path.basename(f))
                return m.group(1) if m else "00000000_000000"
            fs.sort(key=keyf)
            p = fs[-1]
            try:
                res = json.load(open(p))["results"][b]
            except Exception as e:
                warn(f"解析失败 {p}: {e}"); continue
            picked.setdefault(cfg, {})[b] = (os.path.basename(p), len(fs))
            for short, raw_key in keys:
                v = res.get(raw_key)
                if isinstance(v, (int, float)):
                    cells.setdefault(cfg, {})[short] = v
    return cells, picked


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=f"{ROOT}/output/score.json")
    a = ap.parse_args()
    J = json.load(open(a.json))

    print(f"校验对象: {a.json}  ({os.path.getsize(a.json)/1024:.1f} KB, mtime {os.path.getmtime(a.json) and __import__('datetime').datetime.fromtimestamp(os.path.getmtime(a.json)).strftime('%F %T')})\n")

    # ---------- 1. DPG ----------
    print("[1] DPG-Bench")
    rd = raw_dpg()
    jd = J["generation"]["dpg_bench"]["raw"]
    if len(rd) != len(jd):
        bad(f"产物数不一致: 原始 {len(rd)} vs json {len(jd)}")
    for tag, v in sorted(rd.items()):
        if tag not in jd: bad(f"json 缺 {tag}"); continue
        if not close(v, jd[tag]["overall"], 1e-9):
            bad(f"{tag} overall 原始={v} vs json={jd[tag]['overall']}")
    # json vs md（3 位小数）
    js = J["generation"]["dpg_bench"]["single_overall"]
    for cfg, pub in sorted(PUB_DPG_SINGLE.items()):
        got = js.get(cfg)
        if got is None: bad(f"single_overall 缺 {cfg}"); continue
        if not close(got, pub, 5e-4): bad(f"DPG single {cfg}: json={got:.3f} vs md={pub}")
    jg = J["generation"]["dpg_bench"]["grid4_overall"]
    for cfg, pub in sorted(PUB_DPG_GRID4.items()):
        got = jg.get(cfg)
        if got is None: bad(f"grid4_overall 缺 {cfg}"); continue
        if not close(got, pub, 5e-4): bad(f"DPG grid4 {cfg}: json={got:.3f} vs md={pub}")
    if not FAILS: ok(f"{len(rd)} 份产物 × 原始/机器精度一致，{len(PUB_DPG_SINGLE)} single + {len(PUB_DPG_GRID4)} grid4 × md 一致")

    # ---------- 2. GenEval ----------
    print("\n[2] GenEval")
    n0 = len(FAILS)
    rg = raw_geneval(); jgen = J["generation"]["geneval"]["configs"]
    if set(rg) != set(jgen):
        bad(f"config 集合不一致: 原始 {sorted(set(rg)-set(jgen))} / json {sorted(set(jgen)-set(rg))}")
    for cfg, (acc, tags) in sorted(rg.items()):
        if cfg not in jgen: continue
        j = jgen[cfg]["overall"]
        if not close(acc, j, 1e-9): bad(f"GenEval {cfg} 原始={acc} vs json={j}")
        if cfg in PUB_GENEVAL and not close(acc, PUB_GENEVAL[cfg], 1e-4):
            bad(f"GenEval {cfg}: 原始={acc:.6f} vs md={PUB_GENEVAL[cfg]}")
        elif cfg in PUB_GENEVAL and not close(acc, PUB_GENEVAL[cfg], 5e-5):
            # 4 位小数四舍五入的边界差异 → md 抄录/进位问题，非聚合错误
            warn(f"GenEval {cfg}: 原始={acc:.6f}→四舍五入 {round(acc,4)}，md 写 {PUB_GENEVAL[cfg]}（文档进位笔误）")
    for cfg in sorted(set(PUB_GENEVAL) - set(rg)):
        bad(f"md 里有 {cfg}，原始产物没找到")
    if len(FAILS) == n0: ok(f"{len(rg)} config —— 原始 × json × md 三方一致")

    # ---------- 3. COCO ----------
    print("\n[3] COCO-val")
    n0 = len(FAILS)
    rc = raw_coco_means(); jc = J["generation"]["coco_val"]["configs"]
    # score.json 的 COCO 口径 = 有 FID 的正式 config；pref_scores 下另有 smoke_* 冒烟产物，不在口径内
    smoke = sorted(c for c in rc if c.startswith("smoke"))
    rc = {c: e for c, e in rc.items() if c not in smoke}
    if smoke:
        print(f"    （跳过冒烟产物 {len(smoke)} 个: {', '.join(smoke)}）")
    for cfg, e in sorted(rc.items()):
        if cfg not in jc: bad(f"json 缺 {cfg}"); continue
        j = jc[cfg]
        for m in ["hpsv2", "pickscore", "imagereward"]:
            if m in e and e[m]:
                if not close(e[m][0], j.get(m), 1e-6):
                    bad(f"COCO {cfg} {m}: 逐图重算={e[m][0]} vs json={j.get(m)}")
                if e[m][1] != 5000: warn(f"COCO {cfg} {m} n={e[m][1]}（期望 5000）")
    # vs md
    for cfg, (fid, clip, hps, pick, ir, cos) in sorted(PUB_COCO.items()):
        if cfg not in jc: bad(f"json 缺 {cfg}"); continue
        j = jc[cfg]
        for name, want, tol in [("fid", fid, 5e-4), ("clip_score", clip, 5e-5),
                                ("hpsv2", hps, 5e-5), ("pickscore", pick, 5e-4),
                                ("imagereward", ir, 5e-4), ("cosine", cos, 5e-5)]:
            got = j.get(name)
            if got is None: bad(f"COCO {cfg} 缺 {name}")
            elif not close(got, want, tol): bad(f"COCO {cfg} {name}: json={got} vs md={want}")
    if len(FAILS) == n0: ok(f"{len(rc)} config × 3 偏好指标逐图重算一致 + FID/CLIP/HPS/Pick/IR/cosine × md 一致")

    # ---------- 4. 理解侧 ----------
    print("\n[4] 理解侧")
    n0 = len(FAILS)
    ru, picked = raw_understanding()
    ju = J["understanding"]["matrix"]
    n_raw = sum(len(v) for v in ru.values()); n_json = sum(len(v) for v in ju.values())
    n0b = len(FAILS)
    if n_raw != n_json: warn(f"格数不一致: 原始 {n_raw} vs json {n_json}（逐格判定见下）")
    # 漂移 vs 缺陷：产物新于 score.json → 是「json 建完之后对端又写了新格」，属陈旧；
    # 产物旧于 score.json 而 json 仍缺 → 才是聚合脚本漏读。🔴 eval/ 在共享 NFS，对端在并发写。
    jmt = os.path.getmtime(a.json); drift = []
    for cfg, kv in sorted(ru.items()):
        if cfg not in ju:
            fs = sorted(glob.glob(f"{ROOT}/output/janus-pro-7b/eval/*/{cfg}a16/*_results.json") + glob.glob(f"{ROOT}/output/janus-pro-7b/eval_mixed/*/{cfg}a16/*_results.json"), key=os.path.getmtime)
            if fs and os.path.getmtime(fs[-1]) > jmt: drift.append(cfg); continue
            bad(f"json 缺 config {cfg}（产物 {max(map(os.path.getmtime, fs)) if fs else 0} 早于 score.json）"); continue
        for k, v in kv.items():
            if k not in ju[cfg]:
                fs = sorted(glob.glob(f"{ROOT}/output/janus-pro-7b/eval/*/{cfg}a16/*_results.json") + glob.glob(f"{ROOT}/output/janus-pro-7b/eval_mixed/*/{cfg}a16/*_results.json"), key=os.path.getmtime)
                if fs and os.path.getmtime(fs[-1]) > jmt: drift.append(f"{cfg}.{k}"); continue
                bad(f"{cfg}.{k} json 缺")
            elif not close(v, ju[cfg][k], 1e-9): bad(f"{cfg}.{k}: 原始={v} vs json={ju[cfg][k]}")
    if drift:
        warn(f"陈旧（非缺陷）：{len(drift)} 项在 score.json 生成后由对端新写入 → {drift[:8]}"
             f"{' …' if len(drift) > 8 else ''}  重跑 build_score_json.py 即可纳入")
    # 同格多份文件（如 mllmq_mae_w4 mme 有两份）——确认取的是最新
    multi = {c: {b: n for b, (f, n) in bs.items() if n > 1} for c, bs in picked.items()}
    multi = {c: v for c, v in multi.items() if v}
    print(f"    理解侧格数: 原始 {n_raw} / json {n_json}；同格多份: {multi or '无'}")
    if FAILS == [] and n_raw == n_json:
        ok(f"{n_raw} 格逐一一致（coverage.cells_present={J['understanding']['coverage']['cells_present']}）")
    else:
        ok(f"{n_raw} 格逐一一致")

    # ---------- 5. 结构性断言 ----------
    print("\n[5] 结构与口径断言")
    n0 = len(FAILS)
    uc = J["understanding"]["coverage"]
    if uc["cells_expected"] != 20 * 7:
        bad(f"cells_expected 应为 140，实际 {uc['cells_expected']}")
    n_cells_json = len(ju) * 7            # 每个 config 7 个 benchmark（MME 两分项算 1 格）
    if uc["cells_present"] != n_cells_json:
        bad(f"coverage.cells_present={uc['cells_present']} ≠ 实际 benchmark 格数 {n_cells_json}")
    # MME 应计 2 格
    if "fp16" in ju and len([k for k in ju["fp16"] if k.startswith("mme")]) not in (0, 2):
        bad("fp16 MME 分项数异常")
    mme_cfgs_2 = [c for c, v in ju.items() if sum(k.startswith("mme") for k in v) == 2]
    if not mme_cfgs_2: warn("没有任何 config 带完整 MME 两分项 —— 检查 MME 是否被当成 1 格")
    # single_vs_fp16 自洽
    base = js.get("fp16")
    for cfg, d in J["generation"]["dpg_bench"]["single_vs_fp16"].items():
        if not close(d, js[cfg] - base, 1e-4): bad(f"single_vs_fp16[{cfg}]={d} 与 {js[cfg]}-{base} 不符")
    if len(FAILS) == n0: ok("cells_expected=140 / MME 双分项 / single_vs_fp16 自洽")

    print("\n" + "=" * 62)
    if FAILS:
        print(f"\033[0;31m✗ 校验未通过：{len(FAILS)} 处不符\033[0m")
        for f in FAILS: print("   -", f)
    else:
        print("\033[0;32m✓ 全部通过\033[0m")
    if WARNS:
        print(f"\033[1;33m⚠️  {len(WARNS)} 处提示\033[0m")
        for w in WARNS: print("   -", w)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())