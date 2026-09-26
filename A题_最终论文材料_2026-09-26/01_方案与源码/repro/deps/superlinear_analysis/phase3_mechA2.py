# -*- coding: utf-8 -*-
"""Phase 3 机制A(定制版): batch(分量组)主序 × level 次序 的核内子图重标。

- 弱连通分量 = 独立链/链束; batch = 若干分量 (贪心按 op 数装满 B);
- 每核子图序 = [batch1 的 level 子图序列, batch2 的 ...];
- step1 DFS 序经 _prioritize_task_seq 分桶后 = 批内层主序 → M/V 管线跨链交错,
  批间串行 → 活跃张量宽度 ≤ B 的分量, 控制 step2 spill。
- op->核 分配与池方案完全相同, 仅子图标签/顺序不同。
"""
import csv
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

BASE = Path(r"C:/shumo_live/02_求解/A题_2026")
sys.path.insert(0, str(BASE / "fast_eval"))
sys.path.insert(0, str(BASE / "v3_solver"))
sys.path.insert(0, r"C:/shumo_live/a_data/code")
sys.path.insert(0, str(BASE / "n5_push" / "superlinear_analysis"))

from fast_eval_p2 import FastEvalP2  # noqa: E402
from common import op_dag, load_case  # noqa: E402

OUT = BASE / "n5_push" / "superlinear_analysis"


def comp_depth(graph):
    """返回 comp_of, depth (仅非 COPY 算子)。comp 用弱连通分量。"""
    ids, preds, succs = op_dag(graph)
    adj = defaultdict(set)
    for v in ids:
        for w in succs[v]:
            adj[v].add(w)
            adj[w].add(v)
    comp_of = {}
    cid = 0
    for v in sorted(ids):
        if v in comp_of:
            continue
        stack = [v]
        comp_of[v] = cid
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in comp_of:
                    comp_of[w] = cid
                    stack.append(w)
        cid += 1
    depth = {}
    q = deque()
    for v in ids:
        if not preds[v]:
            depth[v] = 0
            q.append(v)
    while q:
        u = q.popleft()
        for w in succs[u]:
            nd = depth[u] + 1
            if w not in depth or nd > depth[w]:
                depth[w] = nd
                q.append(w)
    return comp_of, depth


def relabel(plan, op_core, comp_of, depth, B):
    core_ops = defaultdict(list)
    for op, c in op_core.items():
        core_ops[c].append(op)
    n_cores = len(plan["core_schedules"])
    new_n2s, new_cs = {}, [[] for _ in range(n_cores)]
    sgid = 0
    for c in range(n_cores):
        ops = core_ops.get(c, [])
        # 该核的分量按最小 op id 排序 (稳定), 贪心装 batch
        comps = defaultdict(list)
        for o in ops:
            comps[comp_of[o]].append(o)
        order = sorted(comps, key=lambda k: min(comps[k]))
        batch, size = [], 0
        batches = []
        for k in order:
            sz = len(comps[k])
            if batch and size + sz > B:
                batches.append(batch)
                batch, size = [], 0
            batch.append(k)
            size += sz
        if batch:
            batches.append(batch)
        for batch in batches:
            bset = set(batch)
            bops = [o for o in ops if comp_of[o] in bset]
            maxd = max(depth[o] for o in bops)
            for lev in range(maxd + 1):
                lev_ops = [o for o in bops if depth[o] == lev]
                if not lev_ops:
                    continue
                sgid += 1
                for op in lev_ops:
                    new_n2s[str(op)] = sgid
                new_cs[c].append(sgid)
    return {"node_to_subgraph": new_n2s, "core_schedules": new_cs}


def main():
    # 从 phase1_metrics.csv 拿各例当前方案文件 (自己实测过 mk 的)
    plans = {}
    for fn in ("phase1_metrics.csv", "phase1_metrics_extra.csv"):
        p = OUT / fn
        if not p.exists():
            continue
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("plan_file") and "coverage" not in r["plan_file"]:
                    plans[r["case"]] = r["plan_file"]
    pool = json.load(open(BASE / "audit_20260924_latest" /
                          "final_preview_q2.json", encoding="utf-8"))
    cases = sys.argv[1:]
    Bs = [8, 12, 16, 24, 32, 48, 64, 96]
    results = []
    for case in cases:
        if case not in plans:
            print(f"{case}: no verified plan file in phase1 csv")
            continue
        ppath = BASE / plans[case]
        graph = load_case(case)
        fe = FastEvalP2(graph)
        d = json.load(open(ppath, encoding="utf-8"))
        plan0 = d.get("plan", d)
        sc_mk = json.load(open(BASE / "results" / "singlecore" /
                               f"{case}_sc.json", encoding="utf-8"))["makespan"]
        mk0, _ = fe.evaluate(plan0)
        comp_of, depth = comp_depth(graph)
        core_of_sg = {sg: c for c, sgl in enumerate(plan0["core_schedules"])
                      for sg in sgl}
        op_core = {op: core_of_sg[sg] for op, sg in
                   {int(k): v for k, v in
                    plan0["node_to_subgraph"].items()}.items()}
        best = (mk0, None, None)
        line = [f"{case}: pool mk {mk0} sp {sc_mk/mk0:.3f}"]
        for B in Bs:
            try:
                plb = relabel(plan0, op_core, comp_of, depth, B)
                mk1, info = fe.evaluate(plb)
            except Exception as e:
                line.append(f"B{B}:ERR")
                continue
            d_pct = 100 * (mk1 - mk0) / mk0
            if mk1 < best[0]:
                best = (mk1, B, info)
            line.append(f"B{B}:{mk1}({d_pct:+.1f}%)")
        if best[1]:
            info = best[2]
            line.append(f" BEST B={best[1]} sp {sc_mk/best[0]:.3f} "
                        f"spill {info['spill_added_copy_bytes']/1e6:.2f}MB "
                        f"cross {info['cross_task_traffic']/1e6:.2f}MB")
            results.append({"case": case, "mk0": mk0, "mk_best": best[0],
                            "B": best[1],
                            "d_pct": round(100 * (best[0] - mk0) / mk0, 2),
                            "sp0": round(sc_mk / mk0, 4),
                            "sp_new": round(sc_mk / best[0], 4)})
        print(" | ".join(line), flush=True)
    if results:
        with open(OUT / "phase3_mechA2.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)


if __name__ == "__main__":
    main()
