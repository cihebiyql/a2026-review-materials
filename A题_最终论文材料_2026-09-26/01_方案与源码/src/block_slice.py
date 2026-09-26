# -*- coding: utf-8 -*-
"""block_slice:层块×宽度切片划分(E 类编织 DAG 的第三几何)。

物理:密集层间耦合使得横切串行、纵切付 500c×层数;
块结构 T(k) ≈ W/N + 500×(L/k) + 块内弱边税。
- 块 = 连续 k 层;块内按"前驱多数投票(字节加权)"分组(切弱边);
- 子图 = (块,组);商图块间纯前向(DAG);块内 zigzag 用 SCC 局部合并修复;
- 核内序 = (块序, 组拓扑位次)。
"""
import sys
import os
import json
import warnings
from collections import defaultdict, deque
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get(
    "A2026_SOLVER_DIR",
    r"C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand"))
sys.path.insert(0, r"C:/shumo_live/a_data/code")

from common import load_case, op_dag, cycles_map, tensor_views, ev_p2, ev_p3

SC_DIR = Path(r"C:/shumo_live/02_求解/A题_2026/results/singlecore")


def build_views(g):
    ids, preds, succs = op_dag(g)
    w = cycles_map(g)
    prod, cons, tsize, _ = tensor_views(g)
    tprod = {}
    for o, ts in prod.items():
        for t in ts:
            tprod[t] = o
    ew = defaultdict(float)
    for o, ts in cons.items():
        for t in ts:
            p = tprod.get(t)
            if p is not None and p != o:
                ew[(p, o)] += tsize.get(t, 0)
    pd = defaultdict(int)
    for v in ids:
        for s in succs.get(v, ()):
            pd[s] += 1
    dq = deque(sorted(v for v in ids if pd[v] == 0))
    lvl = {}
    topo = []
    while dq:
        v = dq.popleft()
        topo.append(v)
        lvl[v] = max((lvl.get(p, -1) for p in preds.get(v, ())), default=0) + 1 \
            if preds.get(v) else 0
        for s in succs.get(v, ()):
            pd[s] -= 1
            if pd[s] == 0:
                dq.append(s)
    return ids, preds, succs, w, ew, lvl, topo


def block_slice_plan(g, K=5, k=8, cap_factor=1.35, vote_scale=1e4):
    ids, preds, succs, w, ew, lvl, topo = build_views(g)
    bylvl = defaultdict(list)
    for v in ids:
        bylvl[lvl[v]].append(v)
    L = max(lvl.values()) + 1
    W = sum(w.values())
    # 块划分
    blocks = []
    cur = []
    for l in range(L):
        cur.append(l)
        if len(cur) >= k:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    n2s = {}
    grp_of_sg = {}          # 子图 -> 组号(合并前)
    prev_assign = {}
    sid = 0
    seg_of_sg = []          # 子图 -> 块号
    for bi, levels in enumerate(blocks):
        ops_blk = [v for l in levels for v in bylvl.get(l, ())]
        blk_w = sum(w[v] for v in ops_blk)
        if blk_w < 2 * W / (K * 8):   # 小块整块一组
            for v in ops_blk:
                n2s[v] = sid
            grp_of_sg[sid] = bi % K
            seg_of_sg.append(bi)
            sid += 1
            prev_assign = {}
            continue
        assign = {}
        for l in levels:
            cur_ops = sorted(bylvl.get(l, ()))
            votes = {v: defaultdict(float) for v in cur_ops}
            for v in cur_ops:
                for p in preds.get(v, ()):
                    src = assign.get(p, prev_assign.get(p))
                    if src is not None:
                        votes[v][src] += 1.0 + ew.get((p, v), 0.0) / vote_scale
            tw = sum(w[v] for v in cur_ops)
            cap = tw / K * cap_factor
            gw = defaultdict(float)
            order = sorted(cur_ops, key=lambda v: (-max(votes[v].values(),
                                                       default=0.0), v))
            groups = defaultdict(list)
            for v in order:
                vs = sorted(votes[v].items(), key=lambda kv: -kv[1])
                placed = False
                for c2, _ in vs:
                    if gw[c2] + w[v] <= cap:
                        groups[c2].append(v)
                        gw[c2] += w[v]
                        placed = True
                        break
                if not placed:
                    cmin = min(range(K), key=lambda c: gw[c])
                    groups[cmin].append(v)
                    gw[cmin] += w[v]
            for c2, ms in groups.items():
                for v in ms:
                    assign[v] = c2
            prev_assign = dict(assign)
        for c2 in range(K):
            ms = [v for v in ops_blk if assign.get(v) == c2]
            if not ms:
                continue
            for v in ms:
                n2s[v] = sid
            grp_of_sg[sid] = c2
            seg_of_sg.append(bi)
            sid += 1
    # 商图 SCC 修复(块间前向,环仅块内 zigzag)
    sg_edges = defaultdict(set)
    for v in ids:
        for s in succs.get(v, ()):
            a, b = n2s[v], n2s[s]
            if a != b:
                sg_edges[a].add(b)
    # Tarjan
    index, low, onstk = {}, {}, {}
    stk, comp = [], {}
    nc, counter = 0, 0
    for s0 in range(sid):
        if s0 in index:
            continue
        work = [(s0, iter(sorted(sg_edges[s0])))]
        index[s0] = low[s0] = counter
        counter += 1
        stk.append(s0)
        onstk[s0] = True
        while work:
            x, it = work[-1]
            adv = False
            for y in it:
                if y not in index:
                    index[y] = low[y] = counter
                    counter += 1
                    stk.append(y)
                    onstk[y] = True
                    work.append((y, iter(sorted(sg_edges[y]))))
                    adv = True
                    break
                if onstk.get(y):
                    low[x] = min(low[x], index[y])
            if adv:
                continue
            work.pop()
            if work:
                px = work[-1][0]
                low[px] = min(low[px], low[x])
            if low[x] == index[x]:
                while True:
                    y = stk.pop()
                    onstk[y] = False
                    comp[y] = nc
                    if y == x:
                        break
                nc += 1
    if nc != sid:
        # 合并成员到代表(取块号最小)
        rep = {}
        members = defaultdict(list)
        for s in range(sid):
            members[comp[s]].append(s)
        for c in members.values():
            r = min(c)
            for s in c:
                rep[s] = r
        n2s = {v: rep[s] for v, s in n2s.items()}
        sid = len(members)
    # 子图拓扑位次
    sg_min = defaultdict(lambda: 1 << 60)
    tpos = {v: i for i, v in enumerate(topo)}
    for v in ids:
        sg_min[n2s[v]] = min(sg_min[n2s[v]], tpos[v])
    # 核分配:块内组号均衡放置(按块功贪心最闲核)
    blk_load = defaultdict(float)
    sg_blk_w = defaultdict(float)
    for v in ids:
        sg_blk_w[n2s[v]] += w[v]
    core_of = {}
    blk_ids = defaultdict(list)
    rep_ids = sorted(set(n2s.values()))
    rep_grp = {}
    for s0 in range(sid):
        if s0 in grp_of_sg:
            rep_grp.setdefault(comp.get(s0, s0), grp_of_sg[s0])
    # 用合并后的子图->块(用 sg_min 所在层近似:重新算块)
    lvl_of_sg = {}
    for v in ids:
        sg = n2s[v]
        if sg not in lvl_of_sg or lvl_of_sg[sg] > lvl[v]:
            lvl_of_sg[sg] = lvl[v]
    blkseq = {}
    for bi, levels in enumerate(blocks):
        blkseq[levels[0]] = bi
    for s in range(sid):
        l0 = lvl_of_sg.get(s, 0)
        # 找所在块
        bi = min(range(len(blocks)),
                 key=lambda i: abs(blocks[i][0] - l0))
        blk_ids[bi].append(s)
    for s in rep_ids:
        core_of[s] = rep_grp.get(s, 0) % K
        blk_load[core_of[s]] += sg_blk_w[s]
    cores = [[] for _ in range(K)]
    for s, c in core_of.items():
        cores[c].append(s)
    for c in range(K):
        cores[c].sort(key=lambda s: sg_min[s])
    return {"node_to_subgraph": {str(v): s for v, s in n2s.items()},
            "core_schedules": cores}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--ks", default="2,4,8,16,32")
    a = ap.parse_args()
    out = HERE / "block_slice_out"
    out.mkdir(exist_ok=True)
    posthoc = json.load(open(
        r"C:/shumo_live/02_求解/A题_2026/a_lab/records/POSTHOC_BEST.json"))
    for case in a.cases.split(","):
        g = load_case(case)
        sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
        ev = ev_p3 if a.q == 3 else ev_p2
        best = None
        for k in [int(x) for x in a.ks.split(",")]:
            try:
                plan = block_slice_plan(g, a.K, k=k)
                r, wt = ev(g, plan)
                sp = sc / r["makespan"]
                if best is None or sp > best[1]:
                    best = (k, sp, plan, r["makespan"])
            except Exception as exc:
                print(f"  {case} k={k} fail: {str(exc)[:60]}")
        if best:
            base = posthoc.get(f"{case}|q{a.q}|N{a.K}", 0)
            pool_now = max(base, 0)
            json.dump({"plan": best[2], "mk": best[3], "sp": best[1],
                       "k": best[0]},
                      open(out / f"{case}_q{a.q}_N{a.K}.json", "w"))
            print(f"{case}: k={best[0]} sp={best[1]:.3f} "
                  f"(posthoc {base:.3f}, {'+' if best[1]>base else ''}"
                  f"{(best[1]/base-1)*100 if base else 0:.0f}%)",
                  flush=True)


if __name__ == "__main__":
    main()
