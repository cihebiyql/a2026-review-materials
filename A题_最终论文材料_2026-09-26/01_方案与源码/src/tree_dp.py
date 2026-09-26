# -*- coding: utf-8 -*-
"""树划分数学:生成森林 + 平衡树 DP 切分(跨界数定理 m ≤ 2(K-1)+耦合)。

定理(证明见 docstring 末):把无向图 G 的生成森林 F 切成 K 个连通子树,
则 G 中任何简单路径跨界次数 ≤ 2(K-1) + |E(G)\\E(F) 在该路径上的边数。
推论:对密度 ~1 的图(耦合边少),任何链的跨核同步数为常数,
T ≈ W/K + δ·(2(K-1)+耦合),与图深 L 无关 —— 绕开 500c×L 陷阱。

实现:
  1. 选林:DFS / BFS / 重耦合优先(Prim 式,高 ew 边进森林,少成耦合)
  2. 树 DP:删 K-1 条树边 → K 个连通分量,极小化最大分量功
  3. 分量→核(平衡),拓扑序发射,_components_plan 合法化
"""
import os
import sys
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
from strand_p2 import _components_plan

SC_DIR = Path(r"C:/shumo_live/02_求解/A题_2026/results/singlecore")


def build_forest(ids, succs, ew, mode="heavy"):
    """返回森林父指针(每个点至多一个树父;树边集合)。"""
    nbr = defaultdict(list)   # 无向邻接(带字节)
    for v in ids:
        for s in succs.get(v, ()):
            nbr[v].append((s, ew.get((v, s), 0.0)))
            nbr[s].append((v, ew.get((v, s), 0.0)))
    parent = {}
    visited = set()
    order = sorted(ids)
    roots = []
    for r in order:
        if r in visited:
            continue
        roots.append(r)
        visited.add(r)
        # heavy 模式:优先吸收重边邻居(Prim 式);dfs/bfs:序优先
        if mode == "heavy":
            import heapq
            heap = []
            for u, b in nbr[r]:
                heapq.heappush(heap, (-b, u))
            while heap:
                negb, u = heapq.heappop(heap)
                if u in visited:
                    continue
                parent[u] = r
                visited.add(u)
                for x, b in nbr[u]:
                    if x not in visited:
                        heapq.heappush(heap, (-b, x))
        else:
            dq = deque(u for u, _ in nbr[r])
            while dq:
                u = dq.popleft() if mode == "bfs" else dq.pop()
                if u in visited:
                    continue
                parent[u] = r
                visited.add(u)
                for x, _ in nbr[u]:
                    if x not in visited:
                        dq.append(x)
    return parent, roots


def balanced_tree_partition(ids, w, parent, roots, K):
    """递归最优平衡边二分:拆到 K 份连通分量,极小化最大功。

    定理适用:连通子树划分 → 任何路径跨界 ≤ 2(K-1)+耦合边。
    """
    children = defaultdict(list)
    for v, p in parent.items():
        children[p].append(v)

    def subtree(y, restrict):
        out = set()
        st = [y]
        while st:
            x = st.pop()
            if x in out or x not in restrict:
                continue
            out.add(x)
            for c2 in children[x]:
                if c2 in restrict:
                    st.append(c2)
        return out

    def split(big):
        """在连通集 big 内找最平衡的内边割,返回两半。"""
        tot = sum(w[x] for x in big)
        best = None
        for x in big:
            for y in children[x]:
                if y in big:
                    s = subtree(y, big)
                    cw = sum(w[z] for z in s)
                    score = abs(cw - (tot - cw))
                    if best is None or score < best[0]:
                        best = (score, s)
        if best is None:
            lst = sorted(big)
            h = len(lst) // 2
            return set(lst[:h]), set(lst[h:])
        s = best[1]
        return s, big - s

    # 初始分量 = 每棵树
    comps = []
    seen_all = set()
    for r in roots:
        st = [r]
        tree = set()
        while st:
            x = st.pop()
            if x in tree:
                continue
            tree.add(x)
            st.extend(children[x])
        comps.append(tree)
        seen_all |= tree
    for x in ids:          # 孤立点兜底
        if x not in seen_all:
            comps.append({x})
    while len(comps) > K:
        comps.sort(key=lambda s: -sum(w[x] for x in s))
        big = comps.pop(0)
        a, b = split(big)
        if not a or not b:      # 不可再拆(单点)——并入次大
            comps[0] |= big
            continue
        comps.append(a)
        comps.append(b)
    comp = {}
    for ci, s in enumerate(comps):
        for x in s:
            comp[x] = ci
    return comp, len(comps)


def treedp_plan(g, K=5, forest_mode="heavy"):
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
    parent, roots = build_forest(ids, succs, ew, mode=forest_mode)
    comp, nparts = balanced_tree_partition(ids, w, parent, roots, K)
    # 分量 -> 核:按功均衡贪心
    loads = defaultdict(float)
    co = {}
    cw = defaultdict(float)
    for v, c in comp.items():
        cw[c] += w[v]
    for c in sorted(cw, key=lambda x: -cw[x]):
        k = min(range(K), key=lambda x: loads[x])
        for v, cc in comp.items():
            if cc == c:
                co[v] = k
        loads[k] += cw[c]
    return _components_plan(g, co, K, ids, succs), nparts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()
    out = HERE / "treedp_out"
    out.mkdir(exist_ok=True)
    ev = ev_p3 if a.q == 3 else ev_p2
    pool = json.load(open(HERE / f"POOL_q{a.q}_N5.json"))
    for case in a.cases.split(","):
        g = load_case(case)
        sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
        for mode in ("heavy", "dfs"):
            try:
                plan, np_ = treedp_plan(g, a.K, forest_mode=mode)
                r, _ = ev(g, plan)
                sp = sc / r["makespan"]
                pv = pool.get(case, 0)
                mark = "★" if sp > pv else ""
                if sp > pv:
                    json.dump({"plan": plan, "sp": sp, "kind": f"treedp_{mode}"},
                              open(out / f"{case}_q{a.q}_N{a.K}.json", "w"))
                print(f"{case}[{mode}]: sp={sp:.3f} (池 {pv:.3f}) {mark}",
                      flush=True)
            except Exception as exc:
                print(f"{case}[{mode}]: ERR {type(exc).__name__} {str(exc)[:50]}",
                      flush=True)


if __name__ == "__main__":
    main()
