# -*- coding: utf-8 -*-
"""第一性原理算法:DSC 主导序列聚簇 + 核-外围剥皮分解。

模型(已证):T ≥ max over 链(链功 + 500·跨核边数) + DDR地板。
DSC(Yang&Gerasoulis 1994 思想的忠实实现):
  - 每轮计算含通信的关键路径(同簇边代价=0,跨簇=500+b/60)
  - 只归零主导序列上能缩短 DS 的边(合并簇),簇功 ≤ W/N×slack
  - 收敛后簇→核均衡映射
剥皮核-外围:
  - 迭代删 degree≤1 的点 → 密织核;核给重核(1-2个),外围按拓扑段均衡
两个都产合法方案(SCC 兜底 + Kahn 核内序)。
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

SC_DIR = Path(r"C:/shumo_live/02_求解/A题_2026/results/singlecore")
DELAY = 500.0
BW = 60.0


class Model:
    def __init__(self, g):
        self.ids, self.preds, self.succ = op_dag(g)
        self.w = cycles_map(g)
        prod, cons, tsize, _ = tensor_views(g)
        tprod = {}
        for o, ts in prod.items():
            for t in ts:
                tprod[t] = o
        self.comm = defaultdict(float)   # (u,v) -> 500 + b/60
        for o, ts in cons.items():
            for t in ts:
                p = tprod.get(t)
                if p is not None and p != o:
                    self.comm[(p, o)] += DELAY + tsize.get(t, 0) / BW
        # 拓扑序
        pd = defaultdict(int)
        for v in self.ids:
            for s in self.succ.get(v, ()):
                pd[s] += 1
        dq = deque(sorted(v for v in self.ids if pd[v] == 0))
        self.topo = []
        while dq:
            v = dq.popleft()
            self.topo.append(v)
            for s in sorted(self.succ.get(v, ())):
                pd[s] -= 1
                if pd[s] == 0:
                    dq.append(s)
        self.W = sum(self.w[v] for v in self.ids)

    def cp_with_clusters(self, cl):
        """含通信的关键路径;cl: v->cluster。返回 (DS值, 路径边列表)。"""
        dist = {}
        prev_e = {}
        for v in self.topo:
            best, be = 0.0, None
            for p in self.preds.get(v, ()):
                c = 0.0 if cl[p] == cl[v] else self.comm.get((p, v), DELAY)
                if dist[p] + c > best:
                    best, be = dist[p] + c, (p, v)
            dist[v] = best + self.w[v]
            prev_e[v] = be
        end = max(self.ids, key=lambda v: dist[v])
        path = []
        x = end
        while prev_e[x] is not None:
            path.append(prev_e[x])
            x = prev_e[x][0]
        return dist[end], path[::-1]


def dsc_plan(g, K=5, slack=1.15, max_rounds=400):
    m = Model(g)
    cl = {v: i for i, v in enumerate(m.ids)}
    cw = {i: m.w[v] for i, v in enumerate(m.ids)}
    cap = m.W / K * slack
    for _ in range(max_rounds):
        ds0, path = m.cp_with_clusters(cl)
        # 主导序列上的边,尝试归零(合并簇),取缩短最多者
        best = None
        for (u, v) in path:
            cu, cv = cl[u], cl[v]
            if cu == cv:
                continue
            if cw[cu] + cw[cv] > cap:
                continue
            cl2 = dict(cl)
            for x, c in cl.items():
                if c == cv:
                    cl2[x] = cu
            cw2 = dict(cw)
            cw2[cu] = cw[cu] + cw[cv]
            ds1, _ = m.cp_with_clusters(cl2)
            if best is None or ds1 < best[0]:
                best = (ds1, cl2, cw2)
        if best is None or best[0] >= ds0 - 1e-6:
            break
        _, cl, cw = best
    # 簇 -> 核:均衡 + 拓扑序映射
    clusters = defaultdict(list)
    for v, c in cl.items():
        clusters[c].append(v)
    cl_list = sorted(clusters.items(),
                     key=lambda kv: min(m.topo.index(v) for v in kv[1]))
    loads = [0.0] * K
    co = {}
    for c, vs in cl_list:
        wc = sum(m.w[v] for v in vs)
        # 放到当前最闲核,但保持拓扑连续性(优先放前驱所在核若不超载)
        pref = None
        for v in vs:
            for p in m.preds.get(v, ()):
                if cl[p] != c and co.get(cl[p]) is not None and \
                        loads[co[cl[p]]] + wc <= m.W / K * (slack + 0.5):
                    pref = co[cl[p]]
                    break
            if pref is not None:
                break
        if pref is not None:
            dst = pref
        else:
            dst = min(range(K), key=lambda x: loads[x])
        co[c] = dst
        loads[dst] += wc
    n2s = {v: cl[v] for v in m.ids}
    core_of = {v: co[cl[v]] for v in m.ids}
    return finalize(g, n2s, core_of, K, m)


def peel_plan(g, K=5, core_cores=1):
    m = Model(g)
    # 剥皮:迭代删 degree<=1(无向)
    nbr = defaultdict(set)
    for v in m.ids:
        for s in m.succ.get(v, ()):
            nbr[v].add(s)
            nbr[s].add(v)
    deg = {v: len(nbr[v]) for v in m.ids}
    alive = set(m.ids)
    dq = deque(v for v in m.ids if deg[v] <= 1)
    removed = set()
    while dq:
        v = dq.popleft()
        if v not in alive or v in removed:
            continue
        removed.add(v)
        alive.discard(v)
        for u in nbr[v]:
            if u in alive:
                deg[u] -= 1
                if deg[u] <= 1:
                    dq.append(u)
    core = alive
    periph = set(m.ids) - core
    corew = sum(m.w[v] for v in core)
    perw = m.W - corew
    n2s = {}
    co = {}
    # 核:切成 core_cores 份(拓扑连续段),放前 core_cores 核
    clist = [v for v in m.topo if v in core]
    sid = 0
    seg_n = max(1, len(clist) // core_cores)
    for i in range(0, len(clist), seg_n):
        seg = clist[i:i + seg_n]
        for v in seg:
            n2s[v] = sid
            co[v] = min(i // seg_n, core_cores - 1)
        sid += 1
    # 外围:均衡到其余核(拓扑贪心)
    rest = [c for c in range(K) if c >= core_cores]
    if not rest:
        rest = list(range(K))
    loads = {c: corew if c < core_cores else 0.0 for c in range(K)}
    target = perw / max(1, len(rest))
    for v in m.topo:
        if v in periph:
            c = min(rest, key=lambda x: loads[x])
            n2s[v] = sid
            co[v] = c
            loads[c] += m.w[v]
            sid += 1
    # 外围过碎会多子图;聚成拓扑段
    return finalize(g, n2s, co, K, m)


def finalize(g, n2s, core_of, K, m):
    """连通分量成子图 + SCC 修复 + Kahn 核内序(复用 strand_p2)。"""
    from strand_p2 import _components_plan
    return _components_plan(g, core_of, K, list(m.ids), m.succ)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()
    posthoc = json.load(open(
        r"C:/shumo_live/02_求解/A题_2026/a_lab/records/POSTHOC_BEST.json"))
    out = HERE / "dsc_out"
    out.mkdir(exist_ok=True)
    ev = ev_p3 if a.q == 3 else ev_p2
    for case in a.cases.split(","):
        g = load_case(case)
        sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
        poolv = json.load(open(HERE / f"POOL_q{a.q}_N5.json")).get(case, 0)
        results = {}
        for name, fn in (("dsc", lambda: dsc_plan(g, a.K)),
                         ("peel1", lambda: peel_plan(g, a.K, 1)),
                         ("peel2", lambda: peel_plan(g, a.K, 2))):
            try:
                plan = fn()
                r, _ = ev(g, plan)
                results[name] = (sc / r["makespan"], plan)
            except Exception as exc:
                results[name] = (None, str(exc)[:60])
        line = " ".join(f"{k}={v[0]:.3f}" if v[0] else f"{k}=X" 
                        for k, v in results.items())
        bestk = max((k for k, v in results.items() if v[0]),
                    key=lambda k: results[k][0], default=None)
        if bestk and results[bestk][0] > poolv:
            json.dump({"plan": results[bestk][1],
                       "sp": results[bestk][0], "kind": bestk},
                      open(out / f"{case}_q{a.q}_N5.json", "w"))
        print(f"{case}: {line} | 池 {poolv:.3f}", flush=True)


if __name__ == "__main__":
    main()
