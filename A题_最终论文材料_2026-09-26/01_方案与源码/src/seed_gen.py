# -*- coding: utf-8 -*-
"""失衡种子发生器:耦合密度生长的"重核+外围"结构(agent C 胜解模式的原理化)。

k-core 已证退化(密织均匀)→ 重核改为**实例特定的高耦合邻域生长**:
  从耦合度最高的 op 出发,贪心吸收"与区域耦合最强"的邻居,直到区域功 = α·W;
  区域 → 重核(核0,切拓扑段);外围 → 其余核按拓扑段均衡。
参数:α ∈ {0.3,0.4,0.5,0.6} × 起点 ∈ {最大耦合 op, 最深 CP op, 随机×2}。
合法:复用 _components_plan(分量+SCC+Kahn)。
"""
import os
import sys
import json
import warnings
import heapq
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


def imbalance_seeds(g, K=5, alphas=(0.3, 0.4, 0.5, 0.6), n_random=2):
    ids, preds, succs = op_dag(g)
    w = cycles_map(g)
    W = sum(w[v] for v in ids)
    prod, cons, tsize, _ = tensor_views(g)
    tprod = {}
    for o, ts in prod.items():
        for t in ts:
            tprod[t] = o
    idset = set(ids)
    nbr = defaultdict(lambda: defaultdict(float))   # v -> u -> 耦合字节
    for o, ts in cons.items():
        if o not in idset:
            continue
        for t in ts:
            p = tprod.get(t)
            if p is not None and p != o and p in idset:
                b = tsize.get(t, 0)
                nbr[o][p] += b
                nbr[p][o] += b
    # 拓扑
    pd = defaultdict(int)
    for v in ids:
        for s in succs.get(v, ()):
            pd[s] += 1
    dq = deque(sorted(v for v in ids if pd[v] == 0))
    topo = []
    while dq:
        v = dq.popleft()
        topo.append(v)
        for s in sorted(succs.get(v, ())):
            pd[s] -= 1
            if pd[s] == 0:
                dq.append(s)
    tpos = {v: i for i, v in enumerate(topo)}
    # CP 终点
    dist = {}
    for v in topo:
        dist[v] = w[v] + max((dist.get(p, 0) for p in preds.get(v, ())),
                             default=0)
    cp_end = max(ids, key=lambda v: dist[v])
    # 起点集
    import random as _rnd
    rng = _rnd.Random(11)
    starts = [max(ids, key=lambda v: sum(nbr[v].values())), cp_end]
    starts += [rng.choice(list(ids)) for _ in range(n_random)]

    out = []
    for si, start in enumerate(starts):
        for alpha in alphas:
            target = alpha * W
            region = {start}
            rw = w[start]
            # 堆:(耦合到区域的负值, tpos, v)
            heap = []
            for u, b in nbr[start].items():
                if u not in region:
                    heapq.heappush(heap, (-b, tpos.get(u, 0), u))
            while rw < target and heap:
                negb, _, u = heapq.heappop(heap)
                if u in region:
                    continue
                region.add(u)
                rw += w[u]
                for x, b in nbr[u].items():
                    if x not in region:
                        heapq.heappush(heap, (-b, tpos.get(x, 0), x))
            # 重核 = region(核0),外围均衡
            co = {}
            loads = defaultdict(float)
            for v in region:
                co[v] = 0
            loads[0] = rw
            per = W - rw
            for v in topo:
                if v not in region:
                    c = min(range(1, K), key=lambda x: loads[x])
                    co[v] = c
                    loads[c] += w[v]
            plan = _components_plan(g, co, K, ids, succs)
            out.append((f"imbal_a{int(alpha*100)}_s{si}", plan))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()
    out = HERE / "imbal_out"
    out.mkdir(exist_ok=True)
    ev = ev_p3 if a.q == 3 else ev_p2
    pool = json.load(open(HERE / f"POOL_q{a.q}_N5.json"))
    for case in a.cases.split(","):
        g = load_case(case)
        sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
        pv = pool.get(case, 0)
        best = None
        for tag, plan in imbalance_seeds(g, a.K):
            try:
                r, _ = ev(g, plan)
                sp = sc / r["makespan"]
                if best is None or sp > best[0]:
                    best = (sp, tag, plan)
            except Exception:
                continue
        if best:
            if best[0] > pv:
                json.dump({"plan": best[2], "sp": best[0], "kind": best[1]},
                          open(out / f"{case}_q{a.q}_N{a.K}.json", "w"))
            print(f"{case}: 直接失衡种子 {best[0]:.3f} ({best[1]}) "
                  f"vs 池 {pv:.3f}", flush=True)
        else:
            print(f"{case}: 全部非法", flush=True)


if __name__ == "__main__":
    main()
