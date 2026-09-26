# -*- coding: utf-8 -*-
"""证书目标 MILP:min max(含延迟关键路 DS, 每核负载) —— 三难 Pareto 点的精确解。

变量:x[s,c]∈{0,1}, st[s]≥0 连续, z[e]∈[0,1] 连续(跨核指示), M。
约束:
  Σ_c x[s,c]=1
  st[s] ≥ st[t] + w_t + 500·z[t,s]      (子图 DAG 边 t→s)
  z[t,s] ≥ x[t,c] - x[s,c]  ∀c          (跨核强制 z=1)
  z[t,s] ≥ x[s,c] - x[t,c]  ∀c
  Σ_s w_s·x[s,c] ≤ M                    (负载)
  st[s] + w_s ≤ M
目标:min M。
核内序 = st 升序(天然满足依赖 → 合法)。scipy HiGHS。
"""
import os
import sys
import json
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get(
    "A2026_SOLVER_DIR",
    r"C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand"))
sys.path.insert(0, r"C:/shumo_live/a_data/code")

from common import load_case, op_dag, cycles_map, ev_p2, ev_p3

SC_DIR = Path(r"C:/shumo_live/02_求解/A题_2026/results/singlecore")


def best_seed(case, q=3, K=5):
    best_sp, seed = -1, None
    for d in ("refined7b", "refined7", "refined6", "refined5", "audit_B_best",
              "bxcpu_best", "refined4", "refined3", "refined2", "refined",
              "strand_n5", "arm_g5tf_out", "cross_best"):
        fp = HERE / d / f"{case}_q{q}_N{K}.json"
        if fp.exists():
            try:
                sd = json.load(open(fp))
                if sd.get("plan") and sd.get("sp", 0) > best_sp:
                    best_sp, seed = sd["sp"], sd["plan"]
            except Exception:
                pass
    return seed


def cert_milp_plan(g, seed_plan, K=5, time_limit=120, unit_cap=None):
    ids, preds, succs = op_dag(g)
    w = cycles_map(g)
    n2s = {int(k): v for k, v in seed_plan["node_to_subgraph"].items()}
    # 可选:过粗则按拓扑再分组;过细则合并至 unit_cap
    sg_ops = defaultdict(list)
    for v in ids:
        sg_ops[n2s[v]].append(v)
    sgs = sorted(sg_ops)
    S = len(sgs)
    if unit_cap and S > unit_cap:
        # 按最小拓扑位次排序后贪心合并
        topo_pd = defaultdict(int)
        for v in ids:
            for s2 in succs.get(v, ()):
                topo_pd[s2] += 1
        from collections import deque
        dq = deque(sorted(v for v in ids if topo_pd[v] == 0))
        tpos = {}
        i = 0
        while dq:
            v = dq.popleft()
            tpos[v] = i
            i += 1
            for s2 in sorted(succs.get(v, ())):
                topo_pd[s2] -= 1
                if topo_pd[s2] == 0:
                    dq.append(s2)
        sgs_sorted = sorted(sgs, key=lambda s: min(tpos[v] for v in sg_ops[s]))
        ws = [sum(w[v] for v in sg_ops[s]) for s in sgs_sorted]
        cap = sum(ws) / unit_cap
        merged = {}
        cur, cw = [], 0.0
        for s, wv in zip(sgs_sorted, ws):
            if cur and cw + wv > cap:
                merged[tuple(cur)] = None
                cur, cw = [s], wv
            else:
                cur.append(s)
                cw += wv
        if cur:
            merged[tuple(cur)] = None
        m2 = {}
        for gi, grp in enumerate(merged):
            for s in grp:
                for v in sg_ops[s]:
                    m2[v] = gi
        n2s = m2
        sg_ops = defaultdict(list)
        for v in ids:
            sg_ops[n2s[v]].append(v)
        sgs = sorted(sg_ops)
        S = len(sgs)
    sidx = {s: i for i, s in enumerate(sgs)}
    W = [sum(w[v] for v in sg_ops[s]) for s in sgs]
    # 子图 DAG 边
    edges = set()
    for v in ids:
        for s2 in succs.get(v, ()):
            a, b = n2s[v], n2s.get(s2)
            if b is not None and a != b and a in sidx and b in sidx:
                edges.add((sidx[a], sidx[b]))
    edges = sorted(edges)
    # 每管权
    from common import pipe_map
    pm = pipe_map(g)
    WM = [sum(w[v] for v in sg_ops[s] if pm.get(v) == "PIPE_M") for s in sgs]
    WV = [W[i] - WM[i] for i in range(S)]
    Dur = [max(WM[i], WV[i]) for i in range(S)]
    # 变量布局:x[S*K] | st[S] | z[E] | M
    nx = S * K
    nst = S
    nz = len(edges)
    n = nx + nst + nz + 1
    MID = n - 1
    integrality = np.zeros(n)
    integrality[:nx] = 1
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    ub[:nx] = 1
    rows, cols, vals = [], [], []
    rl, ru = [], []
    r = 0
    # Σ_c x[s,c] = 1
    for s in range(S):
        for c in range(K):
            rows.append(r); cols.append(s * K + c); vals.append(1.0)
        rl.append(1.0); ru.append(1.0); r += 1
    # 依赖:st[b] - st[a] - 500 z >= Dur[a]
    for ei, (a, b) in enumerate(edges):
        rows.append(r); cols.append(nx + b); vals.append(1.0)
        rows.append(r); cols.append(nx + a); vals.append(-1.0)
        rows.append(r); cols.append(nx + nst + ei); vals.append(-500.0)
        rl.append(Dur[a] - 1e-6); ru.append(np.inf); r += 1
    # z 线性化
    for ei, (a, b) in enumerate(edges):
        for c in range(K):
            rows.append(r); cols.append(nx + nst + ei); vals.append(1.0)
            rows.append(r); cols.append(a * K + c); vals.append(-1.0)
            rows.append(r); cols.append(b * K + c); vals.append(1.0)
            rl.append(0.0); ru.append(np.inf); r += 1
            rows.append(r); cols.append(nx + nst + ei); vals.append(1.0)
            rows.append(r); cols.append(b * K + c); vals.append(-1.0)
            rows.append(r); cols.append(a * K + c); vals.append(1.0)
            rl.append(0.0); ru.append(np.inf); r += 1
    # 每管负载 ≤ M
    for c in range(K):
        for pipe_w in (WM, WV):
            for s in range(S):
                rows.append(r); cols.append(s * K + c); vals.append(pipe_w[s])
            rows.append(r); cols.append(MID); vals.append(-1.0)
            rl.append(-np.inf); ru.append(0.0); r += 1
    # st[s] + Dur[s] ≤ M
    for s in range(S):
        rows.append(r); cols.append(nx + s); vals.append(1.0)
        rows.append(r); cols.append(MID); vals.append(-1.0)
        rl.append(-np.inf); ru.append(-Dur[s]); r += 1
    A = sparse.csc_matrix((vals, (rows, cols)), shape=(r, n))
    cons = LinearConstraint(A, np.array(rl), np.array(ru))
    obj = np.zeros(n)
    obj[MID] = 1.0
    res = milp(c=obj, constraints=cons, integrality=integrality,
               bounds=Bounds(lb, ub),
               options=dict(time_limit=time_limit, mip_rel_gap=0.01))
    if not res.success and res.x is None:
        return None
    x = res.x
    assign = {}
    for s in range(S):
        c = int(np.argmax(x[s * K:(s + 1) * K]))
        assign[sgs[s]] = (c, x[nx + s])
    # 发射:核内序 = st 升序
    cores = [[] for _ in range(K)]
    for s in sgs:
        cores[assign[s][0]].append(s)
    for c in range(K):
        cores[c].sort(key=lambda s: assign[s][1])
    return {"node_to_subgraph": {str(v): n2s[v] for v in ids},
            "core_schedules": cores}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--tlim", type=float, default=120)
    a = ap.parse_args()
    out = HERE / "milp_out"
    out.mkdir(exist_ok=True)
    ev = ev_p3 if a.q == 3 else ev_p2
    pool = json.load(open(HERE / f"POOL_q{a.q}_N5.json"))
    for case in a.cases.split(","):
        g = load_case(case)
        sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
        seed = best_seed(case, a.q, a.K)
        if seed is None:
            print(f"{case}: no seed", flush=True)
            continue
        try:
            plan = cert_milp_plan(g, seed, a.K, time_limit=a.tlim,
                                  unit_cap=a.cap)
            if plan is None:
                print(f"{case}: MILP infeasible", flush=True)
                continue
            r, _ = ev(g, plan)
            sp = sc / r["makespan"]
            pv = pool.get(case, 0)
            if sp > pv:
                json.dump({"plan": plan, "mk": r["makespan"], "sp": sp,
                           "kind": "cert_milp"},
                          open(out / f"{case}_q{a.q}_N{a.K}.json", "w"))
            print(f"{case}: MILP sp={sp:.4f} (池 {pv:.3f}, "
                  f"{(sp / pv - 1) * 100 if pv else 0:+.1f}%)", flush=True)
        except Exception as exc:
            print(f"{case}: ERR {type(exc).__name__} {str(exc)[:60]}",
                  flush=True)


if __name__ == "__main__":
    main()
