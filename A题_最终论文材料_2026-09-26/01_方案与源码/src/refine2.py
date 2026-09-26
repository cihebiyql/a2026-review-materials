# -*- coding: utf-8 -*-
"""refine2:真值搜索升级 —— 并行最优移动 + 子图分裂 + SA 接受。

- 批量生成候选(每候选自带 co+n2s,无共享可变状态),多进程并行评估,
  取最优改进(或 SA 概率接受)——突破串行 150-eval 深度天花板;
- 移动算子:单子图搬移 / 连续段搬移 / 子图分裂(重核胖图拓扑对半,
  一半搬别核)——修掉"只搬不裂";
- 合法性:分裂沿拓扑位次切,商图保持 DAG;核内序=子图最小拓扑位次。
"""
import os
import sys
import json
import math
import time
import random
import warnings
import multiprocessing as mp
from collections import defaultdict, deque
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("A2026_SOLVER_DIR",
                                  r"C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand"))
sys.path.insert(0, r"C:/shumo_live/02_求解/A题_2026/fast_eval")
sys.path.insert(0, r"C:/shumo_live/a_data/code")

_G = {}


def _worker_init(q):
    _G["q"] = q
    _G["case"] = None


def _worker_eval(payload):
    case, plan = payload
    try:
        if _G["case"] != case:
            from common import load_case
            from fast_eval_p2 import FastEvalP2, FastEvalP3
            _G["g"] = load_case(case)
            _G["fe"] = FastEvalP3(_G["g"]) if _G["q"] == 3 else FastEvalP2(_G["g"])
            _G["case"] = case
        return _G["fe"].evaluate(plan)[0]
    except Exception as exc:
        if os.environ.get("N5_DEBUG"):
            print("WORKER_EXC:", type(exc).__name__, str(exc)[:100], flush=True)
        return float("inf")


def topo_pos_map(graph):
    from common import op_dag
    ids, preds, succs = op_dag(graph)
    pd = defaultdict(int)
    for v in ids:
        for s in succs.get(v, ()):
            pd[s] += 1
    dq = deque(sorted(v for v in ids if pd[v] == 0))
    pos = {}
    i = 0
    while dq:
        v = dq.popleft()
        pos[v] = i
        i += 1
        for s in sorted(succs.get(v, ())):
            pd[s] -= 1
            if pd[s] == 0:
                dq.append(s)
    for v in ids:
        pos.setdefault(v, i)
    return pos


def refine2(graph, plan, K, case, q, pool, workers=12, iters=200, batch=16,
            seed=0, sa=True, t0_frac=0.04, log=print):
    from common import cycles_map
    rng = random.Random(seed)
    pos = topo_pos_map(graph)
    w = cycles_map(graph)

    from common import op_dag
    _, _, osucc = op_dag(graph)

    def sg_rank(n2s):
        """子图 DAG Kahn 拓扑位次(核内序的合法保证)。"""
        sgedges = defaultdict(set)
        indeg = defaultdict(int)
        nodes = sorted(set(n2s.values()))
        for v, lst in ((v, osucc.get(v, ())) for v in n2s):
            a = n2s[v]
            for s in lst:
                b = n2s.get(s)
                if b is not None and a != b and b not in sgedges[a]:
                    sgedges[a].add(b)
                    indeg[b] += 1
        minpos = defaultdict(lambda: 1 << 60)
        for v, sg in n2s.items():
            minpos[sg] = min(minpos[sg], pos[v])
        dq = deque(sorted((x for x in nodes if indeg[x] == 0),
                          key=lambda x: minpos[x]))
        rk = {}
        i = 0
        while dq:
            x = dq.popleft()
            rk[x] = i
            i += 1
            for y in sorted(sgedges[x], key=lambda y: minpos[y]):
                indeg[y] -= 1
                if indeg[y] == 0:
                    dq.append(y)
        for x in nodes:
            rk.setdefault(x, i)
            i += 1
        return rk

    def emit(co, n2s):
        rk = sg_rank(n2s)
        cores = [[] for _ in range(K)]
        for sg, c in co.items():
            if sg in rk:
                cores[c].append(sg)
        for c in range(K):
            cores[c].sort(key=lambda sg: rk[sg])
        return {"node_to_subgraph": {str(v): sg for v, sg in n2s.items()},
                "core_schedules": cores}

    n2s0 = {int(k): v for k, v in plan["node_to_subgraph"].items()}
    co0 = {}
    for ci, sched in enumerate(plan["core_schedules"]):
        for sg in sched:
            co0[sg] = ci
    cur_co, cur_n2s = dict(co0), dict(n2s0)

    def heavy_core(co, n2s):
        load = defaultdict(float)
        for v, sg in n2s.items():
            load[co[sg]] += w[v]
        return max(range(K), key=lambda c: load[c])

    if True:
        cur_mk = pool.map(_worker_eval,
                          [(case, emit(cur_co, cur_n2s))], chunksize=1)[0]
        best_co, best_n2s, best_mk = dict(cur_co), dict(cur_n2s), cur_mk
        T = max(1.0, cur_mk * t0_frac)
        n_acc = n_split = 0
        for it in range(iters):
            heavy = heavy_core(cur_co, cur_n2s) if it % 3 == 0 \
                else rng.randrange(K)
            ops_of = defaultdict(list)
            if rng.random() < 0.25:  # 分裂候选需要 ops 索引
                for v, sg in cur_n2s.items():
                    ops_of[sg].append(v)
            cands = []  # (co, n2s)
            for _ in range(batch):
                kind = rng.random()
                if kind < 0.22:
                    # op 级窗口搬移:重核上拓扑连续 8~64 个 op 整体迁往别核(新子图)
                    c2 = rng.randrange(K)
                    if c2 == heavy:
                        continue
                    ops_h = sorted((v for v, sg in cur_n2s.items()
                                    if cur_co[sg] == heavy),
                                   key=lambda v: pos[v])
                    if len(ops_h) < 16:
                        continue
                    span = rng.randint(8, min(64, len(ops_h)))
                    p0 = rng.randrange(len(ops_h) - span + 1)
                    window = ops_h[p0:p0 + span]
                    new_id = max(cur_n2s.values()) + 1 + rng.randrange(8)
                    n2s = dict(cur_n2s)
                    for v in window:
                        n2s[v] = new_id
                    co = dict(cur_co)
                    co[new_id] = c2
                    cands.append((co, n2s))
                    continue
                if kind < 0.55:
                    live = set(cur_n2s.values())
                    sgs = [sg for sg, c in cur_co.items()
                           if c == heavy and sg in live]
                    if not sgs:
                        continue
                    sg = rng.choice(sgs)
                    dst = rng.randrange(K)
                    if dst == cur_co[sg]:
                        continue
                    co = dict(cur_co)
                    co[sg] = dst
                    cands.append((co, cur_n2s))
                elif kind < 0.8:
                    live = set(cur_n2s.values())
                    lst = sorted((sg for sg, c in cur_co.items()
                                  if c == heavy and sg in live),
                                 key=lambda sg: min(pos[v] for v, s2
                                                    in cur_n2s.items()
                                                    if s2 == sg))
                    if not lst:
                        continue
                    span = rng.randint(1, min(6, len(lst)))
                    p0 = rng.randrange(len(lst) - span + 1)
                    dst = rng.randrange(K)
                    co = dict(cur_co)
                    for sg in lst[p0:p0 + span]:
                        co[sg] = dst
                    cands.append((co, cur_n2s))
                else:
                    if not ops_of:
                        for v, sg in cur_n2s.items():
                            ops_of[sg].append(v)
                    sgs = [sg for sg in cur_co if cur_co[sg] == heavy
                           and len(ops_of.get(sg, ())) >= 4]
                    if not sgs:
                        continue
                    fat = max(sgs, key=lambda s: sum(w[v] for v in ops_of[s]))
                    oplist = sorted(ops_of[fat], key=lambda v: pos[v])
                    mid = len(oplist) // 2
                    new_id = max(cur_n2s.values()) + 1 + rng.randrange(8)
                    dst = rng.choice([c for c in range(K) if c != heavy])
                    n2s = dict(cur_n2s)
                    for v in oplist[mid:]:
                        n2s[v] = new_id
                    co = dict(cur_co)
                    co[new_id] = dst
                    cands.append((co, n2s))
            if not cands:
                continue
            plans = [(case, emit(co, n2s)) for co, n2s in cands]
            mks = pool.map(_worker_eval, plans, chunksize=1)
            bi = min(range(len(mks)), key=lambda i: mks[i])
            delta = mks[bi] - cur_mk
            acc = delta < -0.5
            if not acc and sa and mks[bi] < 1e17:
                acc = rng.random() < math.exp(-delta / max(T, 1e-9)) \
                    and delta < T
            if acc:
                cur_co, cur_n2s = cands[bi][0], cands[bi][1]
                cur_mk = mks[bi]
                n_acc += 1
                if cur_mk < best_mk:
                    best_co, best_n2s, best_mk = (dict(cur_co),
                                                  dict(cur_n2s), cur_mk)
            T *= 0.985
            if (it & 15) == 0:
                log(f"  it{it} cur={cur_mk:.0f} best={best_mk:.0f} "
                    f"T={T:.0f} acc={n_acc}")
    return emit(best_co, best_n2s), best_mk


def main():
    import argparse
    from common import load_case, ev_p2, ev_p3
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "refined2"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    posthoc = json.load(open(os.environ.get(
        "A2026_POSTHOC",
        r"C:/shumo_live/02_求解/A题_2026/a_lab/records/POSTHOC_BEST.json")))
    pool = mp.Pool(a.workers, initializer=_worker_init, initargs=(a.q,))
    try:
        _run_cases(a, pool, posthoc)
    finally:
        pool.close()
        pool.join()


def _run_cases(a, pool, posthoc):
    from common import load_case, ev_p2, ev_p3
    scdir = Path(os.environ.get(
        "A2026_SC_DIR",
        r"C:/shumo_live/02_求解/A题_2026/results/singlecore"))
    out = Path(a.out)
    for case in a.cases.split(","):
        t0 = time.perf_counter()
        g = load_case(case)
        sc = json.load(open(scdir / f"{case}_sc.json"))["makespan"]
        seed = None
        for d, nm in ((HERE / "audit_B_best", f"{case}_q{a.q}_N{a.K}.json"),
                      (HERE / "arm_g5tf_out", f"{case}_q{a.q}_N{a.K}.json"),
                      (HERE / "cross_best", f"{case}_q{a.q}_N{a.K}.json"),
                      (HERE / "refined", f"{case}_q{a.q}_N{a.K}.json"),
                      (HERE / "strand_n5", f"{case}_q{a.q}_N{a.K}.json"),
                      (HERE / "refined2", f"{case}_q{a.q}_N{a.K}.json"),
                      (HERE / "refined2_q2", f"{case}_q{a.q}_N{a.K}.json"),
                      (HERE / "refined2_light", f"{case}_q{a.q}_N{a.K}.json")):
            fp = d / nm
            if fp.exists():
                sd = json.load(open(fp))
                if sd.get("plan"):
                    seed = sd["plan"]
                    break
        if seed is None:
            for tag in ("v3_main", "v2_cpc"):
                fp = Path(os.environ.get(
                    "A2026_ARCHIVE",
                    r"C:/shumo_live/02_求解/A题_2026/a_lab/registry/"
                    r"plan_archive/plans")) / \
                    f"{case}_q{a.q}_N{a.K}_{tag}.json"
                if fp.exists():
                    d = json.load(open(fp))
                    seed = {"node_to_subgraph": d["node_to_subgraph"],
                            "core_schedules": d["core_schedules"]}
                    break
        if seed is None:
            print(f"{case}: no seed", flush=True)
            continue
        best_plan, best_mk = None, float("inf")
        for sd in range(a.seeds):
            p, mk = refine2(g, seed, a.K, case, a.q, pool=pool,
                            workers=a.workers,
                            iters=a.iters, batch=a.batch, seed=sd)
            if mk < best_mk:
                best_plan, best_mk = p, mk
        r, wt = (ev_p3 if a.q == 3 else ev_p2)(g, best_plan)
        sp = sc / r["makespan"]
        base = posthoc.get(f"{case}|q{a.q}|N{a.K}", float("nan"))
        json.dump({"plan": best_plan, "mk": r["makespan"], "sp": sp},
                  open(out / f"{case}_q{a.q}_N{a.K}.json", "w"))
        print(f"{case}: sp={sp:.4f} (fast={best_mk} "
              f"official={r['makespan']} match={best_mk == r['makespan']}) "
              f"base={base:.3f} "
              f"gain={(sp / base - 1) * 100 if base == base else 0:+.1f}% "
              f"[{time.perf_counter() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
