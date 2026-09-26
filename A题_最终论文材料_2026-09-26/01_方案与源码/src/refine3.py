# -*- coding: utf-8 -*-
"""refine3:顺序感知真值搜索(审计 F4/F5 的 HoL 队头阻塞 + 跨管重排自由度)。

状态 = (n2s, co, orders); orders[core] = 子图显式顺序列表。
算子:
  A 顺序移动:子图在本核序列内改位(合法=合并图无环)
  B 跨核搬移(插入目标核随机位)
  C 子图分裂(拓扑对半,新子图插父邻位)
  D op窗口搬移
合法性:合并图 = 核内串行边 + 子图依赖边,必须无环(Kahn 检查)。
评估:FastEvalP3/P2 bit-exact,批并行,SA 接受。
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
sys.path.insert(0, os.environ.get(
    "A2026_SOLVER_DIR",
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
            if _G["q"] == 1:
                from fast_eval_p1 import FastEvalP1
                _G["g"] = load_case(case)
                _G["fe"] = FastEvalP1(_G["g"])
            elif _G["q"] == 3:
                from fast_eval_p2 import FastEvalP3
                _G["g"] = load_case(case)
                _G["fe"] = FastEvalP3(_G["g"])
            else:
                from fast_eval_p2 import FastEvalP2
                _G["g"] = load_case(case)
                _G["fe"] = FastEvalP2(_G["g"])
            _G["case"] = case
        return _G["fe"].evaluate(plan)[0]
    except Exception as exc:
        if os.environ.get("N5_DEBUG"):
            print("W_EXC:", type(exc).__name__, str(exc)[:80], flush=True)
        return float("inf")


class Ctx:
    def __init__(self, graph, K):
        from common import op_dag, cycles_map
        self.ids, self.preds, self.succ = op_dag(graph)
        self.w = cycles_map(graph)
        self.K = K
        # 子图依赖边(按 op 后继)
        self.sg_dep = defaultdict(set)

    def dep_edges(self, n2s):
        self.sg_dep = defaultdict(set)
        for v in self.ids:
            a = n2s[v]
            for s in self.succ.get(v, ()):
                b = n2s.get(s)
                if b is not None and a != b:
                    self.sg_dep[a].add(b)
        return self.sg_dep

    def legal(self, n2s, co, orders):
        """合并图无环检查:核内串行边 + 依赖边。"""
        dep = self.dep_edges(n2s)
        indeg = defaultdict(int)
        nodes = set(n2s.values())
        succ = defaultdict(set)
        for c, lst in orders.items():
            for x, y in zip(lst, lst[1:]):
                if y not in succ[x]:
                    succ[x].add(y)
                    indeg[y] += 1
        for a in nodes:
            for b in dep[a]:
                if b not in succ[a]:
                    succ[a].add(b)
                    indeg[b] += 1
        dq = deque(x for x in nodes if indeg[x] == 0)
        seen = 0
        while dq:
            x = dq.popleft()
            seen += 1
            for y in succ[x]:
                indeg[y] -= 1
                if indeg[y] == 0:
                    dq.append(y)
        return seen == len(nodes)

    def emit(self, n2s, co, orders):
        cores = [[] for _ in range(self.K)]
        live = set(n2s.values())
        for sg, c in co.items():
            if sg in live:
                cores[c].append(sg)
        for c in range(self.K):
            keep = [sg for sg in orders.get(c, []) if sg in live]
            keep += [sg for sg in cores[c] if sg not in keep]
            cores[c] = keep
        return {"node_to_subgraph": {str(v): sg for v, sg in n2s.items()},
                "core_schedules": cores}


def refine3(graph, plan, K, case, q, pool_obj, workers=12, iters=300,
            batch=12, seed=0, sa=True, t0_frac=0.04, log=print):
    rng = random.Random(seed)
    ctx = Ctx(graph, K)
    n2s0 = {int(k): v for k, v in plan["node_to_subgraph"].items()}
    co0 = {}
    orders0 = defaultdict(list)
    for ci, sched in enumerate(plan["core_schedules"]):
        for sg in sched:
            co0[sg] = ci
            orders0[ci].append(sg)
    cur = (dict(n2s0), dict(co0), {c: list(l) for c, l in orders0.items()})
    topo_pos = {}
    pd = defaultdict(int)
    for v in ctx.ids:
        for s in ctx.succ.get(v, ()):
            pd[s] += 1
    dq = deque(sorted(v for v in ctx.ids if pd[v] == 0))
    i = 0
    while dq:
        v = dq.popleft()
        topo_pos[v] = i
        i += 1
        for s in sorted(ctx.succ.get(v, ())):
            pd[s] -= 1
            if pd[s] == 0:
                dq.append(s)
    for v in ctx.ids:
        topo_pos.setdefault(v, i)

    def evaluate(cands):
        plans = [(case, ctx.emit(*c)) for c in cands]
        return pool_obj.map(_worker_eval, plans, chunksize=1)

    cur_mk = evaluate([cur])[0]
    best, best_mk = cur, cur_mk
    T = max(1.0, cur_mk * t0_frac)
    n_acc = 0
    for it in range(iters):
        cands = []
        n2s, co, orders = cur
        live = set(n2s.values())
        # 每核 live 序列
        ordl = {c: [sg for sg in orders.get(c, []) if sg in live] for c in range(K)}
        for _ in range(batch):
            kind = rng.random()
            c2 = None
            if kind < 0.40:
                # A 顺序移动
                pool_c = [c for c in range(K) if len(ordl[c]) >= 2]
                if not pool_c:
                    continue
                c_src = rng.choice(pool_c)
                lst = ordl[c_src]
                pi = rng.randrange(len(lst))
                sg = lst[pi]
                pj = rng.randrange(len(lst))
                if pj == pi:
                    continue
                nl = lst[:pi] + lst[pi + 1:]
                nl = nl[:pj] + [sg] + nl[pj:]
                n_orders = {c: (nl if c == c_src else list(ordl[c]))
                            for c in range(K)}
                cand = (n2s, co, n_orders)
                if ctx.legal(n2s, co, n_orders):
                    cands.append(cand)
            elif kind < 0.65:
                # B 跨核搬移
                c_src = rng.randrange(K)
                lst = ordl[c_src]
                if not lst:
                    continue
                sg = rng.choice(lst)
                c_dst = rng.choice([c for c in range(K) if c != c_src])
                n_co = dict(co)
                n_co[sg] = c_dst
                dl = ordl[c_dst]
                pj = rng.randrange(len(dl) + 1)
                n_orders = {c: ([x for x in lst if x != sg] if c == c_src
                                else list(dl))
                            for c in range(K)}
                n_orders[c_dst] = n_orders[c_dst][:pj] + [sg] + n_orders[c_dst][pj:]
                if ctx.legal(n2s, n_co, n_orders):
                    cands.append((n2s, n_co, n_orders))
            elif kind < 0.80:
                # C 分裂
                ops_of = defaultdict(list)
                for v, sg in n2s.items():
                    ops_of[sg].append(v)
                cands_c = [c for c in range(K) if ordl[c]]
                if not cands_c:
                    continue
                fat = [sg for sg in ordl[rng.choice(cands_c)]
                       if len(ops_of.get(sg, ())) >= 4]
                if not fat:
                    continue
                sg = rng.choice(fat)
                c_src = co[sg]
                opl = sorted(ops_of[sg], key=lambda v: topo_pos[v])
                mid = len(opl) // 2
                new_id = max(n2s.values()) + 1 + rng.randrange(8)
                n_n2s = dict(n2s)
                for v in opl[mid:]:
                    n_n2s[v] = new_id
                c_dst = rng.choice([c for c in range(K) if c != c_src])
                n_co = dict(co)
                n_co[new_id] = c_dst
                dl = ordl.get(c_dst, [])
                pj = rng.randrange(len(dl) + 1)
                n_orders = {c: [x for x in ordl[c] if x != sg]
                            for c in range(K)}
                n_orders[c_src] = n_orders.get(c_src, [])
                n_orders[c_dst] = list(dl[:pj]) + [new_id] + list(dl[pj:])
                if ctx.legal(n_n2s, n_co, n_orders):
                    cands.append((n_n2s, n_co, n_orders))
            else:
                # D op 窗口
                c_src = rng.randrange(K)
                ops_h = sorted((v for v, sg in n2s.items() if co[sg] == c_src),
                               key=lambda v: topo_pos[v])
                if len(ops_h) < 16:
                    continue
                c_dst = rng.choice([c for c in range(K) if c != c_src])
                span = rng.randint(8, min(64, len(ops_h)))
                p0 = rng.randrange(len(ops_h) - span + 1)
                window = ops_h[p0:p0 + span]
                new_id = max(n2s.values()) + 1 + rng.randrange(8)
                n_n2s = dict(n2s)
                for v in window:
                    n_n2s[v] = new_id
                n_co = dict(co)
                n_co[new_id] = c_dst
                dl = ordl.get(c_dst, [])
                pj = rng.randrange(len(dl) + 1)
                n_orders = {c: list(ordl[c]) for c in range(K)}
                n_orders[c_dst] = list(dl[:pj]) + [new_id] + list(dl[pj:])
                if ctx.legal(n_n2s, n_co, n_orders):
                    cands.append((n_n2s, n_co, n_orders))
        if not cands:
            continue
        mks = evaluate(cands)
        bi = min(range(len(mks)), key=lambda i2: mks[i2])
        delta = mks[bi] - cur_mk
        acc = delta < -0.5
        if not acc and sa and mks[bi] < 1e17 and delta < T:
            acc = rng.random() < math.exp(-delta / max(T, 1e-9))
        if acc:
            cur = cands[bi]
            cur_mk = mks[bi]
            n_acc += 1
            if cur_mk < best_mk:
                best, best_mk = cur, cur_mk
        T *= 0.99
        if (it & 15) == 0:
            log(f"  it{it} cur={cur_mk:.0f} best={best_mk:.0f} T={T:.0f} "
                f"acc={n_acc}")
    return ctx.emit(*best), best_mk


def main():
    import argparse
    from common import load_case, ev_p2, ev_p3
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "refined5"))
    # N5_GREEDY=1(默认) => sa=False 纯贪心链(agent C 战术)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    scdir = Path(os.environ.get(
        "A2026_SC_DIR",
        r"C:/shumo_live/02_求解/A题_2026/results/singlecore"))
    posthoc = json.load(open(os.environ.get(
        "A2026_POSTHOC",
        r"C:/shumo_live/02_求解/A题_2026/a_lab/records/POSTHOC_BEST.json")))
    pool = mp.Pool(a.workers, initializer=_worker_init, initargs=(a.q,))
    try:
        for case in a.cases.split(","):
            t0 = time.perf_counter()
            g = load_case(case)
            sc = json.load(open(scdir / f"{case}_sc.json"))["makespan"]
            seed = None
            best_sp = -1
            for d in ("refined6", "refined5", "audit_B_best", "bxcpu_best", "refined4", "refined3",
                      "refined_mara", "arm_g5tf_out", "cross_best",
                      "refined2", "refined2_light", "refined",
                      "strand_n5", "refined5"):
                fp = HERE / d / f"{case}_q{a.q}_N{a.K}.json"
                if fp.exists():
                    sd = json.load(open(fp))
                    if sd.get("plan") and sd.get("sp", 0) > best_sp:
                        best_sp = sd["sp"]
                        seed = sd["plan"]
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
            for sd_ in range(a.seeds):
                p, mk = refine3(g, seed, a.K, case, a.q, pool_obj=pool,
                                workers=a.workers, iters=a.iters,
                                batch=a.batch, seed=sd_,
                                sa=os.environ.get("N5_GREEDY", "1") != "1")
                if mk < best_mk:
                    best_plan, best_mk = p, mk
            r, wt = (ev_p3 if a.q == 3 else ev_p2)(g, best_plan)
            sp = sc / r["makespan"]
            base = posthoc.get(f"{case}|q{a.q}|N{a.K}", float("nan"))
            json.dump({"plan": best_plan, "mk": r["makespan"], "sp": sp},
                      open(out / f"{case}_q{a.q}_N{a.K}.json", "w"))
            print(f"{case}: sp={sp:.4f} (match={best_mk == r['makespan']}) "
                  f"base={base:.3f} "
                  f"gain={(sp / base - 1) * 100 if base == base else 0:+.1f}% "
                  f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
