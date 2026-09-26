# -*- coding: utf-8 -*-
"""少子图低尾例的 分裂+搬移+顺序 三算子联合真值探针(P3/N5)。

背景:N5 报告下一步#2"子图分裂移动算子(当前只搬不裂)"。
本探针在 refine_real(只搬移+拓扑位次序)之上增加:
  A. 拓扑切割分裂:把大子图按拓扑位一切两半,右半去别的核(合法:原图 DAG,
     切割只产生 左→右 边,收缩图无环);
  B. k 路等分:大子图切 k 段,轮流放 k 个低载核(流水式);
  C. 整子图搬移(与 refine_real 同型);
  D. 核内相邻独立对交换(传递闭包判合法,顺序维度)。
评估 = FastEvalP3(bit-exact);死锁(-2)/校验违例 = 拒绝回退。
种子 = 全部存盘方案中 FastEval 最优者;基线 mk 必须与种子盘面一致。
"""
import sys, os, json, random, time
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:/shumo_live/02_求解/A题_2026/fast_eval")
sys.path.insert(0, r"C:/shumo_live/02_求解/A题_2026/v3_solver")
sys.path.insert(0, r"C:/shumo_live/a_data/code")
warnings_off = __import__("warnings").filterwarnings("ignore")

from common import load_case, ev_p3
from fast_eval_p2 import FastEvalP3

ROOT = Path(r"C:/shumo_live/02_求解/A题_2026")
ARCHIVE = ROOT / "a_lab/registry/plan_archive/plans"
SC_DIR = ROOT / "results/singlecore"
POOL = json.load(open(ROOT / "audit_20260924_latest/POOL_q3_N5.json"))
OUT = HERE / "split_probe_out"; OUT.mkdir(exist_ok=True)


def seed_plans(case, q=3, K=5):
    out = {}
    for tag in ("v3_main", "v2_cpc"):
        fp = ARCHIVE / f"{case}_q{q}_N{K}_{tag}.json"
        if fp.exists():
            d = json.load(open(fp))
            if isinstance(d, dict) and "core_schedules" in d:
                out[str(fp)] = d
            elif isinstance(d, dict) and "plan" in d:
                out[str(fp)] = d["plan"]
    for sub in ("refined", "refined2", "strand_n5", "bxcpu_pull"):
        for fp in (HERE / sub).glob(f"{case}_q{q}_*.json"):
            d = json.load(open(fp))
            if "plan" in d:
                out[str(fp)] = d["plan"]
            elif "core_schedules" in d:
                out[str(fp)] = d
    return out


class Ops:
    """图派生量:拓扑位 / 子图 DAG 边。结构不变时可复用。"""

    def __init__(self, g):
        self.g = g
        outs = defaultdict(list); ins = defaultdict(list)
        for e in g["edges"]:
            outs[e["source"]].append(e["target"]); ins[e["target"]].append(e["source"])
        # 全图 Kahn(含虚拟/张量节点),得拓扑位
        indeg = {n: len(ins[n]) for n in set(ins) | set(outs)}
        dq = deque(sorted([n for n, d in indeg.items() if d == 0]))
        self.pos = {}; seen = 0
        succ_all = {n: outs[n] for n in indeg}
        while dq:
            n = dq.popleft(); self.pos[n] = seen; seen += 1
            for v in succ_all[n]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    dq.append(v)
        self.outs = outs
        self.op_cycle = {o["id"]: o.get("cycles", 0) for o in g["ops"]}
        self.opids = set(self.op_cycle)

    def sg_edges(self, n2s):
        E = set()
        for t, preds in self._tensor_preds().items():
            for u in preds:
                for v in self.outs.get(t, []):
                    if u in self.opids and v in self.opids:
                        a, b = n2s.get(u), n2s.get(v)
                        if a is not None and b is not None and a != b:
                            E.add((a, b))
        return E

    def _tensor_preds(self):
        if not hasattr(self, "_tp"):
            ins = defaultdict(list)
            for e in self.g["edges"]:
                ins[e["target"]].append(e["source"])
            self._tp = {t: ps for t, ps in ins.items() if t not in self.opids}
        return self._tp


class State:
    def __init__(self, plan, K):
        self.K = K
        self.n2s = {int(k): v for k, v in plan["node_to_subgraph"].items()}
        self.sched = [list(c) for c in plan["core_schedules"]]
        while len(self.sched) < K:      # N4/N2 种子补空槽(B.5 允许)
            self.sched.append([])
        self.sg_core = {}
        for ci, c in enumerate(self.sched):
            for sg in c:
                self.sg_core[sg] = ci
        for v in self.n2s.values():
            self.sg_core.setdefault(v, 0)
        self.next_id = max(self.sg_core) + 1

    def nodes_by_sg(self):
        m = defaultdict(list)
        for n, sg in self.n2s.items():
            m[sg].append(n)
        return m

    def emit(self):
        return {"node_to_subgraph": {str(n): s for n, s in self.n2s.items()},
                "core_schedules": [list(c) for c in self.sched]}


def kahn_insert(old_sched, add_sg, E):
    """在保持既有相对顺序的前提下,把 add_sg 插入合法拓扑位。"""
    items = list(old_sched) + [add_sg]
    pos0 = {sg: i for i, sg in enumerate(old_sched)}; pos0[add_sg] = len(old_sched)
    return kahn_pref(items, pos0, E)


def kahn_pref(items, pos0, E):
    """给定候选项与偏好序,输出满足 E 约束的合法拓扑序(偏好序优先)。"""
    indeg = defaultdict(int); succ = defaultdict(list)
    sset = set(items)
    for a, b in E:
        if a in sset and b in sset:
            indeg[b] += 1; succ[a].append(b)
    ready = [s for s in items if indeg[s] == 0]
    out = []
    while ready:
        ready.sort(key=lambda s: (pos0.get(s, 1 << 30), s))
        s = ready.pop(0); out.append(s)
        for v in succ[s]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return out if len(out) == len(items) else None


def resort_core(cs, core, E2):
    """用新依赖边集 E2 重排某核,保持旧相对序(新子图排偏好末尾)。"""
    old = cs.sched[core]
    pos0 = {sg: i for i, sg in enumerate(old)}
    new = kahn_pref(old, pos0, E2)
    if new is None:
        return False
    cs.sched[core] = new
    return True


def reach_pairs(sgs, E):
    succ = defaultdict(set); sset = set(sgs)
    for a, b in E:
        if a in sset and b in sset:
            succ[a].add(b)
    R = {}
    for s in sgs:
        seen = set(); dq = deque([s])
        while dq:
            x = dq.popleft()
            for y in succ[x]:
                if y not in seen:
                    seen.add(y); dq.append(y)
        R[s] = seen
    return R


def probe(case, q=3, K=5, evals_cap=90, wall_s=130, seed=17, verify=True):
    rng = random.Random(seed)
    g = load_case(case)
    ops = Ops(g)
    fe = FastEvalP3(g)
    # ---- 种子:全部存盘方案取 FastEval 最优 ----
    best_seed, mk0 = None, float("inf")
    for fp, pl in seed_plans(case, q, K).items():
        try:
            mk = fe.evaluate(pl)[0]
        except Exception:
            continue
        if mk < mk0:
            mk0, best_seed = mk, pl
    if best_seed is None:
        return {"case": case, "status": "no_seed"}
    st = State(best_seed, K)
    cur_mk = mk0
    sg_work = lambda: {sg: sum(ops.op_cycle.get(n, 0) for n in ns)
                       for sg, ns in st.nodes_by_sg().items()}
    E = ops.sg_edges(st.n2s)
    stats = {"split": [0, 0], "ksplit": [0, 0], "move": [0, 0], "order": [0, 0],
             "evals": 0, "err": {}}
    log = []
    t0 = time.perf_counter()

    def evaluate(cand_st):
        try:
            return fe.evaluate(cand_st.emit())[0]
        except RuntimeError as ex:
            msg = str(ex)[:60]
            stats["err"][msg] = stats["err"].get(msg, 0) + 1
            return None

    while stats["evals"] < evals_cap and time.perf_counter() - t0 < wall_s:
        nb = st.nodes_by_sg(); wk = sg_work()
        kind = rng.choices(["split", "ksplit", "move", "order"], [5, 3, 2, 3])[0]
        cand = None; desc = ""
        if kind in ("split", "ksplit") and E is not None:
            big = sorted(nb, key=lambda s: -wk.get(s, 0))[:3]
            big = [s for s in big if len(nb[s]) >= 4]
            if big:
                src = rng.choice(big)
                nodes = sorted(nb[src], key=lambda n: ops.pos.get(n, 0))
                loads = {c: sum(wk.get(s, 0) for s in st.sg_core if st.sg_core[s] == c)
                         for c in range(K)}
                if kind == "split":
                    frac = rng.choice([0.25, 1 / 3, 0.5, 2 / 3, 0.75])
                    cut = max(1, min(len(nodes) - 1, int(len(nodes) * frac)))
                    dst = sorted(range(K), key=lambda c: loads[c])
                    dst = [c for c in dst if c != st.sg_core[src]][:2]
                    if dst:
                        d = rng.choice(dst)
                        new_sg = st.next_id
                        cs = State(st.emit(), K)
                        for n in nodes[cut:]:
                            cs.n2s[n] = new_sg
                        cs.sg_core[new_sg] = d
                        cs.next_id = new_sg + 1
                        E2 = ops.sg_edges(cs.n2s)
                        cs.sched[d].append(new_sg)
                        okc = resort_core(cs, d, E2) and resort_core(cs, st.sg_core[src], E2)
                        if okc:
                            cand, desc = cs, f"split sg{src}@{frac:.2f}->c{d}"
                else:
                    k = rng.choice([c for c in (2, 3, 4, 5) if c <= len(nodes) // 2 + 1] or [2])
                    size = len(nodes) // k
                    chunks = [nodes[i * size:(i + 1) * size] for i in range(k - 1)] + [nodes[(k - 1) * size:]]
                    dsts = sorted(range(K), key=lambda c: loads[c])[:k]
                    cs = State(st.emit(), K)
                    ok = True
                    for i, ch in enumerate(chunks):
                        if not ch:
                            ok = False; break
                        new_sg = cs.next_id
                        for n in ch:
                            cs.n2s[n] = new_sg
                        cs.sg_core[new_sg] = dsts[i]
                        cs.next_id += 1
                    if ok:
                        cs.sg_core.pop(src, None)  # 原子图已无节点,从核归属中移除
                        E2 = ops.sg_edges(cs.n2s)
                        for c in range(K):
                            members = [s for s, cc in cs.sg_core.items() if cc == c]
                            pos0 = {s: i for i, s in enumerate(cs.sched[c])}
                            newn = kahn_pref(members, pos0, E2)
                            if newn is None:
                                ok = False; break
                            cs.sched[c] = newn
                    if ok:
                        cand, desc = cs, f"ksplit sg{src}x{k}->{dsts}"
        elif kind == "move":
            wk2 = sorted(st.sg_core, key=lambda s: -wk.get(s, 0))
            src_sg = rng.choice(wk2[:4])
            loads = {c: sum(wk.get(s, 0) for s in st.sg_core if st.sg_core[s] == c)
                     for c in range(K)}
            dsts = sorted(range(K), key=lambda c: loads[c])
            dsts = [c for c in dsts if c != st.sg_core[src_sg]][:2]
            if dsts:
                d = rng.choice(dsts)
                src_c = st.sg_core[src_sg]
                cs = State(st.emit(), K)
                cs.sg_core[src_sg] = d
                E2 = ops.sg_edges(cs.n2s)
                cs.sched[src_c] = [s for s in cs.sched[src_c] if s != src_sg]
                cs.sched[d].append(src_sg)
                if resort_core(cs, d, E2) and resort_core(cs, src_c, E2):
                    cand, desc = cs, f"move sg{src_sg}->c{d}"
        else:
            cores = [c for c in range(K) if len(st.sched[c]) >= 2]
            if cores:
                ci = rng.choice(cores)
                R = reach_pairs(st.sched[ci], E)
                i = rng.randrange(len(st.sched[ci]) - 1)
                a, b = st.sched[ci][i], st.sched[ci][i + 1]
                if b not in R.get(a, ()) and a not in R.get(b, ()):
                    cs = State(st.emit(), K)
                    cs.sched[ci][i], cs.sched[ci][i + 1] = cs.sched[ci][i + 1], cs.sched[ci][i]
                    cand, desc = cs, f"order c{ci}:{a}<->{b}"
        if cand is None:
            continue
        stats["evals"] += 1
        mk = evaluate(cand)
        if mk is not None and mk < cur_mk - 0.5:
            st, cur_mk = cand, mk
            E = ops.sg_edges(st.n2s)
            stats[kind if kind in stats else "order"][0] += 1
            log.append({"d": desc, "mk": mk, "ok": 1})
        else:
            stats[kind if kind in stats else "order"][1] += 1
            log.append({"d": desc, "mk": mk, "ok": 0})
    # ---- 官方终验(仅当有改进) ----
    sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
    res = {"case": case, "mk0": mk0, "best": cur_mk,
           "sp0": sc / mk0, "sp_new": sc / cur_mk,
           "pool_sp": POOL.get(case), "stats": stats, "n_moves": len(log)}
    if cur_mk < mk0 - 0.5 and verify:
        try:
            r, wt = ev_p3(g, st.emit())
            res["official_mk"] = r["makespan"]
            res["official_match"] = (r["makespan"] == cur_mk)
            res["official_sp"] = sc / r["makespan"]
        except Exception as ex:
            res["official_error"] = str(ex)[:120]
        json.dump({"plan": st.emit(), "mk": cur_mk, "sp": sc / cur_mk,
                   "seed_mk0": mk0, "log": log},
                  open(OUT / f"{case}_q3_best.json", "w"))
    return res


if __name__ == "__main__":
    cases = sys.argv[1:] or ["case_064", "case_009", "case_040", "case_043",
                             "case_044", "case_068", "case_066", "case_098",
                             "case_100", "case_049"]
    summary = []
    for c in cases:
        cap, wall = (40, 260) if c in ("case_047", "case_071") else (90, 130)
        try:
            r = probe(c, evals_cap=cap, wall_s=wall)
        except Exception as ex:
            r = {"case": c, "status": "ERR:" + repr(ex)[:150]}
        print(json.dumps(r, ensure_ascii=False), flush=True)
        summary.append(r)
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    gain = [(r["case"], r.get("sp_new", 0) - r.get("pool_sp", r.get("sp0", 0)))
            for r in summary if "sp_new" in r]
    print("=== per-case Δsp vs pool ===")
    for c, d in gain:
        print(f"  {c}: {d:+.3f}")
    print(f"=== total Δpoints(均値贡献×100): {sum(d for _, d in gain):+.1f} ===",
          flush=True)
