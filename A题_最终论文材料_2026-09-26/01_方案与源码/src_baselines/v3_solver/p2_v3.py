"""v3 问题2（场景B）求解器：通信/管道感知装箱 + 核内序桶。

场景B 力学（本轮实测）：
- 每核合并为单 Task：无任务接力，但核内四 Pipe 串行 → 均衡目标 = 每核
  每管道负载向量，而非总 cycles；
- 跨核数据边 = COPY_OUT+COPY_IN（2×字节争 DDR）+ 500 cycles 同步延迟：
  深图关键路径每跨核一次 +500（case_064 实测 ~25 次 ≈ 12.5k cycles，解释
  P2=1.19 的瓶颈）→ 装箱须沿关键路径聚簇（前驱亲和）；
- 同核边界零成本：子图边界只作核内序控制桶。

fast 代理 p2_sim：拓扑列表调度，单元时长 = max(ΣM, ΣV) + 拷贝/60，
跨核前驱 +500，全局 DDR 流量地板 total_copy_bytes/60。
"""
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v2_solver"))
from common import (load_case, op_dag, cycles_map, ev_p2, ev_p3, sc_makespan,  # noqa: E402
                    pipe_map, tensor_views, BW)
from p1_v3 import UnitPack, topo_order  # noqa: E402

SC_DIR_DEFAULT = Path(r"C:/shumo_live/02_求解/A题_2026/results/singlecore")
DELAY_B = 500


class P2UnitPack(UnitPack):
    """在签名单元上加 M/V 管道权重与跨单元边字节。"""

    def __init__(self, graph, max_unit_w, order_mode="dfs"):
        super().__init__(graph, max_unit_w, order_mode=order_mode)
        pm = pipe_map(graph)
        w = cycles_map(graph)
        self.um = [0] * self.n_units()
        self.uv = [0] * self.n_units()
        for ui, lst in enumerate(self.unit_ops):
            for v in lst:
                if pm.get(v) == "PIPE_M":
                    self.um[ui] += w[v]
                else:
                    self.uv[ui] += w[v]
        # 单元边字节（经张量；只考虑 eligible op）
        prod, cons, tsize, tpos = tensor_views(graph)
        elig = {o["id"] for o in graph["ops"]
                if o.get("op") not in ("COPY_IN", "COPY_OUT")}
        tprod = {}
        for o, ts in prod.items():
            for t in ts:
                tprod[t] = o
        self.edge_bytes = defaultdict(int)  # (a,b) -> bytes
        for o, ts in cons.items():
            if o not in elig:
                continue
            for t in ts:
                p = tprod.get(t)
                if p is not None and p in elig and p != o and \
                        tpos.get(t) != "DDR":
                    a, b = self.unit_of[p], self.unit_of[o]
                    if a != b:
                        self.edge_bytes[(a, b)] += tsize.get(t, 0)
        # 单元输入字节（图输入，DDR）
        op_type = {o["id"]: o.get("op") for o in graph["ops"]}
        self.uin = [0] * self.n_units()
        for o, ts in cons.items():
            if o not in elig:
                continue
            for t in ts:
                p = tprod.get(t)
                if p is not None and op_type.get(p) == "COPY_IN":
                    self.uin[self.unit_of[o]] += tsize.get(t, 0)


def p2_sim(pk, core_of, K, mte_scale=1.0):
    """场景B fast 代理：最长路 + 每核管道串行 + 跨核延迟 + DDR 地板。"""
    n = pk.n_units()
    pipe_free = [{c: 0.0 for c in range(K)} for _ in range(2)]  # M, V
    mte_free = [0.0] * K
    end = {}
    cross_bytes = 0.0
    total_in = 0.0
    for i in pk.unit_topo:
        c = core_of[i]
        start = 0.0
        for p in pk.upreds[i]:
            lat = 0 if core_of[p] == c else DELAY_B
            b = pk.edge_bytes.get((p, i), 0)
            if core_of[p] != c:
                cross_bytes += b
                # 拷贝对时长：out+in 各 b/60，与 500 延迟取大后串 MTE
                start = max(start, end[p] + DELAY_B + 2 * b / BW * mte_scale)
            else:
                start = max(start, end[p])
        dur = max(pk.um[i], pk.uv[i])
        t0 = max(start, pipe_free[0][c], pipe_free[1][c])
        end[i] = t0 + dur
        pipe_free[0][c] = pipe_free[1][c] = end[i]
        ib = pk.uin[i]
        total_in += ib
        mte_free[c] = max(mte_free[c], start + ib / BW * mte_scale)
        end[i] = max(end[i], mte_free[c])
    sim_mk = max(end.values(), default=0)
    ddr_floor = (2 * cross_bytes + total_in) / BW
    return max(sim_mk, ddr_floor)


def greedy_p2(pk, K, seed=0, w_bal=1.0):
    """拓扑列表调度贪心：就绪单元放到预计完成最早核（管道+延迟+流量）。"""
    rng = random.Random(seed)
    n = pk.n_units()
    core_of = {}
    pf_m = [0.0] * K
    pf_v = [0.0] * K
    pf_mte = [0.0] * K
    end = {}
    cross_b = [0.0] * K
    remaining = {i: set(pk.upreds[i]) for i in range(n)}
    ready = [i for i in range(n) if not remaining[i]]
    ready.sort(key=lambda i: -max(pk.um[i], pk.uv[i]))
    while ready:
        i = ready.pop(0)
        best = None
        for c in range(K):
            start = 0.0
            for p in pk.upreds[i]:
                b = pk.edge_bytes.get((p, i), 0)
                if core_of[p] == c:
                    start = max(start, end[p])
                else:
                    start = max(start, end[p] + DELAY_B + 2 * b / BW)
            dur = max(pk.um[i], pk.uv[i])
            fin = max(start, pf_m[c], pf_v[c]) + dur
            fin = max(fin, pf_mte[c] + pk.uin[i] / BW)
            if best is None or fin < best[0] - 1e-9:
                best = (fin, c)
        fin, c = best
        core_of[i] = c
        start = 0.0
        for p in pk.upreds[i]:
            b = pk.edge_bytes.get((p, i), 0)
            if core_of[p] == c:
                start = max(start, end[p])
            else:
                start = max(start, end[p] + DELAY_B + 2 * b / BW)
        dur = max(pk.um[i], pk.uv[i])
        s0 = max(start, pf_m[c], pf_v[c])
        end[i] = s0 + dur
        pf_m[c] = pf_v[c] = end[i]
        pf_mte[c] = max(pf_mte[c], start + pk.uin[i] / BW)
        end[i] = max(end[i], pf_mte[c])
        for j in pk.usuccs.get(i, ()):
            if i in remaining.get(j, ()):
                remaining[j].discard(i)
                if not remaining[j]:
                    ready.append(j)
        ready.sort(key=lambda x: -max(pk.um[x], pk.uv[x]))
    return core_of


def local_improve_p2(pk, core_of, K, iters=400, seed=0, time_budget=None):
    rng = random.Random(seed)
    n = pk.n_units()
    cur = dict(core_of)
    cur_mk = p2_sim(pk, cur, K)
    best, best_mk = dict(cur), cur_mk
    t0 = time.perf_counter()
    for it in range(iters):
        if time_budget and (it & 15) == 0 and \
                time.perf_counter() - t0 > time_budget:
            break
        i = rng.randrange(n)
        rel = [core_of[p] for p in pk.upreds[i]] + \
              [core_of[s] for s in pk.usuccs.get(i, ())]
        if rel and rng.random() < 0.6:
            c2 = rng.choice(rel)
        else:
            c2 = rng.randrange(K)
        if c2 == cur[i]:
            continue
        cand = dict(cur)
        cand[i] = c2
        mk = p2_sim(pk, cand, K)
        if mk < cur_mk or (mk == cur_mk and rng.random() < 0.3):
            cur, cur_mk = cand, mk
            if mk < best_mk:
                best, best_mk = dict(cand), mk
    return best, best_mk


def unit_plan(pk, core_of, K, bucket="unit"):
    """方案：子图 = 单元（bucket='unit'，核内序=单元拓扑序）或每核一图。"""
    n2s = {}
    rank = pk.rank  # unit_topo 位次
    if bucket == "unit":
        for ui, lst in enumerate(pk.unit_ops):
            sg = rank[ui]
            for v in lst:
                n2s[v] = sg
        cores = [[] for _ in range(K)]
        for ui in range(pk.n_units()):
            cores[core_of[ui]].append(rank[ui])
        for c in range(K):
            cores[c].sort()
        return {"node_to_subgraph": n2s, "core_schedules": cores}
    else:  # 每核一子图
        for ui, lst in enumerate(pk.unit_ops):
            c = core_of[ui]
            for v in lst:
                n2s[v] = c
        return {"node_to_subgraph": n2s,
                "core_schedules": [[c] for c in range(K)]}


def solve_p2(graph, case, sc, K=4, log=print, real_top=3, ls_iters=400,
             extra_plans=None, budget_s=360.0, use_wave=True):
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    soft = t0 + budget_s * 0.5
    ids, preds, succs = op_dag(graph)
    w = cycles_map(graph)
    total = sum(w.values())
    cands = []

    # v2 连续切种子
    from structure_split import structure_aware_plan
    for cpc in (1, 2, 3):
        try:
            planA, _ = structure_aware_plan(graph, K, chunks_per_core=cpc)
            cands.append((f"v2_cpc{cpc}", planA))
        except Exception:
            pass
    if extra_plans:
        cands.extend(extra_plans)

    # 波前种子（结构门控；层内同 motif 局部性对 P2 搬运/L2 同样有利）
    if use_wave:
        try:
            from wave_seed import wave_candidates
            for tag, wplan, meta in wave_candidates(graph, K, log=log):
                cands.append((tag, wplan))
        except Exception as exc:
            log(f"[{case}] p2 wave fail: {exc}")

    # P2 装箱（多粒度 × 双序 × 多种子，时限内早停）
    big = len(ids) > 25000
    for gpc in ((4, 8) if big else (4, 8, 16)):
        for om in ("dfs", "topo"):
            if time.perf_counter() > soft and len(cands) > 3:
                log(f"[{case}] p2 budget: stop portfolio g{gpc}/{om}")
                break
            pk = P2UnitPack(graph, max(1, total // (K * gpc)), order_mode=om)
            if pk.n_units() < K:
                continue
            best_pack, best_mk = None, float("inf")
            for seed in range((3 if big else 6) if pk.n_units() <= 800 else 2):
                co = greedy_p2(pk, K, seed=seed)
                co, mk = local_improve_p2(
                    pk, co, K, iters=min(ls_iters, 4 * pk.n_units()),
                    seed=seed,
                    time_budget=min(5.0, max(1.0, (soft - time.perf_counter())
                                             / 3)) if pk.n_units() > 400 else None)
                if mk < best_mk:
                    best_pack, best_mk = co, mk
            plan = unit_plan(pk, best_pack, K, bucket="unit")
            cands.append((f"p2pack_g{gpc}_{om}", plan))
            plan_s = unit_plan(pk, best_pack, K, bucket="single")
            cands.append((f"p2pack_g{gpc}_{om}_sg", plan_s))
        log(f"[{case}] p2 gpc={gpc}: units={pk.n_units()} sim={best_mk:.0f}")

    # 真评估（P2 评估器即代理；时限内早停，至少 1 个）
    rows = []
    scored = []
    for tag, plan in cands:
        if scored and time.perf_counter() > deadline:
            log(f"[{case}] p2 budget: stop eval at {tag}")
            break
        try:
            r, wt = ev_p2(graph, plan)
            scored.append((r["makespan"], tag, plan, r, wt))
        except Exception as exc:
            log(f"[{case}] {tag} fail: {str(exc)[:60]}")
    scored.sort(key=lambda x: x[0])
    for mk, tag, plan, r, wt in scored[:real_top]:
        dm = r["data_movement_bytes"]
        rows.append({"tag": tag, "plan": plan, "real_mk": mk,
                     "spill_MB": dm["spill_added_copy_bytes"] / 1e6,
                     "added_MB": dm["added_copy_bytes"] / 1e6,
                     "cross_MB": r["cross_task_traffic"] / 1e6})
        log(f"[{case}] {tag}: P2={mk} speedup={sc/mk:.3f} "
            f"spill={dm['spill_added_copy_bytes']/1e6:.1f}MB")
    best = min(rows, key=lambda x: x["real_mk"])
    return {"case": case, "best": best, "rows": rows, "all_scored":
            [(m, t) for m, t, _, _, _ in scored],
            "solve_s": round(time.perf_counter() - t0, 1)}


if __name__ == "__main__":
    for case in ["case_001", "case_050", "case_014", "case_064", "case_071",
                 "case_005"]:
        sc = sc_makespan(case, SC_DIR_DEFAULT)
        g = load_case(case)
        out = solve_p2(g, case, sc)
        print(case, "->", out["best"]["tag"],
              round(sc / out["best"]["real_mk"], 3), f"({out['solve_s']}s)")
