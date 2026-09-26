# -*- coding: utf-8 -*-
"""P2/P3 股流式候选族(N5 冲 5 均值战役)。

机制(源自 2026-09-24 诊断):
- 场景 B 同核零成本、无 Task 屏障;跨核边付 2×字节/60 + 500c 释放延迟;
- E 类深图现状只用到 2 核(v2 凸切)或细单元 CP 累积 500c 跳(pack 细粒);
- 正解 = 股流(长链)单元:整链同核 → CP 跨核跳极少 + 每核负载均衡;
- 修正代理:M/V 分管并行(官方语义:不同 Pipe 可并行),MTE 串行,DDR 地板。

合法口径:每股流一子图(bucket=unit 模式),核内序=股流拓扑位次;
仅 eligible ops 入 node_to_subgraph(与 p1_v3/p2_v3 同口径)。
"""
import os
import sys
import json
import time
import bisect
import random
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLVER_DIR = Path(os.environ.get("A2026_SOLVER_DIR", r"C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand"))
sys.path.insert(0, str(SOLVER_DIR))
from common import (load_case, op_dag, cycles_map, pipe_map, tensor_views,  # noqa: E402
                    ev_p2, ev_p3, BW, DELAY_B)

SC_DIR = Path(os.environ.get("A2026_SC_DIR", r"C:/shumo_live/02_求解/A题_2026/results/singlecore"))


# ---------------- 1) 股流提取:加权路径覆盖 ----------------

class StrandPack:
    """按字节亲和的路径覆盖把 eligible ops 组织成股流单元。"""

    def __init__(self, graph, affinity="byte"):
        ids, preds, succs = op_dag(graph)
        w = cycles_map(graph)
        pm = pipe_map(graph)
        prod, cons, tsize, tpos = tensor_views(graph)
        # op-op 边 + 字节
        ew = defaultdict(float)
        for o, ts in cons.items():
            for t in ts:
                p_list = prod.get(t, ())
                for p in p_list:
                    if p != o:
                        ew[(p, o)] += tsize.get(t, 0)
        self.ids, self.preds, self.succs = set(ids), preds, succs
        self.w, self.pm, self.ew = w, pm, ew
        # 拓扑序与最长剩余权
        indeg = {i: len(preds.get(i, ())) for i in ids}
        dq = deque(v for v in ids if indeg[v] == 0)
        topo = []
        while dq:
            u = dq.popleft()
            topo.append(u)
            for v in succs.get(u, ()):
                indeg[v] -= 1
                if indeg[v] == 0:
                    dq.append(v)
        self.topo = topo
        rem = {}
        for u in reversed(topo):
            rem[u] = w[u] + max((rem.get(v, 0.0) for v in succs.get(u, ())),
                                default=0.0)
        self.rem = rem
        # 图输入字节(单元 uin):消费的张量由原图 COPY_IN 生产
        op_type = {o["id"]: o.get("op") for o in graph["ops"]}
        tprod = {}
        for o, ts in prod.items():
            for t in ts:
                tprod[t] = o
        self.uin_src = defaultdict(float)
        for o, ts in cons.items():
            if o not in self.ids:
                continue
            for t in ts:
                p = tprod.get(t)
                if p is not None and op_type.get(p) == "COPY_IN":
                    self.uin_src[o] += tsize.get(t, 0)
        self.affinity = affinity
        self._cover(affinity)

    def _cover(self, affinity):
        """双向延伸的加权路径覆盖:按剩余权降序起链,前后各延伸。"""
        visited = set()

        def pick(cands, frm, forward):
            outs = [x for x in cands if x not in visited]
            if not outs:
                return None
            if self.affinity == "byte":
                if forward:
                    key = lambda x: (self.ew.get((frm, x), 0.0),
                                     self.rem.get(x, 0.0))
                else:
                    key = lambda x: (self.ew.get((x, frm), 0.0),
                                     self.rem.get(x, 0.0))
            else:
                key = lambda x: (self.rem.get(x, 0.0),
                                 self.ew.get((frm, x), 0.0) if forward
                                 else self.ew.get((x, frm), 0.0))
            return max(outs, key=key)

        strands = []
        order = sorted(self.ids, key=lambda v: -self.rem.get(v, 0.0))
        for v in order:
            if v in visited:
                continue
            chain = [v]
            visited.add(v)
            # 向前延伸
            cur = v
            while True:
                nxt = pick(self.succs.get(cur, ()), cur, True)
                if nxt is None:
                    break
                chain.append(nxt)
                visited.add(nxt)
                cur = nxt
            # 向后延伸
            cur = v
            while True:
                prv = pick(self.preds.get(cur, ()), cur, False)
                if prv is None:
                    break
                chain.insert(0, prv)
                visited.add(prv)
                cur = prv
            strands.append(chain)
        self.strand_ops = strands

    # ---- 分裂:控制最大股流功,保证可均衡 ----
    def split(self, max_w):
        out = []
        for chain in self.strand_ops:
            cur, cur_w = [], 0.0
            for v in chain:
                if cur and cur_w + self.w[v] > max_w:
                    out.append(cur)
                    cur, cur_w = [v], self.w[v]
                else:
                    cur.append(v)
                    cur_w += self.w[v]
            if cur:
                out.append(cur)
        return out

    def build_units(self, max_w):
        chains = self.split(max_w)
        # SCC 凝聚合并:商图(凝聚图)必无环 —— 合法性的构造性保证
        unit_of = {}
        for ui, ch in enumerate(chains):
            for v in ch:
                unit_of[v] = ui
        chains = self._scc_merge(chains, unit_of)
        return self._finalize(chains)

    def topo_units(self, max_w):
        """拓扑序连续段单元:保留 DAG 宽度结构(与链式覆盖互补)。"""
        chains = []
        cur, cw = [], 0.0
        for v in self.topo:
            if cur and cw + self.w[v] > max_w:
                chains.append(cur)
                cur, cw = [v], self.w[v]
            else:
                cur.append(v)
                cw += self.w[v]
        if cur:
            chains.append(cur)
        unit_of = {}
        for ui, ch in enumerate(chains):
            for v in ch:
                unit_of[v] = ui
        chains = self._scc_merge(chains, unit_of)
        return self._finalize(chains)

    def _scc_merge(self, chains, unit_of):
        """单元图 Tarjan SCC;多成员 SCC 合并为一个单元(合并后按拓扑序排)。"""
        n = len(chains)
        adj = [set() for _ in range(n)]
        for v in self.ids:
            ua = unit_of[v]
            for s in self.succs.get(v, ()):
                ub = unit_of[s]
                if ua != ub:
                    adj[ua].add(ub)
        index, low, onstk = {}, {}, {}
        stk, comp = [], [-1] * n
        nc, counter = 0, 0
        for s0 in range(n):
            if s0 in index:
                continue
            work = [(s0, iter(sorted(adj[s0])))]
            index[s0] = low[s0] = counter
            counter += 1
            stk.append(s0)
            onstk[s0] = True
            while work:
                v, it = work[-1]
                advanced = False
                for w in it:
                    if w not in index:
                        index[w] = low[w] = counter
                        counter += 1
                        stk.append(w)
                        onstk[w] = True
                        work.append((w, iter(sorted(adj[w]))))
                        advanced = True
                        break
                    if onstk.get(w):
                        low[v] = min(low[v], index[w])
                if advanced:
                    continue
                work.pop()
                if work:
                    pv = work[-1][0]
                    low[pv] = min(low[pv], low[v])
                if low[v] == index[v]:
                    while True:
                        w = stk.pop()
                        onstk[w] = False
                        comp[w] = nc
                        if w == v:
                            break
                    nc += 1
        if nc == n:
            return chains
        pos = {v: i for i, v in enumerate(self.topo)}
        merged = defaultdict(list)
        for ui, ch in enumerate(chains):
            merged[comp[ui]].extend(ch)
        return [sorted(ch, key=lambda v: pos[v]) for ch in merged.values()]

    def _finalize(self, chains):
        unit_of = {}
        for ui, ch in enumerate(chains):
            for v in ch:
                unit_of[v] = ui
        n = len(chains)
        um = [0.0] * n
        uv = [0.0] * n
        uin = [0.0] * n
        for ui, ch in enumerate(chains):
            for v in ch:
                if self.pm.get(v) == "PIPE_M":
                    um[ui] += self.w[v]
                else:
                    uv[ui] += self.w[v]
                uin[ui] += self.uin_src.get(v, 0.0)
        edge_bytes = defaultdict(float)
        upreds = [set() for _ in range(n)]
        usuccs = [set() for _ in range(n)]
        for v in self.ids:
            ua = unit_of[v]
            for s in self.succs.get(v, ()):
                ub = unit_of[s]
                if ua != ub:
                    edge_bytes[(ua, ub)] += self.ew.get((v, s), 0.0)
                    upreds[ub].add(ua)
                    usuccs[ua].add(ub)
        # 单元拓扑序:Kahn(SCC 合并后必为 DAG),平级按链内最小 op 位次
        pos = {v: i for i, v in enumerate(self.topo)}
        indeg = [len(ps) for ps in upreds]
        dq = deque(sorted((ui for ui in range(n) if indeg[ui] == 0),
                          key=lambda ui: min(pos[v] for v in chains[ui])))
        unit_topo = []
        while dq:
            ui = dq.popleft()
            unit_topo.append(ui)
            for w in sorted(usuccs[ui],
                            key=lambda x: min(pos[v] for v in chains[x])):
                indeg[w] -= 1
                if indeg[w] == 0:
                    dq.append(w)
        rank = {ui: r for r, ui in enumerate(unit_topo)}
        return {"chains": chains, "unit_of": unit_of, "n": n, "um": um,
                "uv": uv, "uin": uin, "edge_bytes": edge_bytes,
                "upreds": upreds, "usuccs": usuccs,
                "unit_topo": unit_topo, "rank": rank}


# ---------------- 2) 修正代理:M/V 分管并行 ----------------

def p2_sim2(u, core_of, K):
    """每核 M/V 管道独立 + MTE 串行 + 跨核 500c/跳 + 2b/60 + DDR 地板。"""
    m_free = [0.0] * K
    v_free = [0.0] * K
    mte_free = [0.0] * K
    end = {}
    cross_total = 0.0
    in_total = 0.0
    eb = u["edge_bytes"]
    for i in u["unit_topo"]:
        c = core_of[i]
        start = 0.0
        cin = 0.0
        for p in u["upreds"][i]:
            if core_of[p] == c:
                start = max(start, end[p])
            else:
                b = eb.get((p, i), 0.0)
                cross_total += b
                cin += b
                start = max(start, end[p] + DELAY_B + 2 * b / BW)
        ib = u["uin"][i] + cin
        in_total += ib
        s_m = max(start, m_free[c])
        s_v = max(start, v_free[c])
        e_mv = max(s_m + u["um"][i], s_v + u["uv"][i])
        e_mte = max(mte_free[c], start) + ib / BW
        end[i] = max(e_mv, e_mte, start)
        m_free[c] = s_m + u["um"][i]
        v_free[c] = s_v + u["uv"][i]
        mte_free[c] = e_mte
    mk = max(end.values(), default=0.0)
    ddr_floor = (2 * cross_total + in_total) / BW
    return max(mk, ddr_floor)


def greedy_assign(u, K, seed=0):
    """HEFT 式列表调度(串行核口径保证均衡):就绪单元放预计最早完成核,
    平手落到累计负载最小的核。跨核代价 500c+2b/60 计入依赖就绪。"""
    rng = random.Random(seed)
    core_load = [0.0] * K
    end = {}
    remaining = {i: set(ps) for i, ps in enumerate(u["upreds"])}
    ready = [i for i in u["unit_topo"] if not remaining[i]]
    ready.sort(key=lambda i: -(u["um"][i] + u["uv"][i]))
    eb = u["edge_bytes"]
    core_of = {}
    while ready:
        i = ready.pop(0)
        dur = u["um"][i] + u["uv"][i]
        best = None
        for c in range(K):
            start = 0.0
            for p in u["upreds"][i]:
                b = eb.get((p, i), 0.0)
                if core_of[p] == c:
                    start = max(start, end[p])
                else:
                    start = max(start, end[p] + DELAY_B + 2 * b / BW)
            fin = max(start, core_load[c]) + dur
            key = (fin, core_load[c], c)
            if best is None or key < best:
                best = key
        fin, load_c, c = best
        start = 0.0
        for p in u["upreds"][i]:
            b = eb.get((p, i), 0.0)
            if core_of[p] == c:
                start = max(start, end[p])
            else:
                start = max(start, end[p] + DELAY_B + 2 * b / BW)
        end[i] = max(start, core_load[c]) + dur
        core_load[c] = end[i]
        core_of[i] = c
        for j in u["usuccs"][i]:
            if i in remaining.get(j, set()):
                remaining[j].discard(i)
                if not remaining[j]:
                    ready.append(j)
        ready.sort(key=lambda x: -(u["um"][x] + u["uv"][x]))
    return core_of


def greedy_affinity(u, K, seed=0, thresh=1.25):
    """前驱亲和贪心:优先跟前驱最多的核走(消除500c乒乓),
    该核超载 thresh×W/K 时才改放最闲核。"""
    rng = random.Random(seed)
    total = sum(u["um"][i] + u["uv"][i] for i in range(u["n"]))
    cap = total / K * thresh
    core_load = [0.0] * K
    end = {}
    remaining = {i: set(ps) for i, ps in enumerate(u["upreds"])}
    ready = [i for i in u["unit_topo"] if not remaining[i]]
    ready.sort(key=lambda i: -(u["um"][i] + u["uv"][i]))
    eb = u["edge_bytes"]
    core_of = {}

    def dep_start(i, c):
        start = 0.0
        for p in u["upreds"][i]:
            b = eb.get((p, i), 0.0)
            if core_of[p] == c:
                start = max(start, end[p])
            else:
                start = max(start, end[p] + DELAY_B + 2 * b / BW)
        return start

    while ready:
        i = ready.pop(0)
        dur = u["um"][i] + u["uv"][i]
        votes = defaultdict(float)
        for p in u["upreds"][i]:
            votes[core_of[p]] += 1.0 + eb.get((p, i), 0.0) / 1e4
        c = None
        if votes:
            vc = max(votes.items(), key=lambda kv: kv[1])[0]
            if core_load[vc] + dur <= cap:
                c = vc
        if c is None:
            c = min(range(K), key=lambda x: (core_load[x], x))
        end[i] = max(dep_start(i, c), core_load[c]) + dur
        core_load[c] = end[i]
        core_of[i] = c
        for j in u["usuccs"][i]:
            if i in remaining.get(j, set()):
                remaining[j].discard(i)
                if not remaining[j]:
                    ready.append(j)
        ready.sort(key=lambda x: -(u["um"][x] + u["uv"][x]))
    return core_of


def local_improve(u, core_of, K, iters=800, seed=0, time_budget=8.0):
    rng = random.Random(seed)
    cur = dict(core_of)
    cur_mk = p2_sim2(u, cur, K)
    best, best_mk = dict(cur), cur_mk
    n = u["n"]
    t0 = time.perf_counter()
    for it in range(iters):
        if (it & 31) == 0 and time.perf_counter() - t0 > time_budget:
            break
        i = rng.randrange(n)
        rel = [core_of[p] for p in u["upreds"][i]] + \
              [core_of[s] for s in u["usuccs"][i]]
        if rel and rng.random() < 0.65:
            c2 = rng.choice(rel)
        else:
            c2 = rng.randrange(K)
        if c2 == cur[i]:
            continue
        cand = dict(cur)
        cand[i] = c2
        mk = p2_sim2(u, cand, K)
        if mk < cur_mk or (mk == cur_mk and rng.random() < 0.3):
            cur, cur_mk = cand, mk
            if mk < best_mk:
                best, best_mk = dict(cand), mk
    return best, best_mk


def strand_plan(u, core_of, K):
    """每股流一子图;核内序=股流拓扑位次。"""
    n2s = {}
    for ui, ch in enumerate(u["chains"]):
        sg = u["rank"][ui]
        for v in ch:
            n2s[v] = sg
    cores = [[] for _ in range(K)]
    for ui in range(u["n"]):
        cores[core_of[ui]].append(u["rank"][ui])
    for c in range(K):
        cores[c].sort()
    return {"node_to_subgraph": n2s, "core_schedules": cores}


def clustered_units(sp, u, K, W, cap_mult=1.3):
    """DSC 式边归零聚簇:按跨核代价降序合并单元簇(消除500c+字节),
    簇功上限 cap_mult×W/K,合并前做簇图可达性环检测。"""
    n = u["n"]
    cap = W / K * cap_mult
    parent = list(range(n))
    cw = [u["um"][i] + u["uv"][i] for i in range(n)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cadj = [set() for _ in range(n)]  # 簇图(随合并动态收缩)
    for i in range(n):
        for j in u["usuccs"][i]:
            cadj[i].add(j)

    def reach(a, b, limit=4000):
        # b簇是否可达a簇(有向) => 合并成环
        if a == b:
            return True
        seen = {b}
        dq = deque([b])
        steps = 0
        while dq and steps < limit:
            x = dq.popleft()
            for y in cadj[x]:
                y = find(y)
                if y == a:
                    return True
                if y not in seen:
                    seen.add(y)
                    dq.append(y)
            steps += 1
        return False

    edges = []
    for i in range(n):
        for j in u["usuccs"][i]:
            b = u["edge_bytes"].get((i, j), 0.0)
            edges.append((DELAY_B + 2 * b / BW, i, j))
    edges.sort(reverse=True)
    for cost, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if cw[ri] + cw[rj] > cap:
            continue
        if reach(ri, rj):
            continue
        parent[rj] = ri
        cw[ri] += cw[rj]
        cadj[ri] |= cadj[rj]
        cadj[ri].discard(ri)
    # 簇 -> 合并链(unit_topo 序拼接)
    groups = defaultdict(list)
    for ui in u["unit_topo"]:
        groups[find(ui)].extend(u["chains"][ui])
    chains = list(groups.values())
    unit_of = {}
    for ci, ch in enumerate(chains):
        for v in ch:
            unit_of[v] = ci
    chains = sp._scc_merge(chains, unit_of)
    return sp._finalize(chains)


def strand_candidates(graph, K=5, log=print):
    """生成候选族:链覆盖 × 3 粒度 + 拓扑段 × 4 粒度,各 2 种子。"""
    W = sum(cycles_map(graph).values())
    out = []
    try:
        sp = StrandPack(graph, affinity="byte")
    except Exception as exc:
        log(f"  strand cover fail: {exc}")
        sp = None
    if sp is not None:
        for gpc in (8, 16, 32):
            base = sp.topo_units(W / (K * gpc))
            if base["n"] < K:
                continue
            for cm in (1.15, 1.3):
                try:
                    u = clustered_units(sp, base, K, W, cap_mult=cm)
                except Exception as exc:
                    log(f"  cluster g{gpc} cm{cm} fail: {str(exc)[:50]}")
                    continue
                if u["n"] < K:
                    continue
                best, best_mk = None, float("inf")
                for seed in (0, 1):
                    co = greedy_assign(u, K, seed=seed)
                    co, mk = local_improve(u, co, K, seed=seed + 7,
                                           time_budget=6.0)
                    if mk < best_mk:
                        best, best_mk = co, mk
                plan = strand_plan(u, best, K)
                out.append((f"clu_g{gpc}_c{int(cm*100)}", plan,
                            {"units": u["n"], "sim": best_mk}))
        for gpc in (1, 2, 4):
            u = sp.build_units(W / (K * gpc))
            if u["n"] < K:
                continue
            best, best_mk = None, float("inf")
            for seed in (0, 1):
                co = greedy_assign(u, K, seed=seed)
                co, mk = local_improve(u, co, K, seed=seed + 7,
                                       time_budget=6.0)
                if mk < best_mk:
                    best, best_mk = co, mk
            plan = strand_plan(u, best, K)
            out.append((f"strand_byte_g{gpc}", plan,
                        {"units": u["n"], "sim": best_mk}))
        for gpc in (4, 8, 16, 32):
            u = sp.topo_units(W / (K * gpc))
            if u["n"] < K:
                continue
            best, best_mk = None, float("inf")
            for seed in (0, 1):
                co = greedy_affinity(u, K, seed=seed)
                co, mk = local_improve(u, co, K, seed=seed + 7,
                                       time_budget=6.0)
                if mk < best_mk:
                    best, best_mk = co, mk
            plan = strand_plan(u, best, K)
            out.append((f"topoaff_g{gpc}", plan,
                        {"units": u["n"], "sim": best_mk}))
    return out


# ---------------- 2b) METIS 字节感知划分(场景B原生目标) ----------------

def metis_plan(graph, K=5, lat_bytes=15000.0, ub_factor=3, seed=1,
               balance_pipe=False):
    """最小割字节+负载均衡的 k 路划分。

    每条切边的代价口径 = 2b/60(带宽) + 500c(延迟) ≈ (b + 15000) 字节;
    顶点权 = op cycles(或 M/V 分管道二元权,metis 支持多约束)。
    子图 = 每核诱导子图的连通分量(天然避免核级双向边成环风险,
    再经 SCC 凝聚兜底);核内序 = 子图拓扑位次。
    """
    import pymetis
    ids, preds, succs = op_dag(graph)
    w = cycles_map(graph)
    pm = pipe_map(graph)
    prod, cons, tsize, _ = tensor_views(graph)
    ew = defaultdict(float)
    for o, ts in cons.items():
        for t in ts:
            for p in prod.get(t, ()):
                if p != o:
                    ew[(p, o)] += tsize.get(t, 0)
    idx = {v: i for i, v in enumerate(ids)}
    n = len(ids)
    adj = [[] for _ in range(n)]
    for (a, b), by in ew.items():
        if a in idx and b in idx:
            adj[idx[a]].append((idx[b], int(by + lat_bytes)))
            adj[idx[b]].append((idx[a], int(by + lat_bytes)))
    for v in ids:
        for s in succs.get(v, ()):
            if (v, s) not in ew and (s, v) not in ew:
                adj[idx[v]].append((idx[s], int(lat_bytes)))
                adj[idx[s]].append((idx[v], int(lat_bytes)))
    # 去重
    for i in range(n):
        seen = {}
        for j, wgt in adj[i]:
            seen[j] = max(seen.get(j, 0), wgt)
        adj[i] = [(j, wgt) for j, wgt in sorted(seen.items())]
    if balance_pipe:
        vwgt = [(int(w[v]), int(w[v]) if pm.get(v) == "PIPE_M" else 0,
                 int(w[v]) if pm.get(v) != "PIPE_M" else 0) for v in ids]
    else:
        vwgt = [int(w[v]) for v in ids]
    xadj = _xadj(adj)
    adjncy, adjwgt = [], []
    for lst in adj:
        for j, wg in lst:
            adjncy.append(j)
            adjwgt.append(wg)
    try:
        ncuts, membership = pymetis.part_graph(
            K, xadj=xadj, adjncy=adjncy, adjwgt=adjwgt, vwgt=vwgt,
            recursive=False, niter=10, seed=seed)
    except TypeError:
        try:
            ncuts, membership = pymetis.part_graph(
                K, xadj=xadj, adjncy=adjncy, adjwgt=adjwgt, vwgt=vwgt,
                recursive=False, niter=10)
        except TypeError:
            ncuts, membership = pymetis.part_graph(
                K, xadj=xadj, adjncy=adjncy, eweights=adjwgt,
                vweights=vwgt, recursive=False)
    # membership -> 每核诱导子图连通分量 = 子图
    core_of = {v: membership[idx[v]] for v in ids}
    return _components_plan(graph, core_of, K, ids, succs)


def _xadj(adj):
    xadj = [0]
    for lst in adj:
        xadj.append(xadj[-1] + len(lst))
    return xadj


def fm_plan(graph, K=5, lat_bytes=15000.0, cap_slack=0.06, passes=8,
            seed=0):
    """FM 式划分:拓扑条带初始化 + 边界 op 移动(字节增益>0 且守负载帽)。
    切边代价口径与 metis_plan 同:(b + lat_bytes)。"""
    import random as _rnd
    rng = _rnd.Random(seed)
    ids, preds, succs = op_dag(graph)
    w = cycles_map(graph)
    prod, cons, tsize, _ = tensor_views(graph)
    ew = defaultdict(float)
    for o, ts in cons.items():
        for t in ts:
            for p in prod.get(t, ()):
                if p != o:
                    ew[(p, o)] += tsize.get(t, 0)
    nbrs = defaultdict(dict)
    for v in ids:
        for s in succs.get(v, ()):
            b = ew.get((v, s), 0.0) + lat_bytes
            nbrs[v][s] = b
            nbrs[s][v] = b
    total = sum(w[v] for v in ids)
    cap = total / K * (1 + cap_slack)
    # 初始化:拓扑序贪心条带
    topo_pos = _topo_pos(ids, succs)
    order = sorted(ids, key=lambda v: topo_pos[v])
    core_of = {}
    loads = [0.0] * K
    cur, c = 0, 0
    for v in order:
        if cur + w[v] > total / K and c < K - 1:
            c += 1
            cur = 0.0
        core_of[v] = c
        loads[c] += w[v]
        cur += w[v]
    # 均衡修复:过载核整段挪给最轻核
    for _ in range(3):
        hi = max(range(K), key=lambda x: loads[x])
        lo = min(range(K), key=lambda x: loads[x])
        if loads[hi] <= cap:
            break
        move = [v for v in order if core_of[v] == hi][-max(1, int((loads[hi]-loads[lo])/2 / max(1.0, w[order[-1]]))):]
        seg = [v for v in order if core_of[v] == hi]
        if not seg:
            break
        take = min(len(seg), max(1, int(len(seg) * 0.2)))
        for v in seg[-take:]:
            core_of[v] = lo
            loads[hi] -= w[v]
            loads[lo] += w[v]
    # FM passes
    for _ in range(passes):
        locked = set()
        moved_any = False
        boundary = [v for v in ids
                    if len({core_of[u] for u in nbrs[v]}) > 1]
        rng.shuffle(boundary)
        for v in boundary:
            if v in locked:
                continue
            a = core_of[v]
            gain = defaultdict(float)
            for u, b in nbrs[v].items():
                gain[core_of[u]] += b if core_of[u] != a else -b
            best = None
            for c2, g in gain.items():
                if c2 == a or g <= 0:
                    continue
                if loads[a] - w[v] < 0 or loads[c2] + w[v] > cap:
                    continue
                if best is None or g > best[1]:
                    best = (c2, g)
            if best:
                c2, _ = best
                core_of[v] = c2
                loads[a] -= w[v]
                loads[c2] += w[v]
                locked.add(v)
                moved_any = True
        if not moved_any:
            break
    return _components_plan(graph, core_of, K, ids, succs)


def plan_to_units(graph, plan):
    """把现有方案的子图结构转成 unit dict(供 p2_sim2 评估与 refinement)。"""
    ids, preds, succs = op_dag(graph)
    w = cycles_map(graph)
    pm = pipe_map(graph)
    prod, cons, tsize, _ = tensor_views(graph)
    ew = defaultdict(float)
    for o, ts in cons.items():
        for t in ts:
            for p in prod.get(t, ()):
                if p != o:
                    ew[(p, o)] += tsize.get(t, 0)
    n2s = {int(k): v for k, v in plan["node_to_subgraph"].items()}
    sgs = sorted(set(n2s.values()))
    sidx = {sg: i for i, sg in enumerate(sgs)}
    n = len(sgs)
    um = [0.0] * n
    uv = [0.0] * n
    for v in ids:
        ui = sidx[n2s[v]]
        if pm.get(v) == "PIPE_M":
            um[ui] += w[v]
        else:
            uv[ui] += w[v]
    edge_bytes = defaultdict(float)
    upreds = [set() for _ in range(n)]
    usuccs = [set() for _ in range(n)]
    for v in ids:
        for s in succs.get(v, ()):
            a, b = sidx[n2s[v]], sidx[n2s[s]]
            if a != b:
                edge_bytes[(a, b)] += ew.get((v, s), 0.0)
                upreds[b].add(a)
                usuccs[a].add(b)
    # 图输入字节
    op_type = {o["id"]: o.get("op") for o in graph["ops"]}
    tprod = {}
    for o, ts in prod.items():
        for t in ts:
            tprod[t] = o
    uin = [0.0] * n
    for o, ts in cons.items():
        if o in set(ids):
            for t in ts:
                p = tprod.get(t)
                if p is not None and op_type.get(p) == "COPY_IN":
                    uin[sidx[n2s[o]]] += tsize.get(t, 0)
    chains = [[v for v in ids if n2s[v] == sg] for sg in sgs]
    topo_pos = _topo_pos(ids, succs)
    indeg = [len(ps) for ps in upreds]
    dq = deque(sorted((ui for ui in range(n) if indeg[ui] == 0),
                      key=lambda ui: min(topo_pos[v] for v in chains[ui])))
    unit_topo = []
    while dq:
        ui = dq.popleft()
        unit_topo.append(ui)
        for x in sorted(usuccs[ui]):
            indeg[x] -= 1
            if indeg[x] == 0:
                dq.append(x)
    rank = {ui: r for r, ui in enumerate(unit_topo)}
    return {"chains": chains, "n": n, "um": um, "uv": uv, "uin": uin,
            "edge_bytes": edge_bytes, "upreds": upreds, "usuccs": usuccs,
            "unit_topo": unit_topo, "rank": rank,
            "sg_ids": sgs, "n2s": n2s}


def refine_plan(graph, plan, K, iters=6000, seed=0, time_budget=20.0,
                unit_map=None):
    """子图级爬山:从现有方案出发,搬移子图(含连续段)到其他核。
    核内序按全局拓扑位次重排(恒合法)。目标 = p2_sim2。"""
    u = unit_map if unit_map is not None else plan_to_units(graph, plan)
    core_of_sg = {}
    for ci, sched in enumerate(plan["core_schedules"]):
        for sg in sched:
            core_of_sg[sg] = ci
    core_of = [core_of_sg.get(sg, 0) for sg in u["sg_ids"]]
    cur = list(core_of)
    cur_mk = p2_sim2(u, cur, K)
    best, best_mk = list(cur), cur_mk
    rng = random.Random(seed)
    by_core = defaultdict(list)
    for ui, c in enumerate(cur):
        by_core[c].append(ui)
    for c in by_core:
        by_core[c].sort(key=lambda ui: u["rank"][ui])
    t0 = time.perf_counter()
    for it in range(iters):
        if (it & 63) == 0 and time.perf_counter() - t0 > time_budget:
            break
        # 选一个源核(偏向负载高的)与一段连续子图
        src = max(range(K), key=lambda c: sum(u["um"][i] + u["uv"][i]
                                              for i in by_core[c]))
        if rng.random() < 0.4:
            src = rng.randrange(K)
        lst = by_core[src]
        if not lst:
            continue
        span = 1 if rng.random() < 0.6 or len(lst) < 2 else rng.randint(2, min(6, len(lst)))
        p0 = rng.randrange(len(lst) - span + 1)
        moving = lst[p0:p0 + span]
        dst = rng.choice([c for c in range(K) if c != src])
        cand = list(cur)
        for ui in moving:
            cand[ui] = dst
        mk = p2_sim2(u, cand, K)
        if mk < cur_mk - 1e-9 or (mk < cur_mk + 1e-9 and rng.random() < 0.25):
            cur, cur_mk = cand, mk
            for ui in moving:
                by_core[src].remove(ui)
                bisect.insort(by_core[dst], ui, key=lambda x: u["rank"][x])
            if mk < best_mk:
                best, best_mk = list(cur), mk
    # 重排发射
    cores = [[] for _ in range(K)]
    for ui, c in enumerate(best):
        cores[c].append(u["sg_ids"][ui])
    for c in range(K):
        cores[c].sort(key=lambda sg: u["rank"][u["sg_ids"].index(sg)]
                      if sg in u["sg_ids"] else 0)
    return {"node_to_subgraph": {str(v): sg for v, sg in u["n2s"].items()},
            "core_schedules": cores}, best_mk


def emit_from_cores(u, core_of, K):
    rank_by_sg = {sg: u["rank"][ui] for ui, sg in enumerate(u["sg_ids"])}
    cores = [[] for _ in range(K)]
    for ui, c in enumerate(core_of):
        cores[c].append(u["sg_ids"][ui])
    for c in range(K):
        cores[c].sort(key=lambda sg: rank_by_sg[sg])
    return {"node_to_subgraph": {str(v): sg for v, sg in u["n2s"].items()},
            "core_schedules": cores}


def refine_real(g, plan, K, fe, evals=250, seed=0, log=print):
    """真值评估驱动的子图搬移爬山(FastEval bit-exact)。
    子图结构不变,只改核归属;核内序=拓扑位次。"""
    u = plan_to_units(g, plan)
    core_of_sg = {}
    for ci, sched in enumerate(plan["core_schedules"]):
        for sg in sched:
            core_of_sg[sg] = ci
    core_of = [core_of_sg.get(sg, 0) for sg in u["sg_ids"]]
    work = [u["um"][i] + u["uv"][i] for i in range(u["n"])]
    by_core = defaultdict(list)
    for ui, c in enumerate(core_of):
        by_core[c].append(ui)
    for c in by_core:
        by_core[c].sort(key=lambda ui: u["rank"][ui])
    cur_mk = fe.evaluate(emit_from_cores(u, core_of, K))[0]
    best, best_mk = list(core_of), cur_mk
    rng = random.Random(seed)
    for it in range(evals):
        src = max(range(K), key=lambda c: sum(work[i] for i in by_core[c]))
        if rng.random() < 0.35:
            src = rng.randrange(K)
        lst = by_core[src]
        if not lst:
            continue
        span = 1 if rng.random() < 0.5 or len(lst) < 2 else rng.randint(2, min(8, len(lst)))
        p0 = rng.randrange(len(lst) - span + 1)
        moving = lst[p0:p0 + span]
        dst = rng.choice([c for c in range(K) if c != src])
        cand = list(core_of)
        for ui in moving:
            cand[ui] = dst
        mk = fe.evaluate(emit_from_cores(u, cand, K))[0]
        if mk < cur_mk - 0.5:
            cur_mk = mk
            core_of = cand
            for ui in moving:
                by_core[src].remove(ui)
                bisect.insort(by_core[dst], ui, key=lambda x: u["rank"][x])
            if mk < best_mk:
                best, best_mk = list(core_of), mk
    return emit_from_cores(u, best, K), best_mk


def metis_candidates(graph, K=5, log=print):
    out = []
    for lat in (15000.0, 3000.0, 60000.0):
        for seed in (1, 2, 3):
            try:
                plan = metis_plan(graph, K=K, lat_bytes=lat, seed=seed)
                out.append((f"metis_lat{int(lat)}_s{seed}", plan, {}))
            except Exception as exc:
                log(f"  metis lat={lat} s={seed} fail: {str(exc)[:60]}")
    return out


def _components_plan(graph, core_of, K, ids, succs):
    """每核诱导子图的连通分量各成一子图(无向连通 => 无双向跨核环风险),
    再做子图 SCC 兜底;核内序 = 子图最小拓扑位次。"""
    from collections import defaultdict
    # 无向邻接(经 contracted succs)
    nbr = defaultdict(set)
    for v in ids:
        for s in succs.get(v, ()):
            nbr[v].add(s)
            nbr[s].add(v)
    comp_id = {}
    cid = 0
    for v in ids:
        if v in comp_id:
            continue
        stack = [v]
        comp_id[v] = cid
        while stack:
            x = stack.pop()
            for y in nbr[x]:
                if core_of[y] == core_of[v] and y not in comp_id:
                    comp_id[y] = cid
                    stack.append(y)
        cid += 1
    # 子图 SCC 兜底(分量间跨核有向边可能成环)
    sg_edges = defaultdict(set)
    for v in ids:
        for s in succs.get(v, ()):
            a, b = comp_id[v], comp_id[s]
            if a != b:
                sg_edges[a].add(b)
    comp_id = _scc_merge_ids(sg_edges, comp_id)
    sg_edges2 = defaultdict(set)
    for v in ids:
        for s in succs.get(v, ()):
            a, b = comp_id[v], comp_id[s]
            if a != b:
                sg_edges2[a].add(b)
    # 子图 DAG Kahn 拓扑序(SCC 凝聚后必无环),核内序取其 restriction
    indeg = defaultdict(int)
    nodes = sorted(set(comp_id.values()))
    for a in nodes:
        for b in sg_edges2[a]:
            indeg[b] += 1
    topo_pos = _topo_pos(ids, succs)
    dq = deque(sorted((x for x in nodes if indeg[x] == 0),
                      key=lambda x: -x))
    pos = {}
    i = 0
    while dq:
        x = dq.popleft()
        pos[x] = i
        i += 1
        for y in sorted(sg_edges2[x]):
            indeg[y] -= 1
            if indeg[y] == 0:
                dq.append(y)
    for x in nodes:
        pos.setdefault(x, i)
        i += 1
    sg_core = {}
    for v in ids:
        sg_core[comp_id[v]] = core_of[v]
    cores = [[] for _ in range(K)]
    for sg, c in sg_core.items():
        cores[c].append(sg)
    for c in range(K):
        cores[c].sort(key=lambda sg: pos[sg])
    n2s = {v: comp_id[v] for v in ids}
    return {"node_to_subgraph": n2s, "core_schedules": cores}


def _topo_pos(ids, succs):
    from collections import deque
    preds = defaultdict(int)
    for v in ids:
        for s in succs.get(v, ()):
            preds[s] += 1
    dq = deque(sorted(v for v in ids if preds[v] == 0))
    pos = {}
    i = 0
    while dq:
        v = dq.popleft()
        pos[v] = i
        i += 1
        for s in sorted(succs.get(v, ())):
            preds[s] -= 1
            if preds[s] == 0:
                dq.append(s)
    for v in ids:
        pos.setdefault(v, i)
        i += 1
    return pos


def _scc_merge_ids(sg_edges, comp_id):
    """子图级 SCC 合并(凝聚必无环)。"""
    nodes = sorted(set(comp_id.values()))
    index, low, onstk = {}, {}, {}
    stk, comp = [], {}
    nc, counter = 0, 0
    adjmap = {x: sorted(sg_edges.get(x, ())) for x in nodes}
    for s0 in nodes:
        if s0 in index:
            continue
        work = [(s0, iter(adjmap[s0]))]
        index[s0] = low[s0] = counter
        counter += 1
        stk.append(s0)
        onstk[s0] = True
        while work:
            v, it = work[-1]
            adv = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stk.append(w)
                    onstk[w] = True
                    work.append((w, iter(adjmap[w])))
                    adv = True
                    break
                if onstk.get(w):
                    low[v] = min(low[v], index[w])
            if adv:
                continue
            work.pop()
            if work:
                pv = work[-1][0]
                low[pv] = min(low[pv], low[v])
            if low[v] == index[v]:
                while True:
                    x = stk.pop()
                    onstk[x] = False
                    comp[x] = nc
                    if x == v:
                        break
                nc += 1
    if nc == len(nodes):
        return comp_id
    return {v: comp[sg] for v, sg in comp_id.items()}




def run_cases(cases, q=3, K=5, budget_s=120.0):
    posthoc = json.load(open(Path(os.environ.get("A2026_POSTHOC", str(SOLVER_DIR.parent.parent / "records" / "POSTHOC_BEST.json"))),
                             encoding="utf-8"))
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        try:
            g = load_case(case)
            sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
            cands = strand_candidates(g, K=K)
            cands += metis_candidates(g, K=K)
            best = None
            for tag, plan, meta in cands:
                try:
                    r, wt = (ev_p3 if q == 3 else ev_p2)(g, plan)
                    mk = r["makespan"]
                    if best is None or mk < best[1]:
                        best = (tag, mk, meta)
                except Exception as exc:
                    print(f"  {tag} eval fail: {str(exc)[:70]}")
            key = f"{case}|q{q}|N{K}"
            base = posthoc.get(key, float("nan"))
            if best:
                sp = sc / best[1]
                gain = sp / base - 1 if base == base else float("nan")
                rows.append({"case": case, "tag": best[0], "mk": best[1],
                             "sp": round(sp, 4),
                             "base": round(base, 4) if base == base else None,
                             "gain%": round(gain * 100, 2) if gain == gain else None,
                             "units": best[2]["units"],
                             "wall": round(time.perf_counter() - t0, 1)})
                print(f"{case}: {best[0]} sp={sp:.3f} base={base:.3f} "
                      f"gain={gain*100:+.1f}% units={best[2]['units']} "
                      f"({rows[-1]['wall']}s)")
            else:
                rows.append({"case": case, "tag": "NONE"})
                print(f"{case}: no legal candidate")
        except Exception as exc:
            rows.append({"case": case, "tag": f"ERR {str(exc)[:60]}"})
            print(f"{case}: ERR {str(exc)[:80]}")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="case_005,case_047,case_086,case_056")
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    args = ap.parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    rows = run_cases(cases, q=args.q, K=args.K)
    out = Path(HERE / f"strand_p2_q{args.q}_N{args.K}.csv")
    with open(out, "w", encoding="utf-8") as f:
        if rows:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(str(v) for v in r.values()) + "\n")
    print("saved:", out)
