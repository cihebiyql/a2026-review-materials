"""v3 问题1（场景A）求解器：签名类装箱 + 边界细化。

核心思想（本轮实验证实）：
1. 场景A 任务激活是全有或全无 → 商图任何一条边都把两个任务串行化。
   所以目标是：商图关键路（Σ任务时长+等待）最小 + 核间均衡。
2. 签名类（op 可达 sink 集合）沿边嵌套递减 → 类序即拓扑序 → 商图天然
   无环，可任意装箱（case_014: cross 仅 20 条边但 v2 连续切把它排成
   深链 → 1.0；类装箱 3.77）。
3. 大类沿类内拓扑二分到粒度上限；装箱单元 = 类/类片。
4. 粒度自适应：装箱后按"全局拓扑序中的同核极大游程"合并为任务，
   少任务少等待少重复读；多种粒度候选并行评估。
5. 组合：v2 连续切种子 + 类装箱种子 → fast 代理 LS → 官方代理 → 真评。
"""
import heapq
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v2_solver"))
from common import (load_case, op_dag, cycles_map, ev_p1, sc_makespan,  # noqa: E402
                    tensor_views, CAP, BW)
from proxy import task_sim, official_proxy  # noqa: E402

SC_DIR_DEFAULT = Path(r"C:/shumo_live/02_求解/A题_2026/results/singlecore")
SAME_WAIT, CROSS_WAIT = 100, 1000


# ---------------- 基础 ----------------

def topo_order(ids, preds, succs):
    indeg = {v: len(preds[v]) for v in ids}
    q = deque(sorted(v for v in ids if indeg[v] == 0))
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for s in sorted(succs.get(u, ())):
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    return order


def sink_sigs(ids, succs, order):
    sig = {}
    for v in reversed(order):
        ss = succs.get(v, ())
        if not ss:
            sig[v] = 1 << (len(sig) if False else 0)  # 占位，下面重算
    # 重新计算：先给 sink 编号
    sinks = [v for v in order if not succs.get(v)]
    sidx = {s: i for i, s in enumerate(sinks)}
    for v in reversed(order):
        ss = succs.get(v, ())
        if not ss:
            sig[v] = 1 << sidx[v]
        else:
            m = 0
            for s in ss:
                m |= sig[s]
            sig[v] = m
    return sig, sinks


def external_bytes_map(ids, preds, succs, graph):
    """每 op 的外部输入字节估计：消费的、由核外（COPY_IN 前）生产的张量。

    场景A 重复读：每任务需自行从 DDR 搬入其图输入（COPY_IN 后的 UB 张量
    若生产者在别的任务）。这里统计 op 消费张量中"生产者不在本 op"且
    "生产者为 COPY_IN（图输入）"的字节。
    """
    prod, cons, tsize, tpos = tensor_views(graph)
    op_type = {o["id"]: o.get("op") for o in graph["ops"]}
    tprod = {}
    for o, ts in prod.items():
        for t in ts:
            tprod[t] = o
    ext = {v: 0 for v in ids}
    for o, ts in cons.items():
        for t in ts:
            p = tprod.get(t)
            if p is not None and op_type.get(p) == "COPY_IN":
                ext[o] += tsize.get(t, 0)
    return ext


# ---------------- 单元构建 ----------------

class UnitPack:
    """装箱单元 = 签名类/类片，附带单元 DAG 与外部输入字节。"""

    def __init__(self, graph, max_unit_w, order_mode="dfs"):
        ids, preds, succs = op_dag(graph)
        self.ids, self.preds, self.succs = ids, preds, succs
        w = cycles_map(graph)
        self.w = w
        # 类内序：DFS 路径跟随（整链聚拢，切边少）或朴素拓扑（分层采样）
        if order_mode == "dfs":
            from structure_split import dfs_priority_topo
            order = dfs_priority_topo(ids, preds, succs)
        else:
            order = topo_order(ids, preds, succs)
        self.pos = {v: i for i, v in enumerate(order)}
        sig, sinks = sink_sigs(ids, succs, order)
        self.n_sinks = len(sinks)
        cls_nodes = defaultdict(list)
        for v in ids:
            cls_nodes[sig[v]].append(v)
        # 大类沿类内 DFS 序切分：切点吸附到跨界边局部极小（窗口内）
        raw_units = []
        for s, lst in cls_nodes.items():
            lst.sort(key=lambda v: self.pos[v])
            tw = sum(w[v] for v in lst)
            nparts = max(1, int(tw // max_unit_w) + (1 if tw % max_unit_w else 0))
            if nparts == 1:
                raw_units.append(list(lst))
                continue
            idset = set(lst)
            from structure_split import crossing_at
            cross = crossing_at(lst, preds, succs, idset)
            prefix = [0]
            for v in lst:
                prefix.append(prefix[-1] + w[v])
            per = tw / nparts
            cuts = [0]
            for piece in range(1, nparts):
                if cuts[-1] >= len(lst) - 1:
                    break
                target = per * piece
                lo = cuts[-1] + 1
                span = max(4, len(lst) // 40)
                center = len(lst) - 1
                for p in range(lo, len(lst)):
                    if prefix[p] >= target:
                        center = p
                        break
                wlo = max(lo, center - span)
                whi = min(len(lst), center + span + 1)
                p = min(range(wlo, max(wlo + 1, whi)),
                        key=lambda q: (cross[q - 1] if q - 1 < len(cross) else 0,
                                       abs(prefix[q] - target), q))
                p = max(p, cuts[-1] + 1)
                cuts.append(p)
            cuts.append(len(lst))
            for j in range(len(cuts) - 1):
                if cuts[j] < cuts[j + 1]:
                    raw_units.append(lst[cuts[j]:cuts[j + 1]])
        # 临时单元 DAG → 单元拓扑序 → 沿该序合并过小相邻单元
        #（拓扑相邻单元合并 = 连续切段变粗，必无环）
        tmp_of = {}
        for ui, lst in enumerate(raw_units):
            for v in lst:
                tmp_of[v] = ui
        n0 = len(raw_units)
        tus = defaultdict(set)
        tpreds = defaultdict(set)
        for u in ids:
            for s2 in succs.get(u, ()):
                a, b = tmp_of[u], tmp_of[s2]
                if a != b:
                    tpreds[b].add(a)
                    tus[a].add(b)
        tin = {i: len(tpreds[i]) for i in range(n0)}
        q = deque(sorted(i for i in range(n0) if tin[i] == 0))
        topo0 = []
        while q:
            u = q.popleft()
            topo0.append(u)
            for s2 in sorted(tus.get(u, ())):
                tin[s2] -= 1
                if tin[s2] == 0:
                    q.append(s2)
        assert len(topo0) == n0
        uw0 = [sum(w[v] for v in lst) for lst in raw_units]
        min_unit_w = max(1, max_unit_w // 4)
        groups = []  # list of list of unit indices (沿拓扑序)
        gw = []
        for ui in topo0:
            if groups and (gw[-1] < min_unit_w or uw0[ui] < min_unit_w) \
                    and gw[-1] + uw0[ui] <= max_unit_w:
                groups[-1].append(ui)
                gw[-1] += uw0[ui]
            else:
                groups.append([ui])
                gw.append(uw0[ui])
        self.unit_ops = []
        for grp in groups:
            lst = []
            for ui in grp:
                lst.extend(raw_units[ui])
            self.unit_ops.append(lst)
        self.unit_of = {}
        for ui, lst in enumerate(self.unit_ops):
            for v in lst:
                self.unit_of[v] = ui
        self.uw = [sum(w[v] for v in lst) for lst in self.unit_ops]
        # 单元 DAG
        self.upreds = defaultdict(set)
        self.usuccs = defaultdict(set)
        for u in ids:
            for s in succs.get(u, ()):
                a, b = self.unit_of[u], self.unit_of[s]
                if a != b:
                    self.upreds[b].add(a)
                    self.usuccs[a].add(b)
        # 单元拓扑序（保证编号合法）
        self.unit_topo = self._unit_topo()
        self.rank = {u: r for r, u in enumerate(self.unit_topo)}
        # 外部字节
        ext = external_bytes_map(ids, preds, succs, graph)
        self.uext = [sum(ext[v] for v in lst) for lst in self.unit_ops]

    def _unit_topo(self):
        n = len(self.unit_ops)
        indeg = {i: len(self.upreds[i]) for i in range(n)}
        q = deque(sorted(i for i in range(n) if indeg[i] == 0))
        out = []
        while q:
            u = q.popleft()
            out.append(u)
            for s in sorted(self.usuccs.get(u, ())):
                indeg[s] -= 1
                if indeg[s] == 0:
                    q.append(s)
        assert len(out) == n
        return out

    def n_units(self):
        return len(self.unit_ops)


# ---------------- 装箱评估（fast 代理）----------------

def eval_pack(pk, core_of, K, kappa, dup_penalty=True):
    n = pk.n_units()
    durations = {}
    for i in range(n):
        dur = kappa * pk.uw[i]
        if dup_penalty and K > 1:
            # 外部输入重复读的带宽税（粗估：均摊到全带宽）
            dur += pk.uext[i] / BW * 1.0
        durations[i] = dur
    core_order = {c: [i for i in pk.unit_topo if core_of[i] == c]
                  for c in range(K)}
    cof = {i: core_of[i] for i in range(n)}
    try:
        return task_sim(durations, cof, core_order, pk.upreds)
    except RuntimeError:
        return float("inf")


def greedy_pack(pk, K, kappa, seed=0, hysteresis=120.0):
    """事件驱动贪心：就绪单元选核 = 最早完成（含跨核等待）；
    平手偏向上一单元所在核（滞回，减少游程碎片）。"""
    rng = random.Random(seed)
    n = pk.n_units()
    remaining = {i: set(pk.upreds[i]) for i in range(n)}
    ready = sorted((i for i in range(n) if not remaining[i]),
                   key=lambda i: -pk.uw[i])
    core_of = {}
    est_end = {}
    core_free = [0.0] * K
    last_core = None
    while ready:
        i = ready.pop(0)
        preds = list(pk.upreds[i])
        best = None
        for c in range(K):
            wait = 0.0
            for p in preds:
                if core_of[p] == c:
                    wait = max(wait, est_end[p])
                else:
                    wait = max(wait, est_end[p] + CROSS_WAIT)
            dur = kappa * pk.uw[i] + pk.uext[i] / BW
            finish = max(core_free[c], wait) + dur
            if c == last_core:
                finish -= hysteresis  # 滞回：同核串行只 +100，切换有切换成本
            if best is None or finish < best[0] - 1e-9:
                best = (finish, c)
        _, c = best
        core_of[i] = c
        start = max(core_free[c],
                    max((est_end[p] + (0 if core_of[p] == c else CROSS_WAIT)
                         for p in preds), default=0))
        dur = kappa * pk.uw[i] + pk.uext[i] / BW
        est_end[i] = start + dur
        core_free[c] = est_end[i]
        last_core = c
        for j in pk.usuccs.get(i, ()):
            if i in remaining.get(j, ()):
                remaining[j].discard(i)
                if not remaining[j]:
                    ready.append(j)
        ready.sort(key=lambda x: -pk.uw[x])
    return core_of


def local_improve(pk, core_of, K, kappa, iters=400, seed=0,
                  time_budget=None):
    rng = random.Random(seed)
    n = pk.n_units()
    cur = dict(core_of)
    cur_mk = eval_pack(pk, cur, K, kappa)
    best, best_mk = dict(cur), cur_mk
    t0 = time.perf_counter()
    for it in range(iters):
        if time_budget and time.perf_counter() - t0 > time_budget:
            break
        i = rng.randrange(n)
        # 定向：偏好前驱/后继所在核
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
        mk = eval_pack(pk, cand, K, kappa)
        if mk < cur_mk or (mk == cur_mk and rng.random() < 0.3):
            cur, cur_mk = cand, mk
            if mk < best_mk:
                best, best_mk = dict(cand), mk
    return best, best_mk


# ---------------- 方案生成（游程合并任务）----------------

def runs_merge_plan(pk, core_of, K, max_tasks_per_core=None):
    """全局单元拓扑序中同核极大游程 = 任务（同核序沿全局序 → 合法）。"""
    seq = pk.unit_topo
    n2s = {}
    sg_id = 0
    prev_core = None
    cores = [[] for _ in range(K)]
    cur_sg = -1
    for i in seq:
        c = core_of[i]
        if c != prev_core or cur_sg < 0:
            cur_sg = sg_id
            sg_id += 1
            prev_core = c
            cores[c].append(cur_sg)
        for v in pk.unit_ops[i]:
            n2s[v] = cur_sg
    return {"node_to_subgraph": n2s, "core_schedules": cores}, sg_id


# ---------------- 主求解 ----------------

def solve_p1(graph, case, sc, K=4, cal_ratio=0.89, log=print,
             real_top=3, ls_iters=400, budget_s=420.0, use_wave=True):
    """budget_s：本题硬时限（秒）。超软线（60%）停发新候选；超硬线
    立即返回当前最优（至少已完成一个真评估——兜底 v2_cpc1）。
    use_wave：种子族 C（波前划分，结构门控自动启用，见 wave_seed.py）。"""
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    ids, preds, succs = op_dag(graph)
    w = cycles_map(graph)
    total = sum(w.values())
    kappa = sc / total
    cands = []  # (tag, plan, fast_mk)

    # ---- 种子族 A：v2 连续切（结构感知；也是超时兜底）----
    sys.path.insert(0, str(HERE.parent / "v2_solver"))
    try:
        from structure_split import structure_aware_plan
        for cpc in (1, 2, 3):
            planA, diag = structure_aware_plan(graph, K, chunks_per_core=cpc)
            cands.append((f"v2_cpc{cpc}", planA, None))
    except Exception as exc:
        log(f"[{case}] v2 seed fail: {exc}")

    # ---- 种子族 C：波前划分（宽层图 relay 盲区补盲；字节预算门控）----
    # 放在装箱族之前生成（生成毫秒级、不占软时限）。
    wave_tax = {}
    if use_wave:
        try:
            from wave_seed import wave_candidates
            for tag, wplan, meta in wave_candidates(graph, K, log=log):
                cands.append((tag, wplan, None))
                wave_tax[tag] = meta["byte_tax"]
            if wave_tax:
                log(f"[{case}] wave seeds: {sorted(wave_tax)}")
        except Exception as exc:
            log(f"[{case}] wave seed fail: {exc}")

    # ---- 种子族 B：签名类装箱（多粒度 × 双序模式 × 多种子，时限内早停）----
    soft = t0 + budget_s * 0.6
    big = len(ids) > 25000
    for gpc in ((4, 8) if big else (4, 8, 16)):
        for om in ("dfs", "topo"):
            if time.perf_counter() > soft and cands:
                log(f"[{case}] budget: stop portfolio at g{gpc}/{om}")
                break
            pk = UnitPack(graph, max(1, total // (K * gpc)), order_mode=om)
            if pk.n_units() < K:
                continue
            best_pack, best_mk = None, float("inf")
            n_seed = (3 if big else 8) if pk.n_units() <= 800 else 2
            for seed in range(n_seed):
                co = greedy_pack(pk, K, kappa, seed=seed)
                co, mk = local_improve(
                    pk, co, K, kappa, iters=min(ls_iters, 3 * pk.n_units()),
                    seed=seed,
                    time_budget=min(6.0, max(1.0, (soft - time.perf_counter())
                                             / max(1, n_seed)))
                    if pk.n_units() > 500 else None)
                if mk < best_mk:
                    best_pack, best_mk = co, mk
            plan, nsg = runs_merge_plan(pk, best_pack, K)
            cands.append((f"pack_g{gpc}_{om}", plan, best_mk))
            log(f"[{case}] gpc={gpc}/{om}: units={pk.n_units()} "
                f"fast={best_mk:.0f} tasks={nsg}")

    # ---- fast 代理打分（复用 task_sim via derive）；波前候选加字节税 ----
    from stub_multicore_cut_and_schedule import derive_multicore_plan
    scored = []
    for tag, plan, fmk in cands:
        try:
            view = derive_multicore_plan(graph, plan)
            w_by_sg = defaultdict(int)
            for v, sg in view["mapping"].items():
                w_by_sg[sg] += w[v]
            durations = {sg: kappa * wt for sg, wt in w_by_sg.items()}
            mk = task_sim(durations, view["core_by_subgraph"],
                          view["core_orders"],
                          {sg: set(ps) for sg, ps in
                           view["subgraph_preds"].items()})
            mk += wave_tax.get(tag, 0.0)   # 波前：边界字节的 DDR 流量税
            scored.append((tag, plan, mk))
        except Exception as exc:
            log(f"[{case}] {tag} derive fail: {str(exc)[:60]}")
    scored.sort(key=lambda x: x[2])
    log(f"[{case}] fast rank: " + ", ".join(
        f"{t}={m:.0f}" for t, _, m in scored[:6]))

    # ---- 官方预算代理 top 候选（时限内早停）----
    # 波前候选保位：fast 代理（κ·Σw）系统性低估装箱族任务的真实 step3
    # 时长（含 M/V 管道串行），把波前压到 top6 之外（case_044 实测：fast
    # 排序波前 74k > pack 42.6k，official 真值反而 wave 52.4k < pack 60k）。
    # → official 名单 = fast top5 + 波前 top2（去重），最终仍由真评估裁决。
    picks = scored[:5]
    picked = {t for t, _, _ in picks}
    for t, p, m in scored:
        if t in wave_tax and t not in picked and len(picks) < 7:
            picks.append((t, p, m))
            picked.add(t)
            if sum(1 for x in picks if x[0] in wave_tax) >= 2:
                break
    ok = []
    for tag, plan, mk in picks:
        if ok and time.perf_counter() > deadline - 60:
            log(f"[{case}] budget: skip official for {tag}")
            break
        try:
            op = official_proxy(graph, plan, cal_ratio=cal_ratio)
            ok.append((tag, plan, op["makespan"], op["spill_MB"]))
        except Exception as exc:
            log(f"[{case}] {tag} official fail: {str(exc)[:60]}")
    if not ok and scored:  # 官方全超时 → 直接用 fast 排序第一
        tag, plan, mk = scored[0]
        ok.append((tag, plan, mk, 0.0))
    ok.sort(key=lambda x: x[2])

    # ---- 真评估 top（至少 1 个；时限内早停）----
    rows = []
    for tag, plan, omk, sp in ok[:real_top]:
        if rows and time.perf_counter() > deadline:
            log(f"[{case}] budget: stop real eval at {tag}")
            break
        r, wt = ev_p1(graph, plan)
        rows.append({"tag": tag, "plan": plan, "real_mk": r["makespan"],
                     "official_mk": omk, "spill_MB": sp,
                     "added_MB": r["data_movement_bytes"]["added_copy_bytes"] / 1e6})
        log(f"[{case}] {tag}: official={omk:.0f} real={r['makespan']} "
            f"speedup={sc/r['makespan']:.3f} spill={sp:.1f}MB")
    best = min(rows, key=lambda x: x["real_mk"])
    return {"case": case, "best": best, "rows": rows,
            "solve_s": round(time.perf_counter() - t0, 1)}


if __name__ == "__main__":
    for case, sc in [("case_001", 233110), ("case_050", 149690),
                     ("case_014", 17698626), ("case_064", 19116),
                     ("case_005", 95618)]:
        g = load_case(case)
        out = solve_p1(g, case, sc)
        print(case, "->", out["best"]["tag"],
              round(sc / out["best"]["real_mk"], 3),
              f"({out['solve_s']}s)")
