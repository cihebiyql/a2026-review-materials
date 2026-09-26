"""v2 求解流水线：候选组合 → 快速代理局部搜索 → 官方代理 → 真评估器。

场景A（问题1）。核心数据结构：拓扑序 + 切点向量 cuts（有序段边界）+
核指派。局部搜索在 (cuts, assignment) 空间，目标 = κ 校准快速代理
makespan（含接力/等待建模）。最终前 M 名做官方预算代理（精确
local_makespan + spill 字节），前 2-3 名真评估；全部真评估样本用于
Spearman(代理, 真值) 检验（阈值 0.7）。
"""
import json
import os
import random
import sys
import time
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from structure_split import (  # noqa: E402
    structure_aware_plan, op_dag, cycles_map, dfs_priority_topo,
    crossing_at, live_peak_estimate, tensor_views, L1_CAP, UB_CAP)
from proxy import task_sim, fast_proxy, official_proxy, SAME_WAIT, CROSS_WAIT  # noqa: E402

ATT = Path(os.environ.get(
    "A2026_ATT",
    r"D:/work/数模竞赛/中文题目_外层/中文题目/A题/"
    r"通用神经网络处理器下的多核调度问题  附件_解压"))


# ---------------- 轻量评估器：直接在 (order, cuts, assign) 上算 ----------------

class CutEvaluator:
    """切点向量 + 核指派的快速评估（不重复 derive_multicore_plan）。"""

    def __init__(self, graph, num_cores):
        ids, preds, succs = op_dag(graph)
        self.w = cycles_map(graph)
        self.order = dfs_priority_topo(ids, preds, succs)
        self.pos = {v: i for i, v in enumerate(self.order)}
        self.n = len(self.order)
        self.cores = num_cores
        # 边表（按位置）
        self.edges = []
        idset = set(ids)
        for u in self.order:
            for s in succs.get(u, ()):
                if s in idset:
                    self.edges.append((self.pos[u], self.pos[s]))
        self.prefix = [0]
        for v in self.order:
            self.prefix.append(self.prefix[-1] + self.w[v])
        self.total = self.prefix[-1]
        self.cross = crossing_at(self.order, preds, succs, idset)
        # spill 预估
        self.prod, self.cons, self.tsize, self.tpos = tensor_views(graph)

    def seg_weight(self, cuts, j):
        return self.prefix[cuts[j + 1]] - self.prefix[cuts[j]]

    def dep_pairs(self, cuts):
        sg_of = [bisect_right(cuts, p) - 1 for p in range(self.n)]
        pairs = set()
        for a, b in self.edges:
            if a < b:  # 拓扑序保证 a<b
                x, y = sg_of[a], sg_of[b]
                if x != y:
                    pairs.add((x, y))
        return pairs, sg_of

    def evaluate(self, cuts, core_of=None, kappa=1.0):
        """返回 (makespan, diag)。cuts: [0, ..., n]；段 j ∈ [cuts[j], cuts[j+1])。"""
        K = len(cuts) - 1
        if core_of is None:  # LPT
            import heapq
            heap = [(0.0, c) for c in range(self.cores)]
            heapq.heapify(heap)
            core_of = {}
            loads = [0.0] * self.cores
            for j in sorted(range(K), key=lambda x: -self.seg_weight(cuts, x)):
                load, c = heapq.heappop(heap)
                core_of[j] = c
                loads[c] = load + self.seg_weight(cuts, j)
                heapq.heappush(heap, (loads[c], c))
        pairs, _ = self.dep_pairs(cuts)
        preds = {j: set() for j in range(K)}
        for a, b in pairs:
            preds[b].add(a)
        durations = {j: kappa * self.seg_weight(cuts, j) for j in range(K)}
        core_order = {c: [j for j in range(K) if core_of[j] == c]
                      for c in range(self.cores)}
        core_of_map = {j: core_of[j] for j in range(K)}
        mk = task_sim(durations, core_of_map, core_order, preds)
        return mk, {"K": K, "dep_edges": len(pairs),
                   "max_seg_w": max(self.seg_weight(cuts, j) for j in range(K))}

    def to_plan(self, cuts, core_of=None):
        K = len(cuts) - 1
        if core_of is None:  # LPT 默认
            import heapq
            heap = [(0.0, c) for c in range(self.cores)]
            heapq.heapify(heap)
            core_of = {}
            for j in sorted(range(K), key=lambda x: -self.seg_weight(cuts, x)):
                load, c = heapq.heappop(heap)
                core_of[j] = c
                heapq.heappush(heap, (load + self.seg_weight(cuts, j), c))
        node2sg = {}
        for j in range(K):
            for p in range(cuts[j], cuts[j + 1]):
                node2sg[self.order[p]] = j
        # 段 id 沿序递增；同核序 = id 序 → 合法
        core_order = {c: [] for c in range(self.cores)}
        for j in range(K):
            core_order[core_of[j]].append(j)
        return {"node_to_subgraph": node2sg,
                "core_schedules": [core_order[c] for c in range(self.cores)]}

    def spill_refine(self, cuts, mem_cap=0.85 * (L1_CAP + UB_CAP)):
        """把驻留峰值估计超限的段对半细切（防大块 spill）。"""
        out = [cuts[0]]
        for j in range(len(cuts) - 1):
            a, b = cuts[j], cuts[j + 1]
            stack = [(a, b)]
            segs = []
            while stack:
                lo, hi = stack.pop()
                mid = (lo + hi) // 2
                if (hi - lo) <= 8 or mid <= lo or live_peak_estimate(
                        self.order[lo:hi], self.prod, self.cons,
                        self.tsize, self.tpos) <= mem_cap or hi - lo <= 16:
                    segs.append((lo, hi))
                else:
                    stack.append((lo, mid))
                    stack.append((mid, hi))
            segs.sort()
            for lo, hi in segs:
                out.append(hi)
        out = sorted(set(out))
        return out


# ---------------- 候选组合 ----------------

def portfolio(ev, num_cores, kappa):
    """生成种子切点向量集合。"""
    seeds = []
    for cpc in (1, 2, 3):
        plan, diag = structure_aware_plan(
            _graph_of(ev), num_cores, chunks_per_core=cpc)
        cuts = _cuts_from_plan(ev, plan)
        mk, d = ev.evaluate(cuts, kappa=kappa)
        seeds.append({"cuts": cuts, "fast_mk": mk, "tag": f"v2_cpc{cpc}",
                      "diag": d})
    # 均匀切（朴素对照种子）
    for K in (num_cores, 2 * num_cores):
        cuts = [round(ev.n * j / K) for j in range(K + 1)]
        cuts[-1] = ev.n
        mk, d = ev.evaluate(cuts, kappa=kappa)
        seeds.append({"cuts": sorted(set(cuts)), "fast_mk": mk,
                      "tag": f"uniform_K{K}", "diag": d})
    return seeds


_GRAPH_HOLDER = {}


def _graph_of(ev):
    return _GRAPH_HOLDER[id(ev)]


def _cuts_from_plan(ev, plan):
    sg_of_pos = []
    for v in ev.order:
        sg_of_pos.append(plan["node_to_subgraph"][v])
    cuts = [0]
    for p in range(1, ev.n):
        if sg_of_pos[p] != sg_of_pos[p - 1]:
            cuts.append(p)
    cuts.append(ev.n)
    return cuts


# ---------------- 局部搜索 ----------------

def local_search(ev, cuts, kappa, iters=120, seed=0, sigma=None):
    """邻域：边界平移 / 段迁移 / 肥段分裂 / 相邻段合并。爬山+轻扰动。"""
    rng = random.Random(seed)
    cur = list(cuts)
    core_of = None  # None → LPT 现算
    cur_mk, _ = ev.evaluate(cur, kappa=kappa)
    best, best_mk = list(cur), cur_mk
    if sigma is None:
        sigma = max(4, ev.n // 64)
    for it in range(iters):
        cand = list(cur)
        move = rng.randrange(4)
        K = len(cand) - 1
        if move == 0 and K >= 2:  # 边界平移
            j = rng.randrange(1, K)
            delta = rng.choice([-1, 1]) * rng.randint(1, sigma)
            new = cand[j] + delta
            if cand[j - 1] + 1 <= new <= cand[j + 1] - 1:
                cand[j] = new
            else:
                continue
        elif move == 1 and K >= 2:  # 肥段分裂（最小跨界处）
            weights = [ev.seg_weight(cand, j) for j in range(K)]
            j = weights.index(max(weights))
            a, b = cand[j], cand[j + 1]
            if b - a < 17:  # v3 修复：窗口 range(a+8,b-8) 在 b-a<=16 为空
                continue
            window = range(a + 8, b - 8)
            mid = min(window, key=lambda p: ev.cross[p - 1])
            cand = cand[:j + 1] + [mid] + cand[j + 1:]
        elif move == 2 and K >= 2:  # 瘦段合并（相邻）
            weights = [ev.seg_weight(cand, j) for j in range(K)]
            j = weights.index(min(weights))
            if K <= 2:
                continue
            m = j if j < K - 1 else j - 1
            cand = cand[:m + 1] + cand[m + 2:]
        else:  # move == 3: 扰动重启附近
            j = rng.randrange(1, len(cand) - 1) if len(cand) > 2 else 0
            if len(cand) <= 2:
                continue
            delta = rng.choice([-1, 1]) * rng.randint(1, 2 * sigma)
            new = cand[j] + delta
            if cand[j - 1] + 1 <= new <= cand[j + 1] - 1:
                cand[j] = new
            else:
                continue
        try:
            mk, _ = ev.evaluate(cand, kappa=kappa)
        except RuntimeError:
            continue
        if mk < cur_mk or rng.random() < 0.05:  # 允许轻随机游走
            cur, cur_mk = cand, mk
            if mk < best_mk:
                best, best_mk = list(cand), mk
    return best, best_mk


# ---------------- 单用例主流程 ----------------

def real_eval_p1(graph, plan):
    """自带真评估（场景A，config 常量与官方 config.txt 一致）。"""
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    t0 = time.time()
    r = evaluate_scene_a(graph, plan, bandwidth=60,
                         capacity={"L1": 524288, "UB": 131072},
                         cross_core_wait=1000, same_core_wait=100)
    return r, time.time() - t0


def load_case(case):
    with open(ATT / "data" / f"{case}.json", encoding="utf-8") as f:
        return json.load(f)


def solve_case(graph, case, num_cores=4, sc_makespan=None,
               cal_ratio=0.89, ls_iters=120, pool_size=6,
               real_eval_top=3, log=print):
    t0 = time.time()
    ev = CutEvaluator(graph, num_cores)
    _GRAPH_HOLDER[id(ev)] = graph
    kappa = (sc_makespan / ev.total) if sc_makespan else 1.0

    seeds = portfolio(ev, num_cores, kappa)
    pool = []
    for s in seeds:
        cuts2 = ev.spill_refine(s["cuts"])
        mk2, _ = ev.evaluate(cuts2, kappa=kappa)
        pool.append({"cuts": cuts2, "fast_mk": mk2, "tag": s["tag"]})
    # 每个种子局部搜索
    for s in sorted(pool, key=lambda x: x["fast_mk"])[:3]:
        ls_cuts, ls_mk = local_search(ev, s["cuts"], kappa, iters=ls_iters)
        ls_cuts = ev.spill_refine(ls_cuts)
        ls_mk2, _ = ev.evaluate(ls_cuts, kappa=kappa)
        pool.append({"cuts": ls_cuts, "fast_mk": ls_mk2,
                     "tag": "ls_" + s["tag"]})
    # 去重 + 排序
    uniq = {}
    for p in pool:
        key = tuple(p["cuts"])
        if key not in uniq or p["fast_mk"] < uniq[key]["fast_mk"]:
            uniq[key] = p
    pool = sorted(uniq.values(), key=lambda x: x["fast_mk"])[:pool_size]
    log(f"[{case}] pool: " + ", ".join(
        f"{p['tag']}={p['fast_mk']:.0f}" for p in pool))

    # 官方预算代理（整图比值校准 cal_ratio）
    for p in pool:
        plan = ev.to_plan(p["cuts"], None)
        try:
            op = official_proxy(graph, plan, cal_ratio=cal_ratio)
            p["official_mk"] = op["makespan"]
            p["spill_MB"] = op["spill_MB"]
            p["per_task_spill_MB"] = op["per_task_spill_MB"]
        except Exception as exc:  # 官方预算失败 → 大罚
            p["official_mk"] = float("inf")
            p["spill_MB"] = 999
            p["official_err"] = str(exc)[:80]
    ok = [p for p in pool if p["official_mk"] < float("inf")]
    ok.sort(key=lambda x: x["official_mk"])

    # spill 驱动再分裂（一轮）：对官方 spill>5MB 的头名候选，把 spill
    # 最大的 ≤2 个段在段内最小跨界边处二分，重建候选（防大块 spill）
    if ok and ok[0].get("spill_MB", 0) > 5:
        p0 = ok[0]
        cuts = list(p0["cuts"])
        spill_by_seg = p0.get("per_task_spill_MB") or {}
        order_by_spill = sorted(range(len(cuts) - 1),
                                key=lambda j: -spill_by_seg.get(j, 0))
        new_cuts = list(cuts)
        added = 0
        for j in order_by_spill:
            if added >= 2 or spill_by_seg.get(j, 0) < 1.0:
                break
            a0, b0 = cuts[j], cuts[j + 1]
            if b0 - a0 < 32:
                continue
            mid = min(range(a0 + 8, b0 - 8), key=lambda q: ev.cross[q - 1])
            new_cuts = sorted(set(new_cuts + [mid]))
            added += 1
        if added:
            cuts_rs = ev.spill_refine(new_cuts)
            mk_rs, _ = ev.evaluate(cuts_rs, kappa=kappa)
            plan_rs = ev.to_plan(cuts_rs, None)
            try:
                op_rs = official_proxy(graph, plan_rs, cal_ratio=cal_ratio)
                cand_rs = {"cuts": cuts_rs, "fast_mk": mk_rs,
                           "tag": "resplit_" + p0["tag"],
                           "official_mk": op_rs["makespan"],
                           "spill_MB": op_rs["spill_MB"],
                           "per_task_spill_MB": op_rs["per_task_spill_MB"]}
                ok.append(cand_rs)
                ok.sort(key=lambda x: x["official_mk"])
                log(f"[{case}] resplit: spill {p0['spill_MB']:.1f}MB -> "
                    f"{op_rs['spill_MB']:.1f}MB, official "
                    f"{p0['official_mk']:.0f} -> {op_rs['makespan']:.0f}")
            except Exception as exc:
                log(f"[{case}] resplit failed: {str(exc)[:60]}")

    # 真评估 top（spill 感知：同量级 makespan 下偏好低 spill）
    rows = []
    for p in ok[:real_eval_top]:
        plan = ev.to_plan(p["cuts"], None)
        r, wt = real_eval_p1(graph, plan)
        p["real_mk"] = r["makespan"]
        p["real_spill_MB"] = r["data_movement_bytes"]["spill_added_copy_bytes"] / 1e6
        p["real_added_MB"] = r["data_movement_bytes"]["added_copy_bytes"] / 1e6
        p["eval_s"] = round(wt, 2)
        rows.append(p)
        log(f"[{case}] {p['tag']}: fast={p['fast_mk']:.0f} "
            f"official={p['official_mk']:.0f} real={r['makespan']} "
            f"spill={p['real_spill_MB']:.1f}MB")
    best = min(rows, key=lambda x: x["real_mk"])
    return {"case": case, "best": best, "rows": rows,
            "solve_seconds": round(time.time() - t0, 1),
            "kappa": kappa, "pool": pool}


if __name__ == "__main__":
    # 冒烟：case_001 / case_050
    for case, sc in [("case_001", 233110), ("case_050", 149690)]:
        g = load_case(case)
        out = solve_case(g, case, sc_makespan=sc)
        print(case, "best real speedup:", round(sc / out["best"]["real_mk"], 3),
              "solve_s:", out["solve_seconds"])
