"""v2 求解器核心：结构感知切分（独立块分组）。

依据（已由两轮独立核验证实）：
- 场景A Makespan ≈ 沿子图依赖链的块时长之和（接力效应，N=3 case_001
  0.990 vs N=4 3.952）→ 第一目标：块间依赖深度最小（理想 0 = 独立块）；
- 大块 spill（case_014 单核 71MB）→ 块大小需驻留峰值上限约束；
- 负载均衡决定并行上限 → LPT 分组。

管线：
1. DFS 路径跟随拓扑序（整链聚拢 → 切边少、净切点多）；
2. 净切点（prefix 无出边跨界）处可"免费"切块；
3. DP 连续划分：代价 = 切边数 + 均衡罚，约束 = 每块驻留峰值上限
   （廉价活张量估计，最终方案再用官方 step2 精查）；
4. LPT 把块组分到 N 核（同核块序沿拓扑 id 递增天然合法）。

本文件不 import 前一 agent 的 solver/baseline。
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

ATT = Path(os.environ.get(
    "A2026_ATT",
    r"D:/work/数模竞赛/中文题目_外层/中文题目/A题/"
    r"通用神经网络处理器下的多核调度问题  附件_解压"))
CODE = ATT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from stub_multicore_cut_and_schedule import (  # noqa: E402
    _build_op_adjacency, _contract_excluded_copy_nodes, EXCLUDED_COPY_TYPES)

L1_CAP = 524288
UB_CAP = 131072


# ---------------- 基础视图 ----------------

def eligible_ids(graph):
    return sorted(o["id"] for o in graph["ops"]
                  if o.get("op") not in EXCLUDED_COPY_TYPES)


def op_dag(graph):
    preds, succs = _build_op_adjacency(graph)
    ids = eligible_ids(graph)
    cp, cs = _contract_excluded_copy_nodes(ids, succs)
    return ids, cp, cs


def cycles_map(graph):
    return {o["id"]: max(1, o.get("cycles", 1)) for o in graph["ops"]
            if o.get("op") not in EXCLUDED_COPY_TYPES}


def tensor_views(graph):
    """op->生产张量 / op->消费张量 / 张量大小。"""
    op_ids = {o["id"] for o in graph["ops"]}
    prod, cons = defaultdict(set), defaultdict(set)
    tsize = {t["id"]: t["size"] for t in graph["tensors"]}
    tpos = {t["id"]: t.get("pos", "UB") for t in graph["tensors"]}
    for e in graph["edges"]:
        s, t = e["source"], e["target"]
        if s in op_ids and t not in op_ids:
            prod[s].add(t)
        elif t in op_ids and s not in op_ids:
            cons[t].add(s)
    return prod, cons, tsize, tpos


# ---------------- 序：DFS 优先级 Kahn 拓扑 ----------------

def dfs_priority_topo(ids, preds, succs, w=None):
    """DFS 优先级的 Kahn 拓扑序。

    先以根出发的 DFS preorder 作优先级（整链聚拢），再跑 Kahn 取
    依赖就绪的最小优先级节点——既保"链连续"又保证输出是严格拓扑序
    （子图 id 沿序递增 → 子图依赖图无环、同核序天然合法）。
    """
    idset = set(ids)
    indeg = {v: sum(p in idset for p in preds[v]) for v in ids}
    rank = {}
    visited = set()
    stack = [v for v in ids if indeg[v] == 0]
    r = 0
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        rank[u] = r
        r += 1
        for s in sorted(succs.get(u, ()), reverse=True):
            if s not in visited:
                stack.append(s)
    nxt = r
    for v in ids:  # 兜底
        if v not in rank:
            rank[v] = nxt
            nxt += 1
    # Kahn with rank priority
    import heapq as _hq
    h = [(rank[v], v) for v in ids if indeg[v] == 0]
    _hq.heapify(h)
    order = []
    while h:
        _, u = _hq.heappop(h)
        order.append(u)
        for s in succs.get(u, ()):
            if s in idset:
                indeg[s] -= 1
                if indeg[s] == 0:
                    _hq.heappush(h, (rank[s], s))
    assert len(order) == len(ids)
    return order


def crossing_at(order, preds, succs, idset):
    """cross[i] = 前 i+1 个点已调度时，跨界（已调度→未调度）边数。O(n+E)。"""
    cross = 0
    out = []
    done = set()
    for v in order:
        done.add(v)
        for s in succs.get(v, ()):
            if s in idset and s not in done:
                cross += 1
        for p in preds.get(v, ()):
            if p in idset and p in done:
                cross -= 1
        out.append(cross)
    return out


# ---------------- 廉价驻留峰值估计 ----------------

def live_peak_estimate(order, prod, cons, tsize, tpos):
    """沿给定序的活张量驻留峰值（UB+L1 合计，DDR 不计）。

    张量生于首个生产者，亡于最后消费者。step2 的精确峰值与其高度相关，
    但更保守（分配策略），故只作内层判据；最终方案用官方 step2 复核。
    """
    last_use = {}
    for i, v in enumerate(order):
        for t in prod.get(v, ()) | cons.get(v, ()):
            last_use[t] = i
    live = 0
    peak = 0
    for i, v in enumerate(order):
        for t in prod.get(v, ()):
            if tpos.get(t) != "DDR":
                live += tsize.get(t, 0)
        for t in cons.get(v, ()):
            if last_use.get(t) == i and tpos.get(t) != "DDR":
                live -= tsize.get(t, 0)
        peak = max(peak, live)
    return peak


# ---------------- 切分主算法 ----------------

def structure_aware_plan(graph, num_cores, chunks_per_core=1,
                         spill_headroom=0.85, snap_window_frac=0.15,
                         balance_weight=0.5):
    """结构感知切分。返回 (plan, diag)。

    - chunks_per_core: 每核目标块数（>1 可减切边税、增调度自由度）
    - spill_headroom: 驻留峰值上限系数 ×(L1+UB)
    - snap_window_frac: 切点吸附窗口（占目标段长的比例）
    """
    ids, preds, succs = op_dag(graph)
    w = cycles_map(graph)
    order = dfs_priority_topo(ids, preds, succs)
    idset = set(ids)
    prod, cons, tsize, tpos = tensor_views(graph)

    n = len(order)
    cross = crossing_at(order, preds, succs, idset)
    K = num_cores * chunks_per_core
    mem_cap = spill_headroom * (L1_CAP + UB_CAP)

    # 前缀周期
    prefix = [0]
    for v in order:
        prefix.append(prefix[-1] + w[v])
    total = prefix[-1]
    target = total / K

    # 贪心切分：到目标重量后在吸附窗口内选切点——先硬均衡窗
    # （段重 ∈ [0.5,1.5]·target），窗内取跨界边最少、平手取最接近目标重
    # 候选切点 = 跨界边数局部极小（含净切点 0）；在其上做 min-max 均衡 DP
    def local_min_candidates():
        cands = []
        for i in range(1, n):
            c = cross[i - 1]
            left = cross[i - 2] if i >= 2 else float("inf")
            right = cross[i] if i < n else float("inf")
            if c <= left and c <= right:
                cands.append(i)
        return cands

    cands = local_min_candidates()
    # 候选太少 → 放宽为跨界边最低 40% 分位的位置；太多 → 等距抽样限 m≤600
    if len(cands) < 2 * K:
        thr = sorted(cross)[: max(1, int(n * 0.4))][-1] if cross else 0
        cands = [i for i in range(1, n) if cross[i - 1] <= thr]
    if len(cands) > 600:
        step = len(cands) / 600
        cands = [cands[int(t * step)] for t in range(600)]
    cands = sorted(set(cands) | {0, n})

    # DP：从候选切点选 K-1 个，最小化最大段重（均衡），平手最小化 Σ切边
    m = len(cands)
    pos_of = {c: j for j, c in enumerate(cands)}
    INF = (float("inf"), float("inf"))
    dp = [[INF] * m for _ in range(K + 1)]
    dp[0][0] = (0.0, 0.0)
    back = [[-1] * m for _ in range(K + 1)]
    for k in range(1, K + 1):
        for j in range(1, m):
            best = INF
            arg = -1
            for t in range(j):
                if dp[k - 1][t] == INF:
                    continue
                a, b = cands[t], cands[j]
                seg_w = prefix[b] - prefix[a]
                prev_max, prev_cut = dp[k - 1][t]
                cur = (max(prev_max, seg_w),
                       prev_cut + cross[b - 1])
                if cur < best:
                    best, arg = cur, t
            dp[k][j] = best
            back[k][j] = arg
    # 终态：第 K 段必须结束在 n
    jn = pos_of[n]
    if dp[K][jn] == INF:
        # 兜底：均匀切
        cuts = [round(n * x / K) for x in range(1, K)] + [n]
    else:
        cuts = [n]
        j = jn
        for k in range(K, 0, -1):
            j = back[k][j]
            cuts.append(cands[j])
        cuts.append(0)
        cuts = sorted(set(c for c in cuts if 0 <= c <= n))
    segs = [(cuts[t], cuts[t + 1]) for t in range(len(cuts) - 1)
            if cuts[t] < cuts[t + 1]]

    # 段→子图；驻留峰值超限的段再对半细切（防大块 spill）
    def refine(a, b, depth=0):
        seg_order = order[a:b]
        if depth >= 6 or b - a <= 8:
            return [(a, b)]
        if live_peak_estimate(seg_order, prod, cons, tsize, tpos) <= mem_cap:
            return [(a, b)]
        mid = (a + b) // 2
        return refine(a, mid, depth + 1) + refine(mid, b, depth + 1)

    final_segs = []
    for a, b in segs:
        final_segs.extend(refine(a, b))

    # 子图 id 沿序递增
    node2sg = {}
    for sg, (a, b) in enumerate(final_segs):
        for v in order[a:b]:
            node2sg[v] = sg

    # LPT：块按周期降序装到最闲核
    seg_w = [(sg, prefix[b] - prefix[a]) for sg, (a, b) in
             enumerate(final_segs)]
    seg_w.sort(key=lambda x: -x[1])
    import heapq as _hq
    heap = [(0.0, c) for c in range(num_cores)]
    _hq.heapify(heap)
    core_of = {}
    core_load = [0.0] * num_cores
    for sg, weight in seg_w:
        load, c = _hq.heappop(heap)
        core_of[sg] = c
        core_load[c] = load + weight
        _hq.heappush(heap, (load + weight, c))
    cores = [[] for _ in range(num_cores)]
    for sg in range(len(final_segs)):  # id 递增入核 → 同核依赖序天然合法
        cores[core_of[sg]].append(sg)

    # 诊断
    dep_pairs = set()
    posmap = node2sg
    for v in order:
        for s in succs.get(v, ()):
            if s in idset and posmap[s] != posmap[v]:
                dep_pairs.add((posmap[v], posmap[s]))

    def dep_depth():
        succ_sg = defaultdict(set)
        for a, b in dep_pairs:
            succ_sg[a].add(b)
        memo = {}

        def d(x):
            if x in memo:
                return memo[x]
            memo[x] = 1
            r = 1 + max((d(y) for y in succ_sg[x]), default=0)
            memo[x] = r
            return r

        return max((d(x) for x in range(len(final_segs))), default=0)

    diag = {"n_segments": len(final_segs), "cut_edges": len(dep_pairs),
            "dep_depth": dep_depth(),
            "per_core_weight": [round(x, 1) for x in core_load]}
    return {"node_to_subgraph": node2sg, "core_schedules": cores}, diag
