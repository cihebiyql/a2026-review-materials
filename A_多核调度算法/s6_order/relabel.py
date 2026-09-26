# -*- coding: utf-8 -*-
"""s6_order.relabel — 排序（核内子图重标）。

在 op->核 分配不变的前提下，重构子图划分与核内顺序：
  - relabel_bl：B×L 网格（batch 容量 B × 层窗口 L），V2 主算子。
    机制：子图标签即核内调度器分桶键；batch 化解锁 M/V 双管并发、
    L 层窗控制张量驻留宽度。
  - relabel_v4：三结构升级（V3）：均衡装箱 balance / 跨核相位 phase /
    层切分 DP（逐层自适应切点，代理代价 = 管道失衡 + 切点惩罚）。
  - relabel_perm：链序可学习版（V4 训练时按采样序重标）。
"""
from collections import defaultdict


def _batches_min_id(comps, comp_list, B):
    order = sorted(comp_list, key=lambda k: min(comps[k]))
    batch, size, out = [], 0, []
    for k in order:
        sz = len(comps[k])
        if batch and size + sz > B:
            out.append(batch)
            batch, size = [], 0
        batch.append(k)
        size += sz
    if batch:
        out.append(batch)
    return out


def _batches_balance(comps, comp_list, B, pipe_of):
    """均衡装箱：每批内 M/V 功互补（重的 M 链配重的 V 链）。"""
    mload = {k: sum(1 for o in comps[k] if pipe_of.get(o, 0))
             for k in comp_list}
    order = sorted(comp_list, key=lambda k: (-mload[k], min(comps[k])))
    batch, size, out = [], 0, []
    remaining = set(comp_list)
    while remaining:
        if not batch:
            k = order[0] if order[0] in remaining else next(iter(remaining))
        else:
            cur_m = sum(mload[k2] for k2 in batch)
            cur_v = sum(len(comps[k2]) - mload[k2] for k2 in batch)
            want_m = cur_v > cur_m
            best_k, best_key = None, None
            for k2 in remaining:
                if (mload[k2] > 0) == want_m or len(batch) == 0:
                    key = (-len(comps[k2]), min(comps[k2]))
                    if best_key is None or key < best_key:
                        best_key, best_k = key, k2
            if best_k is None:
                best_k = next(iter(sorted(remaining,
                                          key=lambda k: min(comps[k]))))
            k = best_k
        batch.append(k)
        size += len(comps[k])
        remaining.discard(k)
        if size >= B and remaining:
            out.append(batch)
            batch, size = [], 0
    if batch:
        out.append(batch)
    return out


def _split_levels(bops, depth, L, dp, pipe_of):
    maxd = max(depth[o] for o in bops)
    if not dp or L is not None:
        out, lev = [], 0
        while lev <= maxd:
            lops = [o for o in bops if lev <= depth[o] < lev + L]
            if lops:
                out.append(lops)
            lev += L
        return out
    # 层切分 DP：段长上限 8，代价 = 管道失衡×段长 + 切点惩罚 λ
    lev_ops = defaultdict(list)
    for o in bops:
        lev_ops[depth[o]].append(o)
    levels = [lev_ops[d] for d in sorted(lev_ops)]
    n = len(levels)
    W = [len(lv) for lv in levels]
    M = [sum(pipe_of.get(o, 0) for o in lv) for lv in levels]
    INF = float('inf')
    lam = 0.35
    f = [INF] * (n + 1)
    back = [0] * (n + 1)
    f[0] = 0.0
    for j in range(1, n + 1):
        for i in range(max(0, j - 8), j):
            m = sum(M[i:j])
            w = sum(W[i:j])
            imb = abs(m - (w - m)) / max(1, w)
            cost = imb * (j - i) + lam
            if f[i] + cost < f[j]:
                f[j] = f[i] + cost
                back[j] = i
    segs, j = [], n
    while j > 0:
        i = back[j]
        segs.append((i, j))
        j = i
    segs.reverse()
    return [[o for lv in levels[i:j] for o in lv] for i, j in segs]


def _core_comps(op_core, comp_of):
    core_ops = defaultdict(list)
    for op, c in op_core.items():
        core_ops[c].append(op)
    per_core = {}
    for c, ops in core_ops.items():
        comps = defaultdict(list)
        for o in ops:
            comps[comp_of[o]].append(o)
        per_core[c] = comps
    return per_core


def _emit(n_cores, op_core, comp_of, batches_of, depth, L, dp, pipe_of):
    core_ops = defaultdict(list)
    for op, c in op_core.items():
        core_ops[c].append(op)
    new_n2s, new_cs = {}, [[] for _ in range(n_cores)]
    sgid = 0
    for c in range(n_cores):
        ops = core_ops.get(c, [])
        comps = defaultdict(list)
        for o in ops:
            comps[comp_of[o]].append(o)
        for batch in batches_of.get(c, []):
            bset = set(batch)
            bops = [o for o in ops if comp_of[o] in bset]
            for lops in _split_levels(bops, depth, L, dp, pipe_of):
                sgid += 1
                for op in lops:
                    new_n2s[str(op)] = sgid
                new_cs[c].append(sgid)
    return {'node_to_subgraph': new_n2s, 'core_schedules': new_cs}


def relabel_bl(n_cores, op_core, comp_of, depth, B, L):
    """V2：B×L 网格重标（链序 = min_id）。"""
    per_core = _core_comps(op_core, comp_of)
    batches_of = {c: _batches_min_id(comps, list(comps), B)
                  for c, comps in per_core.items()}
    return _emit(n_cores, op_core, comp_of, batches_of, depth, L, False, {})


def relabel_v4(n_cores, op_core, comp_of, depth, B, L, pipe_of,
               balance=False, phase=False, dp=False):
    """V3：三结构（均衡装箱/跨核相位/层切分DP）。L=None 且 dp=True 走 DP。"""
    per_core = _core_comps(op_core, comp_of)
    batches_of = {}
    for c, comps in per_core.items():
        batches = (_batches_balance(comps, list(comps), B, pipe_of)
                   if balance else _batches_min_id(comps, list(comps), B))
        if phase and c > 0 and batches:
            rot = c % max(1, len(batches))
            batches = batches[rot:] + batches[:rot]
        batches_of[c] = batches
    return _emit(n_cores, op_core, comp_of, batches_of, depth, L, dp, pipe_of)


def relabel_perm(n_cores, op_core, comp_of, depth, chain_pos, B, L):
    """V4 训练用：链序 = 采样序 chain_pos{链id: 位次}（其余同 relabel_bl）。"""
    core_ops = defaultdict(list)
    for op, c in op_core.items():
        core_ops[c].append(op)
    new_n2s, new_cs = {}, [[] for _ in range(n_cores)]
    sgid = 0
    for c in range(n_cores):
        ops = core_ops.get(c, [])
        comps = defaultdict(list)
        for o in ops:
            comps[comp_of[o]].append(o)
        comp_list = sorted(comps, key=lambda k: chain_pos[k])
        batch, size, batches = [], 0, []
        for k in comp_list:
            sz = len(comps[k])
            if batch and size + sz > B:
                batches.append(batch)
                batch, size = [], 0
            batch.append(k)
            size += sz
        if batch:
            batches.append(batch)
        for batch in batches:
            bset = set(batch)
            bops = [o for o in ops if comp_of[o] in bset]
            maxd = max(depth[o] for o in bops)
            lev = 0
            while lev <= maxd:
                lops = [o for o in bops if lev <= depth[o] < lev + L]
                if lops:
                    sgid += 1
                    for op in lops:
                        new_n2s[str(op)] = sgid
                    new_cs[c].append(sgid)
                lev += L
    return {'node_to_subgraph': new_n2s, 'core_schedules': new_cs}


def relabel_perm_seg(n_cores, op_core, comp_of, depth, B, L):
    """V4 段模式合法重标（辫状图）：批 = 全局排名相邻且同核的段连续串
    （超 B 断批），批内按 L 层窗切。段排名区间不含异核段 ⇒ 商图无环。"""
    pieces = defaultdict(list)
    for o, u in comp_of.items():
        pieces[(u, op_core[o])].append(o)
    seq = sorted(pieces, key=lambda uc: (uc[0], uc[1]))
    batches = defaultdict(list)
    cur_core, cur_batch, cur_size = None, [], 0
    for (u, c) in seq:
        sz = len(pieces[(u, c)])
        if c != cur_core or (cur_batch and cur_size + sz > B):
            if cur_batch:
                batches[cur_core].append(cur_batch)
            cur_batch, cur_size, cur_core = [], 0, c
        cur_batch.append((u, c))
        cur_size += sz
    if cur_batch:
        batches[cur_core].append(cur_batch)

    new_n2s, new_cs = {}, [[] for _ in range(n_cores)]
    sgid = 0
    for c in range(n_cores):
        for batch in batches.get(c, []):
            bops = [o for (u, cc) in batch for o in pieces[(u, cc)]]
            maxd = max(depth[o] for o in bops)
            lev = 0
            while lev <= maxd:
                lops = [o for o in bops if lev <= depth[o] < lev + L]
                if lops:
                    sgid += 1
                    for op in lops:
                        new_n2s[str(op)] = sgid
                    new_cs[c].append(sgid)
                lev += L
    return {'node_to_subgraph': new_n2s, 'core_schedules': new_cs}
