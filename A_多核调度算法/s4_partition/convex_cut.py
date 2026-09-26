# -*- coding: utf-8 -*-
"""s4_partition.convex_cut — 切图（V1 基线构造器）。

沿链跟随拓扑序切 K = N × chunks_per_core 个连续段（凸分区），
按 op 周期做贪心负载均衡（total 模式：总周期；dual：M/V 双管分别均衡，
V1 固定 total，dual 留作消融）。连续段 id 递增 ⇒ 同核序天然合法。
分核策略（s5 轮询）内联于此构造器，s5_assign 提供指派层接口。
"""
from s3_units.topo_units import chain_topo_order


def _weights(cycles, pipes):
    wm = {v: (c if pipes.get(v) == 'PIPE_M' else 0) for v, c in cycles.items()}
    wv = {v: (c - wm[v]) for v, c in cycles.items()}
    return wm, wv


def convex_plan(ids, preds, succs, cycles, pipes, num_cores,
                chunks_per_core=1, balance='total'):
    """V1：拓扑序 + 连续凸块 + 周期均衡 + 轮询分核 → 合法方案。"""
    topo = chain_topo_order(ids, preds, succs)
    K = num_cores * chunks_per_core
    if balance == 'total':
        w1 = w2 = cycles
    else:
        w1, w2 = _weights(cycles, pipes)
    t1 = sum(w1[v] for v in topo) / K
    t2 = sum(w2[v] for v in topo) / K
    n2s = {}
    c1 = c2 = 0.0
    sg = 0
    for i, v in enumerate(topo):
        n2s[v] = sg
        c1 += w1[v]
        c2 += w2[v]
        remain = len(topo) - i - 1
        if sg < K - 1 and remain > 0 and (c1 >= t1 and c2 >= t2):
            sg += 1
            c1 = c2 = 0.0
    core_schedules = [[] for _ in range(num_cores)]
    for s in range(sg + 1):
        core_schedules[s % num_cores].append(s)
    return {'node_to_subgraph': {str(k): v for k, v in n2s.items()},
            'core_schedules': core_schedules}
