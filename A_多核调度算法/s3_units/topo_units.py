# -*- coding: utf-8 -*-
"""s3_units.topo_units — 合法运算顺序与结构单元。

两类输出：
  1) chain_topo_order：链跟随 Kahn 拓扑序（V1 凸块切图的序基础，
     依赖双方尽量相邻 → 少切边；与官方 step1 的确定性序语义兼容）。
  2) build_units：指派/重标用的结构单元 = 链(弱连通分量)；
     辫状图(链过少)退化为按 (depth, id) 拓扑序切的均衡段
     （段在拓扑序上连续 ⇒ 凸性安全）。
"""
from collections import defaultdict
import heapq


def chain_topo_order(ids, preds, succs, follow=True):
    """Kahn + 链跟随。返回覆盖全部 ids 的合法拓扑序。"""
    idset = set(ids)
    indeg = {v: sum(p in idset for p in preds[v]) for v in ids}
    heap = [v for v in ids if indeg[v] == 0]
    heapq.heapify(heap)
    inheap = set(heap)
    order = []
    prefer = None
    while heap:
        if follow and prefer is not None and prefer in inheap:
            u = prefer
            heap.remove(u)
            heapq.heapify(heap)
            inheap.discard(u)
        else:
            u = heapq.heappop(heap)
            inheap.discard(u)
        order.append(u)
        prefer = None
        for w in sorted(succs.get(u, ())):
            if w not in idset:
                continue
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(heap, w)
                inheap.add(w)
                if prefer is None:
                    prefer = w
    assert len(order) == len(ids)
    return order


def build_units(comp_of, depth, min_units=96, max_units=192):
    """链数不足时切拓扑段作为单元（返回 unit_of）。链足够则原样返回。"""
    n_chains = len(set(comp_of.values()))
    if n_chains >= min_units:
        return comp_of
    ops = sorted(comp_of, key=lambda o: (depth[o], o))
    n_seg = min(max_units, max(min_units, len(ops) // 150 or 1))
    size = max(1, -(-len(ops) // n_seg))
    return {o: i // size for i, o in enumerate(ops)}


def op_core_of(plan):
    """从方案提取 op->核 映射（重标器不改变它，V2/V3 沿用）。"""
    n2s = {int(k): v for k, v in plan['node_to_subgraph'].items()}
    core_of_sg = {sg: c for c, sgl in enumerate(plan['core_schedules'])
                  for sg in sgl}
    return {op: core_of_sg[sg] for op, sg in n2s.items()}
