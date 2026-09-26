# Ops 独立版: 无 numba/fast_eval 依赖(enrich 等纯官方评估链路用)
from collections import defaultdict, deque


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


