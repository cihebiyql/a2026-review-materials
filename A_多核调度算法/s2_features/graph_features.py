# -*- coding: utf-8 -*-
"""s2_features.graph_features — DAG 特征分析。

产出算法与实验证据共用的结构特征：
  - 链分解 comp_of（张量为边的弱连通分量；重标/指派层的结构单元）
  - 深度 depth（压缩图上的最长路层数，B×L 重标的 level 依据）
  - 图统计摘要（供 s9 证据/失败分析使用）
depth 用 max 松弛保证为真·最长路层级，
任意边 u→w 恒有 depth[u] < depth[w]——B×L 层窗切分的合法性前提）。
"""
from collections import defaultdict, deque


def comp_depth(ids, preds, succs):
    """弱连通分量分解 + 最长路深度。返回 (comp_of, depth)。"""
    adj = defaultdict(set)
    for v in ids:
        for w in succs[v]:
            adj[v].add(w)
            adj[w].add(v)
    comp_of = {}
    cid = 0
    for v in sorted(ids):
        if v in comp_of:
            continue
        stack = [v]
        comp_of[v] = cid
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in comp_of:
                    comp_of[w] = cid
                    stack.append(w)
        cid += 1
    # 最长路层级（max 松弛）：depth[w] = max(depth[前驱]) + 1。
    # 初值 0 仅给无前驱节点；被改进时重入队，收敛后对任意边 u→w
    # 有 depth[u] < depth[w]，即 depth 是严格拓扑势。
    depth = {}
    q = deque()
    for v in ids:
        if not preds[v]:
            depth[v] = 0
            q.append(v)
    while q:
        u = q.popleft()
        for w in succs[u]:
            nd = depth[u] + 1
            if w not in depth or nd > depth[w]:
                depth[w] = nd
                q.append(w)
    return comp_of, depth


def graph_summary(graph, comp_of, depth):
    """图统计：规模/链数/链富集度/深度/双管负载——证据表与失败分析用。"""
    ops = [o for o in graph['ops'] if o.get('op') not in
           ('COPY_IN', 'COPY_OUT')]
    n_chains = len(set(comp_of.values()))
    wm = sum(max(1, o.get('cycles', 1)) for o in ops
             if o.get('pipe') == 'PIPE_M')
    wv = sum(max(1, o.get('cycles', 1)) for o in ops
             if o.get('pipe') != 'PIPE_M')
    return {
        'n_ops': len(ops), 'n_chains': n_chains,
        'chain_rich': n_chains >= 100,
        'max_depth': max(depth.values()) if depth else 0,
        'WM': wm, 'WV': wv,
        'MV_ratio': round(wm / max(1, wv), 3),
    }
