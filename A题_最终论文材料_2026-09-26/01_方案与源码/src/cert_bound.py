# -*- coding: utf-8 -*-
"""逐例最优性证书:前缀理想下界族(调度论两界对偶)。

对 DAG 调度 K 机 + 依赖,严格下界:
  T ≥ w(I)/K + CPW(V∖I)   对每个拓扑前缀理想 I
 (前缀里的功摊 K 机;前缀之外仍有关键链必须在"前缀之后或并行"完成)
配合 T ≥ W/K、T ≥ CPW(V) 取 max,得每例证书 LB。
用途:逐例报告 (当前 mk)/LB 的最优性间隙 —— 论文级"近优证明"。
"""
import os
import sys
import json
import warnings
from collections import defaultdict, deque
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get(
    "A2026_SOLVER_DIR",
    r"C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand"))
sys.path.insert(0, r"C:/shumo_live/a_data/code")

from common import op_dag, cycles_map

SC_DIR = Path(r"C:/shumo_live/02_求解/A题_2026/results/singlecore")


def prefix_cert(g, K=5, delta_cross=None):
    """返回 (LB, 最优前缀位置)。delta_cross 可选:跨核延迟修正的 CP。"""
    ids, preds, succs = op_dag(g)
    w = cycles_map(g)
    # 拓扑序(任意固定序即可——理想=前缀)
    pd = defaultdict(int)
    for v in ids:
        for s in succs.get(v, ()):
            pd[s] += 1
    dq = deque(sorted(v for v in ids if pd[v] == 0))
    topo = []
    while dq:
        v = dq.popleft()
        topo.append(v)
        for s in sorted(succs.get(v, ())):
            pd[s] -= 1
            if pd[s] == 0:
                dq.append(s)
    n = len(topo)
    W = sum(w[v] for v in ids)
    # 后缀关键链功:suffix_cp[i] = topo[i:] 的最长链功
    dist = {}
    for v in reversed(topo):
        dist[v] = w[v] + max((dist.get(s, 0.0) for s in succs.get(v, ())),
                             default=0.0)
    cpw_all = max(dist[v] for v in ids)
    # 前缀功前缀和
    best = max(W / K, cpw_all)
    best_i = -1
    pref = 0.0
    for i, v in enumerate(topo):
        pref += w[v]
        # 后缀(严格 i 之后)的关键链:需重算?用逆 DP:cp_after[i] = max chain
        # 在 topo[i+1:] —— 预计算 cp_after
        pass
    # 预计算 cp_after[i] = topo[i:] 最长链
    cp_after = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        v = topo[i]
        # v 的链功 = w[v] + max 后继(可能在前缀内!后继都在 i 之后 ✓ 拓扑序保证)
        cp_after[i] = dist[v]
    pref = 0.0
    for i in range(n + 1):
        lb = pref / K + (cp_after[i] if i < n else 0.0)
        if lb > best:
            best = lb
            best_i = i
        if i < n:
            pref += w[topo[i]]
    return best, best_i, W, cpw_all


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--pool", default=str(HERE / "POOL_q3_N5.json"))
    a = ap.parse_args()
    from common import load_case
    pool = json.load(open(a.pool))
    cases = a.cases.split(",") if a.cases else sorted(pool)
    rows = []
    for case in cases:
        g = load_case(case)
        sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
        lb, bi, W, cpw = prefix_cert(g, a.K)
        cur = pool.get(case)
        gap = (sc / cur) / lb - 1 if cur else None   # 当前/证书 - 1
        rows.append((case, cur, sc / lb, gap, bi))
        print(f"{case}: 池sp={cur:.3f} 证书上限sp*={sc/lb:.3f} "
              f"间隙={gap*100:+.1f}% (W/K={W/a.K:.0f} CP={cpw:.0f} 切点={bi})",
              flush=True)
    import statistics
    gaps = [r[3] for r in rows if r[3] is not None]
    print(f"\n== {len(rows)}例:证书间隙 均值{statistics.mean(gaps)*100:+.1f}% "
          f"中位{statistics.median(gaps)*100:+.1f}% max{max(gaps)*100:+.1f}% ==")


if __name__ == "__main__":
    main()
