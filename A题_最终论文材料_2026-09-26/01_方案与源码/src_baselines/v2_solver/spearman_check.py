"""Spearman 检验：代理排序 vs 真评估器排序。

对 5 个中小用例，取候选池（组合种子 + LS 快照，约 8-10 个/例）全部
真评估，计算 ρ(fast, real) 与 ρ(official, real)。阈值 ≥0.7 判可用。
不做 top-3 筛选，避免选择偏差。
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import (CutEvaluator, solve_case, load_case, real_eval_p1,
                      _GRAPH_HOLDER)  # noqa: E402
from structure_split import structure_aware_plan  # noqa: E402
from proxy import official_proxy  # noqa: E402
from local_search_wrap import ls_snapshot  # noqa: E402


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n * n - 1))


def candidates_for(graph, num_cores=4, sc=None):
    """返回候选 plans 池（不去 top，全部）。"""
    ev = CutEvaluator(graph, num_cores)
    _GRAPH_HOLDER[id(ev)] = graph
    kappa = sc / ev.total if sc else 1.0
    outs = []
    seen = set()
    for cpc in (1, 2, 3):
        plan, _ = structure_aware_plan(graph, num_cores, chunks_per_core=cpc)
        cuts = _cuts(ev, plan)
        outs.append((f"v2cpc{cpc}", cuts))
        for it, snap in enumerate(ls_snapshot(ev, cuts, kappa, iters=100)):
            outs.append((f"v2cpc{cpc}_ls{it}", snap))
    for K in (4, 8):
        cuts = [round(ev.n * j / K) for j in range(K + 1)]
        cuts[-1] = ev.n
        outs.append((f"uni{K}", sorted(set(cuts))))
    res = []
    for tag, cuts in outs:
        key = tuple(cuts)
        if key in seen:
            continue
        seen.add(key)
        res.append((tag, cuts))
    return ev, res


def _cuts(ev, plan):
    sg = [plan["node_to_subgraph"][v] for v in ev.order]
    cuts = [0] + [p for p in range(1, ev.n) if sg[p] != sg[p - 1]] + [ev.n]
    return cuts


def main():
    cases = sys.argv[1:] or ["case_001", "case_003", "case_050",
                             "case_062", "case_095"]
    report = {}
    for case in cases:
        sc_path = HERE.parent / "results" / "singlecore" / f"{case}_sc.json"
        sc = json.load(open(sc_path, encoding="utf-8"))["makespan"]
        g = load_case(case)
        ev, cands = candidates_for(g, 4, sc)
        kappa = sc / ev.total
        rows = []
        for tag, cuts in cands:
            cuts2 = ev.spill_refine(list(cuts))
            plan = ev.to_plan(cuts2, None)
            fast_mk, _ = ev.evaluate(cuts2, kappa=kappa)
            off = official_proxy(g, plan, cal_ratio=0.89)
            r, _ = real_eval_p1(g, plan)
            rows.append({"tag": tag, "fast": fast_mk,
                         "official": off["makespan"], "real": r["makespan"]})
        rho_f = spearman([x["fast"] for x in rows], [x["real"] for x in rows])
        rho_o = spearman([x["official"] for x in rows],
                         [x["real"] for x in rows])
        report[case] = {"n_cand": len(rows), "rho_fast": round(rho_f, 3),
                        "rho_official": round(rho_o, 3), "rows": rows}
        print(f"{case}: n={len(rows)} ρ(fast,real)={rho_f:.3f} "
              f"ρ(official,real)={rho_o:.3f}")
    out = HERE.parent / "results" / "v2_spearman.json"
    json.dump(report, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved ->", out)


if __name__ == "__main__":
    main()
