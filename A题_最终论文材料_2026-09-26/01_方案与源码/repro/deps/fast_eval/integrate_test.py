# -*- coding: utf-8 -*-
"""集成测试：用 FastEvalP1 替换 solve_p1 中的 official_proxy 精筛段。

对比：原始（official_proxy）vs 加速（FastEvalP1）的结果一致性和耗时。
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v2_solver"))
sys.path.insert(0, str(HERE.parent / "v3_solver"))

ATT = os.environ.get("A2026_ATT", r"C:/shumo_live/a_data")
sys.path.insert(0, os.path.join(ATT, "code"))

from fast_eval_p1 import FastEvalP1
from common import load_case, sc_makespan, ev_p1
from p1_v3 import solve_p1
import stub_multicore_cut_and_schedule as stub


def solve_p1_fast(graph, case, sc, K=4, budget_s=60.0):
    """用 FastEvalP1 替换 official_proxy 的精筛版 solve_p1。"""
    from common import op_dag, cycles_map
    from p1_v3 import (UnitPack, greedy_pack, local_improve,
                       runs_merge_plan, task_sim)
    from structure_split import structure_aware_plan
    from stub_multicore_cut_and_schedule import derive_multicore_plan
    from collections import defaultdict

    t0 = time.perf_counter()
    deadline = t0 + budget_s
    ids, preds, succs = op_dag(graph)
    w = cycles_map(graph)
    total = sum(w.values())
    kappa = sc / total
    cands = []

    # 种子族 A：v2 连续切
    for cpc in (1, 2, 3):
        planA, diag = structure_aware_plan(graph, K, chunks_per_core=cpc)
        cands.append((f"v2_cpc{cpc}", planA, None))

    # 种子族 B：装箱（只做一轮，演示用）
    pk = UnitPack(graph, max(1, total // (K * 8)), order_mode="dfs")
    if pk.n_units() >= K:
        best_pack, best_mk = None, float("inf")
        for seed in range(3):
            co = greedy_pack(pk, K, kappa, seed=seed)
            co, mk = local_improve(pk, co, K, kappa, iters=200, seed=seed)
            if mk < best_mk:
                best_pack, best_mk = co, mk
        plan, nsg = runs_merge_plan(pk, best_pack, K)
        cands.append((f"pack_g8_dfs", plan, best_mk))

    # ---- fast 代理打分 ----
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
            scored.append((tag, plan, mk))
        except Exception:
            scored.append((tag, plan, fmk or 1e18))
    scored.sort(key=lambda x: x[2])

    # ---- FastEvalP1 精筛（替代 official_proxy）----
    fe = FastEvalP1(graph)
    ok = []
    for tag, plan, mk in scored[:5]:
        if ok and time.perf_counter() > deadline - 10:
            break
        try:
            t1 = time.perf_counter()
            fast_mk, info = fe.evaluate(plan)
            t2 = time.perf_counter()
            ok.append((tag, plan, fast_mk, 0.0, t2 - t1))
        except Exception as exc:
            print(f"  [{case}] {tag} fast fail: {str(exc)[:60]}")
    ok.sort(key=lambda x: x[2])
    return ok


def main():
    data = Path(os.path.join(ATT, "data"))
    sc_dir = Path(HERE.parent / "results" / "singlecore")

    for case in ["case_001", "case_082", "case_050"]:
        graph = json.load(open(data / f"{case}.json", encoding="utf-8"))
        sc = sc_makespan(case, sc_dir)
        print(f"\n== {case} (ops={len(graph['ops'])}, sc={sc})")

        # 方式 1：FastEvalP1 直接评
        fe = FastEvalP1(graph)
        plans = []
        from structure_split import structure_aware_plan
        for cpc in (1, 2):
            plan, _ = structure_aware_plan(graph, 4, chunks_per_core=cpc)
            plans.append((f"v2_cpc{cpc}", plan))

        print(f"  --- FastEvalP1 评估 ---")
        for tag, plan in plans:
            t0 = time.perf_counter()
            mk, info = fe.evaluate(plan)
            dt = time.perf_counter() - t0
            sp = sc / mk
            print(f"  {tag}: mk={mk} speedup={sp:.3f} "
                  f"added={info['added_copy_bytes']/1e6:.1f}MB "
                  f"({dt:.3f}s)")

        # 方式 2：对比 official evaluate_scene_a
        from multicore_cut_evaluate_problem_1 import evaluate_scene_a
        print(f"  --- 官方 evaluate_scene_a 对比 ---")
        for tag, plan in plans:
            t0 = time.perf_counter()
            r = evaluate_scene_a(graph, plan, bandwidth=60,
                                 capacity={"L1": 524288, "UB": 131072},
                                 cross_core_wait=1000, same_core_wait=100)
            dt = time.perf_counter() - t0
            print(f"  {tag}: mk={r['makespan']} "
                  f"({dt:.3f}s)")

        # 方式 3：solve_p1 快速版（演示完整漏斗）
        print(f"  --- solve_p1 快速版（FastEvalP1 精筛）---")
        t0 = time.perf_counter()
        results = solve_p1_fast(graph, case, sc, budget_s=30.0)
        dt = time.perf_counter() - t0
        if results:
            best_tag, best_plan, best_mk, _, eval_t = results[0]
            print(f"  最优: {best_tag} mk={best_mk} speedup={sc/best_mk:.3f} "
                  f"总耗时={dt:.1f}s (精筛={eval_t:.3f}s)")
        print(f"  总耗时: {dt:.1f}s")


if __name__ == "__main__":
    main()
