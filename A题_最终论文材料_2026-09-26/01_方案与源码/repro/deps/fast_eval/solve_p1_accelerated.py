# -*- coding: utf-8 -*-
"""solve_p1 加速版：用 FastEvalP1 替换 official_proxy 精筛段。

保持与 p1_v3.solve_p1 完全相同的搜索逻辑（种子族+装箱+波前），
仅将精筛评估从 official_proxy（调官方 _build_scene_a_tasks）换为
FastEvalP1.evaluate（numba 复刻，结果 bit-exact）。

用法：
  from solve_p1_accelerated import solve_p1_accel
  out = solve_p1_accel(graph, case, sc, K=4, budget_s=420)
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v2_solver"))
sys.path.insert(0, str(HERE.parent / "v3_solver"))

from fast_eval_p1 import FastEvalP1


def solve_p1_accel(graph, case, sc, K=4, cal_ratio=0.89, log=print,
                   real_top=3, ls_iters=400, budget_s=420.0, use_wave=True):
    """与 p1_v3.solve_p1 相同的搜索逻辑，精筛用 FastEvalP1。

    差异仅在：
    - official_proxy(graph, plan) → fe.evaluate(plan)
    - ev_p1(graph, plan) → fe.evaluate(plan)（真评估也是 bit-exact）
    - 不再需要 cal_ratio 校准（复刻直接给出精确值）
    """
    from common import op_dag, cycles_map
    from p1_v3 import (UnitPack, greedy_pack, local_improve,
                       runs_merge_plan, task_sim)
    from stub_multicore_cut_and_schedule import derive_multicore_plan

    t0 = time.perf_counter()
    deadline = t0 + budget_s
    ids, preds, succs = op_dag(graph)
    w = cycles_map(graph)
    total = sum(w.values())
    kappa = sc / total
    cands = []

    # ---- 种子族 A：v2 连续切 ----
    sys.path.insert(0, str(HERE.parent / "v2_solver"))
    try:
        from structure_split import structure_aware_plan
        for cpc in (1, 2, 3):
            planA, diag = structure_aware_plan(graph, K, chunks_per_core=cpc)
            cands.append((f"v2_cpc{cpc}", planA, None))
    except Exception as exc:
        log(f"[{case}] v2 seed fail: {exc}")

    # ---- 种子族 C：波前 ----
    wave_tax = {}
    if use_wave:
        try:
            from wave_seed import wave_candidates
            for tag, wplan, meta in wave_candidates(graph, K, log=log):
                cands.append((tag, wplan, None))
                wave_tax[tag] = meta["byte_tax"]
        except Exception as exc:
            log(f"[{case}] wave seed fail: {exc}")

    # ---- 种子族 B：装箱 ----
    soft = t0 + budget_s * 0.6
    big = len(ids) > 25000
    for gpc in ((4, 8) if big else (4, 8, 16)):
        for om in ("dfs", "topo"):
            if time.perf_counter() > soft and cands:
                break
            pk = UnitPack(graph, max(1, total // (K * gpc)), order_mode=om)
            if pk.n_units() < K:
                continue
            best_pack, best_mk = None, float("inf")
            n_seed = (3 if big else 8) if pk.n_units() <= 800 else 2
            for seed in range(n_seed):
                co = greedy_pack(pk, K, kappa, seed=seed)
                co, mk = local_improve(
                    pk, co, K, kappa, iters=min(ls_iters, 3 * pk.n_units()),
                    seed=seed,
                    time_budget=min(6.0, max(1.0, (soft - time.perf_counter())
                                             / max(1, n_seed)))
                    if pk.n_units() > 500 else None)
                if mk < best_mk:
                    best_pack, best_mk = co, mk
            plan, nsg = runs_merge_plan(pk, best_pack, K)
            cands.append((f"pack_g{gpc}_{om}", plan, best_mk))
            log(f"[{case}] gpc={gpc}/{om}: units={pk.n_units()} "
                f"fast={best_mk:.0f} tasks={nsg}")

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
            mk += wave_tax.get(tag, 0.0)
            scored.append((tag, plan, mk))
        except Exception as exc:
            log(f"[{case}] {tag} derive fail: {str(exc)[:60]}")
    scored.sort(key=lambda x: x[2])

    # 波前保位
    picks = scored[:5]
    picked = {t for t, _, _ in picks}
    for t, p, m in scored:
        if t in wave_tax and t not in picked and len(picks) < 7:
            picks.append((t, p, m))
            picked.add(t)
            if sum(1 for x in picks if x[0] in wave_tax) >= 2:
                break

    # ---- FastEvalP1 精筛（替代 official_proxy）----
    fe = FastEvalP1(graph)
    ok = []
    for tag, plan, mk in picks:
        if ok and time.perf_counter() > deadline - 10:
            log(f"[{case}] budget: skip fast eval for {tag}")
            break
        try:
            t1 = time.perf_counter()
            fast_mk, info = fe.evaluate(plan)
            dt = time.perf_counter() - t1
            ok.append((tag, plan, fast_mk,
                       info["spill_added_copy_bytes"] / 1e6))
        except Exception as exc:
            log(f"[{case}] {tag} fast fail: {str(exc)[:60]}")
    if not ok and scored:
        tag, plan, mk = scored[0]
        ok.append((tag, plan, mk, 0.0))
    ok.sort(key=lambda x: x[2])

    # ---- 输出 top ----
    rows = []
    for tag, plan, omk, sp in ok[:real_top]:
        rows.append({"tag": tag, "plan": plan, "real_mk": omk,
                     "spill_MB": sp})
        log(f"[{case}] {tag}: mk={omk} speedup={sc/omk:.3f} spill={sp:.1f}MB")
    best = min(rows, key=lambda x: x["real_mk"]) if rows else None
    return {"case": case, "best": best, "rows": rows,
            "solve_s": round(time.perf_counter() - t0, 1)}


if __name__ == "__main__":
    import json
    import os
    from common import load_case, sc_makespan

    sc_dir = HERE.parent / "results" / "singlecore"
    for case, sc in [("case_001", 233110), ("case_082", 721041),
                     ("case_050", 149690)]:
        g = load_case(case)
        out = solve_p1_accel(g, case, sc, budget_s=60.0)
        if out["best"]:
            print(f"{case}: {out['best']['tag']} "
                  f"speedup={sc/out['best']['real_mk']:.3f} "
                  f"({out['solve_s']}s)")
