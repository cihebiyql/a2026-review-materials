# -*- coding: utf-8 -*-
"""v4 预算内求解器(题面对齐:可复现、合理时间、2~5核、三指标)。

在 v3_wave 谱系(exp_strand 快照)之上回灌两项已验证技术:
  1. strand 候选族(P2/P3 侧,wm∈{8,12} 变体,生成秒级——辫状族 N5 +7 点来源)
  2. 预算内顺序感知贪心收尾(FastEval bit-exact 目标,时限 cap 秒,
     E 类 +7~15% 来源;最终方案经官方评估器复核)
预算纪律:总墙钟 ≤ budget_s(默认 420s);贪心收尾只在剩余时间富余时启动。

用法:
  py v4_solver.py --case case_016 --q 3 --N 5 [--budget 420] [--out dir]
"""
import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
SNAP = Path(os.environ.get(
    "A2026_SOLVER_DIR",
    r"C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand"))
sys.path.insert(0, str(SNAP))
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:/shumo_live/a_data/code")
sys.path.insert(0, r"C:/shumo_live/02_求解/A题_2026/fast_eval")

from common import load_case, ev_p1, ev_p2, ev_p3  # noqa: E402


def solve_v4(graph, case, sc, q, K, budget_s=420.0, log=print):
    t0 = time.perf_counter()
    from p2_v3 import solve_p2
    from p3_v3 import solve_p3
    import p1_v3
    # ---- 阶段1:基线求解(v3 谱系,含 strand 候选注入) ----
    extra = []
    try:
        from strand_seed import build_strand_plan
        for wm in (8, 12):
            try:
                plan, meta = build_strand_plan(graph, K, wide_min=wm)
                extra.append((f"strand_wm{wm}", plan))
            except Exception:
                pass
    except Exception as exc:
        log(f"[{case}] strand seeds fail: {str(exc)[:50]}")
    base_budget = budget_s * 0.55
    if q == 1:
        os.environ.setdefault("A_LAB_STRAND", "1")
        out2 = p1_v3.solve_p1(graph, case, sc, K=K, log=log,
                              budget_s=base_budget)
        out = out2
    else:
        out2 = solve_p2(graph, case, sc, K=K, log=log,
                        budget_s=base_budget, extra_plans=extra)
        if q == 3:
            out = solve_p3(graph, case, sc, K=K, log=log,
                           budget_s=max(60.0, budget_s * 0.25), p2_out=out2)
        else:
            out = out2
    best_plan = out["best"]["plan"] if out.get("best") else None
    ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[q]
    mk0 = None
    if best_plan is not None:
        r0, _ = ev(graph, best_plan)
        mk0 = r0["makespan"]
    # P1:strand 粗段候选(wm8/12)直接参评——辫状 N5 的关键(P1 重读税需粗段)
    if q == 1:
        try:
            from strand_seed import build_strand_plan
            for wm in (8, 12):
                try:
                    plan_s, _ = build_strand_plan(graph, K, wide_min=wm)
                    rs, _ = ev(graph, plan_s)
                    if mk0 is None or rs["makespan"] < mk0:
                        best_plan, mk0 = plan_s, rs["makespan"]
                except Exception:
                    pass
        except Exception:
            pass
    if best_plan is None or mk0 is None:
        return None
    # ---- 阶段2:顺序感知贪心收尾(FastEval bit-exact) ----
    remain = budget_s - (time.perf_counter() - t0)
    cap = min(max(30.0, remain - 40.0), budget_s * 0.45)
    if cap >= 30:
        try:
            import multiprocessing as mp
            import refine3 as r3
            workers = min(10, os.cpu_count() or 8)
            pool = mp.Pool(workers, initializer=r3._worker_init,
                           initargs=(q,))
            try:
                iters = int(cap / 0.6)
                p2b, mkb = r3.refine3(
                    graph, best_plan, K, case, q, pool_obj=pool,
                    workers=workers, iters=iters, batch=10, seed=0,
                    sa=False, log=lambda *a: None)
            finally:
                pool.close()
                pool.join()
            if mkb is not None and mkb < mk0 - 0.5:
                rb, _ = ev(graph, p2b)
                if rb["makespan"] == mkb or rb["makespan"] < mk0:
                    best_plan = p2b
                    mk0 = min(mk0, rb["makespan"])
        except Exception as exc:
            log(f"[{case}] greedy finish fail: {str(exc)[:60]}")
    return {"plan": best_plan, "mk": mk0,
            "sp": sc / mk0,
            "wall": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--budget", type=float, default=420.0)
    ap.add_argument("--out", default=str(HERE / "v4_out"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    g = load_case(a.case)
    sc = json.load(open(
        rf"C:/shumo_live/02_求解/A题_2026/results/singlecore/{a.case}_sc.json"
    ))["makespan"]
    res = solve_v4(g, a.case, sc, a.q, a.N, budget_s=a.budget)
    if res:
        ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[a.q]
        r, _ = ev(g, res["plan"])
        rec = {"plan": res["plan"], "mk": r["makespan"],
               "sp": sc / r["makespan"],
               "added": r.get("data_movement_bytes", {}).get(
                   "added_copy_bytes", 0),
               "spill": r.get("data_movement_bytes", {}).get(
                   "spill_added_copy_bytes", 0),
               "cache": (r.get("cache") or {}).get("hit_rate")
               if a.q == 3 else None,
               "wall": res["wall"]}
        json.dump(rec, open(out / f"{a.case}_q{a.q}_N{a.N}_s{a.seed}.json",
                            "w"))
        print(f"{a.case} q{a.q} N{a.N}: sp={rec['sp']:.4f} "
              f"added={rec['added']/1e6:.2f}MB "
              f"cache={rec['cache'] if rec['cache'] is not None else '-'} "
              f"[{rec['wall']}s]", flush=True)


if __name__ == "__main__":
    main()
