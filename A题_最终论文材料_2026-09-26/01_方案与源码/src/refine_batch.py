# -*- coding: utf-8 -*-
"""N5 冲 5 均值战役:批量真值驱动精修(FastEval bit-exact 目标)。

对每例:种子 = plan_archive 的 v3_main(备选 v2_cpc)→ 子图搬移爬山
→ 官方评估器终验(bit-exact)→ 落盘方案与成绩。
"""
import os
import sys
import json
import time
import warnings
import multiprocessing as mp
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:/shumo_live/02_求解/A题_2026/fast_eval")
sys.path.insert(0, r"C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand")
sys.path.insert(0, r"C:/shumo_live/a_data/code")

ARCHIVE = Path(os.environ.get("A2026_ARCHIVE", r"C:/shumo_live/02_求解/A题_2026/a_lab/registry/plan_archive/plans"))
SC_DIR = Path(os.environ.get("A2026_SC_DIR", r"C:/shumo_live/02_求解/A题_2026/results/singlecore"))
POSTHOC = Path(os.environ.get("A2026_POSTHOC", r"C:/shumo_live/02_求解/A题_2026/a_lab/records/POSTHOC_BEST.json"))
OUT_DIR = HERE / "refined"
OUT_DIR.mkdir(exist_ok=True)


def load_plan(case, q, K, tag):
    fp = ARCHIVE / f"{case}_q{q}_N{K}_{tag}.json"
    if not fp.exists():
        return None
    d = json.load(open(fp))
    return {"node_to_subgraph": d["node_to_subgraph"],
            "core_schedules": d["core_schedules"]}


def _load_strand(case, q, K):
    fp = HERE / "strand_n5" / f"{case}_q{q}_N{K}.json"
    if not fp.exists():
        return None
    return json.load(open(fp)).get("plan")


def _load_refined(case, q, K):
    fp = OUT_DIR / f"{case}_q{q}_N{K}.json"
    if not fp.exists():
        return None
    d = json.load(open(fp))
    return d.get("plan")


def refine_one(args):
    case, q, K, budget_s = args
    from common import load_case, ev_p2, ev_p3
    from fast_eval_p2 import FastEvalP2, FastEvalP3
    from strand_p2 import refine_real
    t0 = time.perf_counter()
    sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
    g = load_case(case)
    fe = FastEvalP3(g) if q == 3 else FastEvalP2(g)
    # 每例评估耗时探测
    seed_plan = load_plan(case, q, K, "v3_main") or load_plan(case, q, K, "v2_cpc")
    if seed_plan is None:
        return {"case": case, "status": "no_seed"}
    # 多种子取优:archive 两族 + 上轮精修复用(二轮)
    t_enc = time.perf_counter()
    mk0_fast = fe.evaluate(seed_plan)[0]
    dt_eval = time.perf_counter() - t_enc
    for cand in (load_plan(case, q, K, "v2_cpc"),
                 _load_refined(case, q, K),
                 _load_strand(case, q, K)):
        if cand is None:
            continue
        mk_c = fe.evaluate(cand)[0]
        if mk_c < mk0_fast:
            seed_plan, mk0_fast = cand, mk_c
    evals = max(8, min(400, int(budget_s / max(dt_eval, 0.02))))
    best_plan, best_mk = None, float("inf")
    for seed in (0, 1, 2):
        if time.perf_counter() - t0 > budget_s * 1.1:
            break
        p2, mk = refine_real(g, seed_plan, K, fe,
                             evals=max(30, evals // 3), seed=seed)
        if mk < best_mk:
            best_plan, best_mk = p2, mk
    # 官方终验
    ev = ev_p3 if q == 3 else ev_p2
    r, wt = ev(g, best_plan)
    ok = (r["makespan"] == best_mk)
    sp = sc / r["makespan"]
    prev = _load_refined(case, q, K)
    prev_mk = None
    if prev is not None:
        prev_mk = fe.evaluate(prev)[0]
    if prev_mk is None or r["makespan"] < prev_mk:
        json.dump({"plan": best_plan, "mk": r["makespan"], "sp": sp,
                   "fast_match": ok, "seed_mk0": mk0_fast,
                   "wall": round(time.perf_counter() - t0, 1)},
                  open(OUT_DIR / f"{case}_q{q}_N{K}.json", "w"))
    return {"case": case, "mk": r["makespan"], "sp": round(sp, 4),
            "fast_match": ok, "base_fast": mk0_fast,
            "wall": round(time.perf_counter() - t0, 1)}


def main(q=3, K=5, budget_s=150.0, workers=10, cases=None):
    posthoc = json.load(open(POSTHOC))
    cases = cases or [f"case_{i:03d}" for i in range(1, 101)]
    skip_done = "--tail" in sys.argv
    if skip_done:
        cases = [c for c in cases
                 if not (OUT_DIR / f"{c}_q{q}_N{K}.json").exists()]
        print(f"[tail] 剩余 {len(cases)} 例", flush=True)
    tasks = [(c, q, K, budget_s) for c in cases]
    results = []
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(refine_one, tasks):
            if "status" in res:
                print(res["case"], res["status"], flush=True)
                continue
            key = f"{res['case']}|q{q}|N{K}"
            base = posthoc.get(key, float("nan"))
            gain = (res["sp"] / base - 1) * 100 if base == base else float("nan")
            print(f"{res['case']}: {res['sp']:.3f} (base {base:.3f}, "
                  f"{gain:+.1f}%) fast_match={res['fast_match']} "
                  f"[{res['wall']}s]", flush=True)
            results.append(res)
    json.dump(results, open(OUT_DIR / f"summary_q{q}_N{K}.json", "w"))
    sps = [r["sp"] for r in results]
    if sps:
        print(f"\n== 精修 {len(sps)} 例: 均值 {sum(sps)/len(sps):.4f} ==", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--tail", action="store_true")
    a = ap.parse_args()
    cs = a.cases.split(",") if a.cases else None
    main(q=a.q, K=a.K, budget_s=a.budget, workers=a.workers, cases=cs)
