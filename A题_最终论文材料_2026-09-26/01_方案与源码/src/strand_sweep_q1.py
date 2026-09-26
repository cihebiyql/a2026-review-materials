# -*- coding: utf-8 -*-
"""N5 strand 扫描:全 100 例 × wide_min 变体 × {P2,P3} 官方评估。
辫状族已证 1.00→3.44;扫全部用例收同类红利。"""
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
sys.path.insert(0, r"C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand")
sys.path.insert(0, r"C:/shumo_live/a_data/code")

SC_DIR = Path(r"C:/shumo_live/02_求解/A题_2026/results/singlecore")
OUT_DIR = HERE / "strand_n5"
OUT_DIR.mkdir(exist_ok=True)
WMS = (6, 8, 10, 12)


def sweep_one(args):
    case, qs = args
    from strand_seed import build_strand_plan
    from common import load_case, ev_p1, ev_p2, ev_p3
    sc = json.load(open(SC_DIR / f"{case}_sc.json"))["makespan"]
    g = load_case(case)
    out = {"case": case, "sc": sc}
    for q in qs:
        best = None
        for wm in WMS:
            try:
                plan, meta = build_strand_plan(g, 5, wide_min=wm)
                evf = {1: ev_p1, 2: ev_p2, 3: ev_p3}[q]
                r, wt = evf(g, plan)
                sp = sc / r["makespan"]
                if best is None or sp > best["sp"]:
                    best = {"wm": wm, "mk": r["makespan"], "sp": round(sp, 4)}
                    json.dump({"plan": plan, "mk": r["makespan"], "sp": sp,
                               "wm": wm},
                              open(OUT_DIR / f"{case}_q{q}_N5.json", "w"))
            except Exception:
                continue
        out[f"q{q}"] = best
    return out


def main(qs=(1,), workers=6):
    cases = [f"case_{i:03d}" for i in range(1, 101)]
    tasks = [(c, qs) for c in cases]
    rows = []
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(sweep_one, tasks):
            if not res.get("q3") and not res.get("q2"):
                continue
            rows.append(res)
            msg = " ".join(
                f"q{q}: {res[f'q{q}']['sp']:.3f}(wm{res[f'q{q}']['wm']})"
                for q in qs if res.get(f"q{q}"))
            print(f"{res['case']}: {msg}", flush=True)
    json.dump(rows, open(OUT_DIR / "summary.json", "w"))
    for q in qs:
        vals = [r[f"q{q}"]["sp"] for r in rows if r.get(f"q{q}")]
        if vals:
            print(f"== q{q}: {len(vals)}例可跑, 均值 {sum(vals)/len(vals):.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
