"""v3 批量 runner（node1/本机）：三问全管线。

用法（node1）：
  cd /data1/qlyu/a2026/v3_solver && \
  A2026_ATT=/data1/qlyu/a2026/att /data/qlyu/anaconda3/bin/python \
      run_batch_v3.py --cases cases40.txt --sc-dir ../singlecore \
      --out ../../results/v3_batch.jsonl --workers 24

每用例输出一行 JSONL：P1/P2/P3 各自 best（方案、makespan、speedup、
spill、P3 命中率与 L2gain）。
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v2_solver"))
from common import load_case, sc_makespan, ev_p2  # noqa: E402
from p1_v3 import solve_p1  # noqa: E402
from p2_v3 import solve_p2  # noqa: E402
from p3_v3 import solve_p3  # noqa: E402


def one(case, sc_dir, ls_iters):
    case = case.replace(".json", "")
    t0 = time.time()
    sc = sc_makespan(case, sc_dir)
    g = load_case(case)
    n_ops = len(g["ops"])
    quiet = (lambda *a, **k: None)
    out1 = solve_p1(g, case, sc, ls_iters=ls_iters
                    if n_ops < 25000 else max(60, ls_iters // 3), log=quiet,
                    budget_s=420.0)
    out2 = solve_p2(g, case, sc, log=quiet,
                    extra_plans=[("p1best", out1["best"]["plan"])],
                    budget_s=300.0)
    out3 = solve_p3(g, case, sc, log=quiet, p2_out=out2, budget_s=420.0)
    b1, b2, b3 = out1["best"], out2["best"], out3["best"]
    b3h = out3["hitopt"]
    # P3 best 方案的无 L2 对照（同一方案的 P2 评估）
    r2_of_p3, _ = ev_p2(g, b3["plan"])
    return {
        "case": case, "n_ops": n_ops, "sc": sc,
        "p1": {"speedup": round(sc / b1["real_mk"], 3), "mk": b1["real_mk"],
               "tag": b1["tag"], "K": len(b1["plan"]["core_schedules"][0]) + 0,
               "n_sg": len(set(b1["plan"]["node_to_subgraph"].values())),
               "spill_MB": round(b1["spill_MB"], 2),
               "added_MB": round(b1["added_MB"], 2)},
        "p2": {"speedup": round(sc / b2["real_mk"], 3), "mk": b2["real_mk"],
               "tag": b2["tag"], "n_sg": len(set(b2["plan"]["node_to_subgraph"].values())),
               "spill_MB": round(b2["spill_MB"], 2),
               "added_MB": round(b2["added_MB"], 2)},
        "p3": {"speedup": round(sc / b3["real_mk"], 3), "mk": b3["real_mk"],
               "tag": b3["tag"],
               "hit_rate": round(b3["hit_rate"], 4),
               "hit_MB": round(b3["hit_MB"], 2),
               "spill_MB": round(b3["spill_MB"], 2),
               "L2gain_same_plan": round(r2_of_p3["makespan"] / b3["real_mk"], 4),
               "L2gain_vs_p2best": round(b3["L2gain_vs_p2best"], 4)},
        "p3_hitopt": {"tag": b3h["tag"], "mk": b3h["real_mk"],
                      "speedup": round(sc / b3h["real_mk"], 3),
                      "hit_rate": round(b3h["hit_rate"], 4),
                      "hit_MB": round(b3h.get("hit_MB", 0), 2),
                      "L2gain_vs_p2best": round(b3h["L2gain_vs_p2best"], 4)},
        "solve_s": round(time.time() - t0, 1),
        "solve_p1_s": out1["solve_s"], "solve_p2_s": out2["solve_s"],
        "solve_p3_s": out3["solve_s"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ls-iters", type=int, default=400)
    args = ap.parse_args()

    cases = [c for c in Path(args.cases).read_text(encoding="utf-8").split()
             if c]
    done = set()
    if Path(args.out).exists():
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["case"])
                except Exception:
                    pass
    todo = [c for c in cases if c.replace(".json", "") not in done]
    print(f"{len(todo)}/{len(cases)} to run (skip {len(done)})", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex, \
            open(args.out, "a", encoding="utf-8") as fout:
        futs = {ex.submit(one, c, args.sc_dir, args.ls_iters): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            c = futs[fut]
            try:
                row = fut.result()
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{i+1}/{len(todo)}] {row['case']} "
                      f"P1={row['p1']['speedup']} P2={row['p2']['speedup']} "
                      f"P3={row['p3']['speedup']} hit={row['p3']['hit_rate']:.2%} "
                      f"gain={row['p3']['L2gain_vs_p2best']:.3f} "
                      f"{row['solve_s']}s (tot {time.time()-t_start:.0f}s)",
                      flush=True)
            except Exception as exc:
                print(f"[FAIL] {c}: {exc}", flush=True)
                traceback.print_exc()
                fout.write(json.dumps({"case": c.replace(".json", ""),
                                       "error": str(exc)[:200]}) + "\n")
                fout.flush()


if __name__ == "__main__":
    main()
