"""v2 批量 runner（node1/集群用，也可本机）。

用法：
  A2026_ATT=/data/qlyu/tmp_shumo/a2026/att python run_batch.py \
      --cases cases_batch1.txt --sc-dir /data/qlyu/tmp_shumo/a2026/singlecore \
      --out v2_batch1.jsonl --workers 24

每用例：solve_case 全管线（组合+局部搜索+两级代理+真评估 top3），
输出 JSONL 行（含 fast/official/real 三级 makespan → 供 Spearman）。
大图（n_ops>25k）自动降 ls_iters。
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
from pipeline import solve_case, load_case  # noqa: E402


def sc_makespan(case, sc_dir):
    p = Path(sc_dir) / f"{case}_sc.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)["makespan"]


def one(case_json, sc_dir, ls_iters):
    case = case_json.replace(".json", "")
    sc = sc_makespan(case, sc_dir)
    g = load_case(case)
    n_ops = len(g["ops"])
    iters = ls_iters if n_ops < 25000 else max(40, ls_iters // 3)
    out = solve_case(g, case, num_cores=4, sc_makespan=sc,
                     ls_iters=iters, real_eval_top=3, log=lambda *a: None)
    b = out["best"]
    return {
        "case": case, "n_ops": n_ops, "sc": sc,
        "speedup": round(sc / b["real_mk"], 3),
        "best_tag": b["tag"], "K": len(b["cuts"]) - 1,
        "real_mk": b["real_mk"], "fast_mk": round(b["fast_mk"], 1),
        "official_mk": round(b["official_mk"], 1),
        "spill_MB": round(b["real_spill_MB"], 2),
        "added_MB": round(b["real_added_MB"], 2),
        "solve_s": out["solve_seconds"],
        # Spearman 样本：全部真评估过的候选三级对照
        "ranks": [{"tag": p["tag"], "fast": p["fast_mk"],
                   "official": p.get("official_mk"),
                   "real": p.get("real_mk")} for p in out["rows"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--sc-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ls-iters", type=int, default=120)
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
                      f"speedup={row['speedup']} K={row['K']} "
                      f"spill={row['spill_MB']}MB {row['solve_s']}s "
                      f"(total {time.time()-t_start:.0f}s)", flush=True)
            except Exception as exc:
                print(f"[FAIL] {c}: {exc}", flush=True)
                traceback.print_exc()
                fout.write(json.dumps({"case": c.replace(".json", ""),
                                       "error": str(exc)}) + "\n")
                fout.flush()


if __name__ == "__main__":
    main()
