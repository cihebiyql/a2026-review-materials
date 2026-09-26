"""论文补数扫描：40 代表用例 × N∈{2,3,4,5} × 方案{stub,naive,v2} × 问题{1,2,3}。

本程序及代码是在人工智能工具辅助下完成的。
GLM (zai-api/GLM-5.3), 智谱AI(Z.ai), 2026-09-23.

背景：已有 results/v3_scan.csv（stub/naive×P1×100例×N2-5）与
results/v2_batch1.jsonl（v2×P1×40例×N4）。论文正文还需要：
  a) v2 的 1–5 核曲线（N=2,3,5 补跑，N=4 重跑并校验确定性）；
  b) 问题 2（场景B）与问题 3（L2）三方案逐用例数据（本脚本一次性补齐，
     问题3的"无L2基线"即同方案的问题2评估结果）；
  c) 全 100 用例数据体检统计（规模/双管负载/关键路/并行上限），
     存 results/v5_data_stats.json（论文数字一律以本目录落盘为准）。

产出：results/v5_paper_scan.jsonl + results/v5_paper_summary.json
纪律：官方评估器原样调用（eval_lib.evaluate，进程内、不写 trace）；
     v2 求解参数与 run_batch.py 完全一致（确定性，N=4 可与
     v2_batch1.jsonl 逐位对照）。
"""
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "solver"))

from eval_lib import load_graph, evaluate, summarize, stub_plan  # noqa: E402
from naive_convex import naive_convex_plan  # noqa: E402
from pipeline import solve_case, CutEvaluator, _GRAPH_HOLDER  # noqa: E402
from structure_split import op_dag, cycles_map  # noqa: E402

SC_DIR = ROOT / "results" / "singlecore"
OUT = ROOT / "results" / "v5_paper_scan.jsonl"
STATS_OUT = ROOT / "results" / "v5_data_stats.json"
CASES = [c if c.endswith(".json") else c + ".json"
         for c in (HERE / "cases_batch1.txt").read_text(encoding="utf-8").split()]
CORES = (2, 3, 4, 5)


def sc_makespan(case):
    with open(SC_DIR / f"{case[:-5]}_sc.json", encoding="utf-8") as f:
        return json.load(f)["makespan"]


def ev3(graph, plan):
    """三问各评一次，返回 p1/p2/p3 摘要。"""
    out = {}
    for p in (1, 2, 3):
        r, wt = evaluate(graph, plan, p)
        row = summarize(r, p)
        row["eval_s"] = round(wt, 2)
        if p == 3:
            cs = r.get("cache_stats", {})
            row["cache_stats"] = {k: cs.get(k) for k in sorted(cs)
                                  if isinstance(cs.get(k), (int, float))}
        out[f"p{p}"] = row
    return out


def one_unit(case, n):
    g = load_graph(case)
    n_ops = len(g["ops"])
    sc = sc_makespan(case)
    rec = {"case": case[:-5], "n_ops": n_ops, "N": n, "sc": sc}

    # ---- stub（官方随机基线，seed=0 与 v1/v3 扫描一致）----
    plan = stub_plan(g, num_cores=n)
    rec["stub"] = ev3(g, plan)

    # ---- naive（朴素凸划分，与 v3 扫描同参数 K=N）----
    t0 = time.time()
    plan = naive_convex_plan(g, num_cores=n, chunks_per_core=1)
    rec["naive"] = ev3(g, plan)
    rec["naive"]["solve_s"] = round(time.time() - t0, 2)

    # ---- v2（与 run_batch.one 完全同参）----
    t0 = time.time()
    iters = 120 if n_ops < 25000 else max(40, 120 // 3)
    out = solve_case(g, case[:-5], num_cores=n, sc_makespan=sc,
                     ls_iters=iters, real_eval_top=3, log=lambda *a: None)
    b = out["best"]
    evl = CutEvaluator(g, n)
    _GRAPH_HOLDER[id(evl)] = g
    plan = evl.to_plan(b["cuts"], None)
    v2 = {"solve_s": round(time.time() - t0, 1), "K": len(b["cuts"]) - 1,
          "tag": b["tag"],
          "p1": {"makespan": b["real_mk"],
                 "added_copy_bytes": round(b["real_added_MB"] * 1e6),
                 "spill_bytes": round(b["real_spill_MB"] * 1e6)}}
    for p in (2, 3):
        r, wt = evaluate(g, plan, p)
        row = summarize(r, p)
        row["eval_s"] = round(wt, 2)
        if p == 3:
            cs = r.get("cache_stats", {})
            row["cache_stats"] = {k: cs.get(k) for k in sorted(cs)
                                  if isinstance(cs.get(k), (int, float))}
        v2[f"p{p}"] = row
    rec["v2"] = v2
    return rec


def data_stats():
    """100 用例体检：规模/双管负载/纯计算关键路/并行上限/共享张量。"""
    import statistics as st
    from collections import defaultdict

    def pick(vals, q):
        vals = sorted(vals)
        i = min(len(vals) - 1, max(0, round(q * (len(vals) - 1))))
        return vals[i]

    rows = []
    for i in range(1, 101):
        case = f"case_{i:03d}.json"
        g = load_graph(case)
        ids, preds, succs = op_dag(g)
        w = cycles_map(g)
        wm = sum(c for o in g["ops"] if o.get("pipe") == "PIPE_M"
                 for c in [max(1, o.get("cycles", 1))]
                 if o.get("op") not in ("COPY_IN", "COPY_OUT"))
        wv = sum(c for o in g["ops"] if o.get("pipe") == "PIPE_V"
                 for c in [max(1, o.get("cycles", 1))]
                 if o.get("op") not in ("COPY_IN", "COPY_OUT"))
        # 纯计算关键路（op-DAG 最长路，权 = cycles）
        depth = {v: 0 for v in ids}
        indeg = {v: 0 for v in ids}
        for v in ids:
            for s in succs.get(v, ()):
                if s in indeg:
                    indeg[s] += 1
        order = [v for v in ids if indeg[v] == 0]
        cp = 0
        qi = 0
        while qi < len(order):
            u = order[qi]
            qi += 1
            cp = max(cp, depth[u] + w[u])
            for s in succs.get(u, ()):
                if s in depth:
                    depth[s] = max(depth[s], depth[u] + w[u])
                    indeg[s] -= 1
                    if indeg[s] == 0:
                        order.append(s)
        ddr_in = sum(t["size"] for t in g["tensors"]
                     if t.get("pos") == "DDR")
        rows.append({
            "case": case[:-5], "n_ops": len(g["ops"]),
            "n_eligible": len(ids), "n_tensors": len(g["tensors"]),
            "n_edges": len(g["edges"]),
            "W_M": wm, "W_V": wv, "CP": cp,
            "par_cap": max(wm, wv) / cp if cp else None,
            "ddr_bytes": ddr_in,
        })
    caps = [r["par_cap"] for r in rows if r["par_cap"]]
    summary = {
        "n_cases": len(rows),
        "ops": {q: pick([r["n_ops"] for r in rows], q)
                for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
        "edges": {q: pick([r["n_edges"] for r in rows], q)
                  for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
        "W_M": {q: pick([r["W_M"] for r in rows], q)
                for q in (0.25, 0.5, 0.75)},
        "W_V": {q: pick([r["W_V"] for r in rows], q)
                for q in (0.25, 0.5, 0.75)},
        "mv_ratio_median": st.median(r["W_M"] / r["W_V"] if r["W_V"] else 0
                                     for r in rows),
        "mv_ratio_min": min((r["W_M"] / r["W_V"] if r["W_V"] else 0)
                            for r in rows),
        "mv_ratio_max": max((r["W_M"] / r["W_V"] if r["W_V"] else 0)
                            for r in rows),
        "par_cap": {"min": min(caps), "p25": pick(caps, 0.25),
                    "median": st.median(caps), "p75": pick(caps, 0.75),
                    "max": max(caps),
                    "n_ge3": sum(1 for c in caps if c >= 3)},
        "rows": rows,
    }
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    if args.stats_only:
        data_stats()
        print("stats done")
        return

    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["case"], r["N"]))
            except Exception:
                pass
    units = [(c, n) for c in CASES for n in CORES
             if (c[:-5], n) not in done]
    print(f"{len(units)} units to run (skip {len(done)})", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex, \
            open(OUT, "a", encoding="utf-8") as fout:
        futs = {ex.submit(one_unit, c, n): (c, n) for c, n in units}
        for i, fut in enumerate(as_completed(futs)):
            c, n = futs[fut]
            try:
                rec = fut.result()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                sp = rec["sc"] / rec["v2"]["p1"]["makespan"]
                print(f"[{i+1}/{len(units)}] {c} N{n} "
                      f"v2_p1_sp={sp:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
            except Exception as exc:
                print(f"[FAIL] {c} N{n}: {exc}", flush=True)
                traceback.print_exc()
                fout.write(json.dumps({"case": c[:-5], "N": n,
                                       "error": str(exc)}) + "\n")
                fout.flush()
    print(f"all done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
