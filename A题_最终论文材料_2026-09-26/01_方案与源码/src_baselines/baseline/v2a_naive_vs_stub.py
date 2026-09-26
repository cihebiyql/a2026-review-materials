"""V2 猜想A：朴素凸划分能否大幅超越 stub（问题1）。

case_001/050/014 × N∈{2,4} × chunks_per_core∈{1,2,4}，对比 stub。
另记 spill / 切分边界搬运，验证免 spill 直觉。
输出: results/v2a_naive_vs_stub.json
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'solver'))
import eval_lib as EL  # noqa: E402
from naive_convex import naive_convex_plan  # noqa: E402

CASES = ['case_001.json', 'case_050.json', 'case_014.json']


def main():
    out = {}
    for case in CASES:
        graph = EL.load_graph(case)
        sc = EL.evaluate_singlecore_cached(
            graph, HERE / 'results' / 'singlecore' / (case.replace('.json', '_sc.json')))
        rows = {'singlecore_makespan': sc['makespan']}
        # stub 参照
        for ncores in (2, 4):
            plan = EL.stub_plan(graph, num_cores=ncores, seed=0)
            result, wall = EL.evaluate(graph, plan, 1)
            r = EL.summarize(result, 1)
            r.update(speedup=round(sc['makespan'] / result['makespan'], 3),
                     eval_s=round(wall, 1))
            rows['stub_n%d' % ncores] = r
            print(case, 'stub N=%d speedup=%.3f makespan=%d added=%.1fMB' % (
                ncores, r['speedup'], r['makespan'], r['added_copy_bytes'] / 1e6),
                flush=True)
        # 朴素凸划分
        for ncores in (2, 4):
            for cpr in (1, 2, 4):
                t0 = time.perf_counter()
                plan = naive_convex_plan(graph, ncores, chunks_per_core=cpr)
                solve_s = time.perf_counter() - t0
                result, wall = EL.evaluate(graph, plan, 1)
                r = EL.summarize(result, 1)
                r.update(speedup=round(sc['makespan'] / result['makespan'], 3),
                         chunks_per_core=cpr, n_subgraphs=len(plan['core_schedules'][0] + plan['core_schedules'][1]) if len(plan['core_schedules']) > 1 else len(plan['core_schedules'][0]),
                         solve_s=round(solve_s, 2), eval_s=round(wall, 1))
                rows['naive_n%d_c%d' % (ncores, cpr)] = r
                print(case, 'naive N=%d c=%d speedup=%.3f makespan=%d added=%.1fMB spill=%.1fMB solve=%.2fs' % (
                    ncores, cpr, r['speedup'], r['makespan'],
                    r['added_copy_bytes'] / 1e6, r['spill_bytes'] / 1e6, solve_s),
                    flush=True)
        out[case] = rows
    dest = HERE / 'results' / 'v2a_naive_vs_stub.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved ->', dest)


if __name__ == '__main__':
    main()
