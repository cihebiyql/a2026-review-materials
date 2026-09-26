"""V2A-补充：序（chain vs level/BFS）与粒度（块大小）两因素对照，问题1。

链序大块在 case_014/050 失败 → 检验 BFS 序 + 细块（stub 粒度）能否恢复并行。
输出: results/v2a2_order_granularity.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'solver'))
import eval_lib as EL  # noqa: E402
from naive_convex import naive_convex_plan  # noqa: E402

CASES = ['case_001.json', 'case_050.json', 'case_014.json']
NCORES = 4


def main():
    out = {}
    for case in CASES:
        graph = EL.load_graph(case)
        sc = EL.evaluate_singlecore_cached(
            graph, HERE / 'results' / 'singlecore' / (case.replace('.json', '_sc.json')))
        rows = {}
        for order in ('chain', 'level'):
            for csize in (64, 100, 200, 400):
                plan = naive_convex_plan(
                    graph, NCORES, order=order, chunk_size=csize)
                result, wall = EL.evaluate(graph, plan, 1)
                r = EL.summarize(result, 1)
                r['speedup'] = round(sc['makespan'] / result['makespan'], 3)
                r['n_subgraphs'] = sum(len(x) for x in plan['core_schedules'])
                rows['%s_c%d' % (order, csize)] = r
                print(case, order, 'csize=%d' % csize,
                      'speedup=%.3f added=%.1fMB spill=%.1fMB nsg=%d eval=%.1fs' % (
                          r['speedup'], r['added_copy_bytes'] / 1e6,
                          r['spill_bytes'] / 1e6, r['n_subgraphs'], wall),
                      flush=True)
        out[case] = rows
    dest = HERE / 'results' / 'v2a2_order_granularity.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved ->', dest)


if __name__ == '__main__':
    main()
