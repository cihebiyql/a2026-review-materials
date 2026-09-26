"""V4 消融雏形：总量均衡 vs 双资源(M/V分别)均衡（审题报告偏离点3）。

10 个 M/V 比率分散的代表用例，P1，N=4，K=N 与 K=2N 两种粒度。
另记每核 (W_M, W_V) 分布，看双资源均衡是否真的拉平了 pipe 瓶颈。
输出: results/v4_ablation.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'solver'))
import eval_lib as EL  # noqa: E402
from naive_convex import naive_convex_plan, eligible_ids  # noqa: E402

CASES = [16, 50, 47, 84, 12, 53, 76, 30, 91, 14]
NCORES = 4
# 支持命令行覆盖用例列表（分批跑）
if len(sys.argv) > 1:
    CASES = [int(x) for x in sys.argv[1].split(',') if x.strip()]
    PART = sys.argv[2] if len(sys.argv) > 2 else 'part'
else:
    PART = 'full'


def core_mv(graph, plan):
    op_by = {o['id']: o for o in graph['ops']}
    per = defaultdict(lambda: [0, 0])
    for v, sg in plan['node_to_subgraph'].items():
        core = sg % NCORES
        if op_by[v]['pipe'] == 'PIPE_M':
            per[core][0] += op_by[v]['cycles']
        elif op_by[v]['pipe'] == 'PIPE_V':
            per[core][1] += op_by[v]['cycles']
    return {c: tuple(x) for c, x in sorted(per.items())}


def main():
    out = {}
    for cid in CASES:
        case = 'case_%03d.json' % cid
        graph = EL.load_graph(case)
        sc = EL.evaluate_singlecore_cached(
            graph, HERE / 'results' / 'singlecore' / ('case_%03d_sc.json' % cid))
        rows = {'singlecore': sc['makespan']}
        for balance in ('total', 'dual'):
            for cpr in (1, 2):
                plan = naive_convex_plan(graph, NCORES, chunks_per_core=cpr,
                                         balance=balance, order='chain')
                result, wall = EL.evaluate(graph, plan, 1)
                r = EL.summarize(result, 1)
                r['speedup'] = round(sc['makespan'] / result['makespan'], 3)
                r['core_mv'] = core_mv(graph, plan)
                rows['%s_c%d' % (balance, cpr)] = r
                print(case, balance, 'c=%d' % cpr, 'speedup=%.3f coreMV=%s' % (
                    r['speedup'], r['core_mv']), flush=True)
        out[case] = rows
    dest = HERE / 'results' / ('v4_ablation_%s.json' % PART)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved ->', dest)


if __name__ == '__main__':
    main()
