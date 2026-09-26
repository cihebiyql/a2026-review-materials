"""V1a: stub 基线复现 —— case_001/050/014 × 问题1/2/3 × 4核。

与审题报告 §2.6 表对照；单核基准走缓存。
输出: results/v1_stub_baseline.json
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'solver'))
import eval_lib as EL  # noqa: E402

CASES = ['case_001.json', 'case_050.json', 'case_014.json']
PROBLEMS = [1, 2, 3]
NUM_CORES = 4

def main():
    out = {}
    for case in CASES:
        t0 = time.perf_counter()
        graph = EL.load_graph(case)
        n_ops = len(graph['ops'])
        sc = EL.evaluate_singlecore_cached(
            graph, HERE / 'results' / 'singlecore' / (case.replace('.json', '_sc.json')))
        plan = EL.stub_plan(graph, num_cores=NUM_CORES, seed=0)
        n_sg = len(set(plan['node_to_subgraph'].values()))
        rows = {'num_ops': n_ops, 'num_subgraphs': n_sg,
                'singlecore': sc}
        for p in PROBLEMS:
            result, wall = EL.evaluate(graph, plan, p)
            row = EL.summarize(result, p)
            row['eval_seconds'] = round(wall, 2)
            row['speedup'] = round(sc['makespan'] / result['makespan'], 3)
            rows['p%d' % p] = row
            print(case, 'P%d' % p, 'makespan=%d speedup=%.3f eval=%.1fs added=%dB spill=%dB' % (
                result['makespan'], row['speedup'], wall,
                row['added_copy_bytes'], row['spill_bytes']), flush=True)
        rows['total_wall'] = round(time.perf_counter() - t0, 1)
        out[case] = rows
    dest = HERE / 'results' / 'v1_stub_baseline.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved ->', dest)

if __name__ == '__main__':
    main()
