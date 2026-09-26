# -*- coding: utf-8 -*-
"""tools/regen_sc.py — 单核基准缓存再生（可选，包内已带 sc/ 缓存）。

用官方 singlecore_evaluate（一字未改，库方式调用）对全部用例计算单核
Makespan，写入 sc/case_NNN_sc.json。仅当怀疑缓存或想从零验证时运行。
用法：A2026_OFFICIAL=... python3 tools/regen_sc.py [--only case_001,...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import paths  # noqa: F401

from singlecore_evaluate import evaluate_singlecore  # noqa: E402


def main():
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == '--only':
        only = sys.argv[2].split(',')
    out_dir = os.path.join(paths.PIPE_ROOT, 'sc')
    os.makedirs(out_dir, exist_ok=True)
    for case in (only or paths.CASES):
        graph = json.load(open(os.path.join(
            paths.OFFICIAL, 'data', f'{case}.json'),
            encoding='utf-8'))
        r = evaluate_singlecore(
            graph,
            bandwidth=60,
            capacity={'L1': 524288, 'UB': 131072})
        with open(os.path.join(out_dir, f'{case}_sc.json'), 'w') as f:
            json.dump({'makespan': r['makespan'],
                       'source': 'official singlecore_evaluate'}, f)
        print(case, r['makespan'], flush=True)
    print('SC_DONE')


if __name__ == '__main__':
    main()
