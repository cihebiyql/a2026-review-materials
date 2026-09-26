# -*- coding: utf-8 -*-
"""s9_evidence.requalify — 权威证据记录（按团队文档规定字段）。

对 results/plans/ 下**全部终选方案**做一次统一的官方评估器重评，产出
文档要求的记录字段（每行 = 一个 (case, problem, cores, version)）：

  case, problem, cores, version,
  makespan,                    官方评估 Makespan（主指标）
  singlecore_makespan,         单核基准（官方 singlecore_evaluate 缓存, 不变）
  speedup,                     = singlecore / makespan
  added_copy_bytes,            总额外搬运（= partition + spill）
  partition_added_copy_bytes,  边界搬运分量（切图/分核决策引入）
  spill_added_copy_bytes,      换入换出分量（核内容量压力引入）
  cache_hit_rate,              仅 problem=3
  mk_nol2 / added_nol2 / cache_speedup,   仅 problem=3（题面两配置对比）
  official_seconds,            官方评估耗时（区别于算法运行时间 solve_s）
  solve_seconds,               算法运行时间（自 runs jsonl 带出）
  algorithm                    版本阶梯标识(v1/v2/v3/v4)

输出:
  tables/records_official.csv   全部记录（论文附录逐例表的直接数据源）
  tables/records_summary.csv    按 problem×cores×version 分组统计
证据等级: 全部字段出自未改动的官方评估器单遍重评，与求解过程解耦。
用法: PYTHONPATH=. python3 s9_evidence/requalify.py [--workers 16]
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import paths
from pipeline import N_LIST

FIELDS = ['case', 'problem', 'cores', 'version', 'makespan',
          'singlecore_makespan', 'speedup', 'added_copy_bytes',
          'partition_added_copy_bytes', 'spill_added_copy_bytes',
          'cache_hit_rate', 'mk_nol2', 'added_nol2_bytes',
          'cache_speedup', 'official_seconds', 'solve_seconds',
          'algorithm']


def _one(args):
    case, Q, N, ver = args
    fp = os.path.join(paths.RESULTS, 'plans',
                      f'{case}_q{Q}_N{N}_{ver}.json')
    if not os.path.exists(fp):
        return None
    import paths as P
    sys.path.insert(0, P.DEPS)
    from official_eval import ev_p1, ev_p2, ev_p3
    d = json.load(open(fp))
    plan = d['plan']
    graph = json.load(open(os.path.join(
        paths.OFFICIAL, 'data', f'{case}.json')))
    sc = P.sc_makespan(case)
    ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[Q]
    t0 = time.time()
    r, _ = ev(graph, plan)
    dt = round(time.time() - t0, 3)
    dm = r.get('data_movement_bytes', {})
    row = {'case': case, 'problem': Q, 'cores': N, 'version': ver,
           'makespan': r['makespan'], 'singlecore_makespan': sc,
           'speedup': sc / r['makespan'],
           'added_copy_bytes': dm.get('added_copy_bytes'),
           'partition_added_copy_bytes': dm.get('partition_added_copy_bytes'),
           'spill_added_copy_bytes': dm.get('spill_added_copy_bytes'),
           'cache_hit_rate': None, 'mk_nol2': None,
           'added_nol2_bytes': None, 'cache_speedup': None,
           'official_seconds': dt, 'solve_seconds': None,
           'algorithm': {'v1': 'V1 凸块基线', 'v2': 'V2 +B×L重标',
                         'v3': 'V3 +三结构', 'v4': 'V4 +学习式核指派'}[ver]}
    if Q == 3:
        cs = r.get('cache_stats') or {}
        row['cache_hit_rate'] = cs.get('hit_rate')
        r2, _ = ev_p2(graph, plan)
        dm2 = r2.get('data_movement_bytes', {})
        row['mk_nol2'] = r2['makespan']
        row['added_nol2_bytes'] = dm2.get('added_copy_bytes')
        row['cache_speedup'] = (r2['makespan'] / r['makespan']
                                if r['makespan'] else None)
    return row


def _solve_times():
    """从 runs jsonl 取每 (case,Q,N,ver) 最后一条的算法运行时间。"""
    solve = {}
    runs = os.path.join(paths.RESULTS, 'runs')
    if not os.path.isdir(runs):
        return solve
    for fn in os.listdir(runs):
        if not fn.endswith('.jsonl'):
            continue
        try:
            for line in open(os.path.join(runs, fn), encoding='utf-8'):
                d = json.loads(line)
                solve[(d['case'], d['Q'], d['N'], d['version'])] = \
                    d.get('solve_s')
        except Exception:
            pass
    return solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    plans = os.path.join(paths.RESULTS, 'plans')
    tasks = []
    for fn in sorted(os.listdir(plans)) if os.path.isdir(plans) else []:
        if not fn.endswith('.json'):
            continue
        stem = fn[:-5]                    # case_XXX_qQ_NN_ver
        parts = stem.split('_')
        case = f'{parts[0]}_{parts[1]}'
        Q = int(parts[2][1])
        N = int(parts[3][1])
        ver = parts[4]
        tasks.append((case, Q, N, ver))
    print(f'requalify {len(tasks)} plans', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for row in ex.map(_one, tasks):
            if row:
                rows.append(row)
    solve = _solve_times()
    for r in rows:
        r['solve_seconds'] = solve.get(
            (r['case'], r['problem'], r['cores'], r['version']))
    rows.sort(key=lambda r: (r['problem'], r['cores'], r['version'],
                             r['case']))
    out = os.path.join(paths.RESULTS, 'tables')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'records_official.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    # 分组汇总（文档: 按 case/problem/cores 分组统计比较）
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        grp[(r['problem'], r['cores'], r['version'])].append(r)
    srows = []
    for (Q, N, ver), rs in sorted(grp.items()):
        sp = [r['speedup'] for r in rs if r['speedup']]
        srows.append({
            'problem': Q, 'cores': N, 'version': ver, 'n': len(rs),
            'mean_speedup': round(sum(sp) / len(sp), 4) if sp else None,
            'max_speedup': round(max(sp), 4) if sp else None,
            'mean_added_MB': round(sum(r['added_copy_bytes'] for r in rs
                                       if r['added_copy_bytes'] is not None)
                                   / max(1, len(rs)) / 1e6, 3),
            'mean_solve_s': round(sum(r['solve_seconds'] for r in rs
                                      if r['solve_seconds'] is not None)
                                  / max(1, len(rs)), 1),
            'max_solve_s': round(max((r['solve_seconds'] for r in rs
                                      if r['solve_seconds'] is not None),
                                     default=0), 1),
        })
    with open(os.path.join(out, 'records_summary.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    print(f'REQUALIFY_DONE {len(rows)} rows -> records_official.csv,'
          f' records_summary.csv', flush=True)


if __name__ == '__main__':
    main()
