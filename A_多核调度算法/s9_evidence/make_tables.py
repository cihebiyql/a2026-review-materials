# -*- coding: utf-8 -*-
"""s9_evidence.make_tables — 赛题格式逐例表 + 汇总均值。

从 results/runs/*.jsonl 生成：
  tables/percase_q{Q}_N{N}_{ver}.csv
      列: case, makespan(官方), speedup, added_MB, [q3: hit_rate,
          mk_nol2, added_nol2_MB, cache_speedup], solve_time_s,
          official_match, status
  tables/summary.csv
      列: Q, N, version, n, mean_sp, mean_added_MB, mean_solve_s,
          max_solve_s, n_improved, n_fail
同一 (case,Q,N,version) 有多行时取最后一行（断点续跑覆盖旧记录）。
"""
import csv
import json
import os
from collections import defaultdict

import paths

VERS = ['v1', 'v2', 'v3', 'v4']


def load_records():
    """{(case,Q,N,ver): rec} 取每键最后一行。"""
    runs = os.path.join(paths.RESULTS, 'runs')
    recs = {}
    if not os.path.isdir(runs):
        return recs
    for fn in sorted(os.listdir(runs)):
        if not fn.endswith('.jsonl'):
            continue
        try:
            for line in open(os.path.join(runs, fn), encoding='utf-8'):
                d = json.loads(line)
                recs[(d['case'], d['Q'], d['N'], d['version'])] = d
        except Exception:
            continue
    return recs


def mb(b):
    return round(b / 1e6, 3) if isinstance(b, (int, float)) else None


def main():
    recs = load_records()
    out = os.path.join(paths.RESULTS, 'tables')
    os.makedirs(out, exist_ok=True)
    keys = sorted(recs.keys())
    qs = sorted({k[1] for k in keys})
    # 逐例表
    # v4 缺行(崩溃案例跳过) → 继承 v3 记录并标注
    v4_keys = {k for k in keys if k[3] == 'v3'}
    for (case, Q, N, _) in list(v4_keys):
        k4 = (case, Q, N, 'v4')
        if k4 not in recs:
            r = dict(recs[(case, Q, N, 'v3')])
            r['version'] = 'v4'
            r['status'] = 'v4_skipped_flaky_inherit_v3'
            recs[k4] = r
    keys = sorted(recs.keys())
    by_qn = defaultdict(list)
    for k in keys:
        by_qn[(k[1], k[2], k[3])].append(k)
    for (Q, N, ver), ks in sorted(by_qn.items()):
        fp = os.path.join(out, f'percase_q{Q}_N{N}_{ver}.csv')
        cols = (['case', 'makespan', 'speedup', 'added_MB']
                + ([] if Q != 3 else
                   ['hit_rate', 'mk_nol2', 'added_nol2_MB', 'cache_speedup'])
                + ['solve_time_s', 'official_match', 'status'])
        with open(fp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for k in sorted(ks):
                r = recs[k]
                row = {'case': k[0], 'makespan': r.get('official_mk'),
                       'speedup': r.get('sp'),
                       'added_MB': mb(r.get('added_copy_bytes'))}
                if Q == 3:
                    row.update({'hit_rate': r.get('hit_rate'),
                                'mk_nol2': r.get('mk_nol2'),
                                'added_nol2_MB': mb(r.get('added_nol2')),
                                'cache_speedup': r.get('cache_speedup')})
                row['solve_time_s'] = r.get('solve_s')
                row['official_match'] = r.get('official_match')
                row['status'] = r.get('status')
                w.writerow(row)

    # 汇总表
    rows = []
    for (Q, N, ver), ks in sorted(by_qn.items()):
        rs = [recs[k] for k in ks]
        sps = [r['sp'] for r in rs if r.get('sp')]
        adds = [r.get('added_copy_bytes') for r in rs
                if isinstance(r.get('added_copy_bytes'), (int, float))]
        solves = [r.get('solve_s') for r in rs
                  if isinstance(r.get('solve_s'), (int, float))]
        rows.append({
            'Q': Q, 'N': N, 'version': ver, 'n': len(rs),
            'mean_sp': round(sum(sps) / len(sps), 4) if sps else None,
            'mean_added_MB': round(sum(adds) / len(adds) / 1e6, 3)
            if adds else None,
            'mean_solve_s': round(sum(solves) / len(solves), 1)
            if solves else None,
            'max_solve_s': round(max(solves), 1) if solves else None,
            'n_improved': sum(1 for r in rs if r.get('improved')),
            'n_fail': sum(1 for r in rs
                          if str(r.get('status', '')).startswith('FAIL')),
        })
    with open(os.path.join(out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('TABLES_DONE', len(keys), 'records ->', out)


if __name__ == '__main__':
    main()
