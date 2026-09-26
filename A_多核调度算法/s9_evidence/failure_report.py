# -*- coding: utf-8 -*-
"""s9_evidence.failure_report — 失败分析 + 版本增益分布。

  tables/failures.csv : 全部非 ok 记录（崩溃/异常）+ 图特征
  tables/version_gain.csv : 每 (Q,N,version) 的 Δsp 分布
      （相对上一版本；v1 相对自身记 0）+ 零增益例中链富集占比
      （方法边界证据：v4 学习指派仅在链富集图生效的观察）。
"""
import csv
import os
from collections import defaultdict

import paths
from s9_evidence.make_tables import load_records

VERS = ['v1', 'v2', 'v3', 'v4']


def main():
    recs = load_records()
    out = os.path.join(paths.RESULTS, 'tables')
    os.makedirs(out, exist_ok=True)

    fails = [(k, r) for k, r in recs.items()
             if str(r.get('status', '')).startswith('FAIL')]
    with open(os.path.join(out, 'failures.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'Q', 'N', 'version', 'status',
                    'n_ops', 'n_chains'])
        for (case, Q, N, ver), r in sorted(fails):
            g = r.get('graph') or {}
            w.writerow([case, Q, N, ver, r.get('status'),
                        g.get('n_ops'), g.get('n_chains')])
    print(f'failure rows: {len(fails)} -> failures.csv')

    # Δsp 分布
    rows = []
    keys = set(recs.keys())
    groups = defaultdict(list)
    for (case, Q, N, ver) in keys:
        groups[(Q, N)].append((case, ver))
    for (Q, N), items in sorted(groups.items()):
        for vi, ver in enumerate(VERS):
            prev = VERS[vi - 1] if vi > 0 else None
            deltas, rich_improved, rich_total = [], 0, 0
            for (case, v) in items:
                if v != ver:
                    continue
                r = recs[(case, Q, N, ver)]
                sp = r.get('sp')
                if sp is None:
                    continue
                g = r.get('graph') or {}
                if vi == 0:
                    deltas.append(0.0)
                    continue
                rp = recs.get((case, Q, N, prev))
                if rp is None or rp.get('sp') is None:
                    continue
                d = sp - rp['sp']
                deltas.append(d)
                if g.get('chain_rich'):
                    rich_total += 1
                    if d > 1e-9:
                        rich_improved += 1
            if deltas:
                rows.append({
                    'Q': Q, 'N': N, 'version': ver, 'n': len(deltas),
                    'n_improved': sum(1 for d in deltas if d > 1e-9),
                    'mean_dsp': round(sum(deltas) / len(deltas), 5),
                    'max_dsp': round(max(deltas), 5),
                    'gain_chainrich_ratio':
                        f'{rich_improved}/{rich_total}' if rich_total else '',
                })
    if rows:
        with open(os.path.join(out, 'version_gain.csv'), 'w',
                  newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'gain stats -> version_gain.csv ({len(rows)} rows)')


if __name__ == '__main__':
    main()
