# -*- coding: utf-8 -*-
"""s8_evaluate.official — 官方评估器终验与指标行。

ev(Q, graph, plan)：调用未改动的官方评估脚本（库方式），返回统一指标行：
  mk / added_copy_bytes / (q3) hit_rate、无 L2 配置 mk/added
耗时另计（官方评估属裁判环节，不计入算法运行时间）。
"""
import paths  # noqa: F401
import time


def ev(Q, graph, plan):
    from official_eval import ev_p1, ev_p2, ev_p3
    ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[Q]
    r, _ = ev(graph, plan)
    dm = r.get('data_movement_bytes', {})
    row = {'official_mk': r['makespan'],
           'added_copy_bytes': dm.get('added_copy_bytes')}
    if Q == 3:
        cs = r.get('cache_stats') or {}
        row['hit_rate'] = cs.get('hit_rate')
        r2, _ = ev_p2(graph, plan)          # 同方案无 L2 配置
        row['mk_nol2'] = r2['makespan']
        row['added_nol2'] = r2.get(
            'data_movement_bytes', {}).get('added_copy_bytes')
        row['cache_speedup'] = (row['mk_nol2'] / row['official_mk']
                                if row['official_mk'] else None)
    return row


def ev_timed(Q, graph, plan):
    t0 = time.time()
    row = ev(Q, graph, plan)
    row['official_eval_s'] = round(time.time() - t0, 2)
    return row
