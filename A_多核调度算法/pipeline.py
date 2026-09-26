# -*- coding: utf-8 -*-
"""pipeline.py — 单用例全版本主流水线。

对一个 (case, Q) 依次在 N∈N_LIST 上运行版本阶梯 V1→V4（严格串行，
Vk 以 V(k-1) 的最优方案为种子 —— 全部从输入图自举, 全例统一协议）：
  V1 凸块基线   : 拓扑序+连续凸块+周期均衡+轮询分核（纯构造）
  V2 +B×L 重标  : 590 候选网格深扫
  V3 +三结构    : 144 候选（均衡装箱/跨核相位/层切分DP）
  V4 +学习指派  : REINFORCE 50轮×128采样（解码器生成新核指派）
每版本记录：fast_mk（搜索真值）、official_mk（官方终验）、sp、
added/hit_rate/无L2 指标行、solve_s（算法运行时间，赛题"合理时间"口径）。
逐行 append 到 results/runs/{case}_q{Q}.jsonl；终选方案落盘 results/plans/。
"""
import json
import os
import time

import paths
from s1_preprocess.dag_view import load_case, op_dag, cycles_map, pipe_map
from s2_features.graph_features import comp_depth, graph_summary
from s8_evaluate.fasteval_adapter import make_fe
from s8_evaluate.official import ev_timed

N_LIST = [5, 4, 3, 2]


def _emit(rec, case, Q, graph, plan, fast_mk, sc):
    """统一落盘：官方终验 + 指标行 + jsonl append + 方案保存。"""
    rec.update(ev_timed(Q, graph, plan))
    rec['official_match'] = (rec.get('official_mk') == fast_mk)
    os.makedirs(os.path.join(paths.RESULTS, 'runs'), exist_ok=True)
    with open(os.path.join(paths.RESULTS, 'runs',
                           f'{case}_q{Q}.jsonl'), 'a',
              encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    d = os.path.join(paths.RESULTS, 'plans')
    os.makedirs(d, exist_ok=True)
    ver, N = rec['version'], rec['N']
    json.dump({'plan': plan, 'mk': fast_mk,
               'source': f"pipeline {ver} Q={Q} N={N}"},
              open(os.path.join(d, f'{case}_q{Q}_N{N}_{ver}.json'), 'w'))


def _seed(best, case, Q, N, ver):
    """取上一版本种子；独立运行 vk 时从磁盘方案加载（缺则逐级回退）。"""
    if N in best:
        return best[N]
    plans = os.path.join(paths.RESULTS, 'plans')
    order = {'v2': ['v1'], 'v3': ['v2', 'v1'], 'v4': ['v3', 'v2', 'v1']}[ver]
    for v in order:
        fp = os.path.join(plans, f'{case}_q{Q}_N{N}_{v}.json')
        if os.path.exists(fp):
            d = json.load(open(fp))
            return d['plan'], d['mk']
    raise FileNotFoundError(f'no seed plan for {case} q{Q} N{N}')


def run_case_versions(case, Q, n_list=None, versions=('v1', 'v2', 'v3', 'v4'),
                      v4_cfg=None):
    n_list = n_list or N_LIST
    graph = load_case(case)
    ids, preds, succs = op_dag(graph)
    cycles = cycles_map(graph)
    comp_raw, depth = comp_depth(ids, preds, succs)
    summary = graph_summary(graph, comp_raw, depth)
    fe = make_fe(Q, graph)
    sc = paths.sc_makespan(case)

    def base_rec(ver, N, mk, solve_s, status='ok', **extra):
        r = {'case': case, 'Q': Q, 'N': N, 'version': ver,
             'fast_mk': mk, 'sp': sc / mk, 'solve_s': solve_s,
             'graph': summary, 'status': status}
        r.update(extra)
        return r

    best = {}
    for N in n_list:
        t0 = time.time()
        from s4_partition.convex_cut import convex_plan
        plan = convex_plan(ids, preds, succs, cycles, pipe_map(graph), N,
                           chunks_per_core=int(os.environ.get('PIPELINE_CPC', 8)))
        mk, _ = fe.evaluate(plan)
        _emit(base_rec('v1', N, mk, round(time.time() - t0, 2)),
              case, Q, graph, plan, mk, sc)
        best[N] = (plan, mk)

    if 'v2' in versions:
        from s7_search.sweep_grid import sweep_bl
        from s3_units.topo_units import op_core_of
        for N in n_list:
            seed_plan, seed_mk = _seed(best, case, Q, N, 'v1')
            t0 = time.time()
            try:
                op_core = op_core_of(seed_plan)
                mk, plan, cfg, n_ev = sweep_bl(
                    fe, seed_plan, N, op_core, comp_raw, depth)
                status = 'ok'
            except Exception as e:
                mk, plan, cfg, n_ev = seed_mk, seed_plan, None, 0
                status = 'FAIL:' + str(e)[:80]
            _emit(base_rec('v2', N, mk, round(time.time() - t0, 2), status,
                           cfg=cfg, n_eval=n_ev,
                           improved=bool(mk < seed_mk)),
                  case, Q, graph, plan, mk, sc)
            best[N] = (plan, mk)

    if 'v3' in versions:
        from s7_search.sweep_grid import sweep_v4, pipe01_map
        from s3_units.topo_units import op_core_of
        pipe_of = pipe01_map(graph)
        for N in n_list:
            seed_plan, seed_mk = _seed(best, case, Q, N, 'v2')
            t0 = time.time()
            try:
                op_core = op_core_of(seed_plan)
                mk, plan, cfg, n_ev = sweep_v4(
                    fe, seed_plan, N, op_core, comp_raw, depth, pipe_of)
                status = 'ok'
            except Exception as e:
                mk, plan, cfg, n_ev = seed_mk, seed_plan, None, 0
                status = 'FAIL:' + str(e)[:80]
            _emit(base_rec('v3', N, mk, round(time.time() - t0, 2), status,
                           cfg=cfg, n_eval=n_ev,
                           improved=bool(mk < seed_mk)),
                  case, Q, graph, plan, mk, sc)
            best[N] = (plan, mk)

    if 'v4' in versions:
        from s7_search.reinforce import train
        cfg4 = dict(v4_cfg or {})
        cfg4.setdefault('workers', int(os.environ.get('PIPELINE_V4_WORKERS', 16)))
        cfg4.setdefault('iters', int(os.environ.get('PIPELINE_V4_ITERS', 50)))
        for N in n_list:
            seed_plan, seed_mk = _seed(best, case, Q, N, 'v3')
            t0 = time.time()
            try:
                mk, plan, best_B, trec = train(
                    fe, graph, seed_plan, Q,
                    iters=cfg4.get('iters', 50),
                    N=cfg4.get('N', 128),
                    workers=cfg4.get('workers', 16),
                    seed=cfg4.get('seed', 0))
                status = 'ok'
            except Exception as e:
                mk, plan, best_B, trec = (seed_mk, seed_plan, None,
                                          {'err': str(e)[:120]})
                status = 'FAIL:' + str(e)[:80]
            _emit(base_rec('v4', N, mk, round(time.time() - t0, 2), status,
                           best_B=best_B, improved=bool(mk < seed_mk),
                           train=trec),
                  case, Q, graph, plan, mk, sc)
    return True


def run_case_versions_q(case, Q, **kw):
    """子进程入口：单 (case,Q)，全 N、全版本；返回状态串。"""
    try:
        run_case_versions(case, Q, **kw)
        return f'{case}_q{Q} OK'
    except Exception as e:
        import traceback
        return f'{case}_q{Q} ERR:' + repr(e)[:120] + '|' + \
            traceback.format_exc()[-200:]


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('case')
    ap.add_argument('Q', type=int)
    ap.add_argument('--nlist', default='5,4,3,2')
    ap.add_argument('--versions', default='v1,v2,v3,v4')
    a = ap.parse_args()
    print(run_case_versions_q(
        a.case, a.Q,
        n_list=[int(x) for x in a.nlist.split(',')],
        versions=tuple(a.versions.split(','))), flush=True)
