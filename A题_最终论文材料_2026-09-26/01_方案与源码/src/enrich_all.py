# 全指标重测驱动: 对每个基准算法的每 case 重新构造方案(确定性) -> 官方评估器评测 ->
# 记录题面全部指标: Makespan / added_copy_bytes / spill / partition / cross_task /
# cache_hit_rate(q3) / memory_peak_by_core / 求解时间 / 最终方案落盘
import sys, json, os, time, importlib
from collections import defaultdict

ROOT = r'C:/shumo_live/02_求解/A题_2026'
N5 = ROOT + '/n5_push'
sys.path.insert(0, N5)
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p1, ev_p2, ev_p3
from ops_standalone import Ops

OUTJ = N5 + '/baseline_full_metrics.jsonl'
PLANS_DIR = N5 + '/baseline_plans'
os.makedirs(PLANS_DIR, exist_ok=True)

# 加载各基准的构造器(main 保护后 import 安全)
import baselines as B4
import baseline_hinted as BH
import baseline_extra as BX


def build_plan(case, q, algo):
    """确定性重建基准方案。返回 (plan, solve_s) 或 None。"""
    g = load_case(case)
    ops = Ops(g)
    real = [o['id'] for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT')]
    op_cycle = ops.op_cycle
    t0 = time.perf_counter()
    plan = None
    if algo == 'topo_equal':
        plan = B4.b_topo_equal(g, ops, real)
    elif algo == 'greedy_balance':
        plan = B4.b_greedy_balance(g, ops, real, op_cycle)
    elif algo == 'heft_like':
        plan = B4.b_heft(g, ops, real, op_cycle)
    elif algo == 'kl_partition':
        plan = B4.b_kl(g, ops, real, op_cycle)
    elif algo == 'pipe_balance':
        plan = BH.b2_pipe_balance(g, ops, real)
    elif algo == 'shared_input':
        plan = BH.b1_shared_input(g, ops, real)
    elif algo == 'merge_greedy':
        class _OffFE:
            def __init__(self, gg):
                self.g = gg
            def evaluate(self, p):
                ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[q]
                mk = ev(self.g, p)[0]['makespan']
                return (mk, {})
        plan = BH.b3_merge_greedy(g, ops, real, q, _OffFE(g))
    elif algo == 'random_search':
        class _OffFE2:
            def __init__(self, gg):
                self.g = gg
            def evaluate(self, p):
                ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[q]
                mk = ev(self.g, p)[0]['makespan']
                return (mk, {})
        plan = BX.b5_random(g, ops, real, q, _OffFE2(g))
        if plan is None:
            plan = BX.plan_from_core(g, real, {o: i * 5 // len(real) for i, o in enumerate(real)}, ops)
    elif algo == 'core_resident':
        plan = BX.b6_core_resident(g, ops, real, q, None)
    elif algo == 'genetic':
        # GA 太慢不重构造: 用记录里的 mk 直接标(方案另存流程见 baseline_ga 的扩展)
        return None
    dt = time.perf_counter() - t0
    return (plan, dt)


# 目标行来源: 各 jsonl 里已成功的 (case,q,algo)
targets = set()
for fn, algos in ((N5 + '/baseline_results.jsonl', ('topo_equal', 'greedy_balance', 'heft_like', 'kl_partition')),
                  (N5 + '/baseline_hinted.jsonl', ('shared_input', 'pipe_balance', 'merge_greedy')),
                  (N5 + '/baseline_extra.jsonl', ('random_search', 'core_resident'))):
    if not os.path.exists(fn):
        continue
    for l in open(fn, encoding='utf-8'):
        if not l.strip().startswith('{'):
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if 'sp' in r and r.get('algo') in algos:
            targets.add((r['case'], r['q'], r['algo']))

done = set()
if os.path.exists(OUTJ):
    for l in open(OUTJ, encoding='utf-8'):
        try:
            d = json.loads(l)
            if 'mk' in d:
                done.add(d['key'])
        except Exception:
            pass

ONE = os.environ.get('ENRICH_ONE').strip() if os.environ.get('ENRICH_ONE') else None
print(f'enrich targets: {len(targets)}, done: {len(done)}', flush=True)
for (case, q, algo) in ([(ONE.split('|')[0], int(ONE.split('|')[1]), ONE.split('|')[2])] if ONE else sorted(targets)):
    key = f'{case}|q{q}|{algo}'
    if key in done:
        continue
    try:
        res = build_plan(case, q, algo)
        if res is None:
            rec = {'key': key, 'status': 'skip_ga'}
        else:
            plan, dt = res
            g = load_case(case)
            ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[q]
            fb = False
            try:
                ev(g, plan)
            except Exception:
                fb = True
                plan = build_plan(case, q, 'topo_equal')[0]
                dt += build_plan(case, q, 'topo_equal')[1]
            r, wt = ev(g, plan)
            mk = r['makespan']
            dm = r.get('data_movement_bytes', {}) or {}
            cs = r.get('cache_stats', {}) or {}
            peaks = r.get('memory_peak_by_core') or {}
            def _peak_max(p):
                try:
                    vals = []
                    for v in (p.values() if isinstance(p, dict) else p):
                        if isinstance(v, (int, float)): vals.append(v)
                        elif isinstance(v, dict): vals += [x for x in v.values() if isinstance(x, (int, float))]
                    return max(vals) if vals else None
                except Exception:
                    return None
            sc = json.load(open(f'{ROOT}/results/singlecore/{case}_sc.json'))['makespan']
            # 方案落盘(每 case 记录要求)
            json.dump(plan, open(f'{PLANS_DIR}/{case}_q{q}_{algo}.json', 'w'))
            rec = {'key': key, 'case': case, 'q': q, 'algo': algo,
                   'mk': mk, 'sp': round(sc / mk, 4),
                   'added_bytes': dm.get('added_copy_bytes'),
                   'spill_bytes': dm.get('spill_added_copy_bytes'),
                   'partition_bytes': dm.get('partition_added_copy_bytes'),
                   'orig_bytes': dm.get('original_graph_copy_bytes'),
                   'hit_rate': cs.get('hit_rate'),
                   'mem_peak_max': _peak_max(peaks),
                   'solve_s': round(dt, 2), 'eval_s': round(wt, 2), 'fallback': fb}
    except Exception as e:
        rec = {'key': key, 'case': case, 'q': q, 'algo': algo, 'status': 'ERR:' + str(e)[:60]}
    with open(OUTJ, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(rec, ensure_ascii=False, default=str)[:200], flush=True)
print('ENRICH_DONE', flush=True)
