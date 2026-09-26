# 最终形态: 单进程顺序 enrich(无 subprocess/无竞态/无 bash)
import sys, json, os, time
os.environ['A2026_ROOT'] = r'C:/shumo_live/02_求解/A题_2026'
N5 = os.environ['A2026_ROOT'] + '/n5_push'
sys.path.insert(0, N5)
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p1, ev_p2, ev_p3
from ops_standalone import Ops
import baselines as B4
import baseline_hinted as BH
import baseline_extra as BX

OUTJ = N5 + '/baseline_full_metrics.jsonl'
PLANS = N5 + '/baseline_plans'
os.makedirs(PLANS, exist_ok=True)
ROOT = os.environ['A2026_ROOT']


def build_plan(case, q, algo):
    g = load_case(case)
    ops = Ops(g)
    real = [o['id'] for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT')]
    t0 = time.perf_counter()
    if algo == 'topo_equal':
        plan = B4.b_topo_equal(g, ops, real)
    elif algo == 'greedy_balance':
        plan = B4.b_greedy_balance(g, ops, real, ops.op_cycle)
    elif algo == 'heft_like':
        plan = B4.b_heft(g, ops, real, ops.op_cycle)
    elif algo == 'kl_partition':
        plan = B4.b_kl(g, ops, real, ops.op_cycle)
    elif algo == 'pipe_balance':
        plan = BH.b2_pipe_balance(g, ops, real)
    elif algo == 'shared_input':
        plan = BH.b1_shared_input(g, ops, real)
    elif algo == 'merge_greedy':
        ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[q]
        class _O:
            def __init__(s2, gg): s2.g = gg
            def evaluate(s2, p): return (ev(s2.g, p)[0]['makespan'], {})
        plan = BH.b3_merge_greedy(g, ops, real, q, _O(g))
    elif algo == 'random_search':
        ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[q]
        class _O2:
            def __init__(s2, gg): s2.g = gg
            def evaluate(s2, p):
                try:
                    return (ev(s2.g, p)[0]['makespan'], {})
                except Exception:
                    return (1e12, {})
        plan = BX.b5_random(g, ops, real, q, _O2(g))
        if plan is None:
            n = len(real)
            plan = BX.plan_from_core(g, real, {o: i * 5 // n for i, o in enumerate(real)}, ops)
    elif algo == 'core_resident':
        plan = BX.b6_core_resident(g, ops, real, q, None)
    else:
        return None, 0
    return plan, time.perf_counter() - t0


targets = set()
for fn, algos in (('baseline_results.jsonl', ('topo_equal', 'greedy_balance', 'heft_like', 'kl_partition')),
                  ('baseline_hinted.jsonl', ('shared_input', 'pipe_balance', 'merge_greedy')),
                  ('baseline_extra.jsonl', ('random_search', 'core_resident'))):
    fp = os.path.join(N5, fn)
    if not os.path.exists(fp):
        continue
    for l in open(fp, encoding='utf-8'):
        if not l.strip().startswith('{'):
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if 'sp' in r and r.get('algo') in algos:
            targets.add((r['case'], int(r['q']), r['algo']))

done = set()
if os.path.exists(OUTJ):
    for l in open(OUTJ, encoding='utf-8'):
        try:
            d = json.loads(l)
            if 'mk' in d:
                done.add(d['key'])
        except Exception:
            pass
print(f'targets {len(targets)} done {len(done)}', flush=True)
for case, q, algo in sorted(targets):
    key = f'{case}|{q}|{algo}'
    if key in done:
        continue
    try:
        plan, dt = build_plan(case, q, algo)
        g = load_case(case)
        ev = {1: ev_p1, 2: ev_p2, 3: ev_p3}[q]
        fb = False
        try:
            ev(g, plan)
        except Exception:
            fb = True
            plan, dt2 = build_plan(case, q, 'topo_equal')
            dt += dt2
        r, wt = ev(g, plan)
        dm = r.get('data_movement_bytes', {}) or {}
        cs = r.get('cache_stats', {}) or {}
        peaks = r.get('memory_peak_by_core') or {}
        def pm(p):
            try:
                vs = []
                for v in (p.values() if isinstance(p, dict) else p):
                    if isinstance(v, (int, float)): vs.append(v)
                    elif isinstance(v, dict): vs += [x for x in v.values() if isinstance(x, (int, float))]
                return max(vs) if vs else None
            except Exception:
                return None
        sc = json.load(open(f'{ROOT}/results/singlecore/{case}_sc.json'))['makespan']
        json.dump(plan, open(f'{PLANS}/{case}_q{q}_{algo}.json', 'w'))
        rec = {'key': key, 'case': case, 'q': q, 'algo': algo, 'mk': r['makespan'],
               'sp': round(sc / r['makespan'], 4),
               'added_bytes': dm.get('added_copy_bytes'), 'spill_bytes': dm.get('spill_added_copy_bytes'),
               'partition_bytes': dm.get('partition_added_copy_bytes'),
               'orig_bytes': dm.get('original_graph_copy_bytes'), 'hit_rate': cs.get('hit_rate'),
               'mem_peak_max': pm(peaks), 'solve_s': round(dt, 2), 'eval_s': round(wt, 2),
               'fallback': fb}
    except Exception as e:
        rec = {'key': key, 'status': 'ERR:' + str(e)[:60]}
    with open(OUTJ, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    print(key, str(rec.get('sp', rec.get('status', '')))[:24], flush=True)
print('SEQ_FINAL_DONE', flush=True)
