# 014/040 专用:官方评估器直评(FastEval 在此类大图有堆破坏bug),粗网格
import sys, json, os
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/n5_push/superlinear_analysis')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from phase3_mechA2 import comp_depth
from common import load_case, ev_p3
from mechA_v3 import relabel_v3
import glob

BASE = r'C:/shumo_live/02_求解/A题_2026/n5_push'
for case in ('case_014', 'case_040'):
    plan0 = None; best_sp = 0
    for sub in ('refined2', 'refined', 'strand_n5'):
        for fp in glob.glob(f'{BASE}/{sub}/{case}_q3_*.json'):
            d = json.load(open(fp))
            if d.get('sp', 0) > best_sp and 'plan' in d:
                best_sp, plan0 = d['sp'], d['plan']
    for fp in glob.glob(f'{BASE}/superlinear_analysis/{case}_q3_mechA.json') + glob.glob(f'{BASE}/pull_plans/{case}_q3_pull.json'):
        d = json.load(open(fp))
        if d.get('sp', 0) > best_sp and 'plan' in d:
            best_sp, plan0 = d['sp'], d['plan']
    if plan0 is None:
        print(case, 'no_seed'); continue
    graph = load_case(case)
    sc = json.load(open(rf'C:/shumo_live/02_求解/A题_2026/results/singlecore/{case}_sc.json'))['makespan']
    mk0 = ev_p3(graph, plan0)[0]['makespan']
    comp_of, depth = comp_depth(graph)
    core_of_sg = {sg: c for c, sgl in enumerate(plan0['core_schedules']) for sg in sgl}
    op_core = {op: core_of_sg[sg] for op, sg in {int(k): v for k, v in plan0['node_to_subgraph'].items()}.items()}
    best = (mk0, None, None)
    for om in ('min_id', 'size_desc'):
        for L in (3, 4):
            for B in range(8, 384, 16):
                plb = relabel_v3(plan0, op_core, comp_of, depth, B, L, om)
                try:
                    mk1 = ev_p3(graph, plb)[0]['makespan']
                except Exception:
                    continue
                if mk1 < best[0]:
                    best = (mk1, plb, (om, B, L))
    mk_best, pl_best, P = best
    print(f'{case}: mk {mk0} -> {mk_best} P={P} sp {sc/mk0:.4f} -> {sc/mk_best:.4f}', flush=True)
    if pl_best is not None and mk_best < mk0 - 0.5:
        json.dump({'plan': pl_best, 'mk': mk_best, 'sp': sc/mk_best, 'source': f'crasher_official {P}'},
                  open(f'{BASE}/superlinear_analysis/{case}_q3_mechA.json', 'w'))
