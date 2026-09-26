import sys, json, os
sys.path.insert(0, 'superlinear_analysis')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from phase3_mechA2 import comp_depth, relabel
from common import load_case, ev_p3
from fast_eval_p2 import FastEvalP3

def seed_of(case):
    for sub in ('refined2', 'refined', 'strand_n5'):
        fp = os.path.join(r'C:/shumo_live/02_求解/A题_2026/n5_push', sub, f'{case}_q3_N5.json')
        if os.path.exists(fp):
            d = json.load(open(fp))
            if 'plan' in d: return d['plan']
    fp = os.path.join(r'C:/shumo_live/02_求解/A题_2026/n5_push/pull_plans', f'{case}_q3_pull.json')
    if os.path.exists(fp):
        d = json.load(open(fp))
        return d['plan'] if 'plan' in d else d
    return None

def work(case):
    try:
        plan0 = seed_of(case)
        if plan0 is None: return f'{case}: no_seed'
        graph = load_case(case)
        fe = FastEvalP3(graph)
        sc = json.load(open(rf'C:/shumo_live/02_求解/A题_2026/results/singlecore/{case}_sc.json'))['makespan']
        mk0, _ = fe.evaluate(plan0)
        comp_of, depth = comp_depth(graph)
        core_of_sg = {sg: c for c, sgl in enumerate(plan0['core_schedules']) for sg in sgl}
        op_core = {op: core_of_sg[sg] for op, sg in {int(k): v for k, v in plan0['node_to_subgraph'].items()}.items()}
        best = (mk0, None)
        for B in (8, 16, 24, 32, 48, 64, 96):
            plb = relabel(plan0, op_core, comp_of, depth, B)
            mk1, _ = fe.evaluate(plb)
            if mk1 < best[0]: best = (mk1, plb)
        mk_best, pl_best = best
        if pl_best is None or mk_best >= mk0 - 0.5:
            return f'{case}: no_gain'
        mk_off = ev_p3(graph, pl_best)[0]['makespan']
        if mk_off == mk_best:
            json.dump({'plan': pl_best, 'mk': mk_off, 'sp': sc/mk_off, 'source': 'mechA_p3'},
                      open(rf'C:/shumo_live/02_求解/A题_2026/n5_push/superlinear_analysis/{case}_q3_mechA.json', 'w'))
            return f'{case}: GAIN mk {mk0}->{mk_off} sp {sc/mk_off:.4f}'
        return f'{case}: mismatch'
    except Exception as e:
        return f'{case}: ERR {str(e)[:60]}'

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'one':
        print(work(sys.argv[2]), flush=True)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == 'remaining':
        cases = [l.strip() for l in open('mechA_p3_remaining.txt') if l.strip()]
    else:
        shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[1] != 'one' else (0, 1)
        cases = [f'case_{i:03d}' for i in range(1, 101) if i % nshard == shard]
    for c in cases:
        print(work(c), flush=True)
