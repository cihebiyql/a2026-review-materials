# mechA v2: 冠军种子(全源 sp 最高者) + 细 B 扫描; 支持 q1/q2/q3
import sys, json, os, glob
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/n5_push/superlinear_analysis')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from phase3_mechA2 import comp_depth, relabel
from common import load_case, ev_p1, ev_p2, ev_p3
from fast_eval_p1 import FastEvalP1
from fast_eval_p2 import FastEvalP2, FastEvalP3

BASE = r'C:/shumo_live/02_求解/A题_2026/n5_push'
AUD = r'C:/shumo_live/02_求解/A题_2026/audit_20260924_latest'


def champion_seed(case, q):
    """全源 sp 最高的方案文件(含 mechA 自身上轮产出)。"""
    best = (0.0, None)
    pats = [f'{BASE}/superlinear_analysis/{case}_q{q}_mechA.json',
            f'{BASE}/split_probe_out/{case}_q{q}_best.json',
            f'{BASE}/gpu_search_out/{case}_q{q}_gpu.json']
    for sub in ('refined2', 'refined', 'strand_n5'):
        pats.append(f'{BASE}/{sub}/{case}_q{q}_*.json')
    pats.append(f'{BASE}/pull_plans/{case}_q{q}_pull.json')
    pats.append(f'{BASE}/p1_refine_out/{case}_q1_refined.json')
    for pat in pats:
        for fp in glob.glob(pat):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            sp = d.get('sp')
            if sp and sp > best[0] and ('plan' in d):
                best = (sp, d['plan'])
    return best[1]


def work(case, q):
    try:
        plan0 = champion_seed(case, q)
        if plan0 is None:
            return f'{case}: no_seed'
        graph = load_case(case)
        if q == 1:
            fe = FastEvalP1(graph); ev = ev_p1
        elif q == 2:
            fe = FastEvalP2(graph); ev = ev_p2
        else:
            fe = FastEvalP3(graph); ev = ev_p3
        sc = json.load(open(rf'C:/shumo_live/02_求解/A题_2026/results/singlecore/{case}_sc.json'))['makespan']
        mk0, _ = fe.evaluate(plan0)
        comp_of, depth = comp_depth(graph)
        core_of_sg = {sg: c for c, sgl in enumerate(plan0['core_schedules']) for sg in sgl}
        op_core = {op: core_of_sg[sg] for op, sg in {int(k): v for k, v in plan0['node_to_subgraph'].items()}.items()}
        best = (mk0, None)
        for B in range(4, 132, 4):
            plb = relabel(plan0, op_core, comp_of, depth, B)
            mk1, _ = fe.evaluate(plb)
            if mk1 < best[0]:
                best = (mk1, plb)
        mk_best, pl_best = best
        if pl_best is None or mk_best >= mk0 - 0.5:
            return f'{case}: no_gain'
        mk_off = ev(graph, pl_best)[0]['makespan']
        if mk_off == mk_best:
            json.dump({'plan': pl_best, 'mk': mk_off, 'sp': sc/mk_off, 'source': f'mechA_v2_q{q}'},
                      open(rf'{BASE}/superlinear_analysis/{case}_q{q}_mechA.json', 'w'))
            return f'{case}: GAIN mk {mk0}->{mk_off} sp {sc/mk_off:.4f}'
        return f'{case}: mismatch'
    except Exception as e:
        return f'{case}: ERR {str(e)[:60]}'


if __name__ == '__main__':
    mode = sys.argv[1]
    q = int(sys.argv[2])
    if mode == 'one':
        print(work(sys.argv[3], q), flush=True)
    else:
        cases = [l.strip() for l in open(rf'{BASE}/mechA_p3_remaining.txt')] if mode == 'remaining' else \
                [f'case_{i:03d}' for i in range(1, 101)]
        for c in cases:
            print(work(c, q), flush=True)
