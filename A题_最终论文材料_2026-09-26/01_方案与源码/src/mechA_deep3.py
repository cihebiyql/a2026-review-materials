# mechA 深化: B 步长1 细扫 × 层窗口 L(相邻层合并), 对已 GAIN 例压榨
import sys, json, os
from collections import defaultdict
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/n5_push/superlinear_analysis')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from phase3_mechA2 import comp_depth
from common import load_case, ev_p2, ev_p3
from fast_eval_p2 import FastEvalP2, FastEvalP3

BASE = r'C:/shumo_live/02_求解/A题_2026/n5_push'


def relabel_L(plan, op_core, comp_of, depth, B, L):
    core_ops = defaultdict(list)
    for op, c in op_core.items():
        core_ops[c].append(op)
    n_cores = len(plan['core_schedules'])
    new_n2s, new_cs = {}, [[] for _ in range(n_cores)]
    sgid = 0
    for c in range(n_cores):
        ops = core_ops.get(c, [])
        comps = defaultdict(list)
        for o in ops:
            comps[comp_of[o]].append(o)
        order = sorted(comps, key=lambda k: min(comps[k]))
        batch, size = [], 0
        batches = []
        for k in order:
            sz = len(comps[k])
            if batch and size + sz > B:
                batches.append(batch)
                batch, size = [], 0
            batch.append(k)
            size += sz
        if batch:
            batches.append(batch)
        for batch in batches:
            bset = set(batch)
            bops = [o for o in ops if comp_of[o] in bset]
            maxd = max(depth[o] for o in bops)
            lev = 0
            while lev <= maxd:
                lops = [o for o in bops if lev <= depth[o] < lev + L]
                if lops:
                    sgid += 1
                    for op in lops:
                        new_n2s[str(op)] = sgid
                    new_cs[c].append(sgid)
                lev += L
    return {'node_to_subgraph': new_n2s, 'core_schedules': new_cs}


def deep(case, q):
    fp = rf'{BASE}/superlinear_analysis/{case}_q{q}_mechA.json'
    if not os.path.exists(fp):
        return f'{case}: no_prior'
    d = json.load(open(fp))
    plan0 = d['plan']
    graph = load_case(case)
    fe = FastEvalP2(graph) if q == 2 else FastEvalP3(graph)
    ev = ev_p2 if q == 2 else ev_p3
    sc = json.load(open(rf'C:/shumo_live/02_求解/A题_2026/results/singlecore/{case}_sc.json'))['makespan']
    mk0, _ = fe.evaluate(plan0)
    comp_of, depth = comp_depth(graph)
    core_of_sg = {sg: c for c, sgl in enumerate(plan0['core_schedules']) for sg in sgl}
    op_core = {op: core_of_sg[sg] for op, sg in {int(k): v for k, v in plan0['node_to_subgraph'].items()}.items()}
    best = (mk0, None, None)
    for L in (4, 5, 6, 8, 12):
        for B in range(2, 400):
            try:
                plb = relabel_L(plan0, op_core, comp_of, depth, B, L)
                mk1, _ = fe.evaluate(plb)
                if mk1 < best[0]:
                    best = (mk1, plb, (B, L))
            except Exception:
                continue
    mk_best, pl_best, BL = best
    if pl_best is None or mk_best >= mk0 - 0.5:
        return f'{case}: no_gain ({mk0}->{mk_best})'
    mk_off = ev(graph, pl_best)[0]['makespan']
    if mk_off == mk_best:
        json.dump({'plan': pl_best, 'mk': mk_off, 'sp': sc/mk_off,
                   'source': f'mechA_deep B={BL[0]} L={BL[1]}'}, open(fp, 'w'))
        return f'{case}: GAIN mk {mk0}->{mk_off} sp {sc/mk_off:.4f} (B={BL[0]} L={BL[1]})'
    return f'{case}: mismatch'


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        case, q = arg.split(':')
        print(deep(case, int(q)), flush=True)
